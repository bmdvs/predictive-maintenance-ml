"""Work requests on ALARM with a suspected condition from the indicator pattern (rule-based diagnosis).

Rule thresholds are z-scores against the point's healthy spread per indicator and regime (median and
robust scale of the point's model-training hours, log scale for energies). Inputs: cases, the 24-hour
median indicators of the triggering point at case opening, the point's healthy statistics and
controller signals. Output: one work request per case with the suspected condition and component,
the triggering indicators against baseline, the recommended inspection and the shadow-mode flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

INSPECTION = {
    "bearing": "listen, envelope check, temperature check",
    "lubrication": "temperature check, spindle lube check",
    "contamination": "pressure check, inspect rotary union and seals",
    "imbalance": "strobe and vibration pen at 1x",
    "misalignment": "coupling and laser alignment check",
    "looseness": "check base and mounting bolts",
    "cavitation or wear": "coolant level and pressure check",
    "cooler": "check spindle chiller and coolant flow",
    "axis wear": "backlash and axis load check",
    "unclassified": "listen, temperature check",
}
FAULT_LOCATION = {"bpfo": "outer race", "bpfi": "inner race", "bsf": "ball"}
Z_INDICATORS = {
    "env_bpfo": "env_bpfo_1_g2",
    "env_bpfi": "env_bpfi_1_g2",
    "env_bsf": "env_bsf_1_g2",
    "ord_01": "ord_01_g2",
    "ord_02": "ord_02_g2",
    "ord_3_10": "ord_3_10_g2",
    "hf": "hf_rms_g",
    "vel": "vel_rms_mm_s",
}


def log_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ord_3_10_g2"] = d[[f"ord_{o:02d}_g2" for o in range(3, 11)]].sum(axis=1)
    out = pd.DataFrame(index=d.index)
    for k, c in Z_INDICATORS.items():
        out[k] = np.log10(d[c].astype(float) + 1e-12)
    return out


def healthy_stats(train_rows: pd.DataFrame) -> pd.DataFrame:
    """Median and robust scale (MAD x 1.4826) per point, regime and indicator from training hours."""
    li = log_indicators(train_rows).assign(
        point_id=train_rows["point_id"].to_numpy(), regime=train_rows["regime"].to_numpy()
    )
    rows = []
    for (pid, reg), g in li.groupby(["point_id", "regime"]):
        for k in Z_INDICATORS:
            v = g[k].dropna().to_numpy()
            if v.size < 20:
                continue
            med = float(np.median(v))
            sc = float(1.4826 * np.median(np.abs(v - med))) or float(np.std(v)) or 1.0
            rows.append(
                {"point_id": pid, "regime": reg, "indicator": k, "median": med, "scale": sc, "n": int(v.size)}
            )
    return pd.DataFrame(rows)


def zscores(recent: pd.DataFrame, stats: pd.DataFrame, point_id: str) -> dict[str, float]:
    """z per indicator for the 24-hour window before a case opened, in its dominant regime."""
    if recent.empty:
        return {}
    reg = recent["regime"].mode().iat[0]
    li = log_indicators(recent).median()
    st = stats[(stats["point_id"] == point_id) & (stats["regime"] == reg)].set_index("indicator")
    if st.empty:
        st = stats[stats["point_id"] == point_id].groupby("indicator")[["median", "scale"]].median()
    return {
        k: float((li[k] - st.loc[k, "median"]) / st.loc[k, "scale"])
        for k in Z_INDICATORS
        if k in st.index and pd.notna(li[k])
    }


def suspected(
    z: dict[str, float], point_class: str | None, controller: dict[str, Any], zthr: float
) -> tuple[str, str]:
    """Condition and detail from z-scores against the point's healthy spread and controller signals."""
    aux = bool(point_class and not point_class.startswith("spindle"))
    pump = bool(point_class and point_class.endswith("pump"))
    if controller.get("thermistor_residual_alert") and not aux:
        return "cooler", f"thermistor residual {controller.get('thermistor_residual_c', np.nan):.1f} C"
    env = {k: z.get(f"env_{k}", np.nan) for k in FAULT_LOCATION}
    env_hit = {k: v for k, v in env.items() if pd.notna(v) and v >= zthr}
    if env_hit:
        k = max(env_hit, key=env_hit.get)
        return "bearing", f"{FAULT_LOCATION[k]} ({k.upper()} z {env_hit[k]:.1f})"
    o1, o2, harm, hf = (z.get(k, np.nan) for k in ("ord_01", "ord_02", "ord_3_10", "hf"))
    if aux and pd.notna(o2) and o2 >= zthr and (pd.isna(o1) or o2 >= o1):
        return "misalignment", f"2x z {o2:.1f}"
    if aux and pd.notna(o1) and o1 >= zthr:
        return "imbalance", f"1x z {o1:.1f}"
    if aux and pd.notna(harm) and harm >= zthr:
        return "looseness", f"3x-10x z {harm:.1f}"
    if pd.notna(hf) and hf >= zthr:
        drop = controller.get("coolant_drop", 0.0)
        if pump and drop >= 0.05:
            return "cavitation or wear", f"HF z {hf:.1f} with pressure loss {drop:.0%}"
        if drop >= 0.05 and point_class == "spindle_front":
            return "contamination", f"HF z {hf:.1f} with TSC pressure drop {drop:.0%}"
        return "lubrication", f"HF z {hf:.1f}, no fault frequency"
    if controller.get("axis_drift_alert"):
        return "axis wear", f"axis load drift {controller.get('axis_drift_max', np.nan):.0%}"
    return "unclassified", ""


def work_requests(
    cases: pd.DataFrame,
    z_at: dict[str, dict[str, float]],
    point_class: dict[str, str],
    controller_at: dict[str, dict[str, Any]],
    shadow: tuple[pd.Timestamp, pd.Timestamp],
    zthr: float,
) -> pd.DataFrame:
    rows = []
    for c in cases.itertuples():
        pts = [p for p in str(c.trigger_points or "").split(",") if p]
        pid = pts[0] if pts else None
        cond, detail = suspected(
            z_at.get(c.case_id, {}),
            point_class.get(pid) if pid else None,
            controller_at.get(c.case_id, {}),
            zthr,
        )
        rows.append(
            {
                "request_id": f"WR-{c.case_id}",
                "case_id": c.case_id,
                "asset_id": c.asset_id,
                "point_id": pid,
                "created": c.opened,
                "suspected_condition": cond,
                "suspected_detail": detail,
                "triggering_indicators": c.trigger_rules,
                "recommended_inspection": INSPECTION[cond],
                "shadow_mode": bool(shadow[0] <= c.opened < shadow[1]),
            }
        )
    return pd.DataFrame(rows)


CONDITION_FAMILY = {
    "bearing (unplanned)": "bearing",
    "bearing (condition-based)": "bearing",
    "lubrication": "lubrication",
    "imbalance": "imbalance",
    "misalignment": "misalignment",
    "looseness": "looseness",
    "rotary union": "contamination",
    "spindle cooling": "cooler",
    "cooler (compressor)": "cooler",
    "pump seal or impeller": "cavitation or wear",
}


def agreement(
    requests: pd.DataFrame, cases: pd.DataFrame, conds: pd.DataFrame, wo: pd.DataFrame
) -> pd.DataFrame:
    """Per work request, how its case ended: a confirmed condition on the asset while the case was open
    (agreement scored against its finding family), no fault found (the closing inspection), or other."""
    cs = cases.set_index("case_id")
    nff = set(wo.loc[(wo["wo_type"] == "inspection") & wo["no_fault_found"].astype(bool), "wo_id"])
    rows = []
    for r in requests.itertuples():
        c = cs.loc[r.case_id]
        closed = c["closed"] if pd.notna(c["closed"]) else pd.Timestamp.max.tz_localize("UTC")
        hit = conds[
            (conds["asset_key"] == r.asset_id)
            & (conds["credit_ts"] >= r.created)
            & (conds["credit_ts"] <= closed)
        ]
        if len(hit):
            first = hit.sort_values("credit_ts").iloc[0]
            fam = CONDITION_FAMILY.get(first["condition_type"])
            rows.append(
                {
                    "request_id": r.request_id,
                    "outcome": "confirmed",
                    "confirmed": first["condition_type"],
                    "suspected": r.suspected_condition,
                    "agree": fam == r.suspected_condition,
                }
            )
        else:
            outcome = "no fault found" if c.get("closing_wo") in nff else "other"
            rows.append(
                {
                    "request_id": r.request_id,
                    "outcome": outcome,
                    "confirmed": None,
                    "suspected": r.suspected_condition,
                    "agree": None,
                }
            )
    return pd.DataFrame(rows)
