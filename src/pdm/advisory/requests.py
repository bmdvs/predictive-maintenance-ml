"""Work requests on ALARM with a suspected condition from the indicator pattern (rule-based diagnosis).

Inputs: cases, the rolling features of the triggering points at the case time, hourly indicators and
baseline rows, controller daily scores and coolant pressure. Output: one work request per case with
the suspected condition and component, triggering indicators against baseline, the recommended
inspection and the shadow-mode flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DB6 = 10 ** (6 / 10)
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


def suspected(
    feat: dict[str, float], point_class: str | None, controller: dict[str, Any], cfg: dict[str, Any]
) -> tuple[str, str]:
    """Condition and detail from 24-hour ratios to baseline and controller signals."""
    env = {k: feat.get(f"env_{k}_1_ratio_24h", np.nan) for k in FAULT_LOCATION}
    env_hit = {k: v for k, v in env.items() if pd.notna(v) and v >= DB6}
    o1, o2 = feat.get("order_1x_ratio_24h", np.nan), feat.get("order_2x_ratio_24h", np.nan)
    hf, harm = feat.get("hf_rms_g_ratio_24h", np.nan), feat.get("ord_3_10_ratio", np.nan)
    pump = bool(point_class and point_class.endswith("pump"))
    rotating_aux = bool(point_class and not point_class.startswith("spindle"))
    if controller.get("thermistor_residual_alert") and not rotating_aux:
        return "cooler", f"thermistor residual {controller.get('thermistor_residual_c', np.nan):.1f} C"
    if env_hit:
        k = max(env_hit, key=env_hit.get)
        return "bearing", f"{FAULT_LOCATION[k]} ({k.upper()} {env_hit[k]:.1f}x baseline)"
    if rotating_aux and pd.notna(o2) and o2 >= DB6 and (pd.isna(o1) or o2 >= o1):
        return "misalignment", f"2x {o2:.1f}x baseline"
    if rotating_aux and pd.notna(o1) and o1 >= DB6:
        return "imbalance", f"1x {o1:.1f}x baseline"
    if rotating_aux and pd.notna(harm) and harm >= DB6:
        return "looseness", f"3x-10x {harm:.1f}x baseline"
    if pd.notna(hf) and hf >= 1.0 + cfg["rules"]["vendor"]["alert_frac_zone_b_width"]:
        if pump and controller.get("coolant_drop", 0) >= 0.05:
            return "cavitation or wear", f"HF {hf:.2f}x baseline with pressure loss"
        if controller.get("coolant_drop", 0) >= 0.05 and point_class == "spindle_front":
            return (
                "contamination",
                f"HF {hf:.2f}x baseline with TSC pressure drop {controller['coolant_drop']:.0%}",
            )
        return "lubrication", f"HF {hf:.2f}x baseline, no fault frequency"
    if controller.get("thermistor_residual_alert"):
        return "cooler", f"thermistor residual {controller.get('thermistor_residual_c', np.nan):.1f} C"
    if controller.get("axis_drift_alert"):
        return "axis wear", f"axis load drift {controller.get('axis_drift_max', np.nan):.0%}"
    return "unclassified", ""


def work_requests(
    cases: pd.DataFrame,
    features_at: dict[str, dict[str, float]],
    point_class: dict[str, str],
    controller_at: dict[str, dict[str, Any]],
    shadow: tuple[pd.Timestamp, pd.Timestamp],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for c in cases.itertuples():
        pts = [p for p in str(c.trigger_points or "").split(",") if p]
        pid = pts[0] if pts else None
        feat = features_at.get(c.case_id, {})
        cond, detail = suspected(
            feat, point_class.get(pid) if pid else None, controller_at.get(c.case_id, {}), cfg
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
