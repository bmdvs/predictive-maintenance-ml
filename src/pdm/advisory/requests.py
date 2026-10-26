"""Work requests on ALARM with a suspected condition from the indicator pattern (rule-based diagnosis).

Diagnosis reads every signal on the asset over the case's open interval: vibration indicators at all of
the asset's points (daily median of hourly z-scores against the point's healthy spread for its segment
and regime) and the controller signals (TSC pressure drop, spindle thermistor residual, axis-load drift,
compressor discharge temperature residual, as daily z-scores against the training period), taking the
maximum z per indicator. The suspected condition is the first family in the asset kind's fixed
precedence (specific signatures before generic) whose own indicators are at or above the threshold.
Output: one work request per case with the condition, the indicators against baseline, the
recommended inspection and the shadow-mode flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INSPECTION = {
    "bearing": "listen, envelope check, temperature check",
    "lubrication": "temperature check, spindle lube check",
    "rotary union": "pressure check, inspect rotary union and seals",
    "imbalance": "strobe and vibration pen at 1x",
    "misalignment": "coupling and laser alignment check",
    "looseness": "check base and mounting bolts",
    "cavitation or wear": "coolant level and pressure check",
    "spindle cooling": "check spindle chiller and coolant flow",
    "cooler": "check compressor oil cooler and fan",
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
CONTROLLER_INDICATORS = ("tsc_drop", "thermistor", "axis_drift", "discharge")
ROBUST = 1.4826


def log_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ord_3_10_g2"] = d[[f"ord_{o:02d}_g2" for o in range(3, 11)]].sum(axis=1)
    out = pd.DataFrame(index=d.index)
    for k, c in Z_INDICATORS.items():
        out[k] = np.log10(d[c].astype(float) + 1e-12)
    return out


def _robust(v: np.ndarray) -> tuple[float, float]:
    med = float(np.median(v))
    sc = float(ROBUST * np.median(np.abs(v - med))) or float(np.std(v)) or 1.0
    return med, sc


def healthy_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """Median and robust scale per point, regime and indicator."""
    if rows.empty:
        return pd.DataFrame()
    li = log_indicators(rows).assign(point_id=rows["point_id"].to_numpy(), regime=rows["regime"].to_numpy())
    out = []
    for (pid, reg), g in li.groupby(["point_id", "regime"]):
        for k in Z_INDICATORS:
            v = g[k].dropna().to_numpy()
            if v.size >= 20:
                med, sc = _robust(v)
                out.append(
                    {
                        "point_id": pid,
                        "regime": reg,
                        "indicator": k,
                        "median": med,
                        "scale": sc,
                        "n": int(v.size),
                    }
                )
    return pd.DataFrame(out)


def hourly_z(ind: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """z per hour and indicator against the healthy spread of the point's segment in effect at that hour,
    by regime where the segment has it, otherwise pooled over the segment's regimes."""
    parts = []
    for pid, g in ind.groupby("point_id"):
        st = stats[stats["point_id"] == pid]
        if st.empty:
            continue
        starts = st[["segment", "segment_start"]].drop_duplicates().sort_values("segment_start")
        t = np.asarray(pd.to_datetime(g["hour_utc"], utc=True).dt.tz_localize(None), dtype="datetime64[ns]")
        s0 = np.asarray(
            pd.to_datetime(starts["segment_start"], utc=True).dt.tz_localize(None), dtype="datetime64[ns]"
        )
        idx = np.clip(np.searchsorted(s0, t, side="right") - 1, 0, len(s0) - 1)
        li = log_indicators(g).assign(
            segment=starts["segment"].to_numpy()[idx], regime=g["regime"].to_numpy()
        )
        by_reg = st.set_index(["segment", "regime", "indicator"])[["median", "scale"]]
        pooled = st.groupby(["segment", "indicator"])[["median", "scale"]].median()
        z = pd.DataFrame({"point_id": pid, "hour_utc": g["hour_utc"].to_numpy()})
        for k in Z_INDICATORS:
            key_r = pd.MultiIndex.from_arrays([li["segment"], li["regime"], np.full(len(li), k)])
            key_p = pd.MultiIndex.from_arrays([li["segment"], np.full(len(li), k)])
            med = by_reg["median"].reindex(key_r).to_numpy()
            sc = by_reg["scale"].reindex(key_r).to_numpy()
            med = np.where(np.isnan(med), pooled["median"].reindex(key_p).to_numpy(), med)
            sc = np.where(np.isnan(sc), pooled["scale"].reindex(key_p).to_numpy(), sc)
            z[k] = (li[k].to_numpy() - med) / sc
        parts.append(z)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def daily_point_z(hz: pd.DataFrame) -> pd.DataFrame:
    d = hz.assign(day=pd.to_datetime(hz["hour_utc"], utc=True).dt.floor("D"))
    return d.groupby(["point_id", "day"])[list(Z_INDICATORS)].median().reset_index()


def tsc_drop_daily(hourly: pd.DataFrame, ref_days: tuple[int, int]) -> pd.DataFrame:
    """Daily median TSC pressure (pump running) against the median over the reference days before it
    (from ref_days[1] to ref_days[0] days earlier); positive is a drop."""
    h = hourly[hourly["coolant_pressure_bar"] > 0].copy()
    h["local_date"] = (
        pd.to_datetime(h["hour_utc"], utc=True)
        .dt.tz_convert("America/Chicago")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    near, far = ref_days
    out = []
    for mid, g in h.groupby("machine_id"):
        s = g.groupby("local_date")["coolant_pressure_bar"].median()
        ref = s.rolling(f"{far - near}D", min_periods=5).median()
        ref.index = ref.index + pd.Timedelta(days=near)
        ref = ref[~ref.index.duplicated()].reindex(s.index, method="ffill")
        out.append(pd.DataFrame({"entity": mid, "local_date": s.index, "value": (1 - s / ref).to_numpy()}))
    return pd.concat(out, ignore_index=True).assign(indicator="tsc_drop") if out else pd.DataFrame()


def tsc_pressure_daily(hourly: pd.DataFrame, ref_days: tuple[int, int]) -> pd.DataFrame:
    """Per machine and local day: the median of the hourly controller coolant pressure over hours with
    pressure above zero, the number of such hours, the reference median (ref_days[1] to ref_days[0] days
    earlier, as the TSC pressure drop uses) and the drop against it. The hourly value is the silver
    controller table's mean over active cutting time by program, which includes time the pump is off,
    so it follows pressure and the programs' through-spindle coolant duty together."""
    h = hourly[hourly["coolant_pressure_bar"] > 0].copy()
    h["local_date"] = (
        pd.to_datetime(h["hour_utc"], utc=True)
        .dt.tz_convert("America/Chicago")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    near, far = ref_days
    out = []
    for mid, g in h.groupby("machine_id"):
        day = g.groupby("local_date")["coolant_pressure_bar"]
        s = day.median()
        ref = s.rolling(f"{far - near}D", min_periods=5).median()
        ref.index = ref.index + pd.Timedelta(days=near)
        ref = ref[~ref.index.duplicated()].reindex(s.index, method="ffill")
        out.append(
            pd.DataFrame(
                {
                    "machine_id": mid,
                    "local_date": s.index,
                    "pressure_median_bar": s.to_numpy(),
                    "running_hours": day.size().reindex(s.index).to_numpy(),
                    "reference_median_bar": ref.to_numpy(),
                    "drop_share": (1 - s / ref).to_numpy(),
                }
            )
        )
    cols = [
        "machine_id",
        "local_date",
        "pressure_median_bar",
        "running_hours",
        "reference_median_bar",
        "drop_share",
    ]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def controller_z(daily: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """z per entity, day and controller indicator against the entity's training-period days."""
    d = daily.dropna(subset=["value"]).copy()
    d["local_date"] = pd.to_datetime(d["local_date"])
    out = []
    for _key, g in d.groupby(["entity", "indicator"]):
        tr = g.loc[g["local_date"] < train_end, "value"].to_numpy()
        if tr.size < 20:
            continue
        med, sc = _robust(tr)
        out.append(g.assign(z=(g["value"] - med) / sc))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def asset_kind(asset: str) -> str:
    if asset.startswith("CMP-"):
        return "compressor"
    for suffix, kind in (("-SP", "spindle"), ("-TSC", "tsc_pump"), ("-CP", "flood_pump")):
        if asset.endswith(suffix):
            return kind
    return "machine"


def controller_entity(asset: str) -> str:
    return asset if asset.startswith("CMP-") else "-".join(asset.split("-")[:2])


def case_z(
    case: dict, vib: pd.DataFrame, points: list[str], ctrl: pd.DataFrame, end: pd.Timestamp
) -> dict[str, tuple[float, str]]:
    """Maximum z per indicator over the case's open interval, with where it occurred."""
    a = pd.Timestamp(case["opened"]).floor("D")
    b = min(pd.Timestamp(case["closed"]) if pd.notna(case["closed"]) else end, end)
    out: dict[str, tuple[float, str]] = {}
    v = vib[vib["point_id"].isin(points) & (vib["day"] >= a) & (vib["day"] <= b)]
    for k in Z_INDICATORS:
        s = v[["point_id", k]].dropna()
        if len(s):
            i = s[k].idxmax()
            out[k] = (float(s.at[i, k]), str(s.at[i, "point_id"]))
    if ctrl.empty:
        return out
    la = a.tz_convert("America/Chicago").tz_localize(None).normalize()
    lb = b.tz_convert("America/Chicago").tz_localize(None).normalize()
    c = ctrl[
        (ctrl["entity"] == controller_entity(case["asset_id"]))
        & (ctrl["local_date"] >= la)
        & (ctrl["local_date"] <= lb)
    ]
    for k in CONTROLLER_INDICATORS:
        s = c[c["indicator"] == k]
        if len(s):
            i = s["z"].idxmax()
            out[k] = (float(s.at[i, "z"]), str(s.at[i, "entity"]))
    return out


def suspected(z: dict[str, float], kind: str, zthr: float) -> tuple[str, str]:
    """The first family in the asset kind's precedence (specific signatures before generic) whose own
    indicators are all at or above zthr."""

    def hit(*keys: str) -> bool:
        return all(pd.notna(z.get(k, np.nan)) and z[k] >= zthr for k in keys)

    def text(*keys: str) -> str:
        return ", ".join(f"{k} z {z[k]:.1f}" for k in keys)

    env = {k: z.get(f"env_{k}", np.nan) for k in FAULT_LOCATION}
    env_hit = {k: v for k, v in env.items() if pd.notna(v) and v >= zthr}
    bearing = None
    if env_hit:
        k = max(env_hit, key=env_hit.get)
        bearing = ("bearing", f"{FAULT_LOCATION[k]} ({k.upper()} z {env_hit[k]:.1f})")
    rules: list[tuple[str, tuple[str, ...]] | str] = []
    if kind == "spindle":
        rules = [
            ("rotary union", ("tsc_drop", "hf")),
            ("spindle cooling", ("thermistor",)),
            "bearing",
            ("lubrication", ("hf",)),
        ]
    elif kind == "tsc_pump":
        rules = [
            ("cavitation or wear", ("tsc_drop", "hf")),
            "bearing",
            ("imbalance", ("ord_01",)),
            ("looseness", ("ord_3_10",)),
        ]
    elif kind == "flood_pump":
        rules = [
            ("cavitation or wear", ("hf",)),
            "bearing",
            ("imbalance", ("ord_01",)),
            ("looseness", ("ord_3_10",)),
        ]
    elif kind == "compressor":
        rules = [
            ("cooler", ("discharge",)),
            "bearing",
            ("imbalance", ("ord_01",)),
            ("misalignment", ("ord_02",)),
            ("looseness", ("ord_3_10",)),
        ]
    else:
        rules = [("axis wear", ("axis_drift",)), ("spindle cooling", ("thermistor",))]
    for r in rules:
        if r == "bearing":
            if bearing:
                return bearing
            continue
        fam, keys = r
        if hit(*keys):
            return fam, text(*keys)
    return "unclassified", ""


def work_requests(
    cases: pd.DataFrame,
    zs: dict[str, dict[str, tuple[float, str]]],
    shadow: tuple[pd.Timestamp, pd.Timestamp],
    zthr: float,
) -> pd.DataFrame:
    rows = []
    for c in cases.itertuples():
        z = zs.get(c.case_id, {})
        cond, detail = suspected({k: v[0] for k, v in z.items()}, asset_kind(c.asset_id), zthr)
        pts = [p for p in str(c.trigger_points or "").split(",") if p]
        rows.append(
            {
                "request_id": f"WR-{c.case_id}",
                "case_id": c.case_id,
                "asset_id": c.asset_id,
                "point_id": pts[0] if pts else None,
                "created": c.opened,
                "suspected_condition": cond,
                "suspected_detail": detail,
                "triggering_indicators": c.trigger_rules,
                "recommended_inspection": INSPECTION[cond],
                "shadow_mode": bool(shadow[0] <= c.opened < shadow[1]),
                **{f"z_{k}": v[0] for k, v in z.items()},
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
    "rotary union": "rotary union",
    "spindle cooling": "spindle cooling",
    "cooler (compressor)": "cooler",
    "pump seal or impeller": "cavitation or wear",
}


def family_of(condition_type: str, asset_id: str, high_temp_trip: bool = False) -> str | None:
    """Finding family for agreement scoring. An other-machine-stoppage work order on an axis is axis
    wear; a compressor trip is cooler only when the compressor controller logged a high-temperature
    trip for it."""
    if condition_type == "other machine stoppage" and "-AX" in str(asset_id):
        return "axis wear"
    if condition_type == "compressor trip":
        return "cooler" if high_temp_trip else None
    return CONDITION_FAMILY.get(condition_type)


def agreement(
    requests: pd.DataFrame,
    cases: pd.DataFrame,
    conds: pd.DataFrame,
    wo: pd.DataFrame,
    high_temp_trip_wos: set[str] | None = None,
) -> pd.DataFrame:
    """Per work request, how its case ended: a confirmed condition on the asset while the case was open
    (agreement scored against its finding family), no fault found (the closing inspection), or other."""
    cs = cases.set_index("case_id")
    nff = set(wo.loc[(wo["wo_type"] == "inspection") & wo["no_fault_found"].astype(bool), "wo_id"])
    rows = []
    for r in requests.itertuples():
        c = cs.loc[r.case_id]
        hit = conds[
            (conds["asset_key"] == r.asset_id)
            & (conds["credit_ts"] >= r.created)
            & (conds["credit_ts"] <= c["closed"])
        ]
        base = {"request_id": r.request_id, "suspected": r.suspected_condition}
        if len(hit):
            first = hit.sort_values("credit_ts").iloc[0]
            fam = family_of(
                first["condition_type"], first["asset_id"], first["wo_id"] in (high_temp_trip_wos or set())
            )
            rows.append(
                {
                    **base,
                    "outcome": "confirmed",
                    "confirmed": first["condition_type"],
                    "confirmed_family": fam,
                    "agree": fam == r.suspected_condition,
                }
            )
        else:
            outcome = "no fault found" if c.get("closing_wo") in nff else "other"
            rows.append(
                {**base, "outcome": outcome, "confirmed": None, "confirmed_family": None, "agree": None}
            )
    return pd.DataFrame(rows)
