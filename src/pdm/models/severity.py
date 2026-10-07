"""Levels for both methods: WATCH, ALERT and ALARM for the ML Model; ALERT and ALARM for the Vendor Rules.

ALERT, each method's action level, is an episode per asset: persisted exceedances on any of the asset's
points, merged when they overlap or fall within the episode gap; an episode closes the gap after its
last exceedance. ALARM, each method's urgent level, starts inside an open ALERT episode: for the ML
Model at the first hour an ALARM criterion for the asset class holds (config `severity`), for the
Vendor Rules at the first vendor alarm; for both, also when the condition that triggered the ALERT is
met on 14 operating days in a row (days without operation are skipped, an inspection that finds
nothing does not reset the count). WATCH is any unpersisted ML exceedance or a plausibility flag.
(Revision 2, renamed 2026-10-30: was WATCH, ALARM and CRITICAL for the model and WATCH and ALARM for
the vendor rules.)
Inputs: persisted exceedances, scored hours, hourly indicators, controller and compressor hourly
series. Outputs: episodes with their ALARM start and cause, and hourly ALARM criteria per asset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

LOCAL_TZ = "America/Chicago"
DAY = pd.Timedelta(days=1)
MET_SHARE = 0.5
STAGE_BANDS = {"bpfo": "env_bpfo", "bpfi": "env_bpfi", "bsf": "env_bsf"}


def asset_class(asset_id: str) -> str:
    if asset_id.endswith("-SP"):
        return "spindle"
    if asset_id.endswith("-TSC"):
        return "tsc_pump"
    if asset_id.endswith("-CP"):
        return "flood_pump"
    return "compressor"


def _cls_key(cls: str) -> str:
    return "pump" if cls in ("tsc_pump", "flood_pump") else cls


def episodes(signals: pd.DataFrame, gap_days: float) -> pd.DataFrame:
    """ALERT episodes per asset from persisted exceedances (asset_id, point_id, ts, and optionally
    rules): signals on any point of the asset merge when within the gap; an episode closes gap_days
    after its last signal. Carries the points and rules of the opening hour."""
    cols = ["asset_id", "opened", "last", "closed", "points", "trigger_points", "trigger_rules", "signals"]
    if signals.empty:
        return pd.DataFrame(columns=cols)
    gap = pd.Timedelta(days=gap_days)
    s = signals.assign(ts=pd.to_datetime(signals["ts"], utc=True)).sort_values(
        ["asset_id", "ts"], kind="stable"
    )
    if "rules" not in s:
        s["rules"] = ""
    rows = []
    for asset, g in s.groupby("asset_id", sort=True):
        t = g["ts"].to_numpy()
        pts = g["point_id"].fillna("").to_numpy()
        rl = g["rules"].fillna("").astype(str).to_numpy()
        start = 0
        for i in range(1, len(t) + 1):
            if i == len(t) or pd.Timestamp(t[i]) - pd.Timestamp(t[i - 1]) > gap:
                a, b = pd.Timestamp(t[start]), pd.Timestamp(t[i - 1])
                first = t[start:i] == t[start]
                rows.append(
                    {
                        "asset_id": asset,
                        "opened": a,
                        "last": b,
                        "closed": b + gap,
                        "points": ",".join(sorted({p for p in pts[start:i] if p})),
                        "trigger_points": ",".join(sorted({p for p in pts[start:i][first] if p})),
                        "trigger_rules": ",".join(
                            sorted({r for x in rl[start:i][first] for r in x.split(",") if r})
                        ),
                        "signals": i - start,
                    }
                )
                start = i
    out = pd.DataFrame(rows, columns=cols)
    out["opened"] = pd.to_datetime(out["opened"], utc=True)
    out["last"] = pd.to_datetime(out["last"], utc=True)
    out["closed"] = pd.to_datetime(out["closed"], utc=True)
    return out


def _local_day(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(LOCAL_TZ).dt.normalize()


def escalation_time(
    opened: pd.Timestamp,
    closed: pd.Timestamp,
    operating: pd.DataFrame,
    days: int,
) -> pd.Timestamp | None:
    """When an open ALERT escalates by persistence: the first hour at which the triggering condition has
    been met on `days` operating days in a row, counted from the opening day. `operating` holds one row
    per measured hour in the episode (ts, met); a day is operating when it has a measured hour and met
    when at least half of its measured hours meet the condition (step 6 design freeze 2026-10-30; was any
    hour); days without operation are skipped, and an operating day without the condition restarts the
    count."""
    if operating.empty:
        return None
    o = operating[(operating["ts"] >= opened) & (operating["ts"] < closed)]
    if o.empty:
        return None
    o = o.assign(day=_local_day(o["ts"]).to_numpy())
    run = 0
    for _, g in o.groupby("day", sort=True):
        if g["met"].astype(bool).mean() >= MET_SHARE:
            run += 1
            if run >= days:
                return pd.Timestamp(g.loc[g["met"], "ts"].min())
        else:
            run = 0
    return None


def _consecutive(flag: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(len(flag), dtype=bool)
    c = 0
    for i, f in enumerate(flag):
        c = c + 1 if f else 0
        out[i] = c >= n
    return out


def _slope_positive(t: pd.Series, v: np.ndarray, window_h: float) -> np.ndarray:
    """Theil-Sen slope of v over the trailing window (hours) is positive, per row."""
    ts = np.asarray(pd.to_datetime(pd.Series(t), utc=True).dt.tz_localize(None), dtype="datetime64[ns]")
    out = np.zeros(len(v), dtype=bool)
    w = np.timedelta64(int(window_h * 3600), "s")
    j = 0
    for i in range(len(v)):
        while ts[j] < ts[i] - w:
            j += 1
        seg = slice(j, i + 1)
        y = v[seg]
        ok = np.isfinite(y)
        if ok.sum() >= 3:
            x = (ts[seg][ok] - ts[seg][ok][0]) / np.timedelta64(1, "h")
            if np.ptp(x) > 0:
                out[i] = theilslopes(y[ok], x)[0] > 0
    return out


def _rolling_slope_positive(t: pd.Series, v: np.ndarray, days: float) -> np.ndarray:
    """Least-squares slope of v over the trailing days is positive, per row (cheaper than Theil-Sen on
    the longer window)."""
    s = pd.Series(v, index=pd.to_datetime(t, utc=True))
    x = pd.Series((s.index - s.index[0]) / pd.Timedelta(hours=1), index=s.index)
    w = f"{int(days)}D"
    mx, my = x.rolling(w).mean(), s.rolling(w).mean()
    cov = (x * s).rolling(w).mean() - mx * my
    var = (x * x).rolling(w).mean() - mx * mx
    slope = cov / var.where(var > 0)
    return (slope > 0).to_numpy()


def segment_band_levels(ind: pd.DataFrame, segments: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each hour's segment level of the given columns: their median over the segment's re-centering window
    (segments from the ML Model's segment table, level_ready per segment)."""
    out = pd.DataFrame(index=ind.index, columns=columns, dtype=float)
    t = pd.to_datetime(ind["hour_utc"], utc=True)
    s = segments.sort_values("segment_start").reset_index(drop=True)
    starts = pd.to_datetime(s["segment_start"], utc=True)
    for i, r in s.iterrows():
        if pd.isna(r["level_ready"]):
            continue
        a, ready = starts.iat[i], pd.Timestamp(r["level_ready"])
        b = starts.iat[i + 1] if i + 1 < len(s) else pd.Timestamp.max.tz_localize("UTC")
        win = (t >= a) & (t < ready)
        seg = (t >= a) & (t < b)
        med = ind.loc[win, columns].median()
        out.loc[seg, columns] = med.to_numpy(dtype=float)
    return out


def sensor_criteria(
    ind: pd.DataFrame,
    scored: pd.DataFrame,
    segments: pd.DataFrame,
    cfg: dict[str, Any],
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """Hourly ALARM criteria on one point's sensor rows (point class from the rows): temperature
    residual, velocity, the stage-3 and stage-4 bearing signatures and, on pumps, persistent HF energy
    (the cavitation criterion needs the pressure drop as well and is completed per asset). Each
    criterion holds after its persistence. Returns rows (point_id, hour_utc, criterion)."""
    sv = cfg["models"]["severity"]
    if ind.empty:
        return pd.DataFrame(columns=["point_id", "hour_utc", "criterion"])
    d = ind.sort_values("hour_utc").reset_index(drop=True)
    d = d[~d["implausible"].fillna(False).astype(bool)].dropna(subset=["vel_rms_mm_s"]).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=["point_id", "hour_utc", "criterion"])
    pid = str(d["point_id"].iat[0])
    pclass = str(d["point_class"].iat[0])
    cls = (
        "spindle"
        if pclass.startswith("spindle")
        else ("compressor" if pclass.startswith("compressor") else "pump")
    )
    n = sv["alarm_persistence_hours"]
    flags: dict[str, np.ndarray] = {}
    sc = (
        scored.set_index(pd.to_datetime(scored["hour_utc"], utc=True))["temp_resid_c"]
        if len(scored)
        else None
    )
    hours = pd.to_datetime(d["hour_utc"], utc=True)
    resid = (
        hours.map(sc).to_numpy(dtype=float)
        if sc is not None and not sc.index.has_duplicates
        else np.full(len(d), np.nan)
    )
    lim_t = sv["temp_residual_c"][cls]
    if cls == "spindle":
        warm = resid >= lim_t
        rising = _slope_positive(hours, resid, sv["spindle_temp_slope_window_h"]) if warm.any() else warm
        flags["temperature residual"] = _consecutive(warm & rising, n)
    else:
        flags["temperature residual"] = _consecutive(resid >= lim_t, n)
    vel = (
        d["vel_max_dir_mm_s"].to_numpy(dtype=float)
        if "vel_max_dir_mm_s" in d
        else d["vel_rms_mm_s"].to_numpy(dtype=float)
    )
    flags["velocity zone D"] = _consecutive(vel > sv["velocity_mm_s"][cls], n)
    s3 = sv["stage3"]
    band_cols = [f"{c}_{h}_g2" for c in STAGE_BANDS.values() for h in (1, s3["harmonic"])] + [
        "env_bpfi_sb_lo_g2",
        "env_bpfi_sb_hi_g2",
    ]
    band_cols = [c for c in band_cols if c in d]
    toct = sorted(c for c in d.columns if c.startswith("toct_"))
    d["_toct_sum"] = d[toct].sum(axis=1, min_count=1)
    lv = segment_band_levels(d, segments, [*band_cols, "_toct_sum"]) if len(segments) else None
    if lv is not None:
        r = s3["ratio_to_floor"]
        hit = np.zeros(len(d), dtype=bool)
        for key, col in STAGE_BANDS.items():
            c1, c2 = f"{col}_1_g2", f"{col}_{s3['harmonic']}_g2"
            if c1 not in d or c2 not in d:
                continue
            ok = (d[c1] >= r * lv[c1]).to_numpy() & (d[c2] >= r * lv[c2]).to_numpy()
            if key == "bpfi" and "env_bpfi_sb_lo_g2" in d:
                sb = np.maximum(d["env_bpfi_sb_lo_g2"].fillna(0), d["env_bpfi_sb_hi_g2"].fillna(0))
                lsb = np.maximum(lv["env_bpfi_sb_lo_g2"].fillna(0), lv["env_bpfi_sb_hi_g2"].fillna(0))
                ok &= (sb >= r * lsb).to_numpy()
            hit |= np.nan_to_num(ok, nan=False).astype(bool)
        k, m = s3["of_last_hours"]
        cnt = pd.Series(hit.astype(int)).rolling(m, min_periods=m).sum().to_numpy()
        flags["stage 3 bearing signature"] = np.nan_to_num(cnt) >= k
        s4 = sv["stage4"]
        broad = (d["_toct_sum"] >= s4["broadband_ratio"] * lv["_toct_sum"]).to_numpy()
        ser = pd.Series(d["hf_rms_g"].to_numpy(dtype=float), index=hours)
        peak = ser.rolling(f"{s4['peak_window_days']}D").max().to_numpy()
        envs = d[[f"{c}_1_g2" for c in STAGE_BANDS.values() if f"{c}_1_g2" in d]].fillna(0).sum(axis=1)
        epk = pd.Series(envs.to_numpy(), index=hours).rolling(f"{s4['peak_window_days']}D").max().to_numpy()
        fall = (ser.to_numpy() <= (1 - s4["hf_or_envelope_fall_frac"]) * peak) | (
            envs.to_numpy() <= (1 - s4["hf_or_envelope_fall_frac"]) * epk
        )
        rising_v = _rolling_slope_positive(
            hours, d["vel_rms_mm_s"].to_numpy(dtype=float), s4["velocity_slope_days"]
        )
        flags["stage 4 bearing signature"] = _consecutive(
            np.nan_to_num(broad, nan=False).astype(bool) & fall & rising_v, n
        )
    if cls == "pump":
        hf = np.log10(d["hf_rms_g"].to_numpy(dtype=float) + 1e-12)
        base = lv["_hf"] if lv is not None and "_hf" in lv else None
        if base is None:
            lvh = segment_band_levels(d.assign(_hf=hf), segments, ["_hf"]) if len(segments) else None
            rel = hf - (lvh["_hf"].to_numpy(dtype=float) if lvh is not None else np.nanmedian(hf))
        else:
            rel = hf - base
        tr = hours < train_end
        sd = np.nanstd(rel[tr]) if tr.any() else np.nanstd(rel)
        z = rel / (sd if sd > 0 else 1.0)
        flags["_hf_persistent"] = _consecutive(z >= sv["cavitation"]["hf_z"], sv["cavitation"]["hours"])
    rows = [
        pd.DataFrame({"point_id": pid, "hour_utc": hours.to_numpy()[f], "criterion": name})
        for name, f in flags.items()
        if f.any()
    ]
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["point_id", "hour_utc", "criterion"])
    )


def thermistor_hours(minute: pd.DataFrame, limit_c: float, minutes: int) -> pd.DataFrame:
    """Hours in which a machine's spindle thermistor (per-minute mean) has been at or above the limit for
    the given consecutive minutes (machine_id, hour_utc)."""
    m = (
        minute.groupby(["machine_id", "minute_utc"], as_index=False)["spindle_temp_c_mean"]
        .max()
        .sort_values(["machine_id", "minute_utc"])
    )
    out = []
    for mid, g in m.groupby("machine_id"):
        t = np.asarray(pd.to_datetime(g["minute_utc"], utc=True).dt.tz_localize(None), dtype="datetime64[ns]")
        hot = g["spindle_temp_c_mean"].to_numpy(dtype=float) >= limit_c
        contiguous = np.r_[False, np.diff(t) == np.timedelta64(60, "s")]
        run = np.zeros(len(hot), dtype=int)
        c = 0
        for i in range(len(hot)):
            c = (c + 1 if contiguous[i] else 1) if hot[i] else 0
            run[i] = c
        hit = run >= minutes
        if hit.any():
            out.append(
                pd.DataFrame(
                    {"machine_id": mid, "hour_utc": pd.DatetimeIndex(t[hit]).tz_localize("UTC").floor("h")}
                )
            )
    return (
        pd.concat(out, ignore_index=True).drop_duplicates()
        if out
        else pd.DataFrame(columns=["machine_id", "hour_utc"])
    )


def pressure_criteria(tsc: pd.DataFrame, hf_hours: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """TSC pump criteria from the hourly pump-on pressure (machine_id, hour_utc, pressure_bar): pressure
    below the configured bar for the persistence, and sustained cavitation (persistent HF with a pump-on
    pressure drop of at least the configured share against the median 14 to 44 days earlier). hf_hours
    holds the persistent-HF hours per pump asset (asset_id, hour_utc). Returns (asset_id, hour_utc,
    criterion)."""
    sv = cfg["models"]["severity"]
    ref_a, ref_b = cfg["advisory"]["tsc_pressure_reference_days"]
    n = sv["alarm_persistence_hours"]
    rows = []
    for mid, g in tsc.sort_values("hour_utc").groupby("machine_id"):
        t = pd.to_datetime(g["hour_utc"], utc=True)
        p = g["pressure_bar"].to_numpy(dtype=float)
        low = _consecutive(np.nan_to_num(p, nan=np.inf) < sv["tsc_pump_on_bar_min"], n)
        asset = f"{mid}-TSC"
        rows.append(
            pd.DataFrame(
                {"asset_id": asset, "hour_utc": t[low].to_numpy(), "criterion": "TSC pump-on pressure low"}
            )
        )
        day = t.dt.floor("D")
        daily = pd.Series(p, index=day).groupby(level=0).median()
        ref = pd.Series(
            [
                daily[
                    (daily.index >= x - pd.Timedelta(days=ref_b))
                    & (daily.index < x - pd.Timedelta(days=ref_a))
                ].median()
                for x in daily.index
            ],
            index=daily.index,
        )
        drop = (1 - daily / ref) >= sv["cavitation"]["pressure_drop_frac"]
        dropped_days = set(drop[drop.fillna(False)].index)
        hfh = hf_hours[hf_hours["asset_id"] == asset]
        if len(hfh):
            ht = pd.to_datetime(hfh["hour_utc"], utc=True)
            cav = ht[ht.dt.floor("D").isin(dropped_days)]
            rows.append(
                pd.DataFrame(
                    {"asset_id": asset, "hour_utc": cav.to_numpy(), "criterion": "sustained cavitation"}
                )
            )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["asset_id", "hour_utc", "criterion"])
    )


def alarm_starts(
    eps: pd.DataFrame,
    criteria: pd.DataFrame,
    operating: dict[str, pd.DataFrame],
    days: int,
) -> pd.DataFrame:
    """ALARM start per ALERT episode: the first criterion hour inside the episode (criteria rows asset_id,
    hour_utc, criterion), and the persistence escalation; reported separately and combined. `operating`
    maps each asset to its measured hours with the triggering condition (ts, met, and point_id where the
    condition is per point)."""
    out = eps.copy()
    if out.empty:
        for c in ("alarm_criterion_ts", "alarm_criterion", "alarm_escalation_ts", "alarm_ts", "alarm_cause"):
            out[c] = pd.Series(dtype=object)
        return out
    crit = (
        criteria.assign(hour_utc=pd.to_datetime(criteria["hour_utc"], utc=True))
        if len(criteria)
        else criteria
    )
    c_ts, c_name, e_ts = [], [], []
    for r in out.itertuples():
        hit = (
            crit[
                (crit["asset_id"] == r.asset_id)
                & (crit["hour_utc"] >= r.opened)
                & (crit["hour_utc"] < r.closed)
            ].sort_values(["hour_utc", "criterion"])
            if len(crit)
            else crit
        )
        c_ts.append(hit["hour_utc"].iat[0] if len(hit) else pd.NaT)
        c_name.append(hit["criterion"].iat[0] if len(hit) else None)
        op = operating.get(r.asset_id, pd.DataFrame(columns=["ts", "met"]))
        if "point_id" in op and r.trigger_points:
            op = op[op["point_id"].isin(r.trigger_points.split(","))]
        if "rules" in op:
            trig = set(str(r.trigger_rules).split(",")) - {""}
            op = op.assign(
                met=op["rules"].fillna("").map(lambda x, trig=trig: bool(trig & set(x.split(","))))
            )
        e_ts.append(escalation_time(r.opened, r.closed, op, days))
    out["alarm_criterion_ts"] = pd.to_datetime(pd.Series(c_ts, index=out.index), utc=True)
    out["alarm_criterion"] = c_name
    out["alarm_escalation_ts"] = pd.to_datetime(pd.Series(e_ts, index=out.index, dtype=object), utc=True)
    both = out[["alarm_criterion_ts", "alarm_escalation_ts"]]
    out["alarm_ts"] = both.min(axis=1)
    out["alarm_cause"] = np.select(
        [
            out["alarm_criterion_ts"].notna()
            & (
                out["alarm_criterion_ts"]
                <= out["alarm_escalation_ts"].fillna(pd.Timestamp.max.tz_localize("UTC"))
            ),
            out["alarm_escalation_ts"].notna(),
        ],
        [out["alarm_criterion"].fillna("criterion"), "14-day persistence"],
        default=None,
    )
    return out
