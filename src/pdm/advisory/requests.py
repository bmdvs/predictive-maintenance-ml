"""Work requests on ALARM with a suspected condition from the indicator pattern (rule-based diagnosis).

Diagnosis reads every signal on the asset over the case's open interval: vibration indicators at all of
the asset's points (daily median of hourly z-scores against the point's healthy spread for its segment
and regime) and the controller signals (TSC pressure drop, spindle thermistor residual, axis-load drift,
compressor discharge temperature residual, as daily z-scores against the training period), taking the
maximum z per indicator. The suspected condition is the first family in the asset kind's fixed
precedence (specific signatures before generic) whose own indicators are at or above the threshold
(revision 2 order: on spindles rotary union, spindle cooling, bearing, lubrication, contamination, with the
housing temperature separating spindle cooling (temperature without HF) from lubrication (HF with
temperature) and contamination (HF without temperature); on pumps bearing, imbalance, looseness, then
cavitation or wear; compressors unchanged).
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
    "contamination": "inspect seals and rotary union, check for coolant in the bearing area",
    "unclassified": "listen, temperature check",
}
FAULT_LOCATION = {"bpfo": "outer race", "bpfi": "inner race", "bsf": "ball"}
D1_HF_QUIET = 2.0
Z_INDICATORS = {
    "env_bpfo": "env_bpfo_1_g2",
    "env_bpfi": "env_bpfi_1_g2",
    "env_bsf": "env_bsf_1_g2",
    "ord_01": "ord_01_g2",
    "ord_02": "ord_02_g2",
    "ord_3_10": "ord_3_10_g2",
    "hf": "hf_rms_g",
    "vel": "vel_rms_mm_s",
    "temp": "temp_c",
}
CONTROLLER_INDICATORS = ("tsc_drop", "thermistor", "axis_drift", "discharge")
ROBUST = 1.4826


# D2 (2026-11-01, switch advisory.diagnosis_rule: d2): the vibration measures as the change from the
# segment's own level in dB (10 log10 of the ratio for energies in g^2, 20 log10 for RMS amplitudes), so a
# rise reads the same on every point whatever its healthy spread. Kurtosis and crest factor stay as z against
# the segment's healthy spread: they are dimensionless shape statistics whose healthy level differs little
# between points while their spread does, and a ratio in dB of a shape statistic has no physical meaning.
DB_FACTOR = {
    "env_bpfo": 10.0,
    "env_bpfi": 10.0,
    "env_bsf": 10.0,
    "ord_01": 10.0,
    "ord_02": 10.0,
    "ord_3_10": 10.0,
    "hf": 20.0,
    "vel": 20.0,
}
D2_INDICATORS = {**Z_INDICATORS, "crest": "crest_factor", "kurt": "kurtosis"}
FAULT_BANDS = ("env_bpfo", "env_bpfi", "env_bsf")
D2_DEFAULTS: dict[str, float] = {
    "band_min_db": 3.0,
    "dominance_ratio": 2.0,
    "kurt_z": 6.0,
    "crest_z": 6.0,
    "band_rise_db": 3.0,
    "ord1_db": 3.0,
    "kurt_flat_z": 3.0,
    "temp_up_c": 2.0,
    "hf_db": 3.0,
    "ord2_db": 3.0,
    "ord_hi_db": 3.0,
    "tsc_drop_z": 3.0,
}


def log_indicators(df: pd.DataFrame, indicators: dict[str, str] | None = None) -> pd.DataFrame:
    d = df.copy()
    d["ord_3_10_g2"] = d[[f"ord_{o:02d}_g2" for o in range(3, 11)]].sum(axis=1)
    out = pd.DataFrame(index=d.index)
    for k, c in (indicators or Z_INDICATORS).items():
        out[k] = np.log10(d[c].astype(float) + 1e-12)
    return out


def _robust(v: np.ndarray) -> tuple[float, float]:
    med = float(np.median(v))
    sc = float(ROBUST * np.median(np.abs(v - med))) or float(np.std(v)) or 1.0
    return med, sc


def healthy_stats(rows: pd.DataFrame, indicators: dict[str, str] | None = None) -> pd.DataFrame:
    """Median and robust scale per point, regime and indicator."""
    if rows.empty:
        return pd.DataFrame()
    inds = indicators or Z_INDICATORS
    li = log_indicators(rows, inds).assign(
        point_id=rows["point_id"].to_numpy(), regime=rows["regime"].to_numpy()
    )
    out = []
    for (pid, reg), g in li.groupby(["point_id", "regime"]):
        for k in inds:
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


def hourly_z(
    ind: pd.DataFrame, stats: pd.DataFrame, indicators: dict[str, str] | None = None, delta: bool = False
) -> pd.DataFrame:
    """z per hour and indicator against the healthy spread of the point's segment in effect at that hour,
    by regime where the segment has it, otherwise pooled over the segment's regimes. With delta, also the
    change from the segment's level in dB (db_<indicator>) for the vibration measures (D2)."""
    inds = indicators or Z_INDICATORS
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
        li = log_indicators(g, inds).assign(
            segment=starts["segment"].to_numpy()[idx], regime=g["regime"].to_numpy()
        )
        by_reg = st.set_index(["segment", "regime", "indicator"])[["median", "scale"]]
        pooled = st.groupby(["segment", "indicator"])[["median", "scale"]].median()
        z = pd.DataFrame({"point_id": pid, "hour_utc": g["hour_utc"].to_numpy()})
        for k in inds:
            key_r = pd.MultiIndex.from_arrays([li["segment"], li["regime"], np.full(len(li), k)])
            key_p = pd.MultiIndex.from_arrays([li["segment"], np.full(len(li), k)])
            med = by_reg["median"].reindex(key_r).to_numpy()
            sc = by_reg["scale"].reindex(key_r).to_numpy()
            med = np.where(np.isnan(med), pooled["median"].reindex(key_p).to_numpy(), med)
            sc = np.where(np.isnan(sc), pooled["scale"].reindex(key_p).to_numpy(), sc)
            z[k] = (li[k].to_numpy() - med) / sc
            if delta and k in DB_FACTOR:
                z[f"db_{k}"] = DB_FACTOR[k] * (li[k].to_numpy() - med)
        parts.append(z)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def daily_point_z(hz: pd.DataFrame) -> pd.DataFrame:
    d = hz.assign(day=pd.to_datetime(hz["hour_utc"], utc=True).dt.floor("D"))
    cols = [c for c in hz.columns if c not in ("point_id", "hour_utc")]
    return d.groupby(["point_id", "day"])[cols].median().reset_index()


def temp_residual_daily(scores: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    """D2: the temperature residual's change from the segment's own level, in degrees C: the daily median of
    temp_resid_c less its median over the segment's level window (segment start to level ready; the segment's
    first 7 days when it has no level). Columns point_id, day, temp_dc."""
    s = scores[["point_id", "hour_utc", "temp_resid_c"]].dropna().copy()
    s["hour_utc"] = pd.to_datetime(s["hour_utc"], utc=True)
    out = []
    for pid, g in s.groupby("point_id"):
        sg = segs[segs["point_id"] == pid].sort_values("segment_start").drop_duplicates("segment_start")
        if sg.empty:
            continue
        starts = pd.to_datetime(sg["segment_start"], utc=True).to_numpy()
        ready = pd.to_datetime(sg["level_ready"], utc=True)
        ready = ready.where(
            ready.notna(), pd.to_datetime(sg["segment_start"], utc=True) + pd.Timedelta(days=7)
        ).to_numpy()
        h = g["hour_utc"].to_numpy()
        k = np.searchsorted(starts, h, side="right") - 1
        lvl = np.full(len(starts), np.nan)
        v = g["temp_resid_c"].to_numpy(dtype=float)
        for j in range(len(starts)):
            m = (h >= starts[j]) & (h < ready[j])
            if m.any():
                lvl[j] = float(np.median(v[m]))
        dc = np.where(k >= 0, v - lvl[np.clip(k, 0, None)], np.nan)
        out.append(
            pd.DataFrame({"point_id": pid, "day": g["hour_utc"].dt.floor("D").to_numpy(), "temp_dc": dc})
        )
    if not out:
        return pd.DataFrame(columns=["point_id", "day", "temp_dc"])
    d = pd.concat(out, ignore_index=True)
    return d.groupby(["point_id", "day"])["temp_dc"].median().reset_index()


def d2_measures(
    case: dict,
    vib: pd.DataFrame,
    points: list[str],
    temp: pd.DataFrame,
    ctrl: pd.DataFrame,
    end: pd.Timestamp,
) -> dict[str, float]:
    """D2 inputs over the case's open interval (to the case close, or to `end` while it is open, so the
    suspected condition updates while the episode is open): the maximum over days and the asset's points
    of the daily medians of each band, HF and order change in dB, kurtosis and crest factor z and the
    temperature residual change, and the TSC pressure drop z."""
    a = pd.Timestamp(case["opened"]).floor("D")
    b = min(pd.Timestamp(case["closed"]) if pd.notna(case["closed"]) else end, end)
    v = vib[vib["point_id"].isin(points) & (vib["day"] >= a) & (vib["day"] <= b)]
    out: dict[str, float] = {}
    for k in [f"db_{x}" for x in DB_FACTOR] + ["crest", "kurt"]:
        if k in v and v[k].notna().any():
            out[k] = float(v[k].max())
    tt = temp[temp["point_id"].isin(points) & (temp["day"] >= a) & (temp["day"] <= b)] if len(temp) else temp
    if len(tt) and tt["temp_dc"].notna().any():
        out["temp_dc"] = float(tt["temp_dc"].max())
    if len(ctrl):
        la = a.tz_convert("America/Chicago").tz_localize(None).normalize()
        lb = b.tz_convert("America/Chicago").tz_localize(None).normalize()
        c = ctrl[
            (ctrl["entity"] == controller_entity(case["asset_id"]))
            & (ctrl["local_date"] >= la)
            & (ctrl["local_date"] <= lb)
        ]
        s = c[c["indicator"] == "tsc_drop"]
        if len(s):
            out["tsc_drop"] = float(s["z"].max())
    return out


def suspected_d2(m: dict[str, float], kind: str, th: dict[str, float]) -> tuple[str, str]:
    """D2 (2026-11-01): in order, (1) bearing: a dominant fault band (the top band's dB change at least
    band_min_db and at least dominance_ratio times the second band's), or kurtosis or crest factor z sharply
    up; the location is the dominant band; (2) pumps: the bands rising together (two or more at band_rise_db)
    with no dominant band, 1x up (ord1_db) and kurtosis flat (below kurt_flat_z): cavitation or wear;
    (3) spindles with the temperature residual up (temp_up_c): HF up (hf_db) is lubrication, HF flat is
    spindle cooling; (4) the shaft orders as the current rules: imbalance (1x), misalignment (2x,
    compressors), looseness (orders 3 to 10); (5) rotary union: a spindle with a TSC pump-on pressure drop
    (z); otherwise unclassified."""
    g = {k: m.get(f"db_{k}", np.nan) for k in FAULT_BANDS}
    vals = sorted(((v, k) for k, v in g.items() if pd.notna(v)), reverse=True)
    top, topk = vals[0] if vals else (np.nan, "")
    second = vals[1][0] if len(vals) > 1 else np.nan
    dom = (
        pd.notna(top)
        and top >= th["band_min_db"]
        and (top / max(second, 0.1) if pd.notna(second) else np.inf) >= th["dominance_ratio"]
    )
    kz, cz = m.get("kurt", np.nan), m.get("crest", np.nan)
    sharp = (pd.notna(kz) and kz >= th["kurt_z"]) or (pd.notna(cz) and cz >= th["crest_z"])
    if dom or sharp:
        loc = FAULT_LOCATION[topk.removeprefix("env_")] if dom else "location unknown (no dominant band)"
        why = (
            f"{topk.removeprefix('env_').upper()} +{top:.1f} dB, {top / max(second, 0.1):.1f} x the next"
            if dom
            else f"kurtosis z {kz:.1f}, crest z {cz:.1f}"
        )
        return "bearing", f"{loc} ({why})"
    o1 = m.get("db_ord_01", np.nan)
    if kind in ("tsc_pump", "flood_pump"):
        rising = [k for k, v in g.items() if pd.notna(v) and v >= th["band_rise_db"]]
        if (
            len(rising) >= 2
            and pd.notna(o1)
            and o1 >= th["ord1_db"]
            and (pd.isna(kz) or kz < th["kurt_flat_z"])
        ):
            return (
                "cavitation or wear",
                f"bands up together ({', '.join(r.removeprefix('env_').upper() for r in rising)}), "
                f"1x +{o1:.1f} dB, kurtosis z {kz:.1f}",
            )
    td, hf = m.get("temp_dc", np.nan), m.get("db_hf", np.nan)
    if kind == "spindle" and pd.notna(td) and td >= th["temp_up_c"]:
        if pd.notna(hf) and hf >= th["hf_db"]:
            return "lubrication", f"temperature residual +{td:.1f} C, HF +{hf:.1f} dB"
        return "spindle cooling", f"temperature residual +{td:.1f} C, HF flat ({hf:+.1f} dB)"
    o2, o3 = m.get("db_ord_02", np.nan), m.get("db_ord_3_10", np.nan)
    if kind in ("tsc_pump", "flood_pump", "compressor"):
        if pd.notna(o1) and o1 >= th["ord1_db"]:
            return "imbalance", f"1x +{o1:.1f} dB"
        if kind == "compressor" and pd.notna(o2) and o2 >= th["ord2_db"]:
            return "misalignment", f"2x +{o2:.1f} dB"
        if pd.notna(o3) and o3 >= th["ord_hi_db"]:
            return "looseness", f"orders 3 to 10 +{o3:.1f} dB"
    tsc = m.get("tsc_drop", np.nan)
    if kind == "spindle" and pd.notna(tsc) and tsc >= th["tsc_drop_z"]:
        return "rotary union", f"tsc_drop z {tsc:.1f}"
    return "unclassified", ""


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


def tsc_pressure_daily(
    hourly: pd.DataFrame, ref_days: tuple[int, int], column: str = "coolant_pressure_pump_on_bar"
) -> pd.DataFrame:
    """Per machine and local day: the median of the hourly pressure over hours with pressure above zero
    (by default the pressure while the through-spindle coolant pump runs), the number of such hours, the
    reference median (ref_days[1] to ref_days[0] days earlier, as the TSC pressure drop uses) and the drop
    against it."""
    h = hourly[hourly[column] > 0].copy()
    h["local_date"] = (
        pd.to_datetime(h["hour_utc"], utc=True)
        .dt.tz_convert("America/Chicago")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    near, far = ref_days
    out = []
    for mid, g in h.groupby("machine_id"):
        day = g.groupby("local_date")[column]
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


def suspected(z: dict[str, float], kind: str, zthr: float, d1: bool = False) -> tuple[str, str]:
    """The first family in the asset kind's precedence (specific signatures before generic) whose own
    indicators are all at or above zthr. D1 (2026-11-01, switch advisory.d1_precedence): on spindles,
    lubrication ahead of spindle cooling when temperature and HF both reach zthr, and spindle cooling only
    from the thermistor or from a temperature rise with HF below d1_hf_quiet (2); on pumps, cavitation or
    wear ahead of bearing when HF and order 1 rise together."""

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
    hf_up = hit("hf")
    if kind == "spindle" and d1:
        hf = z.get("hf", np.nan)
        hf_quiet = pd.isna(hf) or hf < D1_HF_QUIET
        lube = hit("hf", "temp")
        if not lube and (hit("thermistor") or (hit("temp") and hf_quiet)):
            keys = ("thermistor",) if hit("thermistor") else ("temp",)
            if hit("tsc_drop", "hf"):
                return "rotary union", text("tsc_drop", "hf")
            return "spindle cooling", text(*keys)
        rules = [
            ("rotary union", ("tsc_drop", "hf")),
            "bearing",
            ("lubrication", ("hf", "temp")),
            ("contamination", ("hf",)),
        ]
    elif kind in ("tsc_pump", "flood_pump") and d1 and hit("hf", "ord_01"):
        return "cavitation or wear", text("hf", "ord_01")
    elif kind == "spindle":
        if hit("thermistor") or (hit("temp") and not hf_up):
            keys = ("thermistor",) if hit("thermistor") else ("temp",)
            if hit("tsc_drop", "hf"):
                return "rotary union", text("tsc_drop", "hf")
            return "spindle cooling", text(*keys)
        rules = [
            ("rotary union", ("tsc_drop", "hf")),
            "bearing",
            ("lubrication", ("hf", "temp")),
            ("contamination", ("hf",)),
        ]
    elif kind == "tsc_pump":
        rules = [
            "bearing",
            ("imbalance", ("ord_01",)),
            ("looseness", ("ord_3_10",)),
            ("cavitation or wear", ("tsc_drop", "hf")),
        ]
    elif kind == "flood_pump":
        rules = [
            "bearing",
            ("imbalance", ("ord_01",)),
            ("looseness", ("ord_3_10",)),
            ("cavitation or wear", ("hf",)),
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
    d1: bool = False,
    d2: dict[str, float] | None = None,
    measures: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows = []
    for c in cases.itertuples():
        z = zs.get(c.case_id, {})
        if d2 is not None:
            cond, detail = suspected_d2((measures or {}).get(c.case_id, {}), asset_kind(c.asset_id), d2)
        else:
            cond, detail = suspected({k: v[0] for k, v in z.items()}, asset_kind(c.asset_id), zthr, d1)
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
    "cavitation": "cavitation or wear",
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
