"""Realism checks 19 to 22 for fault conditions and auxiliary-system degradation, and the confirmed
conditions available for evaluation.

Inputs: bronze exports (scalars, controller and compressor minute tables, CMMS work orders), the
truth archive (point hours, events) and configuration. Outputs: CheckResult rows and the
confirmed-conditions table by type and evaluation period, built from bronze CMMS records only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from generate.checks.realism import CheckResult, healthy_rows
from generate.plant_air import seasonal_c
from pdm.config import Register
from pdm.constants import DAYS_PER_YEAR, SECONDS_PER_MINUTE
from pdm.features.burst import ExtractionSettings

DAY = pd.Timedelta(days=1)
TZ = "America/Chicago"
ORDERS = range(1, 11)


def _events(events: pd.DataFrame, name: str) -> pd.DataFrame:
    return events[events["event"] == name].copy()


def _clean(m: pd.DataFrame) -> pd.Series:
    """Rows with no bearing, lubrication, cavitation or wear activity and no radio noise."""
    return (
        (m["stage"] == 0)
        & (m["lube_severity"] == 0)
        & (~m["cavitation"].astype(bool))
        & (m["wear_frac"] == 0)
        & m["radio_noise"].isna()
    )


def _wide(g: pd.DataFrame, half_col: str) -> pd.DataFrame:
    """One row per hour with radial (RMS of x and y) and axial amplitudes by order band."""
    cols = [f"ord_{o:02d}_g2" for o in ORDERS] + [half_col]
    piv = g[g["axis"].isin(["x", "y", "z"])].pivot_table(index="hour_utc", columns="axis", values=cols)
    out = pd.DataFrame(index=piv.index)
    for o in ORDERS:
        c = f"ord_{o:02d}_g2"
        out[f"r{o}"] = np.sqrt((piv[(c, "x")] + piv[(c, "y")]) / 2)
        out[f"z{o}"] = np.sqrt(piv[(c, "z")])
    out["r3_10"] = np.sqrt(sum(out[f"r{o}"] ** 2 for o in range(3, 11)))
    out["r_half"] = np.sqrt((piv[(half_col, "x")] + piv[(half_col, "y")]) / 2)
    return out


def check_fault_signatures(cfg: dict[str, Any], reg: Register, m: pd.DataFrame) -> CheckResult:
    c = cfg["checks"]
    need = c["fault_signature_share_min"]
    min_level = c["condition_hours_min_level"]
    min_hours = c["condition_min_hours"]
    ratio = c["envelope_detection_ratio"]
    env_max = c["healthy_exceed_share_max"]
    keys = c["healthy_env_fault_keys"]
    s = ExtractionSettings.from_config(cfg)
    edges = s.toct_edges
    if "fault_kind" not in m:
        return CheckResult(
            "19", "Fault signatures by band", "", "no condition columns in this archive", None, None
        )
    pts = [p for p in m["point_id"].unique() if not p.split("-")[-1].startswith("SP")]
    rows = []
    for pid in sorted(pts):
        g = m[m["point_id"] == pid]
        res = g[g["axis"] == "res"].set_index("hour_utc")
        kinds = sorted(k for k in res["fault_kind"].dropna().unique() if "+" not in k)
        if not kinds:
            continue
        shaft_hz = float(res["speed_rpm_at_measurement"].median()) / SECONDS_PER_MINUTE
        b = int(np.searchsorted(edges, 0.5 * shaft_hz, side="right")) - 1
        w = _wide(g, f"toct_{b + 1:02d}_g2")
        healthy_h = res.index[healthy_rows(res.reset_index()).to_numpy()]
        h = w.loc[w.index.intersection(healthy_h)]
        base, p95 = h.median(), h.quantile(0.95)
        clean = res.index[_clean(res.reset_index()).to_numpy()]
        for kind in kinds:
            sel = res.index[
                (res["fault_kind"] == kind).to_numpy() & (res["fault_level"] >= min_level).to_numpy()
            ]
            hrs = w.loc[w.index.intersection(sel).intersection(clean)]
            if hrs.empty:
                continue
            if kind == "imbalance":
                inc = hrs.divide(base)
                largest = (inc["r1"] > inc["r2"]) & (inc["r1"] > inc["r3_10"])
                axial = (hrs["z1"] - base["z1"]) < 0.5 * (hrs["r1"] - base["r1"])
                ok_h = largest & axial
            elif kind == "misalignment":
                rz = np.sqrt(hrs["z1"] ** 2 + hrs["z2"] ** 2) - np.sqrt(base["z1"] ** 2 + base["z2"] ** 2)
                rr = np.sqrt(hrs["r1"] ** 2 + hrs["r2"] ** 2) - np.sqrt(base["r1"] ** 2 + base["r2"] ** 2)
                ok_h = (hrs["r2"] > p95["r2"]) & (rz >= 0.5 * rr)
            else:
                ok_h = (hrs["r3_10"] > p95["r3_10"]) & (hrs["r_half"] > p95["r_half"])
            env = (res.loc[hrs.index, [f"res_envr_{k}" for k in keys]] > ratio).any(axis=1)
            rows.append(
                {
                    "point_id": pid,
                    "condition": kind,
                    "hours": len(hrs),
                    "tested": len(hrs) >= min_hours,
                    "signature_share": float(ok_h.mean()),
                    "env_fault_peak_share": float(env.mean()),
                    "r1_increase": float((hrs["r1"] / base["r1"]).median()),
                    "r2_increase": float((hrs["r2"] / base["r2"]).median()),
                    "r3_10_increase": float((hrs["r3_10"] / base["r3_10"]).median()),
                    "half_x_increase": float((hrs["r_half"] / base["r_half"]).median()),
                    "vel_rms_median_mm_s": float(res.loc[hrs.index, "vel_rms_mm_s"].median()),
                    "vel_rms_healthy_median_mm_s": float(
                        res.loc[res.index.intersection(healthy_h), "vel_rms_mm_s"].median()
                    ),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return CheckResult("19", "Fault signatures by band", "", "no condition hours", None, df)
    t = df[df["tested"]]
    ok = (
        bool(((t["signature_share"] >= need) & (t["env_fault_peak_share"] <= env_max)).all())
        if len(t)
        else None
    )
    measured = "; ".join(
        f"{r.point_id} {r.condition} {r.signature_share:.0%} of {r.hours} h, env {r.env_fault_peak_share:.0%}"
        + ("" if r.tested else " (report-only)")
        for r in df.itertuples()
    )
    return CheckResult(
        "19",
        "Fault signatures by band",
        f">= {need:.0%} of condition hours (level >= {min_level}, single condition, no other mechanism; "
        f"tested at >= {min_hours} h): imbalance 1x radial largest increase and axial increment "
        "< 50% of radial; misalignment 2x above healthy p95 and axial increment >= 50% of radial "
        "(1x and 2x); looseness 3x-10x and 0.5x above healthy p95; envelope fault peaks in "
        f"<= {env_max:.0%} of hours",
        measured,
        ok,
        df,
    )


def check_post_reinstallation(
    cfg: dict[str, Any], events: pd.DataFrame, sensor_events: pd.DataFrame
) -> CheckResult:
    c = cfg["checks"]
    p_cfg = cfg["fault_conditions"]["misalignment"]["post_reinstallation_prob"]
    alpha = c["post_reinstallation_binomial_alpha"]
    max_days = c["post_reinstallation_correction_max_days"]
    rein = _events(events, "reinstallation")
    starts = _events(events, "condition_start")
    starts = starts[(starts["kind"] == "misalignment") & (starts["cause"] == "post_reinstallation")]
    clears = _events(events, "condition_cleared")
    clears = clears[(clears["kind"] == "misalignment") & (clears["cause"] == "post_reinstallation")]
    n, k = len(rein), len(starts)
    if n == 0:
        return CheckResult(
            "20",
            "Post-reinstallation misalignment",
            f"share consistent with {p_cfg:.0%} (binomial p > {alpha}); corrected <= {max_days} d on the WO",
            "no compressor reinstallations",
            None,
            None,
        )
    pv = binomtest(k, n, p_cfg).pvalue
    rows, ok_cases = [], True
    for _, st in starts.iterrows():
        r = rein[(rein["asset_id"] == st["asset_id"]) & (rein["ts"] == st["ts"])]
        cl = clears[(clears["asset_id"] == st["asset_id"]) & (clears["ts"] >= st["ts"])].head(1)
        corrected = cl["ts"].iat[0] if not cl.empty else pd.NaT
        days = (corrected - st["ts"]) / DAY if pd.notna(corrected) else np.nan
        same_wo = bool(not r.empty and not cl.empty and r["wo_id"].iat[0] == cl["wo_id"].iat[0])
        pts = [f"{st['asset_id']}-MDE", f"{st['asset_id']}-AE"]
        bl = sensor_events[
            sensor_events["point_id"].isin(pts)
            & (sensor_events["event"] == "baselining_start")
            & (sensor_events["ts"] >= st["ts"])
        ]
        overlap = bool(pd.isna(corrected) or (bl["ts"] < corrected).any())
        good = bool(days <= max_days and same_wo and not overlap)
        ok_cases &= good
        rows.append(
            {
                "asset_id": st["asset_id"],
                "reinstalled": st["ts"],
                "corrected": corrected,
                "days": days,
                "same_wo": same_wo,
                "baseline_before_correction": overlap,
            }
        )
    n_good = sum(r["days"] <= max_days and r["same_wo"] and not r["baseline_before_correction"] for r in rows)
    return CheckResult(
        "20",
        "Post-reinstallation misalignment",
        f"share consistent with {p_cfg:.0%} (binomial p > {alpha}); each corrected <= {max_days} d on the "
        "reinstallation WO; no post-repair baseline starts before the correction",
        f"{k} of {n} compressor reinstallations (p = {pv:.2f}); corrected in time on the same WO with "
        f"no early baseline {n_good}/{k}",
        bool(pv > alpha and ok_cases),
        pd.DataFrame(rows) if rows else None,
    )


def _read_parts(root: Path, pattern: str, columns: list[str]) -> pd.DataFrame:
    files = sorted(root.glob(pattern))
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def check_ingress(
    cfg: dict[str, Any], reg: Register, bronze_root: Path, events: pd.DataFrame, end: pd.Timestamp
) -> CheckResult:
    c = cfg["checks"]
    min_drop = c["tsc_leak_drop_detect_min_frac"]
    nominal = cfg["controller_signals"]["tsc_bar"]["nominal"]
    starts = _events(events, "condition_start")
    clears = _events(events, "condition_cleared")
    leaks = starts[starts["kind"] == "rotary_union_leak"].sort_values("ts")
    ingress = starts[starts["kind"] == "coolant_ingress"].sort_values("ts")
    if leaks.empty:
        return CheckResult("21", "Coolant ingress link", "", "no rotary union leaks", None, None)
    minute: dict[str, pd.DataFrame] = {}
    for mid in sorted({a[:6] for a in leaks["asset_id"]}):
        df = _read_parts(
            bronze_root / "controller" / "minute_agg",
            f"month=*/{mid}.parquet",
            ["minute_utc", "execution", "coolant_pressure_bar_mean"],
        )
        df = df[(df["execution"] == "active") & (df["coolant_pressure_bar_mean"] > 0.5 * nominal)]
        minute[mid] = df.set_index("minute_utc")["coolant_pressure_bar_mean"].sort_index()
    rows, ok = [], True
    for _, lk in leaks.iterrows():
        mid = lk["asset_id"][:6]
        cl = clears[
            (clears["kind"] == "rotary_union_leak")
            & (clears["asset_id"] == lk["asset_id"])
            & (clears["ts"] >= lk["ts"])
        ]
        leak_end = cl["ts"].iat[0] if not cl.empty else end
        ing = ingress[(ingress["asset_id"] == lk["asset_id"]) & ingress["ts"].between(lk["ts"], leak_end)]
        upto = ing["ts"].iat[0] if not ing.empty else leak_end
        s = minute[mid]
        pre = s[(s.index >= lk["ts"] - 7 * DAY) & (s.index < lk["ts"])]
        during = s[(s.index >= lk["ts"]) & (s.index < upto)]
        drop = 1 - during.median() / pre.median() if len(pre) and len(during) else np.nan
        rows.append(
            {
                "asset_id": lk["asset_id"],
                "leak_start": lk["ts"],
                "ingress_start": ing["ts"].iat[0] if not ing.empty else pd.NaT,
                "leak_cleared": cl["ts"].iat[0] if not cl.empty else pd.NaT,
                "configured_drop": lk.get("p_drop", np.nan),
                "measured_drop_before_ingress": drop,
            }
        )
    df = pd.DataFrame(rows)
    orphan = 0
    for _, ig in ingress.iterrows():
        prior = df[(df["asset_id"] == ig["asset_id"]) & (df["leak_start"] <= ig["ts"])]
        if prior.empty:
            orphan += 1
    with_ing = df[df["ingress_start"].notna()]
    ok = orphan == 0 and bool((with_ing["measured_drop_before_ingress"] >= min_drop).all())
    onsets = _events(events, "bearing_onset")
    onsets = onsets[onsets["component_id"].astype(str).str.endswith("SP-F")]
    first_ing = ingress.groupby(ingress["asset_id"].str[:6])["ts"].min()
    hmcs = [mid for mid, v in reg.machines.items() if v["class"] == "hmc"]
    n_after, n_without = 0, 0
    for mid in hmcs:
        t_ing = first_ing.get(mid)
        o = onsets[onsets["component_id"].astype(str).str.startswith(mid)]
        if t_ing is None:
            n_without += len(o)
        else:
            n_after += int((o["ts"] >= t_ing).sum())
            n_without += int((o["ts"] < t_ing).sum())
    year = pd.Timedelta(days=DAYS_PER_YEAR)
    start = pd.Timestamp(str(cfg["meta"]["record"]["start"]), tz=TZ).tz_convert("UTC")
    exp_after = sum((end - t_) / year for t_ in first_ing)
    exp_without = len(hmcs) * (end - start) / year - exp_after
    rate_txt = (
        f"front-set onsets during or after ingress {n_after} in {exp_after:.1f} spindle-years "
        f"({n_after / exp_after:.2f}/yr) vs {n_without} in {exp_without:.1f} without "
        f"({n_without / exp_without:.2f}/yr), report-only"
        if exp_after > 0
        else f"no ingress; front-set onsets {n_without}"
    )
    return CheckResult(
        "21",
        "Coolant ingress link",
        f"every ingress preceded by a rotary union leak with a measured TSC pressure drop >= {min_drop:.1%} "
        "(median active-minute pressure, 7 days before the leak against leak start to ingress); "
        "onset rates report-only",
        f"{len(df)} leaks, {len(ingress)} ingress episodes, ingress without a preceding leak {orphan}; "
        "measured drop before ingress: "
        + (
            ", ".join(f"{r.asset_id} {r.measured_drop_before_ingress:.1%}" for r in with_ing.itertuples())
            or "none"
        )
        + f"; {rate_txt}",
        bool(ok),
        df,
    )


def _episode_rise(
    daily: pd.Series, start: pd.Timestamp, clear: pd.Timestamp | None, end: pd.Timestamp, base_days: int
) -> tuple[float, float]:
    base = daily[(daily.index >= start - base_days * DAY) & (daily.index < start)]
    upto = clear if clear is not None and pd.notna(clear) else end
    during = daily[(daily.index >= start) & (daily.index <= upto)]
    if base.empty or during.empty:
        return np.nan, np.nan
    return float(base.median()), float(during.max() - base.median())


def _velocity_in_range(
    m: pd.DataFrame, pts: list[str], t0: pd.Timestamp, t1: pd.Timestamp, min_days: float = 0.0
) -> bool | None:
    """Episode velocity median within the point's healthy p05-p95; None when the episode is shorter
    than min_days."""
    if (t1 - t0) < min_days * DAY:
        return None
    res = m[(m["axis"] == "res") & m["point_id"].isin(pts)]
    ok = True
    seen = False
    for _, g in res.groupby("point_id"):
        h = g[healthy_rows(g)]
        ep = g[
            g["hour_utc"].between(t0, t1)
            & _clean(g)
            & (g["fault_kind"].isna() if "fault_kind" in g else True)
        ]
        if h.empty or ep.empty:
            continue
        seen = True
        med = ep["vel_rms_mm_s"].median()
        ok &= bool(h["vel_rms_mm_s"].quantile(0.05) <= med <= h["vel_rms_mm_s"].quantile(0.95))
    return ok if seen else None


def check_thermal(
    cfg: dict[str, Any],
    reg: Register,
    bronze_root: Path,
    events: pd.DataFrame,
    m: pd.DataFrame,
    end: pd.Timestamp,
    machine_hours: pd.DataFrame | None = None,
) -> CheckResult:
    c = cfg["checks"]
    tol = c["thermal_residual_tolerance_c"]
    ret_days = c["thermal_return_days"]
    need = c["thermal_return_share_min"]
    base_days = c["thermal_baseline_days"]
    vel_days = c["thermal_velocity_min_days"]
    starts = _events(events, "condition_start")
    clears = _events(events, "condition_cleared")
    rows = []

    fo = starts[starts["kind"] == "cooler_fouling"]
    lo, hi = cfg["oil_cooler_fouling"]["discharge_temp_rise_c"]
    if len(fo):
        mn = _read_parts(
            bronze_root / "compressor_ctrl" / "minute",
            "month=*/compressor_minute.parquet",
            ["compressor_id", "minute_utc", "state", "discharge_temp_c"],
        )
        room = _read_parts(
            bronze_root / "compressor_ctrl" / "room", "month=*/room.parquet", ["minute_utc", "ambient_c"]
        )
        mn = mn.merge(room, on="minute_utc", how="left")
        mn = mn[mn["state"] != "off"].assign(
            hour=mn["minute_utc"].dt.floor("h"), loaded=(mn["state"] == "loaded").astype(float)
        )
        hr = mn.groupby(["compressor_id", "hour"]).agg(
            temp=("discharge_temp_c", "mean"), amb=("ambient_c", "mean"), load=("loaded", "mean")
        )
        dt = cfg["compressor"]["discharge_temp_c"]
        amb_mean = cfg["benign"]["ambient_room_c"]["annual_mean"]
        hr["resid"] = hr["temp"] - (
            dt["base"] + dt["per_c_ambient"] * (hr["amb"] - amb_mean) + dt["per_load_frac"] * hr["load"]
        )
        hr = hr.reset_index()
        hr["day"] = hr["hour"].dt.tz_convert(TZ).dt.normalize().dt.tz_convert("UTC")
        for _, st in fo.iterrows():
            cid = st["asset_id"]
            daily = hr[hr["compressor_id"] == cid].groupby("day")["resid"].median()
            cl = clears[
                (clears["kind"] == "cooler_fouling")
                & (clears["asset_id"] == cid)
                & (clears["ts"] >= st["ts"])
            ].head(1)
            t_clear = cl["ts"].iat[0] if not cl.empty else None
            level = float(cl["level"].iat[0]) if not cl.empty else np.nan
            base, rise = _episode_rise(daily, st["ts"], t_clear, end, base_days)
            returned = None
            if t_clear is not None and pd.notna(base):
                after = daily[
                    (daily.index >= t_clear + ret_days * DAY) & (daily.index < t_clear + (ret_days + 4) * DAY)
                ]
                returned = bool(after.iloc[0] - base <= tol) if len(after) else None
            completed = t_clear is None or level >= 0.999
            within = (
                None
                if pd.isna(rise)
                else bool(rise <= hi + tol and (rise >= lo - tol if completed else True))
            )
            rows.append(
                {
                    "mechanism": "cooler_fouling",
                    "asset_id": cid,
                    "start": st["ts"],
                    "cleared": t_clear,
                    "cleared_by": cl["by"].iat[0] if not cl.empty else None,
                    "configured_rise_c": st.get("p_rise_c", np.nan),
                    "level_at_clear": level,
                    "measured_rise_c": rise,
                    "rise_within_range": within,
                    "returned_by_day3": returned,
                    "velocity_in_healthy_range": _velocity_in_range(
                        m,
                        [f"{cid}-MDE", f"{cid}-AE"],
                        st["ts"],
                        t_clear if t_clear is not None else end,
                        vel_days,
                    ),
                }
            )

    sc = starts[starts["kind"] == "spindle_cooling"]
    lo_s, hi_s = cfg["spindle_cooling"]["thermistor_residual_rise_c"]
    if len(sc):
        cs = cfg["controller_signals"]["spindle_temp"]
        sa = cfg["benign"]["shop_ambient_c"]
        coldest = cfg["benign"]["ambient_room_c"]["coldest_day_of_year"]
        for mid in sorted({a[:6] for a in sc["asset_id"]}):
            df = _read_parts(
                bronze_root / "controller" / "minute_agg",
                f"month=*/{mid}.parquet",
                [
                    "minute_utc",
                    "execution",
                    "seconds_in_state",
                    "spindle_speed_rpm_mean",
                    "spindle_temp_c_mean",
                ],
            )
            df = df[(df["execution"] == "active") & (df["seconds_in_state"] == SECONDS_PER_MINUTE)]
            local = df["minute_utc"].dt.tz_convert(TZ)
            amb = seasonal_c(
                local.dt.dayofyear.to_numpy(float), sa["annual_mean"], sa["annual_amplitude"], coldest
            )
            rated = float(reg.machines[mid]["rated_rpm"])
            expected = amb + cs["rise_c_at_rated"] * df["spindle_speed_rpm_mean"].to_numpy() / rated
            resid = df["spindle_temp_c_mean"].to_numpy() - expected
            daily = (
                pd.Series(resid, index=local.dt.normalize().dt.tz_convert("UTC")).groupby(level=0).median()
            )
            for _, st in sc[sc["asset_id"].str.startswith(mid)].iterrows():
                cl = clears[
                    (clears["kind"] == "spindle_cooling")
                    & (clears["asset_id"] == st["asset_id"])
                    & (clears["ts"] >= st["ts"])
                ].head(1)
                t_clear = cl["ts"].iat[0] if not cl.empty else None
                level = float(cl["level"].iat[0]) if not cl.empty else np.nan
                base, rise = _episode_rise(daily, st["ts"], t_clear, end, base_days)
                completed = t_clear is None or level >= 0.999
                within = (
                    None
                    if pd.isna(rise)
                    else bool(rise <= hi_s + tol and (rise >= lo_s - tol if completed else True))
                )
                rows.append(
                    {
                        "mechanism": "spindle_cooling",
                        "asset_id": st["asset_id"],
                        "start": st["ts"],
                        "cleared": t_clear,
                        "cleared_by": cl["by"].iat[0] if not cl.empty else None,
                        "configured_rise_c": st.get("p_rise_c", np.nan),
                        "level_at_clear": level,
                        "measured_rise_c": rise,
                        "rise_within_range": within,
                        "returned_by_day3": None,
                        "velocity_in_healthy_range": _velocity_in_range(
                            m,
                            [f"{mid}-SP-F", f"{mid}-SP-R"],
                            st["ts"],
                            t_clear if t_clear is not None else end,
                            vel_days,
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return CheckResult("22", "Thermal mechanisms", "", "no fouling or spindle cooling episodes", None, df)
    rise_ok = bool(df["rise_within_range"].dropna().all())
    ret = df.loc[df["mechanism"] == "cooler_fouling", "returned_by_day3"].dropna()
    ret_share = float(ret.mean()) if len(ret) else np.nan
    ret_ok = True if not len(ret) else bool(ret_share >= need)
    vel = df["velocity_in_healthy_range"].dropna()
    vel_ok = bool(vel.all())
    by = df.groupby("mechanism")["rise_within_range"].apply(
        lambda s: f"{int(s.dropna().sum())}/{s.notna().sum()}"
    )
    sp_ok, sp_text = True, ""
    if machine_hours is not None:
        sp_ok, sp_text = spindle_overheat_consistency(cfg, reg, bronze_root, events, machine_hours, end)
    return CheckResult(
        "22",
        "Thermal mechanisms",
        f"measured residual rise within the configured range +/- {tol} C (lower bound only for episodes "
        f"that reached full rise); compressor residual back within {tol} C of its pre-episode level on "
        f"day {ret_days} after cleaning in >= {need:.0%}; velocity median during episodes of at least "
        f"{vel_days} days within the point's healthy p05-p95; every cooler overheat stop has a rendered "
        "thermistor reading at the limit in the stop hour; no healthy cutting hour reaches the limit; "
        "operator notice share consistent with configured; no episode open longer than one PM interval "
        "after 2 C without a recorded PM check or a skipped or late PM explaining it",
        f"rise within range: {', '.join(f'{k} {v}' for k, v in by.items())}; compressor return by day "
        f"{ret_days}: "
        + (f"{int(ret.sum())}/{len(ret)} ({ret_share:.0%})" if len(ret) else "no measurable cleaned episodes")
        + f"; velocity in healthy range {int(vel.sum())}/{len(vel)}"
        + sp_text,
        bool(rise_ok and ret_ok and vel_ok and sp_ok),
        df,
    )


def _hmc_minutes(bronze_root: Path, mid: str) -> pd.DataFrame:
    df = _read_parts(
        bronze_root / "controller" / "minute_agg",
        f"month=*/{mid}.parquet",
        ["minute_utc", "execution", "seconds_in_state", "spindle_temp_c_mean"],
    )
    return df[df["seconds_in_state"] == SECONDS_PER_MINUTE].assign(
        hour_utc=lambda d: d["minute_utc"].dt.floor("h")
    )


def spindle_overheat_consistency(
    cfg: dict[str, Any],
    reg: Register,
    bronze_root: Path,
    events: pd.DataFrame,
    machine_hours: pd.DataFrame,
    end: pd.Timestamp,
) -> tuple[bool, str]:
    """Check 22 additions for the spindle thermistor limit, the operator notice and the PM path."""
    c, sc = cfg["checks"], cfg["spindle_cooling"]
    lim, tol = sc["overheat_limit_c"], c["overheat_reading_tolerance_c"]
    hmcs = [mid for mid, v in reg.machines.items() if v["class"] == "hmc"]
    fails = events[(events["event"] == "unplanned_failure") & (events["mode"] == "spindle_cooler")]
    stop_ok, stop_n, healthy_max = 0, 0, -np.inf
    for mid in hmcs:
        mins = _hmc_minutes(bronze_root, mid)
        hourly_max = mins.groupby("hour_utc")["spindle_temp_c_mean"].max()
        for _, f in fails[fails["asset_id"].str.startswith(mid)].iterrows():
            stop_n += 1
            stop_ok += int(hourly_max.get(f["ts"].floor("h"), -np.inf) >= lim - tol)
        mh = machine_hours[machine_hours["machine_id"] == mid]
        healthy = mh[
            (mh["cut_min"] > 0)
            & (mh["cooler_residual_c"].fillna(0) == 0)
            & (mh["lube_temp_c"].fillna(0) == 0)
            & (mh["spindle_s4"].fillna(0) == 0)
        ]["hour_utc"]
        hm = hourly_max.reindex(pd.DatetimeIndex(healthy)).dropna()
        if len(hm):
            healthy_max = max(healthy_max, float(hm.max()))
    starts = events[(events["event"] == "condition_start") & (events["kind"] == "spindle_cooling")]
    notice = starts["p_notice"].astype(bool) if "p_notice" in starts else pd.Series(dtype=bool)
    p_notice = (
        binomtest(int(notice.sum()), len(notice), sc["operator_notice_prob"]).pvalue
        if len(notice)
        else np.nan
    )
    clears = events[(events["event"] == "condition_cleared") & (events["kind"] == "spindle_cooling")]
    pmc = sc["pm_check"]
    pm = pd.read_parquet(bronze_root / "cmms" / "pm_schedule.parquet")
    pm = pm[pm["pm_task"] == pmc["task"]]
    interval = pd.Timedelta(days=cfg["maintenance"]["pm_program"]["hmc"][pmc["task"]]["interval"])
    gaps, unexplained = 0, 0
    for _, st in starts.iterrows():
        t2 = st["ts"] + pd.Timedelta(
            hours=float(st["ramp_h"]) * pmc["min_residual_c"] / float(st["p_rise_c"])
        )
        cl = clears[(clears["asset_id"] == st["asset_id"]) & (clears["ts"] >= st["ts"])]
        t_end = cl["ts"].iat[0] if not cl.empty else end
        if t_end - t2 <= interval:
            continue
        rows = pm[pm["asset_id"] == st["asset_id"]]
        done = rows[rows["completed"].between(t2, t_end)]
        if not done.empty:
            continue
        gaps += 1
        due = rows[rows["due"].between(t2, t_end)]
        if not (due["outcome"].isin(["skipped", "late"])).any():
            unexplained += 1
    ok = (
        stop_ok == stop_n
        and healthy_max < lim
        and (p_notice > c["operator_notice_binomial_alpha"] if pd.notna(p_notice) else True)
        and unexplained == 0
    )
    text = (
        f"; overheat stops with reading >= {lim - tol:g} C in the stop hour {stop_ok}/{stop_n}; "
        f"healthy cutting hours max {healthy_max:.1f} C (limit {lim:g}); "
        f"operator notice drawn {int(notice.sum())}/{len(notice)} "
        f"(p = {p_notice:.2f}); PM path gaps {gaps}, unexplained {unexplained}"
    )
    return bool(ok), text


def spindle_cooling_outcomes(events: pd.DataFrame, wo_truth: pd.DataFrame) -> dict[str, Any]:
    """Episode outcomes (noticed, found at PM, stopped, open at record end) and alarm-driven inspections."""
    starts = events[(events["event"] == "condition_start") & (events["kind"] == "spindle_cooling")]
    found = events[(events["event"] == "condition_found") & (events["kind"] == "spindle_cooling")]
    clears = events[(events["event"] == "condition_cleared") & (events["kind"] == "spindle_cooling")]
    stops = events[(events["event"] == "unplanned_failure") & (events["mode"] == "spindle_cooler")]
    rows = []
    for _, st in starts.iterrows():
        cl = clears[(clears["asset_id"] == st["asset_id"]) & (clears["ts"] >= st["ts"])].head(1)
        fd = found[(found["asset_id"] == st["asset_id"]) & (found["ts"] >= st["ts"])]
        if cl.empty:
            outcome = "open at record end"
        elif cl["wo_id"].iat[0] in set(stops["wo_id"]):
            outcome = "overheat stop"
        elif not fd.empty and fd["ts"].iat[0] <= cl["ts"].iat[0]:
            outcome = "noticed by operator" if fd["by"].iat[0] == "operator" else "found at PM"
        else:
            outcome = "other"
        rows.append(
            {
                "asset_id": st["asset_id"],
                "start": st["ts"],
                "rise_c": st.get("p_rise_c", np.nan),
                "notice_drawn": bool(st.get("p_notice", False)),
                "outcome": outcome,
                "ended": cl["ts"].iat[0] if not cl.empty else pd.NaT,
                "residual_at_end_c": float(cl["level"].iat[0]) * float(st.get("p_rise_c", np.nan))
                if not cl.empty
                else np.nan,
            }
        )
    alarms = events[events["event"] == "overheat_alarm"]
    opened = events[events["event"] == "alarm_inspection_opened"]
    insp = wo_truth[wo_truth["wo_id"].isin(set(opened["wo_id"]))] if len(opened) else wo_truth.iloc[0:0]
    return {
        "episodes": pd.DataFrame(rows),
        "alarms": len(alarms),
        "alarm_inspections": insp[["wo_id", "asset_id", "opened", "notes"]] if len(insp) else insp,
    }


_BEARING_CODES = {"BRG-WEAR", "BRG-SPALL", "BRG-LUBE", "PMP-BRG", "CMP-BRG-MOTOR", "CMP-BRG-AIREND"}
_TYPE_BY_CODE = {
    "imbalance": "imbalance",
    "misalignment": "misalignment",
    "looseness": "looseness",
    "rotary_union_leak": "rotary union",
    "coolant_ingress": "rotary union",
    "oil_cooler_fouled": "cooler (compressor)",
    "spindle_cooler_fault": "spindle cooling",
    "CRASH-DMG": "crash",
    "PMP-SEAL": "pump seal or impeller",
    "PMP-IMP": "pump seal or impeller",
    "CMP-TRIP": "compressor trip",
    "CMP-VALVE": "compressor trip",
}


def _condition_type(r: Any) -> str | None:
    code = r.failure_code if isinstance(r.failure_code, str) else None
    notes = r.notes if isinstance(r.notes, str) else ""
    if r.wo_type == "inspection":
        return "lubrication" if "lubrication" in notes else None
    if r.wo_type == "pm":
        return _TYPE_BY_CODE.get(code) if code else None
    if r.wo_type not in ("unplanned_repair", "condition_based", "corrective"):
        return None
    if code is None:
        return "unplanned, no code" if r.wo_type == "unplanned_repair" else None
    if code in _BEARING_CODES:
        return "bearing (unplanned)" if r.wo_type == "unplanned_repair" else "bearing (condition-based)"
    if code in _TYPE_BY_CODE:
        return _TYPE_BY_CODE[code]
    return "other machine stoppage" if r.wo_type == "unplanned_repair" else None


def confirmed_conditions(
    cfg: dict[str, Any], bronze_root: Path, events: pd.DataFrame, end: pd.Timestamp
) -> dict[str, Any]:
    """Confirmed conditions by type and evaluation period from bronze CMMS work orders only."""
    wo = pd.read_parquet(bronze_root / "cmms" / "work_orders.parquet")
    wo = wo.assign(ctype=[_condition_type(r) for r in wo.itertuples()]).dropna(subset=["ctype"])
    sp = cfg["splits"]
    local_day = wo["opened"].dt.tz_convert(TZ).dt.date.astype(str)
    period = np.select(
        [
            local_day.between(str(sp["train"][0]), str(sp["train"][1])),
            local_day.between(str(sp["validation"][0]), str(sp["validation"][1])),
            local_day.between(str(sp["shadow"][0]), str(sp["shadow"][1])),
        ],
        ["training", "validation", "shadow"],
        default="outside",
    )
    wo = wo.assign(period=period)
    tab = wo.groupby(["ctype", "period"]).size().unstack(fill_value=0)
    for col in ("training", "validation", "shadow"):
        if col not in tab:
            tab[col] = 0
    tab = tab[["training", "validation", "shadow"]]
    tab["jul2025_jun2026"] = tab["validation"] + tab["shadow"]
    tab = tab.sort_values("jul2025_jun2026", ascending=False)
    total = tab.sum().rename("total")
    tab = pd.concat([tab, total.to_frame().T])
    tab.index.name = "type"
    open_bronze = wo[(wo["closed"].isna()) | (wo["closed"] > end)]
    starts = _events(events, "condition_start")
    clears = _events(events, "condition_cleared")
    open_truth = []
    for _, st in starts.iterrows():
        cl = clears[
            (clears["kind"] == st["kind"])
            & (clears["asset_id"] == st["asset_id"])
            & (clears["ts"] >= st["ts"])
        ]
        if cl.empty:
            open_truth.append(f"{st['kind']} on {st['asset_id']} since {st['ts'].tz_convert(TZ):%Y-%m-%d}")
    return {
        "table": tab.reset_index(),
        "jul2025_jun2026": int(total["jul2025_jun2026"]),
        "open_bronze": open_bronze[["wo_id", "asset_id", "wo_type", "failure_code", "opened", "closed"]],
        "open_truth": open_truth,
    }
