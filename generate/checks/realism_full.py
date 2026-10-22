"""Full-run realism checks 5 to 15 and 17, and the section 16 headline counts.

Inputs: bronze exports, the truth archive and configuration. Output: CheckResult rows and a
headline-count table with the approved count rule applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, poisson

from generate.checks import realism_conditions as rc
from generate.checks.realism import _ISO_KEY, CheckResult, _first_consecutive
from pdm.config import Register
from pdm.constants import DAYS_PER_YEAR, SECONDS_PER_MINUTE
from pdm.ingest.mtconnect_stream import AGG_DECIMALS, aggregate_minutes, compare_aggregates
from pdm.quality.implausible import flag_point
from pdm.rules.iso_zones import zone_boundaries
from pdm.rules.vendor_baseline import VendorRuleSettings
from pdm.rules.vendor_replay import replay_point

DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(weeks=1)


def _machine_ids(reg: Register) -> list[str]:
    return list(reg.machines)


def _is_machine_asset(asset: str, reg: Register) -> bool:
    return any(asset == m or asset.startswith(f"{m}-") for m in reg.machines)


def _instrumented_asset(asset: str) -> bool:
    return asset.startswith("CMP") or (asset.startswith("HMC") and asset.endswith(("-SP", "-TSC", "-CP")))


def failures(events: pd.DataFrame) -> pd.DataFrame:
    cols = ["ts", "asset_id", "mode", "components", "wo_id"]
    return events.reindex(columns=sorted(set(events.columns) | set(cols)))[
        events["event"] == "unplanned_failure"
    ].copy()


def check_lead_times(
    cfg: dict[str, Any], m: pd.DataFrame, events: pd.DataFrame, reg: Register
) -> CheckResult:
    c = cfg["checks"]
    er = c["expected_ranges"]["lead_weeks_median"]
    ratio = c["envelope_detection_ratio"]
    keys = c["healthy_env_fault_keys"]
    fails = failures(events)
    fails = fails[fails["mode"] == "bearing"]
    rows = []
    for _, f in fails.iterrows():
        for comp in str(f["components"]).split(","):
            g = m[
                (m["point_id"] == comp)
                & (m["axis"] == "res")
                & m["radio_noise"].isna()
                & (m["hour_utc"] < f["ts"])
            ]
            g = g[g["stage"] >= 2].sort_values("hour_utc")
            if g.empty:
                continue
            g = g[g["instance_id"] == g["instance_id"].iloc[-1]]
            det = _first_consecutive(g["hour_utc"], (g[[f"res_envr_{k}" for k in keys]] > ratio).any(axis=1))
            cls = reg.point_class(comp)
            group = (
                "pump"
                if cls.endswith("pump")
                else ("compressor" if cls.startswith("compressor") else "spindle")
            )
            rows.append(
                {
                    "component_id": comp,
                    "class": group,
                    "failure": f["ts"],
                    "first_stage2_envelope": det,
                    "lead_weeks": (f["ts"] - det) / WEEK if det is not None else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return CheckResult(
            "5",
            "Lead times",
            "spindle/compressor 4-12 wk, pump 2-6 wk",
            "no failed sensored bearings",
            None,
            df,
        )
    med = df.groupby("class")["lead_weeks"].median()
    ok, parts = True, []
    for cls, val in med.items():
        lo, hi = er[cls]
        parts.append(f"{cls} {val:.1f} wk (n={int((df['class'] == cls).sum())})")
        ok &= bool(lo <= val <= hi)
    return CheckResult(
        "5",
        "Lead times (median, first stage-2 envelope detection to failure)",
        "spindle and compressor 4-12 wk, pump 2-6 wk",
        "; ".join(parts),
        ok,
        df,
    )


MAD_TO_SD = 1.4826
POISSON_90_TAILS = (0.05, 0.95)


def _robust(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.groupby(keys)["vel_rms_mm_s"]
    center = g.median()
    spread = g.apply(lambda s: MAD_TO_SD * np.median(np.abs(s - np.median(s))))
    return pd.DataFrame({"center": center, "spread": spread})


def check_regime(cfg: dict[str, Any], m: pd.DataFrame, sensor_events: pd.DataFrame) -> CheckResult:
    lo, hi = cfg["checks"]["expected_ranges"]["regime_velocity_diff_frac"]
    bands = cfg["regime"]["load_bands_pct"]
    rules = cfg["rules"]
    iso = rules["iso_class_by_point_class"]["spindle"]
    ab, bc, _ = zone_boundaries(iso["group"], iso["foundation"])
    offset = rules["vendor"]["alert_frac_zone_b_width"] * (bc - ab)
    persist = rules["persistence_consecutive"]
    base_days = pd.Timedelta(days=cfg["measurement"]["baselining_days"])

    def band_of(load: pd.Series) -> np.ndarray:
        return np.where(
            load < bands["light"][1], "light", np.where(load >= bands["heavy"][0], "heavy", "medium")
        )

    healthy = (
        (m["stage"] == 0)
        & (m["lube_severity"] == 0)
        & m["radio_noise"].isna()
        & m["point_id"].str.contains("-SP-")
    )
    d = m[healthy & (m["axis"] != "res")].copy()
    d["band"] = band_of(d["load_pct_at_measurement"])
    med = d.groupby("band")["vel_rms_mm_s"].median()
    diff = med.get("heavy", np.nan) / med.get("light", np.nan) - 1

    r = m[healthy & (m["axis"] == "res")].copy()
    r["band"] = band_of(r["load_pct_at_measurement"])
    rows = []
    for pid, g in r.groupby("point_id"):
        g = g.sort_values("hour_utc")
        bounds = sorted(
            sensor_events[
                (sensor_events["point_id"] == pid)
                & sensor_events["event"].isin(["install", "remount", "baselining_start"])
            ]["ts"]
        )
        bounds.append(g["hour_utc"].max() + pd.Timedelta(hours=1))
        ign, cond = 0, 0
        for k in range(len(bounds) - 1):
            gp = g[(g["hour_utc"] >= bounds[k]) & (g["hour_utc"] < bounds[k + 1])]
            win = gp[gp["hour_utc"] < bounds[k] + base_days]
            if win.empty:
                continue
            g_base = win["vel_rms_mm_s"].median()
            b_base = win.groupby("band")["vel_rms_mm_s"].median()
            after = gp[gp["hour_utc"] >= bounds[k] + base_days]
            over_g = (after["vel_rms_mm_s"] >= g_base + offset).to_numpy()
            thr_b = after["band"].map(b_base).fillna(g_base).to_numpy() + offset
            over_b = after["vel_rms_mm_s"].to_numpy() >= thr_b
            for arr, key in ((over_g, "ign"), (over_b, "cond")):
                run, n = 0, 0
                for v in arr:
                    run = run + 1 if v else 0
                    n += int(run == persist)
                if key == "ign":
                    ign += n
                else:
                    cond += n
        rows.append(
            {
                "point_id": pid,
                "healthy_hours": len(g),
                "one_baseline_alerts": ign,
                "per_band_baseline_alerts": cond,
            }
        )
    detail = pd.DataFrame(rows)
    return CheckResult(
        "6",
        "Regime dependence",
        f"heavy vs light {lo:.0%}-{hi:.0%} (pass condition); fixed-offset comparison report-only",
        f"heavy/light {diff:+.0%}; report-only, healthy-hour alerts from the platform offset rule (+{offset:.2f} mm/s, "
        f"{persist} consecutive): one baseline per point {int(detail['one_baseline_alerts'].sum())}, "
        f"per point and load band {int(detail['per_band_baseline_alerts'].sum())}",
        bool(lo <= diff <= hi),
        detail,
    )


def check_lube(cfg: dict[str, Any], m: pd.DataFrame, events: pd.DataFrame) -> CheckResult:
    c = cfg["checks"]
    tol = c["lube_recovery_tolerance_frac"]
    need = c["expected_ranges"]["lube_recovery_share_min"]
    win = pd.Timedelta(hours=c["lube_recovery_window_h"])
    min_h = c["lube_recovery_min_matched_hours"]
    min_eps = c["lube_recovery_min_measurable_episodes"]
    bands = cfg["regime"]["load_bands_pct"]
    post_start = pd.Timedelta(days=c["lube_recovery_post_start_days"])
    stage_min = c["lube_recovery_exclude_bearing_stage_min"]
    starts = events[events["event"] == "lube_episode_start"].sort_values("ts")
    clears = events[events["event"] == "lube_cleared"].sort_values("ts")
    onsets = events[(events["event"] == "bearing_onset") & (events["cause"] == "lubrication")]
    res = m[(m["axis"] == "res") & m["radio_noise"].isna()].copy()
    res["band"] = np.where(
        res["load_pct_at_measurement"] < bands["light"][1],
        "light",
        np.where(res["load_pct_at_measurement"] >= bands["heavy"][0], "heavy", "medium"),
    )

    def compare(g, pre_win, post_win):
        pre = g[g["hour_utc"].between(*pre_win)]
        post = g[g["hour_utc"].between(*post_win)]
        best = None
        for band in set(pre["band"]) & set(post["band"]):
            n = min((pre["band"] == band).sum(), (post["band"] == band).sum())
            if best is None or n > best[1]:
                best = (band, n)
        if best is None or best[1] < min_h:
            return None, (best[1] if best else 0)
        pm = pre[pre["band"] == best[0]]["hf_rms_g"].median()
        qm = post[post["band"] == best[0]]["hf_rms_g"].median()
        return bool(qm <= pm * (1 + tol)), int(best[1])

    rows = []
    for _, s in starts.iterrows():
        asset = s["asset_id"]
        cl = clears[(clears["asset_id"] == asset) & (clears["ts"] >= s["ts"])].head(1)
        pts = [
            p
            for p in res["point_id"].unique()
            if (asset.endswith("-SP") and p.startswith(asset))
            or (asset.startswith("CMP") and p == f"{asset}-AE")
        ]
        window_end = s["ts"] + pd.Timedelta(days=float(s["window_days"]))
        cleared_in_window = not cl.empty and cl["ts"].iat[0] <= window_end
        converted = not onsets[
            onsets["component_id"].astype(str).str.startswith(asset)
            & onsets["ts"].between(window_end - DAY, window_end + DAY)
        ].empty
        rec, matched, bearing = None, 0, None
        if not cl.empty and pts:
            t_clear = cl["ts"].iat[0]
            g = res[res["point_id"].isin(pts)]
            pre_w = (s["ts"] - win, s["ts"])
            post_w = (t_clear + post_start, t_clear + post_start + win)
            in_w = g[g["hour_utc"].between(*pre_w) | g["hour_utc"].between(*post_w)]
            hit = in_w[in_w["stage"] >= stage_min]
            if not hit.empty:
                top = hit.sort_values("stage").iloc[-1]
                bearing = f"stage {int(top['stage'])} bearing on {top['point_id']}"
            else:
                rec, matched = compare(g, pre_w, post_w)
        rows.append(
            {
                "asset_id": asset,
                "start": s["ts"],
                "cleared": cl["ts"].iat[0] if not cl.empty else pd.NaT,
                "cleared_by": cl["by"].iat[0] if not cl.empty else None,
                "cleared_in_window": cleared_in_window,
                "converted": converted,
                "excluded_bearing": bearing,
                "matched_hours": matched,
                "recovered": rec,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return CheckResult("7", "Lubrication episodes", "recovery and conversion", "no episodes", None, df)
    measured_rec = df["recovered"].dropna()
    share = measured_rec.mean() if len(measured_rec) else np.nan
    unclr = df[~df["cleared_in_window"]]
    conv_ok = bool(unclr["converted"].all()) if len(unclr) else True
    report_only = len(measured_rec) < min_eps
    rec_ok = True if report_only else bool(share >= need)
    excl = df[df["excluded_bearing"].notna()]
    return CheckResult(
        "7",
        "Lubrication episodes",
        f"conversion of uncleared episodes (pass/fail); recovery in >= {need:.0%} of measurable episodes "
        f"(pre window 72 h before the start, post window 72 h from clear + {post_start.days} d, same load band, "
        f">= {min_h} matched hours; report-only below {min_eps} measurable; episodes with a bearing at stage "
        f">= {stage_min} on the point in either window reported separately)",
        f"{len(df)} episodes; uncleared {len(unclr)}, converted {int(unclr['converted'].sum())}; recovery measurable "
        f"{len(measured_rec)}, recovered {share:.0%}"
        + (" (report-only)" if report_only else "")
        + (
            "; excluded for a bearing: "
            + "; ".join(f"{r.asset_id} {r.start:%Y-%m} ({r.excluded_bearing})" for r in excl.itertuples())
            if len(excl)
            else ""
        ),
        bool(conv_ok and rec_ok),
        df,
    )


def check_crash(
    cfg: dict[str, Any], events: pd.DataFrame, machine_hours: pd.DataFrame, m: pd.DataFrame
) -> CheckResult:
    alpha = cfg["checks"]["crash_binomial_alpha"]
    props = cfg["crash"]["outcome_proportions"]
    col = events[events["event"] == "collision"]
    n = len(col)
    rows, ok = [], True
    for outcome, p in props.items():
        k = int((col["outcome"] == outcome).sum())
        pv = binomtest(k, n, p).pvalue if n else np.nan
        rows.append(
            {
                "outcome": outcome,
                "count": k,
                "share": k / n if n else np.nan,
                "configured": p,
                "binomial_p": pv,
            }
        )
        ok &= bool(pv > alpha) if n else True
    imm = col[col["outcome"] == "immediate_failure"]
    fails = failures(events)
    crash_f = fails[fails["mode"] == "crash"]
    wo_ok, pre_ok = 0, 0
    for _, cev in imm.iterrows():
        if not crash_f[(crash_f["ts"] >= cev["ts"]) & (crash_f["ts"] <= cev["ts"] + DAY)].empty:
            wo_ok += 1
        mh = machine_hours[
            (machine_hours["machine_id"] == cev["asset_id"])
            & machine_hours["hour_utc"].between(cev["ts"] - 30 * DAY, cev["ts"] - pd.Timedelta(hours=1))
        ]
        if mh.empty or (mh["spindle_stage"].fillna(0) < 2).all():
            pre_ok += 1
    ok &= wo_ok == len(imm) and pre_ok == len(imm)
    return CheckResult(
        "8",
        "Crash outcomes",
        f"shares consistent with configured (binomial p > {alpha}); immediate failures: WO <= 24 h, no stage 2 in prior 30 d",
        f"{n} collisions; "
        + ", ".join(f"{r['outcome']} {r['count']}" for r in rows)
        + f"; immediate with WO {wo_ok}/{len(imm)}, clean prior 30 d {pre_ok}/{len(imm)}",
        ok,
        pd.DataFrame(rows),
    )


def check_censoring(cfg: dict[str, Any], events: pd.DataFrame) -> CheckResult:
    need = cfg["checks"]["expected_ranges"]["censoring_share_before_stage3_min"]
    rep = events[
        (events["event"] == "replacement") & events["reason"].isin(["pm_replacement", "preventive_rebuild"])
    ]
    if rep.empty:
        return CheckResult(
            "9", "Censoring", f">= {need:.0%} before stage 3", "no preventive replacements", None, None
        )
    share = (rep["stage"] < 3).mean()
    return CheckResult(
        "9",
        "Censoring",
        f"preventive replacements before stage 3 in >= {need:.0%}",
        f"{share:.0%} of {len(rep)} component replacements",
        bool(share >= need),
        rep.groupby("reason")["stage"].value_counts().rename("n").reset_index(),
    )


def check_label_noise(cfg: dict[str, Any], wo_truth: pd.DataFrame) -> CheckResult:
    ln = cfg["label_noise"]
    alpha = cfg["checks"]["label_noise_binomial_alpha"]
    lag_lo, lag_hi = cfg["checks"]["label_noise_ranges"]["close_lag_days"]
    u = wo_truth[wo_truth["wo_type"] == "unplanned_repair"]
    if u.empty:
        return CheckResult(
            "10", "Label noise", "binomial vs configured rates", "no unplanned repairs", None, None
        )
    n = len(u)
    k_wrong = int((u["label_noise"] == "wrong_code").sum())
    k_miss = int((u["label_noise"] == "missing_code").sum())
    p_wrong = binomtest(k_wrong, n, ln["wrong_code_frac"]).pvalue
    p_miss = binomtest(k_miss, n, ln["missing_code_frac"]).pvalue
    rep = wo_truth[wo_truth["wo_type"].isin(["unplanned_repair", "condition_based"])]
    lag = (rep["closed"] - rep["started"] - pd.to_timedelta(rep["downtime_hours"], unit="h")) / DAY
    ok = p_wrong > alpha and p_miss > alpha and lag.min() >= lag_lo - 1e-9 and lag.max() <= lag_hi + 1e-9
    return CheckResult(
        "10",
        "Label noise",
        f"wrong and missing shares consistent with {ln['wrong_code_frac']:.0%} and {ln['missing_code_frac']:.0%} "
        f"(binomial p > {alpha}); close lag {lag_lo}-{lag_hi} d",
        f"wrong {k_wrong}/{n} (p = {p_wrong:.2f}), missing {k_miss}/{n} (p = {p_miss:.2f}); close lag "
        f"{lag.min():.2f}-{lag.max():.2f} d",
        bool(ok),
        None,
    )


def benign_episodes(events: pd.DataFrame, benign: pd.DataFrame, heat: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [{"kind": b["event"], "asset_id": b["asset_id"], "ts": b["ts"]} for _, b in benign.iterrows()]
    rem = events[(events["event"] == "benign")]
    rows += [{"kind": r["kind"], "asset_id": r["asset_id"], "ts": r["ts"]} for _, r in rem.iterrows()]
    rows += [{"kind": h["event"], "asset_id": h["asset_id"], "ts": h["ts"]} for h in heat]
    return pd.DataFrame(rows)


def _fisher_after(
    episodes: pd.DataFrame,
    fails: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: pd.Timedelta,
    own_scopes: bool = False,
) -> tuple[int, int, int, int, float]:
    """Failures within the window after episodes against tiled base windows with no episode nearby.
    With own_scopes, base windows come only from the assets that carry episodes."""
    ep = episodes.assign(scope=episodes["asset_id"].str.split("-").str[:2].str.join("-"))
    ep.loc[ep["scope"] == "CMP", "scope"] = "CMP"
    scopes = sorted(set(ep["scope"]) if own_scopes else set(fails["scope"]) | set(ep["scope"]))
    after_fail = sum(
        1
        for _, e in ep.iterrows()
        if not fails[(fails["scope"] == e["scope"]) & fails["ts"].between(e["ts"], e["ts"] + window)].empty
    )
    base_n, base_fail = 0, 0
    for sc in scopes:
        t = start
        while t + window <= end:
            near = ep[(ep["scope"] == sc) & ep["ts"].between(t - window, t + window)]
            if near.empty:
                base_n += 1
                base_fail += int(not fails[(fails["scope"] == sc) & fails["ts"].between(t, t + window)].empty)
            t += window
    table = [[after_fail, len(ep) - after_fail], [base_fail, base_n - base_fail]]
    p = fisher_exact(table)[1] if len(ep) and base_n else np.nan
    return after_fail, len(ep), base_fail, base_n, float(p)


def check_benign(
    cfg: dict[str, Any],
    events: pd.DataFrame,
    episodes: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> CheckResult:
    window = pd.Timedelta(days=cfg["evaluation"]["benign_followup_window_days"])
    alpha = cfg["evaluation"]["fisher_alpha"]
    fails = failures(events)
    fails = fails.assign(scope=fails["asset_id"].str.split("-").str[:2].str.join("-"))
    fails.loc[fails["scope"].str.startswith("CMP"), "scope"] = "CMP"
    a, n, bf, bn, p = _fisher_after(episodes, fails, start, end, window)
    tool = events[events["event"] == "tool_assembly_high_first_use"]
    if "created_in_record" in tool:
        tool = tool[tool["created_in_record"].fillna(False).astype(bool)]
    tool_ep = pd.DataFrame(
        {"kind": "tool_assembly_high_program", "asset_id": tool["asset_id"], "ts": tool["ts"]}
    )
    ta, tn, tbf, tbn, tp = (
        _fisher_after(tool_ep, fails, start, end, window, own_scopes=True)
        if len(tool_ep)
        else (0, 0, 0, 0, np.nan)
    )
    years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
    ok = (p > alpha if pd.notna(p) else True) and (tp > alpha if pd.notna(tp) else True)
    detail = pd.concat(
        [
            episodes.groupby("kind").size().rename("n").reset_index(),
            pd.DataFrame([{"kind": "tool_assembly_high_program (separate test)", "n": tn}]),
        ],
        ignore_index=True,
    )
    return CheckResult(
        "11",
        "Benign anomalies",
        f"failure rate after benign within chance (Fisher p > {alpha}); high tool-assembly programs created in "
        "the record tested the same way from their first use, against base windows on the same machines",
        f"{n} episodes ({n / years:.1f}/yr); failures within 30 d after {a}/{n}; base windows {bf}/{bn}; "
        f"p = {p:.3f}; high tool-assembly programs: failures within 30 d after {ta}/{tn}; base windows "
        f"{tbf}/{tbn}; p = {tp:.3f}",
        bool(ok) if pd.notna(p) else None,
        detail,
    )


def check_sensor_behavior(
    cfg: dict[str, Any], scalars: pd.DataFrame, truth: pd.DataFrame, sensor_events: pd.DataFrame
) -> CheckResult:
    need = cfg["checks"]["expected_ranges"]["implausible_recall_min"]
    res = scalars[(scalars["axis"] == "res") & (~scalars["no_measurement"])]
    flags = pd.concat([flag_point(g, cfg) for _, g in res.groupby("point_id")])
    res = res.join(flags[["implausible", "rule"]])
    res = res.merge(truth[["point_id", "hour_utc", "radio_noise"]], on=["point_id", "hour_utc"], how="left")
    noisy = res["radio_noise"].notna()
    recall = res.loc[noisy, "implausible"].mean() if noisy.any() else np.nan
    precision = (
        res.loc[res["implausible"], "radio_noise"].notna().mean() if res["implausible"].any() else np.nan
    )
    se = sensor_events.groupby("event").size().to_dict()
    gaps = sensor_events[sensor_events["event"] == "battery_gap_start"]
    measured = f"radio-noise rows {noisy.mean():.2%}; flagged recall {recall:.1%}, precision {precision:.1%}; events {se}"
    return CheckResult(
        "13",
        "Sensor behavior",
        f"rates as configured; implausible-row recall >= {need:.0%}",
        measured,
        bool(recall >= need) if pd.notna(recall) else None,
        pd.DataFrame([{"battery_gaps": len(gaps), **se}]),
    )


def check_controller(cfg: dict[str, Any], bronze_root: Path, wo: pd.DataFrame, reg: Register) -> CheckResult:
    root = bronze_root / "controller"
    if not (root / "hz1").exists():
        return CheckResult(
            "14",
            "Controller consistency",
            "aggregates reconcile to 1 Hz",
            "no shadow-quarter 1 Hz rows",
            None,
            None,
        )
    unmatched, worst, files = 0, 0.0, 0
    for f in sorted((root / "hz1").glob("month=*/*.parquet")):
        hz = pd.read_parquet(f).drop(columns=["run_id", "config_hash"])
        stored = pd.read_parquet(root / "minute_agg" / f.parent.name / f.name).drop(
            columns=["run_id", "config_hash"]
        )
        stored = stored[stored["minute_utc"].between(hz["ts_utc"].min().floor("min"), hz["ts_utc"].max())]
        res = compare_aggregates(stored, aggregate_minutes(hz), tolerance=10.0**-AGG_DECIMALS)
        unmatched += int(res["rows_unmatched"])
        worst = max(worst, res["max_abs_diff"])
        files += 1
        del hz, stored
    cmp_ = {
        "rows_unmatched": unmatched,
        "max_abs_diff": worst,
        "within_tolerance": float(worst <= 10.0**-AGG_DECIMALS),
    }
    ev = pd.read_parquet(root / "events")
    stops = ev[ev["stopped_machine"]]
    keys: set[tuple[str, pd.Timestamp]] = set()
    for f in sorted((root / "minute_agg").glob("month=*/*.parquet")):
        a = pd.read_parquet(f, columns=["machine_id", "minute_utc", "execution"])
        a = a[a["execution"] == "interrupted"]
        keys |= set(zip(a["machine_id"], a["minute_utc"], strict=True))
    matched = sum((r.machine_id, r.ts_utc.floor("min")) in keys for r in stops.itertuples())
    lag_max = pd.Timedelta(days=cfg["label_noise"]["close_lag_days"][1])
    u = wo[
        (wo["wo_type"] == "unplanned_repair")
        & wo["asset_id"].map(lambda a: _is_machine_asset(a, reg)).astype(bool)
    ]
    with_stop = 0
    for r in u.itertuples():
        mid = "-".join(r.asset_id.split("-")[:2])
        s = stops[
            (stops["machine_id"] == mid)
            & stops["ts_utc"].between(r.opened - pd.Timedelta(hours=1), r.opened + lag_max)
        ]
        with_stop += int(not s.empty)
    ok = (
        cmp_["rows_unmatched"] == 0
        and cmp_["within_tolerance"] == 1.0
        and matched == len(stops)
        and with_stop == len(u)
    )
    return CheckResult(
        "14",
        "Controller consistency",
        "aggregates reconcile to 1 Hz; stops have state transitions; unplanned WOs have a controller stop",
        f"shadow machine-months {files}, minutes unmatched {int(cmp_['rows_unmatched'])}, max diff {cmp_['max_abs_diff']:.3g}; stops with transition {matched}/{len(stops)}; unplanned WOs with stop {with_stop}/{len(u)}",
        bool(ok),
        None,
    )


def check_erp(
    cfg: dict[str, Any], jobs_truth: pd.DataFrame, schedules: dict[str, pd.DataFrame], wo: pd.DataFrame
) -> CheckResult:
    er = cfg["checks"]["expected_ranges"]
    lo, hi = er["hmc_utilization"]
    tol = er["downtime_reconcile_tolerance"]
    over_cap, overlaps, jobs_n = 0, 0, 0
    cut_by_job = jobs_truth.set_index("job_id")["cutting_hours"]
    for s in schedules.values():
        s = s.sort_values("hour_utc")
        js = s[s["job_id"].notna()].groupby("job_id")["hour_utc"].agg(["min", "max"]).sort_values("min")
        hrs_last = s["hour_utc"].dt.tz_convert(None).to_numpy()[-1]
        starts = js["min"].dt.tz_convert(None).to_numpy()
        ends = js["max"].dt.tz_convert(None).to_numpy()
        overlaps += int((ends[:-1] >= starts[1:]).sum())
        nxt = np.append(starts[1:], hrs_last + np.timedelta64(1, "h"))
        hrs = s["hour_utc"].dt.tz_convert(None).to_numpy()
        sched_min = s["scheduled_min"].to_numpy()
        cum = np.concatenate([[0.0], np.cumsum(sched_min)])
        a = np.searchsorted(hrs, starts)
        b = np.searchsorted(hrs, nxt)
        cap_h = (cum[b] - cum[a]) / SECONDS_PER_MINUTE
        cut = cut_by_job.reindex(js.index).fillna(0.0).to_numpy()
        over_cap += int((cut > cap_h + 1e-6).sum())
        jobs_n += len(js)
    sched_all = pd.concat([s.assign(machine_id=m) for m, s in schedules.items()])
    hmc = [m for m in schedules if m.startswith("HMC")]
    jt = jobs_truth
    util = jt[jt["machine_id"].isin(hmc)]["cutting_hours"].sum() / (
        sched_all[sched_all["machine_id"].isin(hmc)]["scheduled_min"].sum() / SECONDS_PER_MINUTE
    )
    lost_erp = (jt["planned_cutting_hours"] - jt["cutting_hours"]).sum()
    lost_cmms = 0.0
    sched_idx = {m: s.set_index("hour_utc") for m, s in schedules.items()}
    for r in wo[wo["downtime_hours"] > 0].itertuples():
        mid = "-".join(str(r.asset_id).split("-")[:2])
        if mid not in sched_idx:
            continue
        s = sched_idx[mid]
        t0, t1 = r.started, r.started + pd.Timedelta(hours=r.downtime_hours)
        sel = s.loc[(s.index >= t0.floor("h")) & (s.index < t1)]
        lost_cmms += float(sel["cutting_min"].sum()) / SECONDS_PER_MINUTE
    rel = abs(lost_cmms - lost_erp) / lost_erp if lost_erp > 0 else np.nan
    ok = over_cap == 0 and overlaps == 0 and lo <= util <= hi and (rel <= tol if pd.notna(rel) else True)
    return CheckResult(
        "15",
        "ERP consistency",
        f"per-job cutting <= scheduled hours to next job start; no overlapping jobs; HMC utilization {lo:.0%}-{hi:.0%}; "
        f"CMMS downtime vs ERP lost cutting within {tol:.0%}",
        f"{jobs_n} jobs: over capacity {over_cap}, overlapping {overlaps}; HMC utilization {util:.1%}; "
        f"lost cutting ERP {lost_erp:.0f} h vs CMMS {lost_cmms:.0f} h ({rel:.1%})",
        bool(ok),
        None,
    )


def check_replay(
    cfg: dict[str, Any],
    reg: Register,
    scalars: pd.DataFrame,
    sensor_events: pd.DataFrame,
    alerts: pd.DataFrame,
) -> CheckResult:
    rep = []
    for pid in scalars["point_id"].unique():
        k = _ISO_KEY[reg.point_class(pid)]
        s = VendorRuleSettings.for_point_class(cfg, k, "compressor" if k == "compressor" else k)
        rep += replay_point(pid, scalars[scalars["point_id"] == pid], sensor_events, s, cfg)
    r = pd.DataFrame(rep)
    key = ["point_id", "alert_ts", "level", "rules"]
    if r.empty and alerts.empty:
        return CheckResult(
            "17", "Rule floor reproduces vendor alert log", "exact match", "no alerts", True, None
        )
    m = alerts[key].merge(r[key], how="outer", indicator=True)
    counts = m["_merge"].value_counts().to_dict()
    ok = counts.get("left_only", 0) == 0 and counts.get("right_only", 0) == 0
    return CheckResult(
        "17",
        "Rule floor reproduces vendor alert log",
        "every alert reproduced from bronze; no extras",
        f"matched {counts.get('both', 0)}, log only {counts.get('left_only', 0)}, replay only {counts.get('right_only', 0)}",
        bool(ok),
        None,
    )


def headline_counts(
    cfg: dict[str, Any],
    reg: Register,
    events: pd.DataFrame,
    scalars: pd.DataFrame,
    wo: pd.DataFrame,
    alerts: pd.DataFrame,
    schedules: dict[str, pd.DataFrame],
    machine_hours: pd.DataFrame,
    episodes: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    shadow: tuple[pd.Timestamp, pd.Timestamp],
    confirmed_total: int | None = None,
) -> pd.DataFrame:
    er = cfg["checks"]["expected_ranges"]
    exp = cfg["checks"]["count_expectations"]
    years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
    f = failures(events)
    mach = (
        f[f["asset_id"].map(lambda a: _is_machine_asset(a, reg)).astype(bool)]
        if len(f)
        else f.reindex(columns=["ts", "asset_id", "mode"])
    )
    powered_h = sum(float(s["scheduled_min"].sum()) for s in schedules.values()) / SECONDS_PER_MINUTE
    down_h = float((machine_hours["down"]).sum())
    rep = events[events["event"] == "replacement"]
    insp = wo[wo["wo_type"] == "inspection"]
    months = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR / 12)
    instr_assets = 14
    sh0, sh1 = shadow
    in_sh = mach[mach["ts"].between(sh0, sh1)]
    uninstr = [m for m, v in reg.machines.items() if v["class"] != "hmc"]
    lube = events[(events["event"] == "lube_episode_start") & events["asset_id"].str.endswith("-SP")]
    q = {
        "unplanned_stoppages_machines": len(mach),
        "fleet_mtbf_running_h": (powered_h - down_h) / max(len(mach), 1),
        "hmc_spindle_bearing_failures": int(
            ((f["mode"] == "bearing") & f["asset_id"].str.match(r"HMC-\d+-SP$")).sum()
        ),
        "hmc_spindle_preventive_rebuilds": int(
            (rep["reason"].eq("preventive_rebuild") & rep["component_id"].str.endswith("SP-F")).sum()
        ),
        "crash_spindle_failures_no_precursor": int((f["mode"] == "crash").sum()),
        "hmc_pump_unplanned_failures": int(f["asset_id"].str.match(r"HMC-\d+-(TSC|CP)$").sum()),
        "compressor_unplanned_events": int(f["asset_id"].str.startswith("CMP").sum()),
        "axis_ball_screw_replacements": int((f["mode"] == "ball_screw_or_guide").sum()),
        "hmc_lube_episodes_per_year": len(lube) / years,
        "benign_episodes_per_year": len(episodes) / years,
        "alert_inspections_nff_share": float(insp["no_fault_found"].mean()) if len(insp) else np.nan,
        "rule_alerts_per_instrumented_asset_month": len(alerts) / (instr_assets * months),
        "shadow_failures_instrumented": int(in_sh["asset_id"].map(_instrumented_asset).astype(bool).sum()),
        "shadow_failures_uninstrumented": int(
            in_sh["asset_id"].map(lambda a: "-".join(a.split("-")[:2]) in uninstr).astype(bool).sum()
        ),
        "no_measurement_share": float(
            scalars.drop_duplicates(["point_id", "hour_utc"])["no_measurement"].mean()
        ),
    }
    cs = events[events["event"] == "condition_start"]

    def n_cond(kind: str, cause: str | None = None) -> int:
        sel = cs["kind"] == kind
        if cause is not None:
            sel &= cs["cause"] == cause
        return int(sel.sum())

    q.update(
        {
            "imbalance_episodes": n_cond("imbalance"),
            "misalignment_gradual_episodes": n_cond("misalignment", "gradual"),
            "misalignment_post_reinstallation": n_cond("misalignment", "post_reinstallation"),
            "looseness_episodes": n_cond("looseness"),
            "rotary_union_leaks": n_cond("rotary_union_leak"),
            "coolant_ingress_episodes": n_cond("coolant_ingress"),
            "cooler_fouling_episodes": n_cond("cooler_fouling"),
            "spindle_cooling_episodes": n_cond("spindle_cooling"),
        }
    )
    if confirmed_total is not None:
        q["confirmed_conditions_jul2025_jun2026"] = confirmed_total
    rows = []
    sc = scalars.drop_duplicates(["point_id", "hour_utc"])
    duty = next(c for c, v in reg.compressors.items() if v["role"] == "duty")
    classes = {
        "spindles": sc["point_id"].str.contains("-SP-"),
        "tsc_pumps": sc["point_id"].str.endswith("-TSC"),
        "flood_pumps": sc["point_id"].str.endswith("-CP"),
        "duty_compressor": sc["point_id"].str.startswith(duty),
        "standby_compressor": sc["point_id"].str.startswith("CMP") & ~sc["point_id"].str.startswith(duty),
    }
    col = events[events["event"] == "collision"]
    for qname, val in (
        ("collisions_per_machine_year_effective", len(col) / (len(reg.machines) * years)),
        ("brinelling_events", int((col["outcome"] == "brinelling").sum())),
        ("immediate_spindle_failures_from_collision", int((col["outcome"] == "immediate_failure").sum())),
    ):
        rng_txt = f"{er[qname][0]}-{er[qname][1]}" if qname in er else "reported"
        rows.append(
            {
                "quantity": qname,
                "expected": rng_txt,
                "measured": round(val, 2) if isinstance(val, float) else val,
                "status": ("in range" if er[qname][0] <= val <= er[qname][1] else "OUT OF RANGE")
                if qname.startswith("collisions")
                else "reported",
                "basis": "including multipliers" if qname.startswith("collisions") else "",
            }
        )
    for name, mask in classes.items():
        rows.append(
            {
                "quantity": f"no_measurement_share_{name}",
                "expected": "about 0.5" if name == "standby_compressor" else "reported",
                "measured": round(float(sc.loc[mask, "no_measurement"].mean()), 3),
                "status": "reported",
                "basis": "scheduled hours",
            }
        )
    for k, v in q.items():
        lo, hi = er[k]
        inside = lo <= v <= hi
        status = "in range" if inside else "OUT OF RANGE"
        if not inside and k in exp:
            a, b = poisson.ppf(POISSON_90_TAILS[0], exp[k]), poisson.ppf(POISSON_90_TAILS[1], exp[k])
            status = (
                "outside range, within expected spread (rule 8)"
                if a <= v <= b
                else "OUT OF RANGE and outside expected spread"
            )
        rows.append(
            {
                "quantity": k,
                "expected": f"{lo}-{hi}",
                "measured": round(v, 3) if isinstance(v, float) else v,
                "status": status,
                "basis": BASIS.get(k, ""),
            }
        )
    out = pd.DataFrame(rows)
    return pd.concat(
        [
            out[~out["quantity"].str.startswith("no_measurement_share_")],
            out[out["quantity"].str.startswith("no_measurement_share_")],
        ],
        ignore_index=True,
    )


BASIS = {
    "fleet_mtbf_running_h": "powered hours minus downtime",
    "no_measurement_share": "scheduled hours",
    "confirmed_conditions_jul2025_jun2026": "bronze CMMS, all types incl. all unplanned failures",
}


def check_band_placement(cfg: dict[str, Any], events: pd.DataFrame, n_points: int) -> CheckResult:
    need = cfg["checks"]["band_contains_resonance_min_points"]
    sel = events[events["event"] == "envelope_band_selected"].copy()
    if sel.empty:
        return CheckResult(
            "18", "Envelope band placement", f">= {need} of {n_points} points", "no selections", None, None
        )
    sel["contained"] = (sel["band_lo_hz"] <= sel["resonance_hz"]) & (sel["resonance_hz"] <= sel["band_hi_hz"])
    per = (
        sel.groupby("point_id")
        .agg(
            selections=("contained", "size"),
            contained_all=("contained", "all"),
            resonance_hz=("resonance_hz", lambda s: ";".join(f"{v:.0f}" for v in s)),
            bands=("band_lo_hz", lambda s: ""),
            housing_gain_db=("housing_gain_db", "first"),
        )
        .reset_index()
    )
    per["bands"] = per["point_id"].map(
        sel.groupby("point_id").apply(
            lambda g: ";".join(
                f"{a:.0f}-{b:.0f}" for a, b in zip(g["band_lo_hz"], g["band_hi_hz"], strict=True)
            )
        )
    )
    good = int(per["contained_all"].sum())
    return CheckResult(
        "18",
        "Envelope band placement",
        f"every selected band contains the configured resonance on >= {need} of {n_points} points",
        f"{good} of {len(per)} points contained at every selection ({int(sel['contained'].sum())}/{len(sel)} selections)",
        bool(good >= need),
        per,
    )


def degradation_episodes(truth: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Instances that reached stage 2, by instrumented asset class and how they ended."""
    s2 = (
        truth[truth["stage"] >= 2]
        .groupby(["point_id", "instance_id"])
        .size()
        .reset_index()[["point_id", "instance_id"]]
    )
    rep = events[events["event"] == "replacement"][["instance_id", "reason"]].dropna(subset=["instance_id"])
    s2 = s2.merge(rep, on="instance_id", how="left")

    def cls(pid: str) -> str:
        if "-SP-" in pid:
            return "HMC spindle"
        if pid.endswith("-TSC"):
            return "TSC pump"
        if pid.endswith("-CP"):
            return "flood pump"
        return "compressor"

    outcome = {
        "unplanned_bearing": "unplanned failure",
        "planned_repair": "condition-based repair (confirmed on WO)",
    }
    s2["asset_class"] = s2["point_id"].map(cls)
    s2["outcome"] = s2["reason"].map(outcome).fillna(s2["reason"].fillna("open at record end"))
    tab = s2.groupby(["asset_class", "outcome"]).size().unstack(fill_value=0)
    return tab.reset_index()


RELIABILITY_KEYS = [
    "unplanned_stoppages_machines",
    "fleet_mtbf_running_h",
    "hmc_spindle_bearing_failures",
    "hmc_spindle_preventive_rebuilds",
    "crash_spindle_failures_no_precursor",
    "hmc_pump_unplanned_failures",
    "compressor_unplanned_events",
    "axis_ball_screw_replacements",
]


def run_full_checks(
    cfg: dict[str, Any],
    reg: Register,
    bronze_root: Path,
    truth_dir: Path,
    scalars: pd.DataFrame,
    truth: pd.DataFrame,
    events: pd.DataFrame,
    sensor_events: pd.DataFrame,
    m: pd.DataFrame,
) -> tuple[list[CheckResult], dict[str, Any]]:
    tz = "America/Chicago"
    start = pd.Timestamp(str(cfg["meta"]["record"]["start"]), tz=tz).tz_convert("UTC")
    end = pd.Timestamp(str(cfg["meta"]["record"]["end"]), tz=tz).tz_convert("UTC") + DAY
    shadow = (
        pd.Timestamp(str(cfg["meta"]["shadow"]["start"]), tz=tz).tz_convert("UTC"),
        pd.Timestamp(str(cfg["meta"]["shadow"]["end"]), tz=tz).tz_convert("UTC") + DAY,
    )
    event_cols = [
        "ts",
        "event",
        "asset_id",
        "mode",
        "components",
        "component_id",
        "instance_id",
        "cause",
        "defect",
        "reason",
        "stage",
        "outcome",
        "by",
        "window_days",
        "kind",
        "point_id",
        "wo_id",
        "band_lo_hz",
        "band_hi_hz",
        "resonance_hz",
        "housing_gain_db",
        "level",
        "p_drop",
        "p_rise_c",
        "program",
    ]
    events = events.reindex(columns=sorted(set(events.columns) | set(event_cols)))
    parts = truth_dir / "parts"
    schedules = {
        f.stem.removeprefix("schedule_"): pd.read_parquet(f) for f in sorted(parts.glob("schedule_*.parquet"))
    }
    machine_hours = pd.read_parquet(truth_dir / "machine_hours.parquet")
    wo_truth = pd.read_parquet(truth_dir / "work_orders_cmms_truth.parquet")
    wo = pd.read_parquet(bronze_root / "cmms" / "work_orders.parquet")
    jobs_truth = pd.read_parquet(truth_dir / "jobs_truth.parquet")
    benign = pd.read_parquet(truth_dir / "benign.parquet")
    alerts_path = bronze_root / "cm_platform" / "alerts" / "alerts.parquet"
    alerts = (
        pd.read_parquet(alerts_path)
        if alerts_path.exists()
        else pd.DataFrame(columns=["point_id", "alert_ts", "level", "rules"])
    )
    episodes = benign_episodes(
        events,
        benign[benign["event"] != "compressor_room_heat"] if "event" in benign else benign,
        benign[benign["event"] == "compressor_room_heat"].to_dict("records") if "event" in benign else [],
    )
    confirmed = rc.confirmed_conditions(cfg, bronze_root, events, end)
    head = headline_counts(
        cfg,
        reg,
        events,
        scalars,
        wo,
        alerts,
        schedules,
        machine_hours,
        episodes,
        start,
        end,
        shadow,
        confirmed["jul2025_jun2026"],
    )
    rel = head[head["quantity"].isin(RELIABILITY_KEYS)]
    bad = rel["status"].str.startswith("OUT OF RANGE and")
    amend = rel["status"].str.startswith("outside range")
    c12 = CheckResult(
        "12",
        "Fleet reliability",
        "section 16 ranges; count rule for small counts",
        "; ".join(
            f"{r.quantity} {r.measured}" + (f" ({r.basis})" if r.basis else "") for r in rel.itertuples()
        )
        + (f"; amend under rule 8: {', '.join(rel[amend]['quantity'])}" if amend.any() else ""),
        bool(not bad.any() and not (rel["status"] == "OUT OF RANGE").any()),
        rel,
    )
    results = [
        check_lead_times(cfg, m, events, reg),
        check_regime(cfg, m, sensor_events),
        check_lube(cfg, m, events),
        check_crash(cfg, events, machine_hours, m),
        check_censoring(cfg, events),
        check_label_noise(cfg, wo_truth),
        check_benign(cfg, events, episodes, start, end),
        c12,
        check_sensor_behavior(cfg, scalars, truth, sensor_events),
        check_controller(cfg, bronze_root, wo, reg),
        check_erp(cfg, jobs_truth, schedules, wo),
    ]
    tail = [
        check_replay(cfg, reg, scalars, sensor_events, alerts),
        check_band_placement(cfg, events, len(reg.sensors)),
        rc.check_fault_signatures(cfg, reg, m),
        rc.check_post_reinstallation(cfg, events, sensor_events),
        rc.check_ingress(cfg, reg, bronze_root, events, end),
        rc.check_thermal(cfg, reg, bronze_root, events, m, end),
    ]
    return results + tail, {
        "headline": head,
        "episodes": degradation_episodes(truth, events),
        "confirmed": confirmed,
    }
