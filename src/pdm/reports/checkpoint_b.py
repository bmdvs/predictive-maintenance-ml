"""Checkpoint (b) tables that sit beside the evaluation (revision 2, step 6 item 7): signal tests by
condition type, false episodes by context, monthly healthy hours above threshold, robustness rows, the
candidate and gate tables, healthy training hours and re-centering events per point, work-request
diagnosis, exhibit data and the timeline for all 28 assets.

Inputs: gold and silver tables written by the pipeline and the ML Model tasks. Output: gold.cpb_* tables.
python -m pdm.reports.checkpoint_b writes them from the local parity store.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pdm import ml_tasks, splits, tasks
from pdm.advisory import requests as adv
from pdm.evaluation import rev2 as r2
from pdm.models import nbm, severity
from pdm.store import Store

DAY = pd.Timedelta(days=1)
SIGNAL = {
    "bearing (unplanned)": ("env_bpfo", "env_bpfi", "env_bsf"),
    "bearing (condition-based)": ("env_bpfo", "env_bpfi", "env_bsf"),
    "lubrication": ("hf",),
    "imbalance": ("ord_01",),
    "misalignment": ("ord_02",),
    "looseness": ("ord_3_10",),
    "rotary union": ("hf",),
    "pump seal or impeller": ("hf", "vel"),
}


def _u(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return tasks._utc_cols(df, [c for c in cols if c in df])


def _healthy_mask(
    hours: pd.Series, asset: str, conds: pd.DataFrame, wo: pd.DataFrame, window: float
) -> np.ndarray:
    """Hours outside the catch window before any confirmed condition on the asset and outside open
    inspection or repair work on it."""
    t = pd.to_datetime(hours, utc=True)
    ok = np.ones(len(t), dtype=bool)
    for c in conds.loc[conds["asset_key"] == asset, "credit_ts"]:
        ok &= ~((t >= c - pd.Timedelta(days=window)) & (t < c + DAY)).to_numpy()
    w = wo[wo["asset_id"].str.startswith(asset) & wo["wo_type"].isin(tasks.INSPECTION_OR_REPAIR)]
    for a, b in zip(w["opened"], w["closed"], strict=True):
        ok &= ~((t >= a) & (t < (b if pd.notna(b) else t.max() + DAY))).to_numpy()
    return ok


def _open_at(eps: pd.DataFrame, asset: str, days: pd.Series) -> np.ndarray:
    e = eps[eps["asset_id"] == asset]
    d = pd.to_datetime(days, utc=True)
    on = np.zeros(len(d), dtype=bool)
    for r in e.itertuples():
        on |= ((d + DAY > r.opened) & (d < r.closed)).to_numpy()
    return on


def signal_tests(
    store: Store, cfg: dict[str, Any], lev: pd.DataFrame, conds: pd.DataFrame, wo: pd.DataFrame
) -> pd.DataFrame:
    """For each condition type with at least 3 conditions in the shadow year: the share of the type's
    signal days (the type's own indicator at z of 3 or more on the asset, within the catch window before
    confirmation) on which each method had an ALERT open, against the share of healthy days on the same
    assets (outside every catch window and open work) with an ALERT open."""
    b = splits.boundaries(cfg)
    window = float(cfg["evaluation"]["catch"]["opened_within_days"])
    zthr = float(cfg["evaluation"]["advisory_z"])
    cp = r2.conditions_in(conds, b["shadow"])
    types = cp["condition_type"].value_counts()
    types = [t for t, n in types.items() if n >= 3 and t in SIGNAL]
    if not types:
        return pd.DataFrame()
    stats = _u(store.read_pd("gold", "point_segment_stats"), ["segment_start"])
    cols = ["point_id", "hour_utc", "regime", "implausible", *adv.Z_INDICATORS.values()]
    cols = [c for c in cols if c != "ord_3_10_g2"] + [f"ord_{o:02d}_g2" for o in range(3, 11)]
    ind = store.read_pd("silver", "hourly_indicators", sorted(set(cols)))
    ind["hour_utc"] = tasks._utc(ind["hour_utc"])
    ind = ind[
        (ind["hour_utc"] >= b["shadow"][0])
        & (ind["hour_utc"] < b["shadow"][1])
        & ~ind["implausible"].fillna(False).astype(bool)
    ]
    z = adv.daily_point_z(adv.hourly_z(ind, stats))
    asset_of = tasks._points(store).set_index("point_id")["asset_id"]
    z["asset_id"] = z["point_id"].map(asset_of)
    rows = []
    for ctype in types:
        keys = SIGNAL[ctype]
        cs = cp[cp["condition_type"] == ctype]
        for method in ("ml_model", "vendor_rules"):
            e = lev[(lev["period"] == "shadow") & (lev["method"] == method)]
            sig_on, sig_n, h_on, h_n = 0, 0, 0, 0
            for asset, g in cs.groupby("asset_key"):
                za = z[z["asset_id"] == asset].groupby("day")[list(keys)].max().max(axis=1)
                if za.empty:
                    continue
                days = pd.Series(za.index)
                pre = np.zeros(len(days), dtype=bool)
                for t in g["credit_ts"]:
                    pre |= ((days >= t - pd.Timedelta(days=window)) & (days < t)).to_numpy()
                present = pre & (za.to_numpy() >= zthr)
                healthy = _healthy_mask(days, asset, conds, wo, window)
                on = _open_at(e, asset, days)
                sig_on += int((on & present).sum())
                sig_n += int(present.sum())
                h_on += int((on & healthy).sum())
                h_n += int(healthy.sum())
            rows.append(
                {
                    "condition_type": ctype,
                    "method": method,
                    "signal": ",".join(keys),
                    "conditions": len(cs),
                    "signal_days": sig_n,
                    "signal_days_alerted_share": sig_on / sig_n if sig_n else np.nan,
                    "healthy_days": h_n,
                    "healthy_days_alerted_share": h_on / h_n if h_n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def false_by_context(
    store: Store, cfg: dict[str, Any], lev: pd.DataFrame, conds: pd.DataFrame, wo: pd.DataFrame
) -> pd.DataFrame:
    """False ALERT episodes in the shadow year by the context at their opening on the machine: a program
    new in the shadow year (not run on the machine in training), a known program, within 30 days after a
    re-centering (repair or remount), or a benign event (a remount, or a titanium job); each against the
    context's share of healthy shadow-year hours."""
    b = splits.boundaries(cfg)
    window = float(cfg["evaluation"]["catch"]["opened_within_days"])
    chh = store.read_pd("silver", "controller_hourly", ["machine_id", "hour_utc", "program", "active_s"])
    chh["hour_utc"] = tasks._utc(chh["hour_utc"])
    chh = chh[chh["active_s"] > 0]
    trained = chh[chh["hour_utc"] < b["shadow"][0]].groupby("machine_id")["program"].agg(set).to_dict()
    sh = chh[(chh["hour_utc"] >= b["shadow"][0]) & (chh["hour_utc"] < b["shadow"][1])].copy()
    sh["new_program"] = [
        p not in trained.get(m, set()) for m, p in zip(sh["machine_id"], sh["program"], strict=True)
    ]
    jobs = store.read_pd("bronze", "erp_jobs", ["machine_id", "date", "material"])
    jobs["date"] = pd.to_datetime(jobs["date"]).dt.date
    ti = set(
        zip(
            jobs.loc[jobs["material"] == "titanium", "machine_id"],
            jobs.loc[jobs["material"] == "titanium", "date"],
            strict=True,
        )
    )
    sh["titanium"] = [
        (m, d) in ti
        for m, d in zip(sh["machine_id"], sh["hour_utc"].dt.tz_convert(tasks.LOCAL_TZ).dt.date, strict=True)
    ]
    segs = _u(store.read_pd("gold", "ml_segments"), ["segment_start"])
    segs["machine_id"] = segs["point_id"].str[:6]
    se = _u(store.read_pd("bronze", "cm_sensor_events", ["point_id", "ts", "event"]), ["ts"])
    remounts = se[se["event"] == "remount"].assign(machine_id=lambda d: d["point_id"].str[:6])

    def context(machine: str, t: pd.Timestamp, hour_row: pd.Series | None) -> str:
        if (
            (remounts["machine_id"] == machine) & (remounts["ts"] <= t) & (remounts["ts"] > t - 30 * DAY)
        ).any():
            return "benign event"
        if hour_row is not None and bool(hour_row["titanium"]):
            return "benign event"
        if (
            (segs["machine_id"] == machine)
            & (segs["segment_start"] <= t)
            & (segs["segment_start"] > t - 30 * DAY)
            & ~segs["reason"].astype(str).str.startswith("install")
        ).any():
            return "after re-centering"
        if hour_row is None:
            return "no program (compressor)"
        return "new program" if bool(hour_row["new_program"]) else "known program"

    idx = sh.set_index(["machine_id", "hour_utc"]).sort_index()
    rows = []
    for method in ("ml_model", "vendor_rules"):
        e = lev[(lev["period"] == "shadow") & (lev["method"] == method)]
        e = e[(e["opened"] >= b["shadow"][0]) & (e["opened"] < b["shadow"][1])].reset_index(drop=True)
        f = e[r2.false_mask(e.assign(point_id=""), conds).to_numpy()] if len(e) else e
        for r in f.itertuples():
            m = r.asset_id[:6]
            h = r.opened.floor("h")
            hr = idx.loc[(m, h)] if (m, h) in idx.index else None
            if isinstance(hr, pd.DataFrame):
                hr = hr.iloc[0]
            rows.append(
                {"method": method, "context": context(m, r.opened, hr if not m.startswith("CMP") else None)}
            )
    fe = pd.DataFrame(rows)
    hh = []
    for m, g in sh.groupby("machine_id"):
        ok = _healthy_mask(g["hour_utc"], f"{m}-SP", conds, wo, window)
        for (_, t), row in zip(
            g.loc[ok, ["machine_id", "hour_utc"]].itertuples(index=False), g[ok].itertuples(), strict=True
        ):
            hh.append(context(m, t, pd.Series({"titanium": row.titanium, "new_program": row.new_program})))
    share = pd.Series(hh).value_counts(normalize=True) if hh else pd.Series(dtype=float)
    out = (
        fe.groupby(["method", "context"]).size().rename("false_episodes").reset_index()
        if len(fe)
        else pd.DataFrame(columns=["method", "context", "false_episodes"])
    )
    out["share_of_false"] = out["false_episodes"] / out.groupby("method")["false_episodes"].transform("sum")
    out["share_of_healthy_hours"] = out["context"].map(share)
    return out


def monthly_above(store: Store, cfg: dict[str, Any], conds: pd.DataFrame, wo: pd.DataFrame) -> pd.DataFrame:
    """Monthly share of healthy shadow-year hours with the ML Model's score above its threshold, by asset
    class (healthy: outside every catch window and open work)."""
    window = float(cfg["evaluation"]["catch"]["opened_within_days"])
    sc = store.read_pd(
        "gold", "nbm_scores", ["point_id", "asset_id", "hour_utc", "state", "score", "threshold"]
    )
    sc["hour_utc"] = tasks._utc(sc["hour_utc"])
    sc = sc[sc["state"] == "scored"]
    parts = []
    for a, g in sc.groupby("asset_id"):
        ok = _healthy_mask(g["hour_utc"], a, conds, wo, window)
        parts.append(g[ok].assign(asset_class=severity.asset_class(a)))
    h = pd.concat(parts, ignore_index=True)
    h["month"] = h["hour_utc"].dt.tz_convert(tasks.LOCAL_TZ).dt.strftime("%Y-%m")
    h["above"] = h["score"] > h["threshold"]
    return (
        h.groupby(["asset_class", "month"])
        .agg(healthy_hours=("above", "size"), above_share=("above", "mean"))
        .reset_index()
    )


def robustness(store: Store, cfg: dict[str, Any], lev: pd.DataFrame, conds: pd.DataFrame) -> pd.DataFrame:
    """Robustness rows on the shadow year: leave-one-asset-out range of the catch share and false rate,
    persistence of 3, 6 and 12 hours, catch windows of 60, 90 and 120 days, with and without the 14-day
    rule; and from the folds, the band-only candidate as an ablation and a single threshold across speed
    bands."""
    b = splits.boundaries(cfg)
    span = b["shadow"]
    ctx = tasks._context(store)
    pts = tasks._points(store)
    n = pts["asset_id"].nunique()
    S = lev[lev["period"] == "shadow"]
    cp = r2.conditions_in(conds, span)
    rows = []
    for method, label in (("ml_model", "ML Model"), ("vendor_rules", "Vendor Rules")):
        e = r2.alert_view(S[S["method"] == method])
        for w in (60, 90, 120):
            r, _ = r2.method_row(label, "ALERT", e, cp, conds, span, n, w, ctx, cfg)
            rows.append(
                {
                    "row": f"catch window {w} days",
                    "method": label,
                    "caught": r["caught"],
                    "conditions": r["conditions"],
                    "caught_share": r["caught_share"],
                    "false_per_asset_month": r["false_per_asset_month"],
                }
            )
        shares, rates = [], []
        for a in sorted(pts["asset_id"].unique()):
            r, _ = r2.method_row(
                label,
                "ALERT",
                e[e["asset_id"] != a],
                cp[cp["asset_key"] != a],
                conds,
                span,
                n - 1,
                120,
                ctx,
                cfg,
            )
            shares.append(r["caught_share"])
            rates.append(r["false_per_asset_month"])
        rows.append(
            {
                "row": "leave one asset out (range)",
                "method": label,
                "caught_share_min": np.nanmin(shares),
                "caught_share_max": np.nanmax(shares),
                "false_min": np.nanmin(rates),
                "false_max": np.nanmax(rates),
            }
        )
        for esc in (True, False):
            av = r2.alarm_view(S[S["method"] == method], esc)
            f = cp[cp["unplanned"].astype(bool)]
            r, _ = r2.method_row(label, "ALARM", av, f, conds, span, n, 120, ctx, cfg)
            rows.append(
                {
                    "row": f"ALARM {'with' if esc else 'without'} the 14-day rule",
                    "method": label,
                    "caught": r["caught"],
                    "conditions": r["conditions"],
                    "caught_share": r["caught_share"],
                    "false_per_asset_month": r["false_per_asset_month"],
                }
            )
    ex = store.read_pd("gold", "nbm_exceedances")
    ex["ts"] = tasks._utc(ex["ts"])
    op = float(cfg["models"]["nbm"]["threshold_percentile"])
    for k in ml_tasks.PERSISTENCES:
        g = ex[(ex["persistence_hours"] == k) & np.isclose(ex["sweep"], op)]
        e = severity.episodes(
            g[["asset_id", "point_id", "ts"]], cfg["evaluation"]["catch"]["episode_gap_days"]
        )
        r, _ = r2.method_row("ML Model", "ALERT", e.assign(point_id=""), cp, conds, span, n, 120, ctx, cfg)
        rows.append(
            {
                "row": f"persistence {k} hours",
                "method": "ML Model",
                "caught": r["caught"],
                "conditions": r["conditions"],
                "caught_share": r["caught_share"],
                "false_per_asset_month": r["false_per_asset_month"],
            }
        )
    sel = store.read_pd("gold", "ml_candidates")
    champ = sel[sel["champion"].astype(bool)].iloc[0]
    alt_mode = "single" if champ["threshold_mode"] == "speed_band" else "speed_band"
    alt = sel[
        (sel["candidate"] == champ["candidate"])
        & (sel["hidden"] == champ["hidden"])
        & (sel["epochs"] == champ["epochs"])
        & (sel["threshold_mode"] == alt_mode)
    ]
    band_only = (
        sel[
            (sel["candidate"] == "autoencoder_third_octave")
            & (sel["threshold_mode"] == champ["threshold_mode"])
        ]
        .sort_values("skill", ascending=False)
        .head(1)
    )
    for label, d in (
        ("selected settings (folds)", sel[sel["champion"].astype(bool)]),
        (f"threshold mode {alt_mode} (folds)", alt),
        ("band-only candidate, autoencoder on the one-third-octave bands (folds)", band_only),
    ):
        for r in d.itertuples():
            rows.append(
                {
                    "row": label,
                    "method": "ML Model",
                    "settings": r.settings,
                    "caught_share": r.caught_share_at_budget,
                    "chance_share": r.chance_share_at_budget,
                    "skill": r.skill,
                    "false_per_asset_month": r.false_per_asset_month_operating,
                }
            )
    return pd.DataFrame(rows)


def gates_local(store: Store, cfg: dict[str, Any], model: Any) -> pd.DataFrame:
    """The four promotion gates on the local model (the platform registry evaluates the same gates on the
    registered version): criterion, measured, pass."""
    from pdm import registry

    tables = {
        "ml_train_scores": store.read_pd("gold", "ml_train_scores"),
        "ml_train_segments": store.read_pd("gold", "ml_train_segments"),
        "ml_candidates": store.read_pd("gold", "ml_candidates"),
        "ml_settings": store.read_pd("gold", "ml_settings"),
        "points": tasks._points(store),
    }
    tables["ml_train_scores"]["hour_utc"] = tasks._utc(tables["ml_train_scores"]["hour_utc"])
    if getattr(model, "healthy_stats", None) is None:
        model.healthy_stats = store.read_pd("gold", "point_healthy_stats")
    tags = {
        "challenger": "true",
        "settings": model.settings_key,
        "model_version": tables["ml_settings"]["model_version"].iat[0],
    }
    gates = registry.evaluate_gates(model, tags, tables, registry.training_prediction(model, store, cfg))
    crit = {
        "selection": "the registered run is the selection rule's pick on both folds combined",
        "parity": "the registered model's training scores match the training run's within 1e-9",
        "points": "every point's model and re-centering levels present",
        "false_episodes": "false episodes per asset-month on the last six months of training within twice the fold rate",
    }
    return pd.DataFrame(
        [
            {"gate": g.name, "criterion": crit.get(g.name, ""), "measured": g.detail, "pass": g.passed}
            for g in gates
        ]
    )


def per_point(store: Store) -> pd.DataFrame:
    """Healthy training hours per sensor point and its re-centering events (segments after training)."""
    m = store.read_pd("gold", "ml_models")
    segs = store.read_pd("gold", "ml_segments")
    rc = (
        segs[~segs["trained"].astype(bool)]
        .groupby("point_id")
        .agg(
            recentering_events=("segment", "size"),
            plausibility_flags=("plausibility_flag", "sum"),
            without_level=("level_ready", lambda s: int(s.isna().sum())),
        )
    )
    out = m[["point_id", "point_class", "n_train", "segments_trained"]].merge(rc, on="point_id", how="left")
    return out.fillna({"recentering_events": 0, "plausibility_flags": 0, "without_level": 0})


def diagnosis(store: Store, cfg: dict[str, Any], conds: pd.DataFrame) -> pd.DataFrame:
    """Work-request diagnosis on the ML Model's shadow-year cases: suspected condition against the confirmed
    finding, by the level the case reached."""
    b = splits.boundaries(cfg)
    a = tasks._agreement(store, cfg, conds, b["shadow"])
    if a.empty:
        return a
    cs = store.read_pd("gold", "cases")[["case_id", "level"]]
    wr = store.read_pd("gold", "work_requests")[["request_id", "case_id"]]
    return a.merge(wr, on="request_id", how="left").merge(cs, on="case_id", how="left")


def timeline(
    store: Store, cfg: dict[str, Any], lev: pd.DataFrame, conds: pd.DataFrame, wo: pd.DataFrame
) -> pd.DataFrame:
    """Timeline rows for all 28 assets over the record: ALERT and ALARM periods by method, Controller
    Health periods, inspections with findings, PM findings and failures."""
    pts = tasks._points(store)
    assets = sorted(pts["asset_id"].unique())
    rows = []
    for r in lev[lev["period"].isin(["training", "shadow"])].itertuples():
        m = "ML Model" if r.method == "ml_model" else "Vendor Rules"
        rows.append(
            {
                "asset_id": r.asset_id,
                "kind": f"{m} ALERT",
                "start": r.opened,
                "end": r.closed,
                "detail": r.period,
            }
        )
        if pd.notna(r.alarm_ts):
            rows.append(
                {
                    "asset_id": r.asset_id,
                    "kind": f"{m} ALARM",
                    "start": r.alarm_ts,
                    "end": r.closed,
                    "detail": str(r.alarm_cause),
                }
            )
    ch = ml_tasks._ch_episodes(store, cfg)
    for r in ch[ch["asset_id"].isin(assets)].itertuples():
        rows.append(
            {
                "asset_id": r.asset_id,
                "kind": "Controller Health",
                "start": r.opened,
                "end": r.closed,
                "detail": r.trigger_rules,
            }
        )
    w = wo[wo["asset_id"].isin(assets)]
    for r in w[w["wo_type"] == "inspection"].itertuples():
        nff = r.no_fault_found
        detail = "open" if pd.isna(nff) else ("no fault found" if bool(nff) else f"found {r.failure_code}")
        rows.append(
            {
                "asset_id": r.asset_id,
                "kind": "inspection",
                "start": r.opened,
                "end": r.closed,
                "detail": detail,
            }
        )
    for r in w[(w["wo_type"] == "pm") & w["failure_code"].notna()].itertuples():
        rows.append(
            {
                "asset_id": r.asset_id,
                "kind": "PM finding",
                "start": r.opened,
                "end": r.closed,
                "detail": r.failure_code,
            }
        )
    for r in conds[conds["unplanned"].astype(bool) & conds["asset_key"].isin(assets)].itertuples():
        rows.append(
            {
                "asset_id": r.asset_key,
                "kind": "failure",
                "start": r.credit_ts,
                "end": r.closed,
                "detail": str(r.failure_code),
            }
        )
    return pd.DataFrame(rows).sort_values(["asset_id", "start", "kind"]).reset_index(drop=True)


def exhibits(store: Store, cfg: dict[str, Any], model: Any, conds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Exhibit data: a score trace around the earliest-warned condition the ML Model caught (score, the
    threshold by speed band, WATCH, ALERT and ALARM starts, confirmation); the retained bursts nearest
    confirmation for one lubrication, one bearing, one pump seal or impeller and one rotary union with
    ingress condition; and the model's rebuild against its input on one healthy and one faulty hour of the
    same point (autoencoder and PCA candidates; other candidates have no rebuild)."""
    ct = _u(store.read_pd("gold", "eval_conditions"), ["credit_ts"])
    lev = _u(store.read_pd("gold", "level_episodes"), ["opened", "closed", "alarm_ts"])
    sc = store.read_pd("gold", "nbm_scores")
    sc["hour_utc"] = tasks._utc(sc["hour_utc"])
    out: dict[str, pd.DataFrame] = {}
    caught = ct[ct["ml_alert_caught"]].sort_values("ml_alert_warning_days", ascending=False)
    if len(caught):
        c = caught.iloc[0]
        a, t = c["asset_key"], c["credit_ts"]
        trace = sc[(sc["asset_id"] == a) & (sc["hour_utc"] >= t - 150 * DAY) & (sc["hour_utc"] < t + 7 * DAY)]
        e = lev[
            (lev["method"] == "ml_model")
            & (lev["period"] == "shadow")
            & (lev["asset_id"] == a)
            & (lev["opened"] <= t)
            & (lev["closed"] > t)
        ]
        watch = trace[(trace["state"] == "scored") & (trace["score"] > trace["threshold"])]["hour_utc"].min()
        marks = pd.DataFrame(
            [
                {"mark": "WATCH start", "ts": watch},
                {"mark": "ALERT start", "ts": e["opened"].min() if len(e) else pd.NaT},
                {"mark": "ALARM start", "ts": e["alarm_ts"].min() if len(e) else pd.NaT},
                {"mark": "confirmation", "ts": t, "wo_id": c["wo_id"], "condition_type": c["condition_type"]},
            ]
        )
        out["cpb_exhibit_trace"] = trace[
            ["point_id", "hour_utc", "speed_band", "state", "score", "threshold"]
        ]
        out["cpb_exhibit_trace_marks"] = marks
    bi = store.read_pd("bronze", "cm_burst_index", ["point_id", "hour_utc", "reason", "path"])
    bi["hour_utc"] = tasks._utc(bi["hour_utc"])
    asset_of = tasks._points(store).set_index("point_id")["asset_id"]
    bi["asset_id"] = bi["point_id"].map(asset_of)
    want = {
        "lubrication": ["lubrication"],
        "bearing": ["bearing (unplanned)", "bearing (condition-based)"],
        "pump seal or impeller": ["pump seal or impeller"],
        "rotary union with ingress": ["rotary union"],
    }
    rows = []
    for label, types in want.items():
        cs = ct[ct["condition_type"].isin(types)]
        if label == "rotary union with ingress":
            cs = cs[cs["failure_code"] == "SP-COOLANT-INGRESS"]
        for c in cs.sort_values("credit_ts").itertuples():
            bb = bi[(bi["asset_id"] == c.asset_key) & (bi["hour_utc"] <= c.credit_ts)]
            if len(bb):
                nb = bb.iloc[(c.credit_ts - bb["hour_utc"]).argsort().iloc[0]]
                rows.append(
                    {
                        "exhibit": label,
                        "wo_id": c.wo_id,
                        "asset_id": c.asset_key,
                        "confirmed": c.credit_ts,
                        "point_id": nb["point_id"],
                        "burst_hour": nb["hour_utc"],
                        "offset_days": (c.credit_ts - nb["hour_utc"]) / DAY,
                        "reason": nb["reason"],
                        "path": nb["path"],
                    }
                )
                break
    out["cpb_exhibit_bursts"] = pd.DataFrame(rows)
    out["cpb_exhibit_rebuild"] = rebuild_example(store, cfg, model, ct, sc)
    return out


def rebuild_example(
    store: Store, cfg: dict[str, Any], model: Any, ct: pd.DataFrame, sc: pd.DataFrame
) -> pd.DataFrame:
    """Input against the model's rebuild, per feature, on a healthy and a faulty hour of one point: the
    point of the first caught condition, its highest-scored hour before confirmation, and a healthy hour of
    the same segment and speed band."""
    caught = ct[ct["ml_alert_caught"]].sort_values("credit_ts")
    for c in caught.itertuples():
        s = sc[
            (sc["asset_id"] == c.asset_key)
            & (sc["state"] == "scored")
            & (sc["hour_utc"] < c.credit_ts)
            & (sc["hour_utc"] >= c.credit_ts - 120 * DAY)
        ]
        if s.empty:
            continue
        bad = s.loc[s["score"].idxmax()]
        pm = model.models.get(bad["point_id"])
        if pm is None or nbm.model_type(pm.settings.candidate) not in ("autoencoder", "pca"):
            return pd.DataFrame(
                {
                    "note": [
                        f"the selected candidate ({pm.settings.candidate if pm else 'none'}) has no rebuild of its input"
                    ]
                }
            )
        same = sc[
            (sc["point_id"] == bad["point_id"])
            & (sc["segment"] == bad["segment"])
            & (sc["speed_band"] == bad["speed_band"])
            & (sc["state"] == "scored")
        ]
        good = same.iloc[(same["score"] - same["score"].median()).abs().argsort().iloc[0]]
        ind = store.read_pd("silver", "hourly_indicators")
        ind = ind[ind["point_id"] == bad["point_id"]]
        ind["hour_utc"] = tasks._utc(ind["hour_utc"])
        win = _u(store.read_pd("silver", "healthy_windows"), ["start"])
        win = win[win["point_id"] == bad["point_id"]]
        rows = []
        for label, h in (("healthy", good["hour_utc"]), ("faulty", bad["hour_utc"])):
            r = ind[ind["hour_utc"] <= h]
            amb = None
            s1, _ = nbm.score_point(
                pm, r, win, cfg, h, h + pd.Timedelta(hours=1), model.plausibility_factor, 3, amb
            )
            clean = nbm.prepare(r, amb)
            clean[nbm.TEMP_RESID] = pm.temp_fit.residual(clean)
            x, cols = nbm.features(clean, pm.settings.candidate)
            bands = nbm.speed_band(clean, pm.rated_rpm, cfg)
            keys = nbm.operating_key(clean, bands)
            segs = nbm.segment_levels(
                clean, x, keys, win.sort_values("start").reset_index(drop=True), pm.offsets, cfg
            )
            seg = nbm.segment_ids(clean["hour_utc"], list(win.sort_values("start")["start"]))
            i = len(clean) - 1
            lvl = segs[seg[i]].level
            raw = np.column_stack([x[i : i + 1] - lvl, nbm._one_hot(clean.iloc[i : i + 1], bands[i : i + 1])])
            z = pm.scaler.transform(raw)
            rec = (
                pm.model.predict(z)
                if nbm.model_type(pm.settings.candidate) == "autoencoder"
                else pm.model.inverse_transform(pm.model.transform(z))
            )
            for j, name in enumerate(cols):
                rows.append(
                    {
                        "hour": label,
                        "hour_utc": h,
                        "point_id": bad["point_id"],
                        "feature": name,
                        "input_z": float(z[0, j]),
                        "rebuild_z": float(rec[0, j]),
                        "score": float(s1["score"].iloc[-1]) if len(s1) else np.nan,
                    }
                )
        return pd.DataFrame(rows)
    return pd.DataFrame({"note": ["no caught condition with scored hours"]})


def build(store: Store, cfg: dict[str, Any], model: Any) -> dict[str, int]:
    conds = tasks._conditions(store, cfg)
    wo = tasks._wo(store)
    lev = ml_tasks._levels(store)
    tables = {
        "cpb_signal_tests": signal_tests(store, cfg, lev, conds, wo),
        "cpb_false_by_context": false_by_context(store, cfg, lev, conds, wo),
        "cpb_monthly_above_threshold": monthly_above(store, cfg, conds, wo),
        "cpb_robustness": robustness(store, cfg, lev, conds),
        "cpb_gates": gates_local(store, cfg, model),
        "cpb_per_point": per_point(store),
        "cpb_diagnosis": diagnosis(store, cfg, conds),
        "cpb_timeline": timeline(store, cfg, lev, conds, wo),
        **exhibits(store, cfg, model, conds),
    }
    out = {}
    for name, df in tables.items():
        df = df if len(df) else pd.DataFrame({"note": ["no rows"]})
        store.write_pd(df, "gold", name)
        out[name] = len(df)
    return out


def main() -> None:
    import pickle

    from pdm import log, paths
    from pdm.config import load_config
    from pdm.spark import local_session

    log.configure()
    cfg = load_config()
    cfg["evaluation"]["holdout_shadow"] = False
    spark = local_session()
    store = Store(spark, local_root=paths.LOCAL_GOLD_DIR)
    with (paths.LOCAL_GOLD_DIR / "ml_model.pkl").open("rb") as fh:
        model = pickle.load(fh)["model"]  # noqa: S301 (the local run's own model file)
    print(build(store, cfg, model))
    spark.stop()


if __name__ == "__main__":
    main()
