"""ML Model job task bodies (revision 2): the training run (expanding-window selection on two folds inside
training, the selection rule on both folds combined, the champion's refit on all of training) and the
scoring run over the shadow year with the frozen model.

Training reads nothing at or after the training end; scoring needs the holdout lifted. Tables go to gold
(ml_candidates, ml_fold_scores, ml_train_scores, ml_models, ml_segments, ml_settings, point_healthy_stats,
nbm_scores, nbm_exceedances). (Revision 2; originally per-segment challengers refit monthly.)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from pdm import log, splits, tasks
from pdm.advisory import requests as adv
from pdm.evaluation import events as ev
from pdm.evaluation import rev2 as r2
from pdm.models import nbm, severity, temperature
from pdm.store import Store

LOG = log.get_logger("pdm.ml_tasks")
DAY = pd.Timedelta(days=1)
PERSISTENCES = (3, 6, 12)


# ---------------------------------------------------------------- inputs


def _utc(s: pd.Series) -> pd.Series:
    return tasks._utc(s)


def load_inputs(store: Store, cfg: dict[str, Any], end: pd.Timestamp) -> dict[str, Any]:
    """Everything the ML Model needs before `end` (UTC): hourly indicators, segment starts (healthy
    windows), sanity flags, confirmed conditions, work orders, points, rated speeds, the compressor-room
    ambient by hour and the Vendor Rules' persisted exceedances at the operating setting."""
    ind = store.read_pd("silver", "hourly_indicators")
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind = ind[ind["hour_utc"] < end].reset_index(drop=True)
    win = tasks._utc_cols(store.read_pd("silver", "healthy_windows"), ["start", "end"])
    win = win[win["start"] < end].reset_index(drop=True)
    sanity = (
        store.read_pd("gold", "baseline_sanity")
        if store.exists("gold", "baseline_sanity")
        else pd.DataFrame()
    )
    if len(sanity):
        sanity = sanity.merge(win[["window_id", "start", "end"]], on="window_id", how="left")
    conds = tasks._conditions(store, cfg)
    wo = tasks._wo(store)
    pts = tasks._points(store)
    mach = store.read_pd("bronze", "reg_machines").set_index("machine_id")
    rated = mach["rated_rpm"].astype(float).to_dict()
    room = store.read_pd("bronze", "cmp_room", ["minute_utc", "ambient_c"])
    room["minute_utc"] = _utc(room["minute_utc"])
    ambient = temperature.hourly_ambient(room[room["minute_utc"] < end])
    return {
        "ind": ind,
        "win": win,
        "sanity": sanity,
        "conds": conds,
        "wo": wo,
        "pts": pts,
        "rated": rated,
        "ambient": ambient,
        "vendor": vendor_signals(store, cfg, pts, end),
    }


def vendor_signals(store: Store, cfg: dict[str, Any], pts: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """The Vendor Rules' persisted exceedances at the operating setting: the vendor baseline rules
    (velocity, HF and envelope at +25% of the zone B width or +6 dB, and the temperature limits) and the
    ISO zone rule (zone C alert, zone D alarm), with the asset and the rules met at each hour."""
    rf = store.read_pd("gold", "rule_floor_exceedances")
    op = cfg["rules"]["vendor"]["alert_frac_zone_b_width"]
    rf = rf[(rf["method"] == "iso") | ((rf["method"] == "vendor") & (np.isclose(rf["sweep"], op)))].copy()
    rf["ts"] = _utc(rf["ts"])
    rf = rf[rf["ts"] < end]
    if "rules" not in rf:
        rf["rules"] = rf["level"]
    return rf.merge(pts[["point_id", "asset_id"]], on="point_id", how="left")


def excluded_intervals(
    asset: str,
    conds: pd.DataFrame,
    wo: pd.DataFrame,
    flagged: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: dict[str, Any],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Training exclusions known at the cutoff: the days before each confirmed condition, open inspection
    and repair work, and baseline windows flagged by the sanity check."""
    before = pd.Timedelta(days=cfg["models"]["nbm"]["exclude_before_condition_days"])
    c = conds[(conds["asset_key"] == asset) & (conds["credit_ts"] < cutoff)]
    out = [(t - before, t + DAY) for t in c["credit_ts"]]
    w = wo[
        wo["asset_id"].str.startswith(asset)
        & wo["wo_type"].isin(tasks.INSPECTION_OR_REPAIR)
        & (wo["opened"] < cutoff)
    ]
    out += [(a, b if pd.notna(b) else cutoff) for a, b in zip(w["opened"], w["closed"], strict=True)]
    if len(flagged):
        out += list(zip(flagged["start"], flagged["end"], strict=True))
    return out


def _point_args(inp: dict[str, Any], cfg: dict[str, Any], cutoff: pd.Timestamp) -> list[dict[str, Any]]:
    flagged_all = (
        inp["sanity"][inp["sanity"]["flagged"].astype(bool)] if len(inp["sanity"]) else pd.DataFrame()
    )
    out = []
    for pid, rows in inp["ind"].groupby("point_id", sort=True):
        asset = str(rows["asset_id"].iat[0])
        machine = str(rows["machine_id"].iat[0]) if "machine_id" in rows else asset[:6]
        win = inp["win"][inp["win"]["point_id"] == pid]
        fl = flagged_all[flagged_all["point_id"] == pid] if len(flagged_all) else flagged_all
        out.append(
            {
                "point_id": pid,
                "asset_id": asset,
                "rows": rows.reset_index(drop=True),
                "windows": win[["point_id", "start", "reason"]].reset_index(drop=True),
                "excluded": excluded_intervals(asset, inp["conds"], inp["wo"], fl, cutoff, cfg),
                "rated": inp["rated"].get(machine),
                "ambient": inp["ambient"]
                if str(rows["point_class"].iat[0]).startswith("compressor")
                else None,
            }
        )
    return out


def _parallel(fn: Any, jobs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[Any]:
    n = cfg["models"]["nbm"].get("selection_n_jobs", 1)
    if n == 1 or len(jobs) < 2:
        return [fn(**j) for j in jobs]
    from joblib import Parallel, delayed

    n = os.cpu_count() or 1 if n == -1 else n
    return Parallel(n_jobs=n, backend="loky")(delayed(fn)(**j) for j in jobs)


def _fit_job(
    point_id: str,
    asset_id: str,
    rows: pd.DataFrame,
    windows: pd.DataFrame,
    excluded: list[tuple[pd.Timestamp, pd.Timestamp]],
    rated: float | None,
    ambient: pd.Series | None,
    settings: list[nbm.Settings],
    cfg: dict[str, Any],
    fit_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[str, dict[str, nbm.PointModel], dict[str, Any]]:
    models, info = nbm.fit_point(rows, windows, settings, cfg, fit_start, cutoff, excluded, rated, ambient)
    return point_id, models, {**info, "asset_id": asset_id, "excluded_intervals": len(excluded)}


def _score_job(
    point_id: str,
    asset_id: str,
    rows: pd.DataFrame,
    windows: pd.DataFrame,
    ambient: pd.Series | None,
    models: dict[str, nbm.PointModel],
    cfg: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    factor: float,
    pcts: list[float],
    full: bool,
) -> dict[str, Any]:
    """Score one point over a span with each settings' model: the persisted exceedances at every sweep
    percentile and persistence, and (full) the scored rows and segments."""
    nb = cfg["models"]["nbm"]
    out: dict[str, Any] = {"point_id": point_id, "asset_id": asset_id, "exceed": {}, "scored_hours": {}}
    for key, m in models.items():
        s, segs = nbm.score_point(m, rows, windows, cfg, start, end, factor, nb["top_contributors"], ambient)
        if s.empty:
            continue
        out["scored_hours"][key] = int((s["state"] == "scored").sum())
        for p in pcts:
            for k in PERSISTENCES:
                ex = nbm.exceedance_hours(s, f"thr_{p:g}", k)
                out["exceed"][(key, float(p), k)] = ex["ts"].to_numpy()
        if full:
            out.setdefault("scores", {})[key] = s.assign(asset_id=asset_id)
            out.setdefault("segments", {})[key] = segs.assign(asset_id=asset_id)
    return out


# ---------------------------------------------------------------- selection metrics


def asset_episodes(times: dict[str, np.ndarray], gap_days: float) -> pd.DataFrame:
    """ALERT episodes per asset from persisted exceedance times per asset."""
    rows = [
        pd.DataFrame({"asset_id": a, "point_id": "", "ts": pd.to_datetime(t, utc=True)})
        for a, t in times.items()
        if len(t)
    ]
    sig = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["asset_id", "point_id", "ts"])
    return severity.episodes(sig, gap_days)


def asset_conditions(conds: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    """Confirmed conditions on the instrumented assets credited in [a, b), matched by asset (R3.9 item 3)."""
    c = conds[conds["instrumented"] & (conds["credit_ts"] >= a) & (conds["credit_ts"] < b)].copy()
    c["point_scope"] = None
    return c


def summarize(
    method: str,
    level: str,
    eps: pd.DataFrame,
    conds: pd.DataFrame,
    conds_all: pd.DataFrame,
    period: tuple[pd.Timestamp, pd.Timestamp],
    n_assets: int,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    window_days: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    w = window_days if window_days is not None else cfg["evaluation"]["catch"]["opened_within_days"]
    ca = conds_all.assign(point_scope=None)
    return ev.summarize(
        method,
        level,
        eps,
        conds,
        ca,
        period,
        n_assets,
        w,
        ctx["inspection_hours_each"],
        ctx["margin"],
        cfg["evaluation"]["plannable_min_warning_days"],
    )


def _curve(
    results: list[dict[str, Any]],
    key: str,
    pcts: list[float],
    persistence: int,
    conds: pd.DataFrame,
    conds_all: pd.DataFrame,
    period: tuple[pd.Timestamp, pd.Timestamp],
    n_assets: int,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
) -> pd.DataFrame:
    gap = cfg["evaluation"]["catch"]["episode_gap_days"]
    rows = []
    for p in pcts:
        times: dict[str, list[np.ndarray]] = {}
        for r in results:
            t = r["exceed"].get((key, float(p), persistence))
            if t is not None and len(t):
                times.setdefault(r["asset_id"], []).append(t)
        eps = asset_episodes({a: np.concatenate(v) for a, v in times.items()}, gap)
        row, _ = summarize("ml_model", f"{p:g}", eps, conds, conds_all, period, n_assets, cfg, ctx)
        rows.append({**row, "percentile": float(p), "asset_months": asset_months(period, n_assets)})
    return pd.DataFrame(rows)


def asset_months(period: tuple[pd.Timestamp, pd.Timestamp], n_assets: int) -> float:
    return n_assets * (period[1] - period[0]) / pd.Timedelta(days=365.25 / 12)


def vendor_episodes(
    vendor: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp, cfg: dict[str, Any]
) -> pd.DataFrame:
    v = vendor[(vendor["ts"] >= a) & (vendor["ts"] < b)]
    return severity.episodes(
        v[["asset_id", "point_id", "ts", "rules"]], cfg["evaluation"]["catch"]["episode_gap_days"]
    )


def combine(curves: list[pd.DataFrame]) -> pd.DataFrame:
    """Pool fold curves point-wise: caught over conditions, chance weighted by conditions, false episodes
    over asset-months."""
    c = pd.concat(curves, ignore_index=True)
    c["chance_x"] = c["chance_share"].fillna(0.0) * c["conditions"]
    g = c.groupby("percentile", as_index=False)[
        ["conditions", "caught", "chance_x", "false_episodes", "asset_months"]
    ].sum()
    g["caught_share"] = g["caught"] / g["conditions"].replace(0, np.nan)
    g["chance_share"] = g["chance_x"] / g["conditions"].replace(0, np.nan)
    g["false_per_asset_month"] = g["false_episodes"] / g["asset_months"].replace(0, np.nan)
    return g.drop(columns=["chance_x"])


def curve_floor(
    curves: dict[str, pd.DataFrame], budget: float, sweep: list[float], extension: list[float]
) -> float:
    """The operating curve's lowest percentile: the configured sweep's lowest, extended down the extension
    list until every candidate's pooled curve reaches the budget (its highest false-episode rate at or
    above the floor is at least the budget); the extension's last value if none does."""
    floor = min(sweep)
    for p in sorted(extension, reverse=True):
        if all(
            float(c.loc[c["percentile"] >= floor, "false_per_asset_month"].max()) >= budget
            for c in curves.values()
        ):
            return floor
        floor = p
    return floor


def select_champion(sel: pd.DataFrame, n_conditions: int, margin_conditions: int) -> str:
    """Highest skill (catch share minus chance at the Vendor Rules' false-episode budget, both folds
    combined); among settings within margin_conditions of it, the simplest."""
    best = sel["skill"].max()
    close = sel[(best - sel["skill"]) * n_conditions <= margin_conditions + 1e-9]
    order = close.sort_values("complexity_rank", kind="stable")
    return str(order["settings"].iat[0])


# ---------------------------------------------------------------- training run


def _fit_all(
    args: list[dict[str, Any]],
    settings: list[nbm.Settings],
    cfg: dict[str, Any],
    fit_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[dict[str, dict[str, nbm.PointModel]], pd.DataFrame]:
    keys = ("point_id", "asset_id", "rows", "windows", "excluded", "rated", "ambient")
    jobs = [
        {
            **{k: a[k] for k in keys},
            "settings": settings,
            "cfg": cfg,
            "fit_start": fit_start,
            "cutoff": cutoff,
        }
        for a in args
    ]
    fits = _parallel(_fit_job, jobs, cfg)
    models = {pid: m for pid, m, _ in fits if m}
    info = pd.DataFrame([i for _, _, i in fits])
    return models, info


def _score_all(
    args: list[dict[str, Any]],
    models: dict[str, dict[str, nbm.PointModel]],
    cfg: dict[str, Any],
    span: tuple[pd.Timestamp, pd.Timestamp],
    factor: float,
    pcts: list[float],
    full: bool,
) -> list[dict[str, Any]]:
    keys = ("point_id", "asset_id", "rows", "windows", "ambient")
    jobs = [
        {
            **{k: a[k] for k in keys},
            "models": models[a["point_id"]],
            "cfg": cfg,
            "start": span[0],
            "end": span[1],
            "factor": factor,
            "pcts": pcts,
            "full": full,
        }
        for a in args
        if a["point_id"] in models
    ]
    return _parallel(_score_job, jobs, cfg)


def _factor(models: dict[str, dict[str, nbm.PointModel]], cfg: dict[str, Any]) -> tuple[float, int]:
    pl = cfg["models"]["nbm"]["plausibility"]
    first = [next(iter(m.values())) for m in models.values() if m]
    return nbm.plausibility_factor(first, pl["factor_quantile"], pl.get("factor") or 1.45)


def _frames(results: list[dict[str, Any]], key: str, part: str) -> pd.DataFrame:
    fr = [r[part][key] for r in results if key in r.get(part, {})]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def persisted_episodes(
    scores: pd.DataFrame, thr_col: str, persistence: int, cfg: dict[str, Any]
) -> pd.DataFrame:
    """ALERT episodes per asset from scored hours (columns point_id, asset_id, hour_utc, state, score and
    the threshold column)."""
    cols = ["point_id", "ts", "ratio", "top_contributors"]
    ex = nbm.exceedance_hours(scores, thr_col, persistence) if len(scores) else pd.DataFrame(columns=cols)
    if len(ex):
        ex = ex.merge(scores[["point_id", "asset_id"]].drop_duplicates(), on="point_id")
    else:
        ex = ex.assign(asset_id=pd.Series(dtype=object))
    sig = ex.reindex(columns=["asset_id", "point_id", "ts"])
    return severity.episodes(sig, cfg["evaluation"]["catch"]["episode_gap_days"])


def _fold_row(
    k: int, s: nbm.Settings, op: pd.Series, at: dict[str, Any], vrow: dict[str, Any], factor: float
) -> dict[str, Any]:
    return {
        "fold": k,
        "settings": s.key,
        "candidate": s.candidate,
        "threshold_mode": s.threshold_mode,
        "hidden": "-".join(map(str, s.hidden)),
        "epochs": s.epochs,
        "conditions": int(op["conditions"]),
        "caught_at_operating": int(op["caught"]),
        "caught_share_operating": float(op["caught_share"]),
        "false_per_asset_month_operating": float(op["false_per_asset_month"]),
        "caught_share_at_budget": at["caught_share"],
        "chance_share_at_budget": at["chance_share"],
        "within_curve": bool(at["within_curve"]),
        "vendor_false_per_asset_month": vrow["false_per_asset_month"],
        "vendor_caught_share": vrow["caught_share"],
        "plausibility_factor": factor,
    }


def train_task(
    store: Store, cfg: dict[str, Any], experiment: str | None = None, job_run_id: str | None = None
) -> dict[str, Any]:
    """The ML Model's training run: every setting in the selection grid fit on both expanding-window folds,
    scored on each fold's span and judged against the Vendor Rules' false-episode budget on the same span;
    the selection rule on both folds combined picks the settings; the champion is refit on all of
    training. Reads nothing at or after the training end."""
    tasks._no_autolog(experiment)
    nb = cfg["models"]["nbm"]
    b = splits.boundaries(cfg)
    cutoff = splits.training_cutoff(cfg)
    inp = load_inputs(store, cfg, b["train"][1])
    ctx = tasks._context(store)
    grid = nbm.settings_grid(cfg)
    op_p = float(nb["threshold_percentile"])
    pcts = nbm.sweep_percentiles(cfg)
    n_assets = inp["pts"]["asset_id"].nunique()
    persist = int(cfg["models"]["severity"]["alert_persistence_hours"])
    fold_rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, int], list[pd.DataFrame]] = {}
    fold_models: dict[int, Any] = {}
    fold_args: dict[int, Any] = {}
    vendor_rows: list[dict[str, Any]] = []
    factors: dict[int, tuple[float, int]] = {}
    for f in splits.folds(cfg):
        k = f["fold"]
        LOG.info("fold_start fold=%s cutoff=%s", k, f["cutoff"])
        args = _point_args(inp, cfg, f["cutoff"])
        models, _ = _fit_all(args, grid, cfg, f["fit_start"], f["cutoff"])
        factor, n_ratios = _factor(models, cfg)
        factors[k] = (factor, n_ratios)
        results = _score_all(args, models, cfg, f["score"], factor, pcts, False)
        conds_f = asset_conditions(inp["conds"], *f["score"])
        veps = vendor_episodes(inp["vendor"], *f["score"], cfg)
        vrow, _ = summarize(
            "vendor_rules", "ALERT", veps, conds_f, inp["conds"], f["score"], n_assets, cfg, ctx
        )
        vendor_rows.append({**vrow, "fold": k, "asset_months": asset_months(f["score"], n_assets)})
        for s in grid:
            for pk in PERSISTENCES:
                c = _curve(results, s.key, pcts, pk, conds_f, inp["conds"], f["score"], n_assets, cfg, ctx)
                curves.setdefault((s.key, pk), []).append(c.assign(fold=k))
        fold_models[k], fold_args[k] = models, args
        LOG.info("fold_done fold=%s factor=%.3f", k, factor)
    vend = pd.DataFrame(vendor_rows)
    budget = float(vend["false_episodes"].sum() / vend["asset_months"].sum())
    floor = curve_floor(
        {s.key: combine(curves[(s.key, persist)]) for s in grid},
        budget,
        [float(p) for p in nb["sweep_percentiles"]],
        [float(p) for p in nb.get("sweep_extension_percentiles", [])],
    )
    LOG.info("curve floor percentile %s for budget %.4f", floor, budget)
    for s in grid:
        for c in curves[(s.key, persist)]:
            k = int(c["fold"].iat[0])
            cc = c[c["percentile"] >= floor]
            vrow = vend[vend["fold"] == k].iloc[0].to_dict()
            at = ev.at_budget(cc, vrow["false_per_asset_month"])
            op = cc[cc["percentile"] == op_p].iloc[0]
            fold_rows.append(_fold_row(k, s, op, at, vrow, factors[k][0]))
    pool = set(nb.get("selection_candidates") or nb["candidates"])
    sel_rows = []
    for s in grid:
        comb = combine(curves[(s.key, persist)])
        comb = comb[comb["percentile"] >= floor]
        at = ev.at_budget(comb, budget)
        op = comb[comb["percentile"] == op_p].iloc[0]
        sel_rows.append(
            {
                "settings": s.key,
                "candidate": s.candidate,
                "threshold_mode": s.threshold_mode,
                "hidden": "-".join(map(str, s.hidden)),
                "epochs": s.epochs,
                "conditions": int(op["conditions"]),
                "caught_share_at_budget": at["caught_share"],
                "chance_share_at_budget": at["chance_share"],
                "skill": at["caught_share"] - at["chance_share"],
                "within_curve": bool(at["within_curve"]),
                "caught_share_operating": float(op["caught_share"]),
                "false_per_asset_month_operating": float(op["false_per_asset_month"]),
                "vendor_false_per_asset_month": budget,
                "vendor_caught_share": float(vend["caught"].sum() / max(vend["conditions"].sum(), 1)),
                "input_set": nbm.input_set(s.candidate),
                "pool": "selection" if s.candidate in pool else "ablation",
                "curve_false_max": float(comb["false_per_asset_month"].max()),
                "curve_floor_percentile": floor,
            }
        )
    sel = pd.DataFrame(sel_rows)
    order = sorted(grid, key=nbm.complexity)
    sel["complexity_rank"] = sel["settings"].map({s.key: i for i, s in enumerate(order)})
    in_pool = sel[sel["pool"] == "selection"]
    champion_key = select_champion(
        in_pool, int(in_pool["conditions"].max()), nb["champion_margin_conditions"]
    )
    sel["champion"] = sel["settings"] == champion_key
    champ = next(s for s in grid if s.key == champion_key)
    LOG.info("champion settings=%s", champion_key)
    fold_scores, fold_segs = [], []
    for f in splits.folds(cfg):
        k = f["fold"]
        m = {
            pid: {champion_key: ms[champion_key]} for pid, ms in fold_models[k].items() if champion_key in ms
        }
        res = _score_all(fold_args[k], m, cfg, f["score"], factors[k][0], pcts, True)
        fold_scores.append(_frames(res, champion_key, "scores").assign(fold=k))
        fold_segs.append(_frames(res, champion_key, "segments").assign(fold=k))
    rob = [c.assign(settings=key, persistence_hours=pk) for (key, pk), cs in curves.items() for c in cs]
    del fold_models
    args = _point_args(inp, cfg, cutoff)
    final, info = _fit_all(args, [champ], cfg, b["train"][0], cutoff)
    factor, n_ratios = _factor(final, cfg)
    res = _score_all(args, final, cfg, (b["train"][0], cutoff), factor, pcts, True)
    train_scores = _frames(res, champion_key, "scores")
    train_segs = _frames(res, champion_key, "segments")
    models = {pid: ms[champion_key] for pid, ms in final.items()}
    ref_a, ref_b = splits.reference_window(cfg)
    ts_utc = pd.to_datetime(train_scores["hour_utc"], utc=True)
    ref_eps = persisted_episodes(train_scores[ts_utc >= ref_a], "threshold", persist, cfg)
    ref_conds = asset_conditions(inp["conds"], ref_a, ref_b)
    ref_row, _ = summarize(
        "ml_model", "ALERT", ref_eps, ref_conds, inp["conds"], (ref_a, ref_b), n_assets, cfg, ctx
    )
    fold_rate = float(sel.loc[sel["champion"], "false_per_asset_month_operating"].iat[0])
    hs = healthy_stats_table(train_scores, inp["ind"])
    version = tasks.model_version(
        "nbm", {"settings": champion_key, "factor": factor, "info": info.astype(str).to_dict("records")}
    )
    model_info = pd.DataFrame(
        [
            {
                "point_id": pid,
                "point_class": m.point_class,
                "settings": champion_key,
                "n_train": m.n_train,
                "segments_trained": m.segments_trained,
                "temp_fit": str(m.temp_fit.to_dict()),
                "threshold_operating": m.thresholds["single"]["_point"][op_p],
                "speed_bands": ",".join(x for x in m.thresholds["speed_band"] if x != "_point"),
            }
            for pid, m in sorted(models.items())
        ]
    )
    settings_row = pd.DataFrame(
        [
            {
                "settings": champion_key,
                "candidate": champ.candidate,
                "threshold_mode": champ.threshold_mode,
                "hidden": "-".join(map(str, champ.hidden)),
                "epochs": champ.epochs,
                "plausibility_factor": factor,
                "plausibility_ratios": n_ratios,
                "curve_floor_percentile": floor,
                "selection_candidates": ",".join(sorted(pool)),
                "fold1_plausibility_factor": factors.get(1, (np.nan, 0))[0],
                "fold2_plausibility_factor": factors.get(2, (np.nan, 0))[0],
                "vendor_budget_false_per_asset_month": budget,
                "fold_false_per_asset_month": fold_rate,
                "reference_false_per_asset_month": ref_row["false_per_asset_month"],
                "reference_window": f"{ref_a:%Y-%m-%d} to {ref_b:%Y-%m-%d}",
                "model_version": version,
            }
        ]
    )
    from pdm.models.nbm_pyfunc import NBMPointModels

    model = NBMPointModels(champion_key, models, factor, cfg, hs)
    champ_row = sel[sel["champion"]].iloc[0]
    run = tasks._mlflow_log(
        experiment,
        {
            "settings": champion_key,
            "candidate": champ.candidate,
            "threshold_mode": champ.threshold_mode,
            "hidden": "-".join(map(str, champ.hidden)),
            "epochs": champ.epochs,
            "plausibility_factor": factor,
            "alert_persistence_hours": persist,
            "threshold_percentile": op_p,
            "seed": cfg["models"]["seed"],
        },
        {
            "skill": float(champ_row["skill"]),
            "caught_share_at_budget": float(champ_row["caught_share_at_budget"]),
            "fold_false_per_asset_month": fold_rate,
            "reference_false_per_asset_month": float(ref_row["false_per_asset_month"]),
            "vendor_budget_false_per_asset_month": budget,
        },
        {
            "name": f"ml_model {champion_key}",
            "layer": "ml_model",
            "candidate": champ.candidate,
            "settings": champion_key,
            "challenger": "true",
            "model_version": version,
            "levels": "WATCH,ALERT,ALARM",
            **({"pdm.job_run_id": str(job_run_id)} if job_run_id else {}),
        },
        context=tasks.run_context(store, cfg, TRAIN_INPUTS) if experiment else None,
        extra=_log_model(model, sel, signature_example(inp["ind"], inp["win"])),
        lineage=(store, TRAIN_INPUTS),
    )
    store.write_pd(pd.DataFrame(fold_rows), "gold", "ml_candidates_fold")
    store.write_pd(sel.assign(model_version=version), "gold", "ml_candidates")
    store.write_pd(pd.concat(rob, ignore_index=True), "gold", "ml_selection_curves")
    store.write_pd(vend, "gold", "ml_vendor_fold")
    store.write_pd(
        pd.concat(fold_scores, ignore_index=True).assign(model_version=version), "gold", "ml_fold_scores"
    )
    store.write_pd(pd.concat(fold_segs, ignore_index=True), "gold", "ml_fold_segments")
    store.write_pd(
        train_scores.assign(model_version=version, settings=champion_key), "gold", "ml_train_scores"
    )
    store.write_pd(train_segs.assign(model_version=version), "gold", "ml_train_segments")
    store.write_pd(model_info.assign(model_version=version), "gold", "ml_models")
    store.write_pd(info.astype(str), "gold", "ml_training_info")
    store.write_pd(settings_row, "gold", "ml_settings")
    store.write_pd(hs.assign(model_version=version), "gold", "point_healthy_stats")
    return {
        "champion": champion_key,
        "version": version,
        "mlflow_run": run,
        "model": model,
        "selection": sel,
        "plausibility_factor": factor,
        "fold_rate": fold_rate,
        "reference_rate": float(ref_row["false_per_asset_month"]),
    }


TRAIN_INPUTS = [
    ("silver", "hourly_indicators"),
    ("silver", "healthy_windows"),
    ("silver", "confirmed_conditions"),
    ("silver", "points"),
    ("gold", "baseline_sanity"),
    ("gold", "rule_floor_exceedances"),
    ("bronze", "cmms_work_orders"),
    ("bronze", "cmp_room"),
]


def healthy_stats_table(scores: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    """Healthy spread per point, segment and regime from the training hours the champion scored, for the
    advisory z-scores (median and robust scale of the log indicators)."""
    if scores.empty:
        return pd.DataFrame()
    s = scores[scores["state"] == "scored"][["point_id", "hour_utc", "segment"]]
    rows = ind.merge(s, on=["point_id", "hour_utc"])
    parts = []
    for (_pid, seg), g in rows.groupby(["point_id", "segment"]):
        st = adv.healthy_stats(g)
        if len(st):
            parts.append(st.assign(segment=int(seg), segment_start=g["hour_utc"].min()))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def signature_example(ind: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """A small model input for the registered signature: one point's first 30 measured hours with its
    segment starts."""
    pid = sorted(ind["point_id"].unique())[0]
    x = ind[ind["point_id"] == pid].sort_values("hour_utc").head(30).copy()
    starts = win.loc[win["point_id"] == pid, "start"].sort_values()
    x["segment_starts"] = ";".join(str(t) for t in starts)
    x["hour_utc"] = pd.to_datetime(x["hour_utc"], utc=True).dt.tz_localize(None)
    x["implausible"] = x["implausible"].fillna(False).astype(bool)
    return x.reset_index(drop=True)


def _log_model(model: Any, sel: pd.DataFrame, example: pd.DataFrame | None = None) -> Any:
    def body() -> None:
        import tempfile
        from pathlib import Path

        import mlflow

        import pdm
        from pdm.models.nbm_pyfunc import HEALTHY_STATS

        tasks._log_table(sel, "candidate_selection")
        with tempfile.TemporaryDirectory() as d:
            hs_path = str(Path(d) / f"{HEALTHY_STATS}.parquet")
            model.healthy_stats.to_parquet(hs_path)
            kwargs = {
                "python_model": model,
                "artifacts": {HEALTHY_STATS: hs_path},
                "code_paths": [str(Path(pdm.__file__).parent)],
                "pip_requirements": ["numpy", "pandas", "scipy", "scikit-learn", "pyyaml", "pyarrow"],
            }
            if example is not None:
                from mlflow.models import infer_signature

                kwargs["signature"] = infer_signature(example, model.predict(None, example))
            try:
                mlflow.pyfunc.log_model(name="model", **kwargs)
            except TypeError:
                mlflow.pyfunc.log_model(artifact_path="model", **kwargs)

    return body


# ---------------------------------------------------------------- scoring run and levels

CONTROLLER_CRITERIA = (
    "TSC pump-on pressure low",
    "thermistor at limit",
    "sustained cavitation",
    "discharge temperature",
)
ALARM_CRITERIA = {
    "spindle": {
        "temperature residual",
        "thermistor at limit",
        "stage 3 bearing signature",
        "stage 4 bearing signature",
        "velocity zone D",
    },
    "tsc_pump": {
        "velocity zone D",
        "stage 4 bearing signature",
        "temperature residual",
        "sustained cavitation",
    },
    "flood_pump": {"velocity zone D", "stage 4 bearing signature", "temperature residual"},
    "compressor": {
        "velocity zone D",
        "stage 4 bearing signature",
        "temperature residual",
        "discharge temperature",
    },
}


def score_task(store: Store, cfg: dict[str, Any], model: Any, model_version: str) -> dict[str, Any]:
    """Score the shadow year with the frozen ML Model (the registered champion in the job) and write
    gold.nbm_scores, nbm_exceedances (every sweep percentile and persistence), ml_segments (every segment
    with its re-centering and plausibility outcome) and the shadow-year segment stats for the advisory."""
    if cfg["evaluation"].get("holdout_shadow", True):
        raise ValueError("scoring the shadow year needs evaluation.holdout_shadow false")
    b = splits.boundaries(cfg)
    inp = load_inputs(store, cfg, b["shadow"][1])
    sc, segs = model.score_span(inp["ind"], inp["win"], b["shadow"][0], b["shadow"][1], inp["ambient"])
    assets = inp["pts"].set_index("point_id")["asset_id"]
    sc["asset_id"] = sc["point_id"].map(assets)
    segs["asset_id"] = segs["point_id"].map(assets)
    pcts = nbm.sweep_percentiles(cfg)
    parts = []
    for p in pcts:
        for k in PERSISTENCES:
            ex = nbm.exceedance_hours(sc, f"thr_{p:g}", k)
            if len(ex):
                parts.append(ex.assign(sweep=p, persistence_hours=k))
    ex_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["point_id", "ts"])
    ex_all["asset_id"] = ex_all["point_id"].map(assets)
    train_stats = model.healthy_stats
    if train_stats is None or len(train_stats) == 0:
        train_stats = tasks._utc_cols(store.read_pd("gold", "point_healthy_stats"), ["segment_start"])
    hs = shadow_segment_stats(train_stats, segs, inp["ind"])
    store.write_pd(sc.assign(model_version=model_version, settings=model.settings_key), "gold", "nbm_scores")
    store.write_pd(ex_all.assign(method="ml_model", model_version=model_version), "gold", "nbm_exceedances")
    store.write_pd(segs.assign(model_version=model_version), "gold", "ml_segments")
    store.write_pd(hs.assign(model_version=model_version), "gold", "point_segment_stats")
    flagged = segs[segs["plausibility_flag"].astype(bool)] if len(segs) else segs
    return {
        "model_version": model_version,
        "settings": model.settings_key,
        "scored_hours": int((sc["state"] == "scored").sum()),
        "segments_shadow": int((~segs["trained"].astype(bool)).sum()) if len(segs) else 0,
        "plausibility_flags": len(flagged),
        "exceedances": len(ex_all),
    }


def shadow_segment_stats(train_stats: pd.DataFrame, segs: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    """Advisory healthy stats for every segment: training segments as trained; a segment re-centered in the
    shadow year takes its median from its re-centering window and its scale from the point's training
    segments (the frozen spread, re-centered)."""
    if train_stats is None or train_stats.empty or segs.empty:
        return train_stats if train_stats is not None else pd.DataFrame()
    out = [train_stats]
    scale = train_stats.groupby(["point_id", "regime", "indicator"])["scale"].median()
    for r in segs[~segs["trained"].astype(bool)].itertuples():
        if pd.isna(r.level_ready):
            continue
        t = pd.to_datetime(ind["hour_utc"], utc=True)
        w = ind[(ind["point_id"] == r.point_id) & (t >= r.segment_start) & (t < r.level_ready)]
        st = adv.healthy_stats(w) if len(w) >= 20 else pd.DataFrame()
        if st.empty:
            continue
        st["scale"] = [
            scale.get((p, g, i), s)
            for p, g, i, s in zip(st["point_id"], st["regime"], st["indicator"], st["scale"], strict=True)
        ]
        out.append(st.assign(segment=int(r.segment), segment_start=r.segment_start))
    return pd.concat(out, ignore_index=True)


def _controller_series(store: Store, cfg: dict[str, Any], end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Hourly controller inputs to the ALARM criteria: thermistor hours at the limit per HMC, the pump-on
    TSC pressure per HMC and the compressor discharge temperature hours at the warning."""
    from pyspark.sql import functions as F

    sv = cfg["models"]["severity"]
    hmc = set(store.read_pd("bronze", "reg_machines").query("`class` == 'hmc'")["machine_id"])
    mn = (
        store.read("bronze", "ctrl_minute")
        .where(F.col("machine_id").isin(sorted(hmc)))
        .select("machine_id", "minute_utc", "spindle_temp_c_mean")
    )
    hot = mn.where(F.col("spindle_temp_c_mean") >= sv["thermistor_alarm_c"]).toPandas()
    hot["minute_utc"] = _utc(hot["minute_utc"])
    therm = severity.thermistor_hours(
        hot[hot["minute_utc"] < end], sv["thermistor_alarm_c"], sv["thermistor_persistence_minutes"]
    )
    chh = store.read_pd(
        "silver", "controller_hourly", ["machine_id", "hour_utc", "coolant_pressure_pump_on_bar"]
    )
    chh["hour_utc"] = _utc(chh["hour_utc"])
    chh = chh[chh["machine_id"].isin(hmc) & (chh["hour_utc"] < end)]
    tsc = (
        chh.groupby(["machine_id", "hour_utc"])["coolant_pressure_pump_on_bar"]
        .median()
        .rename("pressure_bar")
        .reset_index()
    )
    cm = (
        store.read("bronze", "cmp_minute")
        .where(F.col("discharge_temp_c") >= sv["discharge_temp_c"])
        .select("compressor_id", "minute_utc")
        .toPandas()
    )
    cm["hour_utc"] = _utc(cm["minute_utc"]).dt.floor("h")
    dis = cm[cm["hour_utc"] < end][["compressor_id", "hour_utc"]].drop_duplicates()
    return {"thermistor": therm, "tsc": tsc, "discharge": dis}


def ml_alarm_criteria(
    scores: pd.DataFrame,
    segs: pd.DataFrame,
    ind: pd.DataFrame,
    ctrl: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """ALARM criterion hours per asset for the ML Model (asset_id, hour_utc, criterion), by asset class."""
    train_end = splits.training_cutoff(cfg)
    parts = []
    hf_hours = []
    for pid, rows in ind.groupby("point_id", sort=True):
        asset = str(rows["asset_id"].iat[0])
        s = scores[scores["point_id"] == pid]
        g = segs[segs["point_id"] == pid] if len(segs) else segs
        c = severity.sensor_criteria(rows, s, g, cfg, train_end)
        if c.empty:
            continue
        c["asset_id"] = asset
        hf_hours.append(c[c["criterion"] == "_hf_persistent"])
        cls = severity.asset_class(asset)
        keep = c[c["criterion"].isin(ALARM_CRITERIA[cls])]
        if cls == "compressor":
            keep = keep[
                (keep["criterion"] != "stage 4 bearing signature") | keep["point_id"].str.endswith("-AE")
            ]
        parts.append(keep)
    th = ctrl["thermistor"]
    if len(th):
        parts.append(
            pd.DataFrame(
                {
                    "asset_id": th["machine_id"] + "-SP",
                    "hour_utc": th["hour_utc"],
                    "criterion": "thermistor at limit",
                }
            )
        )
    hfh = (
        pd.concat(hf_hours, ignore_index=True) if hf_hours else pd.DataFrame(columns=["asset_id", "hour_utc"])
    )
    parts.append(severity.pressure_criteria(ctrl["tsc"], hfh[["asset_id", "hour_utc"]], cfg))
    d = ctrl["discharge"]
    if len(d):
        parts.append(
            pd.DataFrame(
                {
                    "asset_id": d["compressor_id"],
                    "hour_utc": d["hour_utc"],
                    "criterion": "discharge temperature",
                }
            )
        )
    out = pd.concat(
        [p.reindex(columns=["asset_id", "hour_utc", "criterion"]) for p in parts], ignore_index=True
    )
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True)
    allowed = [
        c in ALARM_CRITERIA[severity.asset_class(a)]
        for a, c in zip(out["asset_id"], out["criterion"], strict=True)
    ]
    out = out[np.asarray(allowed, dtype=bool)]
    return out.drop_duplicates().sort_values(["asset_id", "hour_utc", "criterion"]).reset_index(drop=True)


def ml_levels(
    scores: pd.DataFrame,
    criteria: pd.DataFrame,
    cfg: dict[str, Any],
    thr_col: str = "threshold",
    persistence: int | None = None,
) -> pd.DataFrame:
    """The ML Model's ALERT episodes per asset with their ALARM start (criterion and 14-day escalation), on
    the full criteria (the operating result) and on the sensor criteria only (alarm_sensor_*: without the
    criteria that read controller signals, for the comparison with the Vendor Rules)."""
    sv = cfg["models"]["severity"]
    k = persistence or int(sv["alert_persistence_hours"])
    eps = persisted_episodes(scores, thr_col, k, cfg)
    s = scores[scores["state"] == "scored"]
    operating = {
        a: pd.DataFrame(
            {
                "ts": pd.to_datetime(g["hour_utc"], utc=True).to_numpy(),
                "point_id": g["point_id"].to_numpy(),
                "met": (g["score"] > g[thr_col]).to_numpy(),
            }
        )
        for a, g in s.groupby("asset_id")
    }
    days = int(sv["escalation_days"])
    full = severity.alarm_starts(eps, criteria, operating, days)
    sensor = severity.alarm_starts(
        eps, criteria[~criteria["criterion"].isin(CONTROLLER_CRITERIA)], operating, days
    )
    return _with_sensor(full, sensor).assign(method="ml_model")


def _with_sensor(full: pd.DataFrame, sensor: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "alarm_criterion_ts",
        "alarm_criterion",
        "alarm_ts",
        "alarm_cause",
        "alarm_with14_ts",
        "alarm_with14_cause",
    ]
    out = full.copy()
    for c in cols:
        v = (
            sensor[c].reset_index(drop=True).set_axis(out.index)
            if len(sensor)
            else pd.Series(index=out.index, dtype=object)
        )
        out[c.replace("alarm_", "alarm_sensor_", 1)] = v
    for c in ("alarm_sensor_criterion_ts", "alarm_sensor_ts", "alarm_sensor_with14_ts"):
        out[c] = pd.to_datetime(out[c], utc=True)
    return out


def vendor_levels(vendor: pd.DataFrame, ind: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """The Vendor Rules' ALERT episodes per asset (vendor alert or alarm, ISO zone C or D, temperature
    warning or alarm) with their ALARM start: the first alarm-level exceedance in the episode, or the
    14-day escalation on the criterion that fired."""
    sv = cfg["models"]["severity"]
    eps = severity.episodes(
        vendor[["asset_id", "point_id", "ts", "rules"]], cfg["evaluation"]["catch"]["episode_gap_days"]
    )
    al = vendor[vendor["level"] == "alarm"]
    crit = pd.DataFrame(
        {"asset_id": al["asset_id"], "hour_utc": al["ts"], "criterion": "vendor " + al["rules"].astype(str)}
    )
    meas = ind[["point_id", "asset_id", "hour_utc"]].copy()
    meas["hour_utc"] = pd.to_datetime(meas["hour_utc"], utc=True)
    rules = vendor.groupby(["point_id", "ts"])["rules"].agg(
        lambda x: ",".join(sorted({r for v in x for r in str(v).split(",") if r}))
    )
    meas = meas.merge(
        rules.rename("rules").reset_index().rename(columns={"ts": "hour_utc"}),
        on=["point_id", "hour_utc"],
        how="left",
    )
    operating = {
        a: pd.DataFrame(
            {
                "ts": g["hour_utc"].to_numpy(),
                "point_id": g["point_id"].to_numpy(),
                "rules": g["rules"].fillna("").to_numpy(),
            }
        )
        for a, g in meas.groupby("asset_id")
    }
    full = severity.alarm_starts(eps, crit, operating, int(sv["escalation_days"]))
    return _with_sensor(full, full).assign(method="vendor_rules")


def _scores_for_levels(store: Store, held_out: bool) -> pd.DataFrame:
    parts = [store.read_pd("gold", "ml_train_scores").assign(period="training")]
    if not held_out and store.exists("gold", "nbm_scores"):
        parts.append(store.read_pd("gold", "nbm_scores").assign(period="shadow"))
    s = pd.concat(parts, ignore_index=True)
    s["hour_utc"] = _utc(s["hour_utc"])
    return s


def levels_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    """WATCH, ALERT and ALARM for both methods: ALERT episodes per asset with their ALARM start (criterion
    and 14-day escalation, reported apart), on each fold's scored span, on training (in-sample, for the
    false ALARM rate on healthy training hours before the shadow year is scored) and, once the holdout is
    lifted, on the shadow year. Writes gold.alarm_criteria, gold.level_episodes and gold.watch_hours."""
    held = bool(cfg["evaluation"].get("holdout_shadow", True))
    b = splits.boundaries(cfg)
    end = b["train"][1] if held else b["shadow"][1]
    inp = load_inputs(store, cfg, end)
    scores = _scores_for_levels(store, held)
    segs = store.read_pd("gold", "ml_train_segments")
    if not held and store.exists("gold", "ml_segments"):
        segs = pd.concat([segs, store.read_pd("gold", "ml_segments")], ignore_index=True)
    segs = tasks._utc_cols(segs, ["segment_start", "level_ready"]).drop_duplicates(
        ["point_id", "segment_start"], keep="last"
    )
    ctrl = _controller_series(store, cfg, end)
    crit = ml_alarm_criteria(scores, segs, inp["ind"], ctrl, cfg)
    parts = []
    periods: list[tuple[str, tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame]] = []
    fs = store.read_pd("gold", "ml_fold_scores")
    fs["hour_utc"] = _utc(fs["hour_utc"])
    for f in splits.folds(cfg):
        periods.append((f"fold {f['fold']}", f["score"], fs[fs["fold"] == f["fold"]]))
    periods.append(
        ("training", (b["train"][0], splits.training_cutoff(cfg)), scores[scores["period"] == "training"])
    )
    if not held:
        periods.append(("shadow", b["shadow"], scores[scores["period"] == "shadow"]))
    for name, (a, z), sc in periods:
        ml = ml_levels(sc, crit, cfg)
        vd = vendor_levels(
            inp["vendor"][(inp["vendor"]["ts"] >= a) & (inp["vendor"]["ts"] < z)], inp["ind"], cfg
        )
        parts += [ml.assign(period=name), vd.assign(period=name)]
    lev = pd.concat(parts, ignore_index=True)
    watch = scores[
        (scores["state"] == "plausibility")
        | ((scores["state"] == "scored") & (scores["score"] > scores["threshold"]))
    ]
    watch = watch[["point_id", "asset_id", "hour_utc", "state", "period"]].assign(
        reason=np.where(watch["state"] == "plausibility", "plausibility flag", "unpersisted exceedance")
    )
    store.write_pd(crit, "gold", "alarm_criteria")
    store.write_pd(lev, "gold", "level_episodes")
    if not held:
        cases = ml_cases(lev[(lev["period"] == "shadow") & (lev["method"] == "ml_model")])
        store.write_pd(cases, "gold", "cases")
        store.write_pd(tasks._work_requests(store, cases, cfg), "gold", "work_requests")
        store.write_pd(tasks._tsc_pressure_daily(store, cfg), "gold", "tsc_pressure_daily")
    store.write_pd(watch.drop(columns=["state"]), "gold", "watch_hours")
    return {
        "criteria_hours": crit.groupby("criterion").size().to_dict(),
        "episodes": {f"{p} {m}": int(n) for (p, m), n in lev.groupby(["period", "method"]).size().items()},
        "alarms": {
            f"{p} {m}": int(n)
            for (p, m), n in lev[lev["alarm_ts"].notna()].groupby(["period", "method"]).size().items()
        },
    }


def ml_cases(eps: pd.DataFrame) -> pd.DataFrame:
    """The ML Model's ALERT episodes as cases for the queue and the advisory (one work request each), with
    the level reached (ALERT, or ALARM with its start and cause)."""
    c = eps.copy().reset_index(drop=True)
    c["case_id"] = [
        f"{a}|{pd.Timestamp(o):%Y%m%d%H}" for a, o in zip(c["asset_id"], c["opened"], strict=True)
    ]
    c["first_alarm"] = c["opened"]
    c["layers"] = "ML Model"
    c["trigger_rules"] = "ml_model:" + c["trigger_points"].astype(str)
    c["level"] = np.where(c["alarm_ts"].notna(), "ALARM", "ALERT")
    return c


# ---------------------------------------------------------------- evaluation

HOURS_PER_TECHNICIAN_YEAR = 2080.0


def _quarters(span: tuple[pd.Timestamp, pd.Timestamp]) -> list[tuple[str, tuple[pd.Timestamp, pd.Timestamp]]]:
    a = span[0].tz_convert(tasks.LOCAL_TZ).tz_localize(None)
    out = []
    for q in range(4):
        s = (a + pd.DateOffset(months=3 * q)).tz_localize(tasks.LOCAL_TZ).tz_convert("UTC")
        e = (a + pd.DateOffset(months=3 * q + 3)).tz_localize(tasks.LOCAL_TZ).tz_convert("UTC")
        out.append((f"shadow Q{q + 1}", (s, min(e, span[1]))))
    return out


def _levels(store: Store) -> pd.DataFrame:
    lev = store.read_pd("gold", "level_episodes")
    return tasks._utc_cols(
        lev,
        [
            "opened",
            "last",
            "closed",
            "alarm_ts",
            "alarm_criterion_ts",
            "alarm_escalation_ts",
            "alarm_sensor_ts",
            "alarm_sensor_criterion_ts",
            "alarm_with14_ts",
            "alarm_sensor_with14_ts",
        ],
    )


def _ch_episodes(store: Store, cfg: dict[str, Any]) -> pd.DataFrame:
    hmc = set(store.read_pd("bronze", "reg_machines").query("`class` == 'hmc'")["machine_id"])
    ce = store.read_pd("gold", "controller_health_exceedances")
    sig = r2.controller_signals(ce, hmc, 1.0)
    return r2.controller_episodes(sig, cfg["evaluation"]["catch"]["episode_gap_days"])


def _ml_curve_eps(
    store: Store, cfg: dict[str, Any], span: tuple[pd.Timestamp, pd.Timestamp]
) -> dict[float, pd.DataFrame]:
    """The ML Model's ALERT episodes at every sweep percentile over a span (shadow-year exceedances at the
    ALERT persistence)."""
    k = int(cfg["models"]["severity"]["alert_persistence_hours"])
    ex = store.read_pd("gold", "nbm_exceedances")
    ex["ts"] = _utc(ex["ts"])
    ex = ex[(ex["persistence_hours"] == k) & (ex["ts"] >= span[0]) & (ex["ts"] < span[1])]
    gap = cfg["evaluation"]["catch"]["episode_gap_days"]
    floor = float(store.read_pd("gold", "ml_settings")["curve_floor_percentile"].iat[0])
    ex = ex[ex["sweep"] >= floor]
    return {
        float(p): severity.episodes(g[["asset_id", "point_id", "ts"]], gap) for p, g in ex.groupby("sweep")
    }


def evaluation_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    """The ML Model against the Vendor Rules, with Controller Health beside them: method rows at ALERT and
    ALARM (with and without the 14-day rule) by period (each fold's span, the shadow year and its quarters),
    the matched-budget comparison with 90% intervals from resampling whole conditions, the per-condition
    table with the four groups and Controller Health, the R3.6 value table, the combined monitoring line,
    catch by condition type, Controller Health on instrumented assets and uninstrumented machines, the
    inspection workload by month and each ALERT episode's outcome."""
    held = bool(cfg["evaluation"].get("holdout_shadow", True))
    b = splits.boundaries(cfg)
    lev = _levels(store)
    conds = tasks._conditions(store, cfg)
    ctx = tasks._context(store)
    wo = tasks._wo(store)
    pts = tasks._points(store)
    n_assets = pts["asset_id"].nunique()
    machines = store.read_pd("bronze", "reg_machines")["machine_id"].nunique()
    window = float(cfg["evaluation"]["catch"]["opened_within_days"])
    gap = cfg["evaluation"]["catch"]["episode_gap_days"]
    ch = _ch_episodes(store, cfg)
    periods = [(f"fold {f['fold']}", f["score"]) for f in splits.folds(cfg)]
    periods.append(("training", (b["train"][0], splits.training_cutoff(cfg))))
    if not held:
        periods += [("shadow", b["shadow"]), *_quarters(b["shadow"])]
    rows, by_type, ch_rows = [], [], []
    outcomes: list[pd.DataFrame] = []
    conf = r2.confirmations(conds, wo)
    for name, span in periods:
        lev_name = "shadow" if name.startswith("shadow") else name
        L = lev[lev["period"] == lev_name]
        L = L[(L["opened"] >= span[0] - pd.Timedelta(days=window)) & (L["opened"] < span[1])]
        cp = r2.conditions_in(conds, span)
        fails = cp[cp["unplanned"].astype(bool)]
        for method, label in (("ml_model", "ML Model"), ("vendor_rules", "Vendor Rules")):
            e = L[L["method"] == method]
            r, caught = r2.method_row(
                label, "ALERT", r2.alert_view(e), cp, conds, span, n_assets, window, ctx, cfg
            )
            rows.append(
                {**r, **r2.false_two_ways(r2.alert_view(e), conds, wo, span, n_assets), "period": name}
            )
            by_type.append(caught.assign(period=name))
            for esc, sensor, lvl in (
                (False, False, "ALARM"),
                (True, False, "ALARM with the 14-day rule (tested variant)"),
                (False, True, "ALARM on sensor criteria"),
                (True, True, "ALARM on sensor criteria with the 14-day rule (tested variant)"),
            ):
                av = r2.alarm_view(e, esc, sensor)
                ra, _ = r2.method_row(label, lvl, av, fails, conds, span, n_assets, window, ctx, cfg)
                only14 = (
                    int(
                        (
                            e["alarm_escalation_ts"].notna()
                            & e["alarm_sensor_criterion_ts" if sensor else "alarm_criterion_ts"].isna()
                        ).sum()
                    )
                    if esc
                    else 0
                )
                ao = r2.alarm_outcomes(e, conf, wo, span, b["shadow"][1], esc, sensor)
                n_false = int(ao["false_alarm"].sum())
                n_nothing = int((ao["outcome"] == r2.ALARM_OUTCOMES[3]).sum())
                ra.update(
                    {
                        "episodes": len(ao),
                        "false_episodes": n_false,
                        "false_per_asset_month": n_false / r2.asset_months(span, n_assets),
                        "unrelated_first": n_false - n_nothing,
                        "false_excl_unrelated": n_nothing,
                        "false_excl_unrelated_per_asset_month": n_nothing / r2.asset_months(span, n_assets),
                        "pending": int((ao["outcome"] == r2.ALARM_OUTCOMES[4]).sum()),
                    }
                )
                rows.append({**ra, "period": name, "alarms_from_14_day_rule_alone": only14})
                outcomes.append(ao.assign(period=name, method=label, level=lvl))
        r, _ = r2.method_row(
            "Controller Health", "ALERT", ch.assign(point_id=""), cp, conds, span, n_assets, window, ctx, cfg
        )
        ch_rows.append({**r, "period": name, "scope": "instrumented assets"})
        cu = r2.conditions_in(conds, span, instrumented=False)
        r, _ = r2.method_row(
            "Controller Health", "ALERT", ch.assign(point_id=""), cu, conds, span, machines, window, ctx, cfg
        )
        ch_rows.append({**r, "period": name, "scope": "uninstrumented machines"})
    out: dict[str, Any] = {}
    methods = pd.DataFrame(rows)
    store.write_pd(methods, "gold", "eval_methods")
    ao_all = pd.concat(outcomes, ignore_index=True) if outcomes else pd.DataFrame()
    store.write_pd(ao_all, "gold", "eval_alarm_outcomes")
    store.write_pd(false_alarms_by_criterion(ao_all), "gold", "eval_false_alarms_by_criterion")
    store.write_pd(pd.concat(by_type, ignore_index=True), "gold", "eval_catch_by_condition")
    store.write_pd(pd.DataFrame(ch_rows), "gold", "eval_controller_health")
    out["methods"] = methods[
        ["period", "method", "level", "conditions", "caught", "false_per_asset_month"]
    ].to_dict("records")
    if held:
        return out
    span = b["shadow"]
    S = lev[lev["period"] == "shadow"]
    ml, vd = S[S["method"] == "ml_model"], S[S["method"] == "vendor_rules"]
    cp = r2.conditions_in(conds, span)
    ct = r2.condition_table(cp, ml, vd, ch, window)
    sc = store.read_pd(
        "gold", "nbm_scores", ["point_id", "asset_id", "hour_utc", "state", "score", "threshold"]
    )
    sc["hour_utc"] = _utc(sc["hour_utc"])
    gaps = sc[["asset_id", "hour_utc"]]
    ct["ml_miss_reason"] = [
        None if r["ml_alert_caught"] else r2.miss_reason(r, "ml_model", sc, window, gaps)
        for _, r in ct.iterrows()
    ]
    vend_ex = load_inputs(store, cfg, span[1])["vendor"]
    vsc = gaps.assign(state="scored", score=np.nan, threshold=np.nan)
    vkeys = set(zip(vend_ex["asset_id"], vend_ex["ts"].dt.floor("h"), strict=True))
    vsc["vendor_over"] = [(a, t) in vkeys for a, t in zip(vsc["asset_id"], vsc["hour_utc"], strict=True)]
    ct["vendor_miss_reason"] = [
        None if r["vendor_alert_caught"] else r2.miss_reason(r, "vendor_rules", vsc, window, gaps)
        for _, r in ct.iterrows()
    ]
    store.write_pd(ct, "gold", "eval_conditions")
    curve_eps = _ml_curve_eps(store, cfg, span)
    curve = []
    for p, e in sorted(curve_eps.items()):
        r, _ = r2.method_row(
            "ML Model", f"{p:g}", e.assign(point_id=""), cp, conds, span, n_assets, window, ctx, cfg
        )
        curve.append({**r, "percentile": p})
    curve = pd.DataFrame(curve)
    vrow = methods[
        (methods["period"] == "shadow")
        & (methods["method"] == "Vendor Rules")
        & (methods["level"] == "ALERT")
    ].iloc[0]
    at = r2.at_budget(curve, float(vrow["false_per_asset_month"]))
    op_p = float(cfg["models"]["nbm"]["threshold_percentile"])
    conds_all = conds.reset_index(drop=True)
    eval_idx = np.asarray([conds_all.index[conds_all["wo_id"] == w][0] for w in cp["wo_id"]], dtype=int)
    v_alert = r2.alert_view(vd)
    v_alert = v_alert[(v_alert["opened"] >= span[0]) & (v_alert["opened"] < span[1])]
    vcatch = (
        ev.catch_table(r2.alert_view(vd), cp, window)["caught"].to_numpy(dtype=bool)
        if len(vd)
        else np.zeros(len(cp), bool)
    )
    ml_c, ml_cov = {}, {}
    for p, e in curve_eps.items():
        ee = e.assign(point_id="")
        ml_c[p] = (
            ev.catch_table(ee, cp, window)["caught"].to_numpy(dtype=bool)
            if len(ee)
            else np.zeros(len(cp), bool)
        )
        inside = ee[(ee["opened"] >= span[0]) & (ee["opened"] < span[1])]
        ml_cov[p] = r2._cover(inside, conds_all)
    boot = r2.bootstrap(
        vcatch,
        r2._cover(v_alert, conds_all),
        ml_c,
        ml_cov,
        op_p,
        eval_idx,
        len(conds_all),
        r2.asset_months(span, n_assets),
        int(cfg["evaluation"].get("bootstrap_draws", 2000)),
        int(cfg["models"]["seed"]),
    )
    mrow = methods[
        (methods["period"] == "shadow") & (methods["method"] == "ML Model") & (methods["level"] == "ALERT")
    ].iloc[0]
    boot.update(
        r2.asset_bootstrap(
            r2.alert_view(vd),
            curve_eps,
            op_p,
            cp,
            conds,
            span,
            sorted(pts["asset_id"].unique()),
            window,
            int(cfg["evaluation"].get("bootstrap_draws", 2000)),
            int(cfg["models"]["seed"]),
        )
    )
    matched = pd.DataFrame(
        [
            {
                "vendor_false_per_asset_month": float(vrow["false_per_asset_month"]),
                "vendor_caught_share": float(vrow["caught_share"]),
                "vendor_chance_share": float(vrow["chance_share"]),
                "ml_caught_share_at_budget": at["caught_share"],
                "ml_chance_share_at_budget": at["chance_share"],
                "within_curve": bool(at["within_curve"]),
                "catch_diff": at["caught_share"] - float(vrow["caught_share"]),
                "ml_false_per_asset_month": float(mrow["false_per_asset_month"]),
                "false_diff": float(mrow["false_per_asset_month"]) - float(vrow["false_per_asset_month"]),
                **boot,
            }
        ]
    )
    store.write_pd(curve, "gold", "eval_ml_curve")
    store.write_pd(matched, "gold", "eval_matched")
    store.write_pd(
        r2.value_table(ct, wo, ctx["margin"], cfg["evaluation"]["plannable_min_warning_days"]),
        "gold",
        "eval_value",
    )
    groups = (
        ct.groupby("group")
        .agg(
            conditions=("wo_id", "count"),
            ch_caught=("ch_caught", "sum"),
            median_ml_days_vs_vendor=("ml_days_vs_vendor", "median"),
        )
        .reset_index()
    )
    store.write_pd(groups, "gold", "eval_groups")
    caught_any = ct["ml_alert_caught"] | ct["vendor_alert_caught"] | ct["ch_caught"]
    n_false, rate = r2.merged_false_rate(
        [r2.alert_view(ml), r2.alert_view(vd), ch], conds, gap, span, n_assets
    )
    vd_only = int((ct["vendor_alert_caught"] | ct["ch_caught"]).sum())
    v_false, v_rate = r2.merged_false_rate([r2.alert_view(vd), ch], conds, gap, span, n_assets)
    combined = pd.DataFrame(
        [
            {
                "combination": "Vendor Rules and Controller Health",
                "caught": vd_only,
                "conditions": len(ct),
                "false_episodes": v_false,
                "false_per_asset_month": v_rate,
            },
            {
                "combination": "ML Model added to the Vendor Rules and Controller Health",
                "caught": int(caught_any.sum()),
                "conditions": len(ct),
                "false_episodes": n_false,
                "false_per_asset_month": rate,
            },
        ]
    )
    combined["line"] = [
        f"caught {r.caught} of {r.conditions} conditions at {r.false_per_asset_month:.2f} "
        "false episodes per asset-month"
        for r in combined.itertuples()
    ]
    store.write_pd(combined, "gold", "eval_combined")
    store.write_pd(inspection_workload(ml, vd, conds, wo, span, ctx, cfg), "gold", "eval_inspection_monthly")
    store.write_pd(alert_outcomes(S, conds, wo, span), "gold", "eval_alert_outcomes")
    out.update(
        {
            "matched": matched.to_dict("records"),
            "groups": groups.to_dict("records"),
            "combined": combined["line"].tolist(),
        }
    )
    return out


def inspection_workload(
    ml: pd.DataFrame,
    vd: pd.DataFrame,
    conds: pd.DataFrame,
    wo: pd.DataFrame,
    span: tuple[pd.Timestamp, pd.Timestamp],
    ctx: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Inspection hours on false ALERT episodes by month: the Vendor Rules' from the actual CMMS inspection
    labor on the asset during the episode; the ML Model's as false episodes times the median inspection
    labor (an estimate); each as a share of the technicians' available hours."""
    avail = len(cfg["cmms"]["technicians"]) * HOURS_PER_TECHNICIAN_YEAR / 12.0
    insp = wo[wo["wo_type"] == "inspection"]
    rows = []
    for method, lev in (("Vendor Rules", vd), ("ML Model", ml)):
        e = r2.alert_view(lev)
        e = e[(e["opened"] >= span[0]) & (e["opened"] < span[1])].reset_index(drop=True)
        if e.empty:
            continue
        f = e[r2.false_mask(e, conds).to_numpy()]
        for r in f.itertuples():
            if method == "Vendor Rules":
                w = insp[
                    insp["asset_id"].str.startswith(r.asset_id)
                    & (insp["opened"] >= r.opened)
                    & (insp["opened"] < r.closed)
                ]
                h, basis = float(w["labor_hours"].sum()), "actual CMMS inspection labor"
            else:
                h, basis = ctx["inspection_hours_each"], "estimate: median inspection labor"
            rows.append(
                {
                    "method": method,
                    "month": f"{r.opened.tz_convert(tasks.LOCAL_TZ):%Y-%m}",
                    "hours": h,
                    "basis": basis,
                }
            )
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    m = d.groupby(["method", "basis", "month"], as_index=False).agg(
        false_episodes=("hours", "size"), hours=("hours", "sum")
    )
    m["share_of_available"] = m["hours"] / avail
    m["available_hours"] = avail
    return m


def alert_outcomes(
    S: pd.DataFrame, conds: pd.DataFrame, wo: pd.DataFrame, span: tuple[pd.Timestamp, pd.Timestamp]
) -> pd.DataFrame:
    """Each shadow-year ALERT episode's outcome by method: condition confirmed (a confirmed condition on the
    asset while it was open); repair or PM reached the asset first for an unrelated reason (a repair,
    rebuild or PM work order on the asset while open, with no confirmed condition, as for ALARM); nothing
    found (an inspection while open closed with no fault found, or no work at all); pending at the record end
    (open at the end with nothing confirmed)."""
    rep = wo[wo["wo_type"].isin((*r2.REPAIRS, "pm", "rebuild_order"))]
    insp = wo[wo["wo_type"] == "inspection"]
    rows = []
    for r in S[(S["opened"] >= span[0]) & (S["opened"] < span[1])].itertuples():
        c = conds[
            (conds["asset_key"] == r.asset_id)
            & (conds["credit_ts"] >= r.opened)
            & (conds["credit_ts"] < r.closed)
        ]
        if len(c):
            out, ref = "condition confirmed", ",".join(c["wo_id"])
        else:
            w = rep[
                rep["asset_id"].str.startswith(r.asset_id)
                & (rep["opened"] >= r.opened)
                & (rep["opened"] < r.closed)
            ]
            i = insp[
                insp["asset_id"].str.startswith(r.asset_id)
                & (insp["opened"] >= r.opened)
                & (insp["opened"] < r.closed)
            ]
            if len(w):
                out, ref = (
                    "repair or PM reached the asset first for an unrelated reason",
                    ",".join(w["wo_id"]),
                )
            elif r.closed > span[1]:
                out, ref = "pending at the record end", ""
            else:
                out, ref = "nothing found", ",".join(i["wo_id"])
        rows.append(
            {
                "method": r.method,
                "asset_id": r.asset_id,
                "opened": r.opened,
                "closed": r.closed,
                "level": "ALARM" if pd.notna(r.alarm_ts) else "ALERT",
                "outcome": out,
                "work_orders": ref,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- monitoring

MONITORED = ("ML Model", "Vendor Rules", "Controller Health")


def status_changes(eps: pd.DataFrame) -> pd.DataFrame:
    """Per asset, the ML Model's level over time from its episodes: ALERT at the opening, ALARM at the
    ALARM start, OK at the close."""
    rows = []
    for r in eps.itertuples():
        rows.append((r.asset_id, r.opened, "ALERT"))
        if pd.notna(r.alarm_ts):
            rows.append((r.asset_id, r.alarm_ts, "ALARM"))
        rows.append((r.asset_id, r.closed, "OK"))
    out = pd.DataFrame(rows, columns=["asset_id", "ts", "status"])
    return out.sort_values(["asset_id", "ts"], kind="stable").reset_index(drop=True)


def _shadow_months(cfg: dict[str, Any]) -> list[str]:
    b = splits.boundaries(cfg)
    first = b["shadow"][0].tz_convert(tasks.LOCAL_TZ)
    last = (b["shadow"][1] - pd.Timedelta(hours=1)).tz_convert(tasks.LOCAL_TZ)
    return [str(d) for d in pd.period_range(first.strftime("%Y-%m"), last.strftime("%Y-%m"), freq="M")]


def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> list[str]:
    first, last = a.tz_convert(tasks.LOCAL_TZ), (b - pd.Timedelta(hours=1)).tz_convert(tasks.LOCAL_TZ)
    return [str(d) for d in pd.period_range(first.strftime("%Y-%m"), last.strftime("%Y-%m"), freq="M")]


def monitoring_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    """Monthly monitoring over the twelve shadow-year months, reported and not acted on: event performance
    and alert rates by method, drift of the indicators and the ML Model score against references from the
    last six months of training (re-centered after each repair or remount: the re-centering window plus
    the first scored days with no open episode on the point), data quality, and the retraining rule's
    verdict each month. Also gold.scoring_log for the inference profile, with its baseline from the
    reference window, and a refresh of the data profiles. Skipped while the shadow year is held out.
    (Revision 2; originally the shadow quarter, with references from the validation period.)"""
    from pdm.monitoring import performance as mon

    if cfg["evaluation"].get("holdout_shadow", True):
        return {"skipped": "shadow year held out"}
    b = splits.boundaries(cfg)
    end = b["shadow"][1]
    ref_a, ref_b = splits.reference_window(cfg)
    months = _shadow_months(cfg)
    window = cfg["evaluation"]["alert_precision_window_days"]
    catch_w = float(cfg["evaluation"]["catch"]["opened_within_days"])
    conds = tasks._conditions(store, cfg)
    ctx = tasks._context(store)
    pts = tasks._points(store)
    n_assets = pts["asset_id"].nunique()
    machines = store.read_pd("bronze", "reg_machines")["machine_id"].nunique()
    lev = _levels(store)
    ch = _ch_episodes(store, cfg)
    ct = tasks._utc_cols(store.read_pd("gold", "eval_conditions"), ["credit_ts"])
    caught = pd.concat(
        [
            pd.DataFrame(
                {
                    "method": "ML Model",
                    "wo_id": ct["wo_id"],
                    "caught": ct["ml_alert_caught"],
                    "warning_days": ct["ml_alert_warning_days"],
                }
            ),
            pd.DataFrame(
                {
                    "method": "Vendor Rules",
                    "wo_id": ct["wo_id"],
                    "caught": ct["vendor_alert_caught"],
                    "warning_days": ct["vendor_alert_warning_days"],
                }
            ),
            pd.DataFrame(
                {
                    "method": "Controller Health",
                    "wo_id": ct["wo_id"],
                    "caught": ct["ch_caught"],
                    "warning_days": ct["ch_warning_days"],
                }
            ),
        ],
        ignore_index=True,
    )
    S = lev[lev["period"] == "shadow"]
    eps = {
        "ML Model": S[S["method"] == "ml_model"],
        "Vendor Rules": S[S["method"] == "vendor_rules"],
        "Controller Health": ch[(ch["opened"] >= b["shadow"][0]) & (ch["opened"] < end)],
    }
    fa = pd.concat(
        [
            tasks.alert_rows(
                e[(e["opened"] >= b["shadow"][0]) & (e["opened"] < end)], conds, end, window
            ).assign(method=m)
            for m, e in eps.items()
        ],
        ignore_index=True,
    )
    fa["ts"] = pd.to_datetime(fa["ts"], utc=True)
    credit = conds.drop_duplicates("wo_id").set_index("wo_id")["credit_ts"]
    n_by = {"ML Model": n_assets, "Vendor Rules": n_assets, "Controller Health": machines}
    perf = mon.event_performance(caught, fa, credit, n_by)
    grid = pd.DataFrame([(m, mo) for m in MONITORED for mo in months], columns=["method", "month"])
    perf = grid.merge(perf[perf["month"].isin(months)], on=["method", "month"], how="left")
    for c in ("conditions", "alerts"):
        perf[c] = perf[c].fillna(0).astype(int)
    perf["alerts_per_asset_month"] = perf["alerts_per_asset_month"].fillna(0.0)
    store.write_pd(perf.assign(as_of=str(end)), "monitoring", "event_performance")

    # references: the last six months of training
    tr = lev[lev["period"] == "training"]
    ref_eps = {
        "ML Model": tr[tr["method"] == "ml_model"],
        "Vendor Rules": tr[tr["method"] == "vendor_rules"],
        "Controller Health": ch,
    }
    ref_conds = r2.conditions_in(conds, (ref_a, ref_b))
    reference = {}
    for m, e in ref_eps.items():
        n = machines if m == "Controller Health" else n_assets
        r, _ = r2.method_row(
            m,
            "ALERT",
            r2.alert_view(e) if m != "Controller Health" else e.assign(point_id=""),
            ref_conds,
            conds,
            (ref_a, ref_b),
            n,
            catch_w,
            ctx,
            cfg,
        )
        reference[m] = {
            "recall": r["caught_share"],
            "alerts_per_asset_month": r["episodes_per_asset_month"]
            * (n_assets if m != "Controller Health" else machines)
            / max(n, 1),
        }
    store.write_pd(
        pd.DataFrame(
            [
                {"method": m, **v, "reference_window": f"{ref_a:%Y-%m-%d} to {ref_b:%Y-%m-%d}"}
                for m, v in reference.items()
            ]
        ),
        "monitoring",
        "references",
    )

    # drift on the re-centered references
    feats = list(nbm.VECTOR_SOURCE)
    ind = store.read_pd("silver", "hourly_indicators", ["point_id", "hour_utc", "implausible", *feats])
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind = ind[(ind["hour_utc"] < end) & ~ind["implausible"].fillna(False).astype(bool)]
    base_months = _months_between(ref_a, ref_b)
    sc = store.read_pd(
        "gold",
        "nbm_scores",
        ["point_id", "asset_id", "hour_utc", "state", "score", "threshold", "model_version"],
    )
    trs = store.read_pd(
        "gold",
        "ml_train_scores",
        ["point_id", "asset_id", "hour_utc", "state", "score", "threshold", "model_version"],
    )
    allsc = pd.concat([trs, sc], ignore_index=True)
    allsc["hour_utc"] = _utc(allsc["hour_utc"])
    allsc = allsc[allsc["state"] == "scored"]
    segs = tasks._utc_cols(store.read_pd("gold", "ml_segments"), ["segment_start", "level_ready"])
    days = int(cfg["monitoring"]["drift"]["reference_scored_days"])
    asset_of = pts.set_index("point_id")["asset_id"].to_dict()
    rebase: dict[str, list[Any]] = {}
    ml_eps = eps["ML Model"]
    for pid, g in segs[(segs["segment_start"] >= ref_a) & segs["level_ready"].notna()].groupby("point_id"):
        e = ml_eps[ml_eps["asset_id"] == asset_of.get(pid)]
        intervals = list(zip(e["opened"], e["closed"].clip(upper=end), strict=True))
        wins = [(a, r, True) for a, r in zip(g["segment_start"], g["level_ready"], strict=True)]
        rebase[pid] = mon.drift_references(
            wins, allsc.loc[allsc["point_id"] == pid, "hour_utc"], intervals, end, days
        )
    sc_drift = allsc.rename(columns={"score": mon.SCORE_FEATURE})
    drift = pd.concat(
        [
            mon.feature_drift(ind, feats, base_months, months, rebase),
            mon.feature_drift(sc_drift, [mon.SCORE_FEATURE], base_months, months, rebase),
        ],
        ignore_index=True,
    ).assign(as_of=str(end))
    store.write_pd(drift, "monitoring", "feature_drift")
    decisions = pd.concat(
        [mon.retraining_decision(perf, drift, reference, cfg, m) for m in MONITORED], ignore_index=True
    )
    store.write_pd(decisions.assign(as_of=str(end)), "monitoring", "retraining_decision")
    recenter = segs[(segs["segment_start"] >= b["shadow"][0]) & ~segs["trained"].astype(bool)]
    store.write_pd(recenter, "monitoring", "recentering_events")

    log = tasks.scoring_log(
        allsc,
        status_changes(pd.concat([tr[tr["method"] == "ml_model"], ml_eps], ignore_index=True)),
        conds,
        "threshold",
        ref_a,
        end,
        window,
        cfg,
    )
    store.write_pd(log, "gold", "scoring_log")
    base = (log["hour_utc"] >= ref_a) & (log["hour_utc"] < ref_b)
    store.write_pd(log[base].reset_index(drop=True), "gold", "scoring_log_baseline")
    profiles = tasks._refresh_profiles(store)
    return {
        "months": len(months),
        "event_performance_rows": len(perf),
        "drift_rows": len(drift),
        "retrain_months": decisions.loc[decisions["retrain"].astype(bool), ["method", "month"]].to_dict(
            "records"
        ),
        "recentering_events": len(recenter),
        "scoring_log_rows": len(log),
        "profiles": profiles,
    }


def false_alarms_by_criterion(outcomes: pd.DataFrame) -> pd.DataFrame:
    """False ALARMs by the criterion that started them, per period, method and level, from the ALARM
    outcome table (vendor alarm-level rules grouped as one criterion)."""
    if outcomes.empty:
        return outcomes
    o = outcomes.copy()
    o["criterion"] = (
        o["cause"]
        .fillna("criterion")
        .astype(str)
        .str.replace(r"^vendor .*", "vendor alarm level", regex=True)
    )
    g = o.groupby(["period", "method", "level", "criterion"])
    return g.agg(alarms=("false_alarm", "size"), false_alarms=("false_alarm", "sum")).reset_index()
