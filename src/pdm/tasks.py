"""Job task bodies: train and score the per-point models and controller health, survival, the
supervised gate, prognosis, fusion, advisory, evaluation and monitoring.

Each task reads its inputs from the Store and writes its output tables back; job entry points under
jobs/ and the local parity runner call these functions. MLflow logging is optional (tracking off in
tests). While the shadow quarter is held out, no task writes anything computed on it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from pdm import log, splits
from pdm.advisory import requests as adv
from pdm.evaluation import events as ev
from pdm.features.controller import compressor_hourly
from pdm.fusion.status import cases as make_cases
from pdm.fusion.status import status_changes
from pdm.models import controller_health as ch
from pdm.models import nbm, prognosis, supervised, survival
from pdm.rules.floor import ISO_KEY
from pdm.rules.iso_zones import zone_boundaries
from pdm.store import Store

LOG = log.get_logger("pdm.tasks")
LOCAL_TZ = "America/Chicago"
DAY = pd.Timedelta(days=1)
INSTRUMENTED = r"^(HMC-\d+-(SP|TSC|CP)|CMP-\d+)$"
INSPECTION_OR_REPAIR = ("inspection", "unplanned_repair", "condition_based", "corrective")


def _utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def _utc_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = _utc(df[c])
    return df


def model_version(name: str, payload: Any) -> str:
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{name}-{h}"


def _mlflow_log(
    experiment: str | None, params: dict[str, Any], metrics: dict[str, float], tags: dict[str, str]
) -> str | None:
    if experiment is None:
        return None
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=tags.get("name")) as run:
        mlflow.log_params(
            {k: (v if isinstance(v, (int, float)) else str(v)[:500]) for k, v in params.items()}
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})
        mlflow.set_tags(tags)
        return run.info.run_id


def horizon(cfg: dict[str, Any]) -> pd.Timestamp:
    """End of the data any task may use: the shadow start while the shadow quarter is held out."""
    b = splits.boundaries(cfg)
    return b["shadow"][0] if cfg["evaluation"].get("holdout_shadow", True) else b["shadow"][1]


def _wo(store: Store) -> pd.DataFrame:
    return _utc_cols(store.read_pd("bronze", "cmms_work_orders"), ["opened", "started", "closed"])


def _points(store: Store) -> pd.DataFrame:
    return store.read_pd("silver", "points", ["point_id", "asset_id", "machine_id", "point_class"])


def _points_by_asset(pts: pd.DataFrame) -> dict[str, list[str]]:
    return pts.groupby("asset_id")["point_id"].apply(sorted).to_dict()


def _conditions(store: Store, cfg: dict[str, Any]) -> pd.DataFrame:
    c = _utc_cols(store.read_pd("silver", "confirmed_conditions"), ["confirmed_ts", "closed"])
    wo = _wo(store).set_index("wo_id")
    pm_started = c["wo_id"].map(wo["started"])
    c["credit_ts"] = (
        c["confirmed_ts"].where(~c["pm_finding"].astype(bool), pm_started).fillna(c["confirmed_ts"])
    )
    mclass = store.read_pd("bronze", "reg_machines").set_index("machine_id")["class"].to_dict()
    c["machine_class"] = c["machine_id"].map(mclass).fillna("compressor")
    c["instrumented"] = c["asset_id"].str.match(INSTRUMENTED)
    return ev.condition_scope(c, _points_by_asset(_points(store)))


# ---------------------------------------------------------------- per-point models


def _excluded(
    asset: str,
    conds: pd.DataFrame,
    wo: pd.DataFrame,
    rebase: list[pd.Timestamp],
    flagged: pd.DataFrame,
    cfg: dict[str, Any],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    nb = cfg["models"]["nbm"]
    before = pd.Timedelta(days=nb["exclude_before_condition_days"])
    rb = pd.Timedelta(days=nb["rebaseline_days"])
    out = [(t - before, t + DAY) for t in conds.loc[conds["asset_key"] == asset, "credit_ts"]]
    w = wo[wo["asset_id"].str.startswith(asset) & wo["wo_type"].isin(INSPECTION_OR_REPAIR)]
    out += [(a, b if pd.notna(b) else a + DAY) for a, b in zip(w["opened"], w["closed"], strict=True)]
    out += [(s, s + rb) for s in rebase]
    out += list(zip(flagged["start"], flagged["end"], strict=True))
    return out


SIMPLICITY = ("pca_t2_spe", "mahalanobis", "isolation_forest", "autoencoder_third_octave")


def select_champion(sel: pd.DataFrame, n_conditions: int, margin_conditions: int) -> str:
    """Highest skill (catch share minus chance); among candidates within margin_conditions of it, the
    simplest."""
    best = sel["skill"].max()
    close = sel[(best - sel["skill"]) * n_conditions <= margin_conditions + 1e-9]
    return min(close["candidate"], key=SIMPLICITY.index)


def _segment_stats(clean: pd.DataFrame, plan: pd.DataFrame, keep: np.ndarray, min_rows: int) -> pd.DataFrame:
    """Healthy spread per segment: the base model's training hours in the install segment, and each
    refit segment's re-baseline window."""
    t = clean["hour_utc"]
    parts = []
    for k, w in plan.iterrows():
        if k == 0:
            seg_end = plan["start"].iat[1] if len(plan) > 1 else t.max() + DAY
            rows = clean[keep & (t < seg_end).to_numpy()]
            rows = rows if len(rows) >= min_rows else clean[keep]
        elif w["refit"]:
            rows = clean[((t >= w["start"]) & (t < w["window_end"])).to_numpy()]
        else:
            continue
        st = adv.healthy_stats(rows)
        if len(st):
            parts.append(st.assign(segment=int(k), segment_start=w["start"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def nbm_task(store: Store, cfg: dict[str, Any], experiment: str | None = None) -> dict[str, Any]:
    nb = cfg["models"]["nbm"]
    end = horizon(cfg)
    cutoff = splits.training_cutoff(cfg)
    rb = pd.Timedelta(days=nb["rebaseline_days"])
    ind = store.read_pd("silver", "hourly_indicators")
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind = ind[ind["hour_utc"] < end]
    win = _utc_cols(store.read_pd("silver", "healthy_windows"), ["start", "end"])
    win = win[win["start"] < end]
    sanity = store.read_pd("gold", "baseline_sanity")
    flagged_ids = set(sanity.loc[sanity["flagged"].astype(bool), "window_id"])
    conds = _conditions(store, cfg)
    wo = _wo(store)
    rf = _episodes_from(store.read_pd("gold", "rule_floor_exceedances"), _points(store), cfg)
    candidates = list(nb["candidates"])
    scores: dict[str, list[pd.DataFrame]] = {c: [] for c in candidates}
    seg_rows, healthy = [], []
    for pid, rows in ind.groupby("point_id"):
        rows = rows.sort_values("hour_utc").reset_index(drop=True)
        asset = rows["asset_id"].iat[0]
        plan = nbm.segment_plan(win[win["point_id"] == pid], flagged_ids, end, rb)
        rebase = list(plan.loc[plan["reason"] != "install", "start"])
        excl = _excluded(asset, conds, wo, rebase, plan[plan["flagged"]], cfg)
        for cand in candidates:
            s, infos = nbm.fit_score_rebaselined(
                rows, cand, cfg, cutoff, excl, plan, end, nb["top_contributors"]
            )
            if len(s):
                scores[cand].append(s.assign(asset_id=asset))
            if cand == candidates[0]:
                seg_rows += [{**i, "asset_id": asset, "excluded_intervals": len(excl)} for i in infos]
                clean = rows[~rows["implausible"].fillna(False).astype(bool)].dropna(subset=["vel_rms_mm_s"])
                clean = clean.reset_index(drop=True)
                keep = nbm.training_mask(clean["hour_utc"], cutoff, excl)
                healthy.append(_segment_stats(clean, plan, keep, nb["min_train_hours"]))
    pts = _points(store)
    train_conds = conds[(conds["credit_ts"] < cutoff) & conds["instrumented"]]
    period = (splits.boundaries(cfg)["train"][0], cutoff)
    n_assets = pts["asset_id"].nunique()
    vend_op = cfg["rules"]["vendor"]["alert_frac_zone_b_width"]
    vend = rf[(rf["method"] == "rule_floor") & (rf["sweep"] == vend_op)]
    budget, _ = _summary("rule_floor", vend, train_conds, conds, period, n_assets, cfg, store)
    sel_rows, exceed = [], {}
    gap = cfg["evaluation"]["catch"]["episode_gap_days"]
    for cand in candidates:
        s = pd.concat(scores[cand], ignore_index=True) if scores[cand] else pd.DataFrame()
        curve = []
        for p in nb["sweep_percentiles"]:
            ex = nbm.exceedance_hours(s, p, nb["persistence_hours"]) if len(s) else pd.DataFrame()
            ex = ex.merge(pts[["point_id", "asset_id"]], on="point_id") if len(ex) else ex
            exceed[(cand, p)] = ex
            ep = (
                ev.episodes(ex[["asset_id", "point_id", "ts"]], gap)
                if len(ex)
                else ev.episodes(pd.DataFrame(), 0)
            )
            row, _ = _summary("nbm", ep, train_conds, conds, period, n_assets, cfg, store)
            curve.append({**row, "level": p})
        c = pd.DataFrame(curve)
        at = ev.at_budget(c, budget["false_per_asset_month"])
        op = c[c["level"] == nb["threshold_percentile"]].iloc[0]
        inside = bool(at["within_curve"])
        caught = at["caught_share"] if inside else float(op["caught_share"])
        chance = at["chance_share"] if inside else float(op["chance_share"])
        sel_rows.append(
            {
                "candidate": cand,
                "basis": "vendor training budget" if inside else "operating point",
                "training_caught": caught,
                "training_chance": chance,
                "skill": caught - chance,
                "operating_false_per_asset_month": float(op["false_per_asset_month"]),
                "curve_false_min": float(c["false_per_asset_month"].min()),
                "curve_false_max": float(c["false_per_asset_month"].max()),
                "vendor_training_false_per_asset_month": budget["false_per_asset_month"],
                "training_conditions": len(train_conds),
            }
        )
    sel = pd.DataFrame(sel_rows).sort_values("skill", ascending=False, kind="stable").reset_index(drop=True)
    champion = select_champion(sel, len(train_conds), nb["champion_margin_conditions"])
    info = pd.DataFrame(seg_rows)
    version = model_version("nbm", {"champion": champion, "info": info.to_dict("records")})
    sc = pd.concat(scores[champion], ignore_index=True)
    ex_all = pd.concat(
        [
            exceed[(champion, p)].assign(sweep=p)
            for p in nb["sweep_percentiles"]
            if len(exceed[(champion, p)])
        ],
        ignore_index=True,
    )
    run = _mlflow_log(
        experiment,
        {"champion": champion, "segments_refit": int((info["model"] == "refit").sum())},
        {f"training_skill_{r.candidate}": r.skill for r in sel.itertuples()},
        {"name": "nbm", "model_version": version},
    )
    hs = pd.concat(healthy, ignore_index=True)
    store.write_pd(sc.assign(model_version=version, candidate=champion), "gold", "nbm_scores")
    store.write_pd(
        ex_all.assign(method="nbm", level="alarm", model_version=version), "gold", "nbm_exceedances"
    )
    store.write_pd(
        sel.assign(champion=sel["candidate"] == champion, model_version=version), "gold", "nbm_candidates"
    )
    store.write_pd(info.astype({"models": str}).assign(model_version=version), "gold", "nbm_models")
    store.write_pd(hs.assign(model_version=version), "gold", "point_healthy_stats")
    b = splits.boundaries(cfg)
    periods = {"training": (b["train"][0], cutoff), "validation": (b["validation"][0], end)}
    diag = nbm.exceedance_diagnosis(sc, win, nb["threshold_percentile"], periods, cfg)
    store.write_pd(diag.assign(model_version=version), "gold", "nbm_exceedance_diagnosis")
    return {
        "champion": champion,
        "selection": sel,
        "version": version,
        "mlflow_run": run,
        "segments": info["model"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------- controller health
def controller_health_task(
    store: Store, cfg: dict[str, Any], experiment: str | None = None
) -> dict[str, Any]:
    h = store.read_pd("silver", "controller_hourly")
    h["hour_utc"] = _utc(h["hour_utc"])
    h = h[h["hour_utc"] < horizon(cfg)]
    rated = (
        store.read_pd("bronze", "reg_machines").set_index("machine_id")["rated_rpm"].astype(float).to_dict()
    )
    train_end = splits.training_cutoff(cfg).tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    scores, models = [], {}
    for mid, g in h.groupby("machine_id"):
        s, model = ch.daily_scores(g.sort_values("hour_utc").reset_index(drop=True), rated[mid], train_end)
        scores.append(s)
        models[mid] = model.coef.tolist()
    sc = pd.concat(scores, ignore_index=True)
    ex = pd.concat(
        [ch.exceedances(sc, cfg, k) for k in cfg["models"]["controller_health"]["sweep_scale"]],
        ignore_index=True,
    )
    version = model_version("controller_health", models)
    _mlflow_log(
        experiment,
        {"machines": len(models)},
        {"exceedance_days_scale_1": float((ex["sweep"] == 1.0).sum())},
        {"name": "controller_health", "model_version": version},
    )
    store.write_pd(sc.assign(model_version=version), "gold", "controller_health_daily")
    store.write_pd(ex.assign(model_version=version), "gold", "controller_health_exceedances")
    end_naive = horizon(cfg).tz_convert("UTC").tz_localize(None)
    cmp_h = compressor_hourly(store.read("bronze", "cmp_minute"), store.read("bronze", "cmp_room"))
    cmp_h = cmp_h.where(cmp_h.hour_utc < end_naive).toPandas()
    cmp_h["hour_utc"] = _utc(cmp_h["hour_utc"])
    dd = ch.discharge_daily(
        cmp_h, splits.training_cutoff(cfg), cfg["models"]["controller_health"]["discharge_min_load_minutes"]
    )
    store.write_pd(dd.assign(model_version=version), "gold", "compressor_discharge_daily")
    return {
        "exceedance_days": ex.groupby("sweep").size().to_dict(),
        "version": version,
        "machines": len(models),
    }


def _controller_signals(ex: pd.DataFrame, hmc: set[str]) -> pd.DataFrame:
    ex = ex.reset_index(drop=True)
    ts = pd.to_datetime(ex["alert_date"]).dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC") + pd.Timedelta(
        hours=23
    )
    asset = [
        f"{m}-SP" if (r == "thermistor_residual" and m in hmc) else m
        for m, r in zip(ex["machine_id"], ex["rule"], strict=True)
    ]
    return pd.DataFrame(
        {
            "method": "controller",
            "sweep": ex["sweep"].to_numpy(),
            "asset_id": asset,
            "machine_id": ex["machine_id"].to_numpy(),
            "point_id": None,
            "ts": ts,
            "level": "alarm",
            "rules": ex["rule"].to_numpy(),
        }
    )


# ---------------------------------------------------------------- survival, supervised gate, prognosis
def survival_task(store: Store, cfg: dict[str, Any], experiment: str | None = None) -> dict[str, Any]:
    b = splits.boundaries(cfg)
    lives = survival.lifetimes(
        store.read_pd("bronze", "reg_machines"),
        store.read_pd("bronze", "reg_pumps"),
        store.read_pd("bronze", "reg_hmc_spindles"),
        store.read_pd("bronze", "reg_compressors"),
        _wo(store),
        _utc_cols(store.read_pd("bronze", "ctrl_events").query("event_type == 'collision'"), ["ts_utc"]),
        _utc_cols(store.read_pd("bronze", "cmms_pm_schedule"), ["due", "completed"]),
        (b["train"][0], horizon(cfg)),
    )
    res = survival.fit_models(lives)
    summary = pd.DataFrame(
        [
            {"metric": "lives", "value": res["n"]},
            {"metric": "events", "value": res["events"]},
            {"metric": "concordance_weibull_aft", "value": res["concordance_aft"]},
            {"metric": "concordance_cox", "value": res["concordance_cox"]},
            {"metric": "integrated_brier_ipcw_weibull_aft", "value": res["ibs_ipcw"]},
            {"metric": "integrated_brier_ipcw_kaplan_meier_reference", "value": res["ibs_ipcw_km"]},
            {"metric": "ibs_time_low_days", "value": res["ibs_times"][0]},
            {"metric": "ibs_time_high_days", "value": res["ibs_times"][1]},
        ]
        + [
            {"metric": f"median_life_days_{k}", "value": v}
            for k, v in res["median_life_days_by_class"].items()
        ]
    )
    version = model_version("component_survival", summary.to_dict("records"))
    _mlflow_log(
        experiment,
        {"lives": res["n"]},
        {
            "concordance_aft": res["concordance_aft"],
            "concordance_cox": res["concordance_cox"],
            "ibs_ipcw": res["ibs_ipcw"],
        },
        {"name": "survival", "model_version": version},
    )
    store.write_pd(lives.assign(model_version=version), "gold", "component_lives")
    store.write_pd(summary.assign(model_version=version), "gold", "survival_summary")
    store.write_pd(
        res["aft"]
        .summary.reset_index()
        .astype({"param": str, "covariate": str})
        .assign(model_version=version),
        "gold",
        "survival_aft_coefficients",
    )
    classes = survival.per_class_weibull(lives, cfg["models"]["survival"]["interval"])
    store.write_pd(classes.assign(model_version=version), "gold", "survival_class_weibull")
    return {
        "class_weibull": classes,
        "covariate_models_supported": bool(res["ibs_ipcw"] < res["ibs_ipcw_km"]),
        **{
            k: res[k]
            for k in ("n", "events", "concordance_aft", "concordance_cox", "ibs_ipcw", "ibs_ipcw_km")
        },
    }


def supervised_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    conds = _conditions(store, cfg)
    gate = supervised.training_gate(conds[conds["unplanned"]], splits.training_cutoff(cfg), cfg)
    store.write_pd(pd.DataFrame([gate]), "gold", "supervised_gate")
    return gate


def prognosis_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["models"]["prognosis"]
    b = splits.boundaries(cfg)
    cutoff = splits.training_cutoff(cfg)
    zones = {
        k: zone_boundaries(v["group"], v["foundation"])
        for k, v in cfg["rules"]["iso_class_by_point_class"].items()
    }
    ind = store.read_pd(
        "silver",
        "hourly_indicators",
        ["point_id", "asset_id", "point_class", "hour_utc", "vel_max_dir_mm_s", "implausible"],
    )
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind = ind[~ind["implausible"].fillna(False).astype(bool) & ~ind["point_class"].str.startswith("spindle")]
    ind = ind[ind["hour_utc"] < horizon(cfg)]
    rf = _episodes_from(store.read_pd("gold", "rule_floor_exceedances"), _points(store), cfg)
    rf = rf[
        (rf["method"] == "rule_floor") & (rf["sweep"] == cfg["rules"]["vendor"]["alert_frac_zone_b_width"])
    ]
    conds = _conditions(store, cfg)
    fails = conds[
        conds["unplanned"]
        & conds["condition_type"].str.startswith("bearing")
        & conds["instrumented"]
        & ~conds["asset_id"].str.contains("-SP")
        & (conds["credit_ts"] < horizon(cfg))
    ]
    window = pd.Timedelta(days=cfg["evaluation"]["catch"]["opened_within_days"])
    insts, skipped = [], []
    for f in fails.itertuples():
        m = (
            ev._scope_mask(rf, f.asset_key, f.point_scope)
            & (rf["opened"] <= f.credit_ts)
            & (rf["closed"] > f.credit_ts)
            & (rf["opened"] >= f.credit_ts - window)
        )
        hit = rf[m].sort_values("opened")
        if hit.empty:
            skipped.append(
                {"wo_id": f.wo_id, "asset_id": f.asset_id, "reason": "no rule-floor episode open at failure"}
            )
            continue
        det = hit["opened"].iat[0]
        pid = hit["point_id"].iat[0] or (f.point_scope or "")
        g = ind[(ind["point_id"] == pid)]
        if g.empty:
            skipped.append({"wo_id": f.wo_id, "asset_id": f.asset_id, "reason": "no velocity rows"})
            continue
        before = g[(g["hour_utc"] < det) & (g["hour_utc"] >= det - pd.Timedelta(days=14))]["vel_max_dir_mm_s"]
        if before.empty:
            skipped.append(
                {"wo_id": f.wo_id, "asset_id": f.asset_id, "reason": "no velocity rows before detection"}
            )
            continue
        phi = float(before.median())
        daily = (
            g[(g["hour_utc"] >= det) & (g["hour_utc"] < f.credit_ts)]
            .set_index("hour_utc")["vel_max_dir_mm_s"]
            .resample("1D")
            .median()
            .dropna()
        )
        thr = zones[ISO_KEY[g["point_class"].iat[0]]][2]
        period = (
            "training"
            if f.credit_ts < cutoff
            else ("validation" if f.credit_ts >= b["validation"][0] else "gap")
        )
        insts.append(
            (period, f.wo_id, prognosis.Instance(f.asset_id, det.floor("D"), f.credit_ts, daily, phi, thr))
        )
    train = [i for per, _, i in insts if per == "training"]
    preds = []
    if len(train) >= 2:
        for per, wo_id, i in insts:
            others = [t for t in train if t is not i]
            prior = prognosis.prior_from_training(others)
            preds.append(
                prognosis.predict(
                    i, prior, [0.25, 0.5, 0.75], p["interval"], 2000, cfg["models"]["seed"]
                ).assign(
                    period=per,
                    wo_id=wo_id,
                    prior="leave-one-out" if per == "training" else "training",
                    threshold_mm_s=i.threshold,
                    phi_mm_s=i.phi,
                )
            )
    pred = prognosis.scores(pd.concat(preds, ignore_index=True), p["alpha"]) if preds else pd.DataFrame()
    store.write_pd(pred if len(pred) else pd.DataFrame({"note": ["no instances"]}), "gold", "prognosis")
    store.write_pd(
        pd.DataFrame(skipped) if skipped else pd.DataFrame({"note": ["none"]}), "gold", "prognosis_skipped"
    )
    out = {"instances": len(insts), "training_instances": len(train), "skipped": len(skipped)}
    for per in ("training", "validation"):
        v = pred[pred["period"] == per] if len(pred) else pred
        out[per] = {
            "predictions": len(v),
            "alpha_lambda_share": float(v["within_alpha"].mean()) if len(v) else np.nan,
            "interval_coverage": float(v["covered"].mean()) if len(v) else np.nan,
            "interval_width_median_days": float(
                v.loc[np.isfinite(v["interval_width_days"]), "interval_width_days"].median()
            )
            if len(v)
            else np.nan,
        }
    return out


# ---------------------------------------------------------------- episodes, fusion, advisory
def _episodes_from(ex: pd.DataFrame, pts: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Alert episodes per method and sweep from persisted exceedances; the rule floor joins the vendor
    rules at each offset with the ISO zone rule."""
    gap = cfg["evaluation"]["catch"]["episode_gap_days"]
    ex = ex.copy()
    ex["ts"] = _utc(ex["ts"])
    ex = ex[ex["ts"] < horizon(cfg)]
    if "asset_id" not in ex:
        ex = ex.merge(pts[["point_id", "asset_id"]], on="point_id", how="left")
    parts = []
    for (m, s), g in ex.groupby(["method", "sweep"]):
        parts.append(ev.episodes(g[["asset_id", "point_id", "ts"]], gap).assign(method=m, sweep=s))
    iso = ex[ex["method"] == "iso"]
    for s, g in ex[ex["method"] == "vendor"].groupby("sweep"):
        both = pd.concat([g, iso], ignore_index=True)
        parts.append(
            ev.episodes(both[["asset_id", "point_id", "ts"]], gap).assign(method="rule_floor", sweep=s)
        )
    return pd.concat(parts, ignore_index=True)


def _daily_first(sig: pd.DataFrame) -> pd.DataFrame:
    d = sig.assign(day=sig["ts"].dt.floor("D"), point_key=sig["point_id"].fillna(""))
    return (
        d.sort_values("ts")
        .drop_duplicates(["method", "asset_id", "point_key", "level", "day"])
        .drop(columns=["day", "point_key"])
    )


def signals(store: Store, cfg: dict[str, Any], nbm_level: float, controller_scale: float) -> pd.DataFrame:
    """Unified signals for fusion at one operating level: rule floor at the vendor setting, per-point
    model exceedances at the given percentile, controller exceedances at the given scale (first
    signal per day per source)."""
    pts = _points(store)
    rf = store.read_pd("gold", "rule_floor_exceedances")
    vend = cfg["rules"]["vendor"]["alert_frac_zone_b_width"]
    rf = rf[(rf["method"] == "iso") | ((rf["method"] == "vendor") & (rf["sweep"] == vend))]
    rf = rf.merge(pts, on="point_id", how="left").assign(rules=lambda d: d["method"] + ":" + d["level"])
    na = store.read_pd("gold", "nbm_exceedances")
    na = (
        na[na["sweep"] == nbm_level]
        .drop(columns=["asset_id"], errors="ignore")
        .merge(pts, on="point_id", how="left")
    )
    na = na.assign(rules="nbm:" + na["top_contributors"].astype(str))
    hmc = set(store.read_pd("bronze", "reg_machines").query("`class` == 'hmc'")["machine_id"])
    ce = store.read_pd("gold", "controller_health_exceedances")
    cs = _controller_signals(ce[ce["sweep"] == controller_scale], hmc)
    cols = ["method", "asset_id", "machine_id", "point_id", "ts", "level", "rules"]
    parts = [rf[cols], na[cols], cs[cols]]
    for part in parts:
        part["ts"] = _utc(part["ts"])
    out = pd.concat(parts, ignore_index=True)
    out = out[out["ts"] < horizon(cfg)]
    return _daily_first(out).sort_values("ts").reset_index(drop=True)


def _fused_cases(
    sig: pd.DataFrame, wo: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = horizon(cfg)
    changes = status_changes(sig, cfg)
    cs = make_cases(changes, sig, wo[wo["closed"] < end], cfg["models"]["fusion"]["case_quiet_close_days"])
    if len(cs):
        cs["opened"] = _utc(cs["opened"])
        cs["closed"] = _utc(cs["closed"])
    return changes, cs


def fusion_advisory_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    nb = cfg["models"]["nbm"]
    op_scale = 1.0
    wo = _wo(store)
    all_cases, out = [], {}
    for p in nb["sweep_percentiles"]:
        sig = signals(store, cfg, p, op_scale)
        changes, cs = _fused_cases(sig, wo, cfg)
        all_cases.append(cs.assign(sweep=p))
        out[f"cases_{p:g}"] = len(cs)
        if p == nb["threshold_percentile"]:
            store.write_pd(sig, "gold", "signals")
            store.write_pd(changes, "gold", "asset_status")
            op_cases = cs
    store.write_pd(pd.concat(all_cases, ignore_index=True), "gold", "cases_sweep")
    store.write_pd(op_cases, "gold", "cases")
    wr = _work_requests(store, op_cases, cfg)
    store.write_pd(wr, "gold", "work_requests")
    out["work_requests"] = len(wr)
    return out


def _controller_daily_z(store: Store, cfg: dict[str, Any]) -> pd.DataFrame:
    end = horizon(cfg)
    train_end = splits.training_cutoff(cfg).tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    d = store.read_pd("gold", "controller_health_daily")
    parts = [
        pd.DataFrame(
            {"entity": d["machine_id"], "local_date": d["local_date"], "indicator": k, "value": d[c]}
        )
        for k, c in (("thermistor", "thermistor_residual_c"), ("axis_drift", "axis_drift_max"))
        if c in d
    ]
    chh = store.read_pd("silver", "controller_hourly", ["machine_id", "hour_utc", "coolant_pressure_bar"])
    chh["hour_utc"] = _utc(chh["hour_utc"])
    chh = chh[chh["hour_utc"] < end]
    a = cfg["advisory"]["tsc_pressure_reference_days"]
    parts.append(adv.tsc_drop_daily(chh, (a[0], a[1])))
    cd = store.read_pd("gold", "compressor_discharge_daily")
    parts.append(
        pd.DataFrame(
            {
                "entity": cd["compressor_id"],
                "local_date": cd["local_date"],
                "indicator": "discharge",
                "value": cd["discharge_residual_c"],
            }
        )
    )
    return adv.controller_z(pd.concat(parts, ignore_index=True), train_end)


def _work_requests(store: Store, cs: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    end = horizon(cfg)
    stats = _utc_cols(store.read_pd("gold", "point_healthy_stats"), ["segment_start"])
    cols = ["point_id", "hour_utc", "regime", "implausible", *adv.Z_INDICATORS.values()]
    cols += [f"ord_{o:02d}_g2" for o in range(3, 11)]
    ind = store.read_pd("silver", "hourly_indicators", sorted(set(cols) - {"ord_3_10_g2"}))
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind = ind[~ind["implausible"].fillna(False).astype(bool) & (ind["hour_utc"] < end)]
    vib = adv.daily_point_z(adv.hourly_z(ind, stats))
    ctrl = _controller_daily_z(store, cfg)
    pts_by_asset = _points_by_asset(_points(store))
    zs = {
        c["case_id"]: adv.case_z(c, vib, pts_by_asset.get(c["asset_id"], []), ctrl, end)
        for c in cs.to_dict("records")
    }
    b = splits.boundaries(cfg)
    return adv.work_requests(cs, zs, b["shadow"], cfg["evaluation"]["advisory_z"])


# ---------------------------------------------------------------- evaluation
def _context(store: Store) -> dict[str, Any]:
    wo = _wo(store)
    insp = wo[(wo["wo_type"] == "inspection")]
    margin = (
        store.read_pd("bronze", "erp_contribution_margin")
        .set_index("machine_class")["contribution_margin_per_hour_usd"]
        .to_dict()
    )
    return {"inspection_hours_each": float(insp["labor_hours"].median()), "margin": margin}


_CTX: dict[int, dict[str, Any]] = {}


def _summary(
    method: str,
    ep: pd.DataFrame,
    conds: pd.DataFrame,
    conds_all: pd.DataFrame,
    period: tuple[pd.Timestamp, pd.Timestamp],
    n_assets: int,
    cfg: dict[str, Any],
    store: Store,
    level: Any = "",
) -> tuple[dict, pd.DataFrame]:
    ctx = _CTX.setdefault(id(store), _context(store))
    return ev.summarize(
        method,
        str(level),
        ep,
        conds,
        conds_all,
        period,
        n_assets,
        cfg["evaluation"]["catch"]["opened_within_days"],
        ctx["inspection_hours_each"],
        ctx["margin"],
        cfg["maintenance"]["planned_repair_lead_days"][0],
    )


def evaluation_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    """Validation-period evaluation: operating curves with chance columns per method, the comparison
    at the vendor rules' operating point, alerts per asset-month and diagnosis agreement."""
    b = splits.boundaries(cfg)
    period = b["validation"]
    pts = _points(store)
    conds = _conditions(store, cfg)
    conds_p = conds[(conds["credit_ts"] >= period[0]) & (conds["credit_ts"] < period[1])]
    sensor = conds_p[conds_p["instrumented"]]
    no_sensor = conds_p[~conds_p["instrumented"]]
    n_instr = pts["asset_id"].nunique()
    n_mach = store.read_pd("bronze", "reg_machines")["machine_id"].nunique()
    vend_op = cfg["rules"]["vendor"]["alert_frac_zone_b_width"]
    nb = cfg["models"]["nbm"]
    rf = _episodes_from(store.read_pd("gold", "rule_floor_exceedances"), pts, cfg)
    nx = store.read_pd("gold", "nbm_exceedances")
    nx["ts"] = _utc(nx["ts"])
    nbm_ep = pd.concat(
        [
            ev.episodes(
                g[["asset_id", "point_id", "ts"]], cfg["evaluation"]["catch"]["episode_gap_days"]
            ).assign(sweep=p)
            for p, g in nx.groupby("sweep")
        ],
        ignore_index=True,
    )
    hmc = set(store.read_pd("bronze", "reg_machines").query("`class` == 'hmc'")["machine_id"])
    cx = _controller_signals(store.read_pd("gold", "controller_health_exceedances"), hmc)
    cx["ts"] = _utc(cx["ts"])
    ctrl_ep = pd.concat(
        [
            ev.episodes(
                g[["asset_id", "point_id", "ts"]], cfg["evaluation"]["catch"]["episode_gap_days"]
            ).assign(sweep=s)
            for s, g in cx.groupby("sweep")
        ],
        ignore_index=True,
    )
    cs = store.read_pd("gold", "cases_sweep")
    cs = _utc_cols(cs, ["opened", "closed"])
    cs["closed"] = cs["closed"].fillna(horizon(cfg))
    case_ep = cs.assign(point_id=cs["trigger_points"].fillna(""), last=cs["opened"])
    instr_assets = set(pts["asset_id"])

    specs = []
    for s in sorted(rf.loc[rf["method"] == "vendor", "sweep"].unique()):
        specs.append(("vendor", s, rf[(rf["method"] == "vendor") & (rf["sweep"] == s)], sensor, n_instr))
        specs.append(
            ("rule_floor", s, rf[(rf["method"] == "rule_floor") & (rf["sweep"] == s)], sensor, n_instr)
        )
    specs.append(("iso", 0.0, rf[rf["method"] == "iso"], sensor, n_instr))
    for s, g in nbm_ep.groupby("sweep"):
        specs.append(("nbm", s, g, sensor, n_instr))
    for s, g in ctrl_ep.groupby("sweep"):
        specs.append(("controller", s, g, conds_p, n_mach))
        specs.append(
            ("controller_uninstrumented", s, g[~g["asset_id"].isin(instr_assets)], no_sensor, n_mach)
        )
    for s, g in case_ep.groupby("sweep"):
        specs.append(("model_layer", s, g, conds_p, n_instr + n_mach))
        specs.append(("model_layer_instrumented", s, g[g["asset_id"].isin(instr_assets)], sensor, n_instr))
        specs.append(
            ("model_layer_uninstrumented", s, g[~g["asset_id"].isin(instr_assets)], no_sensor, n_mach)
        )
    rows, caught = [], []
    for method, s, ep, cnd, n in specs:
        r, c = _summary(method, ep, cnd, conds, period, n, cfg, store, s)
        rows.append({**r, "sweep": float(s), "n_assets": n})
        caught.append(c.assign(sweep=float(s)))
    curves = pd.DataFrame(rows)
    caught = pd.concat(caught, ignore_index=True)
    operating = {
        "vendor": vend_op,
        "rule_floor": vend_op,
        "iso": 0.0,
        "nbm": nb["threshold_percentile"],
        "controller": 1.0,
        "controller_uninstrumented": 1.0,
        "model_layer": nb["threshold_percentile"],
        "model_layer_instrumented": nb["threshold_percentile"],
        "model_layer_uninstrumented": nb["threshold_percentile"],
    }
    curves["operating_point"] = [
        operating[m] == s for m, s in zip(curves["method"], curves["sweep"], strict=True)
    ]
    budget = float(
        curves.loc[
            (curves["method"] == "rule_floor") & curves["operating_point"], "false_per_asset_month"
        ].iat[0]
    )
    matched = []
    for m in ("vendor", "nbm", "controller", "model_layer", "model_layer_instrumented"):
        c = curves[curves["method"] == m]
        at = ev.at_budget(c, budget)
        near = c.iloc[(c["false_per_asset_month"] - budget).abs().argsort().iat[0]]
        matched.append(
            {
                "method": m,
                "budget_false_per_asset_month": budget,
                **{f"interp_{k}": v for k, v in at.items()},
                "nearest_sweep": near["sweep"],
                "nearest_false_per_asset_month": near["false_per_asset_month"],
                "nearest_caught_share": near["caught_share"],
                "nearest_chance_share": near["chance_share"],
            }
        )
    matched = pd.DataFrame(matched)
    near_instr = float(matched.loc[matched["method"] == "model_layer_instrumented", "nearest_sweep"].iat[0])
    head_rows = [
        ("rule_floor", vend_op),
        ("model_layer_instrumented", nb["threshold_percentile"]),
        ("model_layer_instrumented", near_instr),
        ("controller_uninstrumented", 1.0),
        ("model_layer_uninstrumented", nb["threshold_percentile"]),
    ]
    headline = curves[
        [any((m == r.method) and (s == r.sweep) for m, s in head_rows) for r in curves.itertuples()]
    ]
    head_types = caught[
        [any((m == r.method) and (s == r.sweep) for m, s in head_rows) for r in caught.itertuples()]
    ]
    by_type = (
        head_types.groupby(["method", "sweep", "condition_type"])
        .agg(
            conditions=("wo_id", "count"),
            caught=("caught", "sum"),
            warning_days_median=("warning_days", "median"),
            warning_days_min=("warning_days", "min"),
            warning_days_max=("warning_days", "max"),
        )
        .reset_index()
    )
    agree = _agreement(store, cfg, conds, period)
    store.write_pd(curves, "gold", "evaluation_curves_validation")
    store.write_pd(matched, "gold", "evaluation_matched_validation")
    store.write_pd(headline, "gold", "evaluation_headline_validation")
    store.write_pd(by_type, "gold", "evaluation_headline_by_type_validation")
    store.write_pd(caught, "gold", "evaluation_events_validation")
    store.write_pd(
        agree if len(agree) else pd.DataFrame({"note": ["no work requests"]}),
        "gold",
        "diagnosis_agreement_validation",
    )
    return {
        "curves": curves,
        "matched": matched,
        "headline": headline,
        "by_type": by_type,
        "agreement": agree,
        "conditions": len(conds_p),
        "sensor_conditions": len(sensor),
    }


def _agreement(
    store: Store, cfg: dict[str, Any], conds: pd.DataFrame, period: tuple[pd.Timestamp, pd.Timestamp]
) -> pd.DataFrame:
    wr = store.read_pd("gold", "work_requests")
    if wr.empty:
        return wr
    wr["created"] = _utc(wr["created"])
    wr = wr[(wr["created"] >= period[0]) & (wr["created"] < period[1])]
    cs = _utc_cols(store.read_pd("gold", "cases"), ["opened", "closed"])
    wo = _wo(store)
    a = adv.agreement(wr, cs, conds, wo)
    return a.merge(wr[["request_id", "asset_id", "point_id", "created", "suspected_detail"]], on="request_id")


# ---------------------------------------------------------------- monitoring
def monitoring_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    """Monthly event performance, feature drift and the retraining decision over the shadow quarter;
    skipped while the shadow quarter is held out."""
    if cfg["evaluation"].get("holdout_shadow", True):
        return {"skipped": "shadow quarter held out until design freeze"}
    return {
        "skipped": "holdout_shadow is false; monitoring is rebuilt on the evaluation tables at design freeze"
    }
