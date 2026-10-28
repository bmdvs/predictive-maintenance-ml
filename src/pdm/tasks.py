"""Job task bodies: train the per-point model challengers, controller health, survival and the supervised
gate; score the per-point models from the registered champion; prognosis, fusion, advisory, evaluation
and monitoring.

Each task reads its inputs from the Store and writes its output tables back; job entry points under
jobs/ and the local parity runner call these functions. MLflow logging is optional (tracking off in
tests). While the shadow quarter is held out, no task writes anything computed on it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
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
SURVIVAL_INPUTS = [
    ("bronze", "reg_machines"),
    ("bronze", "reg_pumps"),
    ("bronze", "reg_hmc_spindles"),
    ("bronze", "reg_compressors"),
    ("bronze", "cmms_work_orders"),
    ("bronze", "ctrl_events"),
    ("bronze", "cmms_pm_schedule"),
]


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


def run_context(
    store: Store | None, cfg: dict[str, Any] | None, inputs: list[tuple[str, str]]
) -> dict[str, Any]:
    """MLflow parameters every training run carries: the split definition and holdout cap, the config hash
    of the source data, the hash of the configuration used and the Delta version of each input table."""
    if store is None or cfg is None:
        return {}
    b = splits.boundaries(cfg)
    ctx: dict[str, Any] = {
        "split_train": f"{b['train'][0]:%Y-%m-%d} to {b['train'][1]:%Y-%m-%d} (exclusive, UTC)",
        "split_validation": f"{b['validation'][0]:%Y-%m-%d} to {b['validation'][1]:%Y-%m-%d}",
        "split_shadow": f"{b['shadow'][0]:%Y-%m-%d} to {b['shadow'][1]:%Y-%m-%d}",
        "split_gap_days": cfg["splits"]["gap_days"],
        "training_cutoff_utc": str(splits.training_cutoff(cfg)),
        "data_horizon_utc": str(horizon(cfg)),
        "holdout_shadow": bool(cfg["evaluation"].get("holdout_shadow", True)),
        "config_hash_used": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[
            :16
        ],
    }
    if store.exists("bronze", "cm_scalars"):
        src = (
            store.read("bronze", "cm_scalars").select("run_id", "config_hash").distinct().limit(3).toPandas()
        )
        ctx["source_run_id"] = ",".join(sorted(src["run_id"].astype(str)))
        ctx["source_config_hash"] = ",".join(sorted(src["config_hash"].astype(str)))
    for schema, table in inputs:
        ctx[f"input_version.{schema}.{table}"] = store.version(schema, table)
    return ctx


def _no_autolog(experiment: str | None) -> None:
    """Databricks turns on MLflow autologging by default, which would log a run for every model fit inside
    a task; training tasks log only the runs they create themselves."""
    if experiment is None:
        return
    import mlflow

    mlflow.autolog(disable=True)


def _ensure_experiment_parent(experiment: str) -> None:
    """MLflow does not create the workspace folder an experiment path lives in; create it when tracking
    to Databricks."""
    import mlflow

    parent = experiment.rsplit("/", 1)[0]
    if not parent.startswith("/") or not str(mlflow.get_tracking_uri()).startswith("databricks"):
        return
    from databricks.sdk import WorkspaceClient

    WorkspaceClient().workspace.mkdirs(parent)


def _mlflow_log(
    experiment: str | None,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str],
    context: dict[str, Any] | None = None,
    extra: Callable[[], None] | None = None,
    nested: bool = False,
    lineage: tuple[Store, list[tuple[str, str]]] | None = None,
) -> str | None:
    """One MLflow run with parameters (plus the run context), metrics, tags and input tables; extra runs
    inside it (artifacts, models, nested runs)."""
    if experiment is None:
        return None
    import mlflow

    if not nested:
        _ensure_experiment_parent(experiment)
        mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=tags.get("name"), nested=nested) as run:
        allp = {**(context or {}), **params}
        mlflow.log_params({k: (v if isinstance(v, (int, float)) else str(v)[:500]) for k, v in allp.items()})
        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(float(v))}
        )
        mlflow.set_tags(tags)
        if lineage is not None:
            log_inputs(*lineage)
        if extra is not None:
            extra()
        return run.info.run_id


def log_inputs(store: Store, inputs: list[tuple[str, str]]) -> None:
    """Input tables as MLflow datasets, so a registered model shows its lineage: the Delta table with its
    version where its history is readable, otherwise (pipeline materialized views) the table by name."""
    if not store.catalog:
        return
    import mlflow

    for schema, table in inputs:
        name = store.name(schema, table)
        try:
            ds = mlflow.data.load_delta(table_name=name)
        except Exception:  # noqa: BLE001
            ds = mlflow.data.from_spark(store.read(schema, table), table_name=name)
        mlflow.log_input(ds, context="training")


def _log_table(df: pd.DataFrame, name: str) -> None:
    import mlflow

    mlflow.log_text(df.to_csv(index=False), f"{name}.csv")


def horizon(cfg: dict[str, Any]) -> pd.Timestamp:
    """End of the data any task may use: the shadow start while the shadow quarter is held out;
    otherwise the end of the local month `evaluation.as_of` (YYYY-MM, inside the shadow quarter), or the
    record end when as_of is not set."""
    b = splits.boundaries(cfg)
    ev_cfg = cfg["evaluation"]
    as_of = ev_cfg.get("as_of")
    if ev_cfg.get("holdout_shadow", True):
        if as_of:
            raise ValueError("evaluation.as_of needs evaluation.holdout_shadow false")
        return b["shadow"][0]
    if not as_of:
        return b["shadow"][1]
    start = pd.Timestamp(f"{as_of}-01", tz=LOCAL_TZ)
    end = (start + pd.offsets.MonthBegin(1)).tz_convert("UTC")
    if start.tz_convert("UTC") < b["shadow"][0] or end > b["shadow"][1]:
        raise ValueError(f"evaluation.as_of {as_of} is outside the shadow quarter")
    return end


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


def _nbm_run_extra(
    candidate: str,
    model: Any,
    infos: dict[str, list[dict[str, Any]]],
    sel: pd.DataFrame,
    ind: pd.DataFrame,
) -> Callable[[], None]:
    """Inside a candidate's run: one nested run per point, the selection table, and the per-point models
    with their version histories and healthy stats as one pyfunc model (registered by the promote task as
    pdm.ml.nbm_point_models)."""

    def body() -> None:
        import tempfile

        import mlflow
        from mlflow.models import infer_signature

        import pdm
        from pdm.models.nbm_pyfunc import HEALTHY_STATS

        for pid, info in sorted(infos.items()):
            fitted = [i for i in info if i.get("model") in ("segment fit", "window refit")]
            _mlflow_log(
                "nested",
                {"point_id": pid, "segments": len(info), "segments_fitted": len(fitted)},
                {
                    "n_train_hours": float(
                        sum(i.get("n_train_final") or i.get("n_train") or 0 for i in fitted)
                    ),
                    "monthly_updates": float(sum(i.get("monthly_updates") or 0 for i in fitted)),
                },
                {"name": f"{candidate} {pid}", "point_id": pid, "candidate": candidate},
                extra=lambda info=info: _log_table(pd.DataFrame(info).astype(str), "segments"),
                nested=True,
            )
        _log_table(sel, "candidate_selection")
        cols = ["point_id", "hour_utc", "regime", "implausible", *nbm.VECTOR_SOURCE, *nbm.toct_columns(ind)]
        example = ind[cols].dropna(subset=["vel_rms_mm_s"]).head(20).reset_index(drop=True)
        example["hour_utc"] = example["hour_utc"].dt.tz_localize(None)
        example["implausible"] = example["implausible"].fillna(False).astype(bool)
        signature = infer_signature(example, model.predict(None, example))
        with tempfile.TemporaryDirectory() as d:
            hs_path = str(Path(d) / f"{HEALTHY_STATS}.parquet")
            model.healthy_stats.to_parquet(hs_path)
            kwargs = {
                "python_model": model,
                "artifacts": {HEALTHY_STATS: hs_path},
                "code_paths": [str(Path(pdm.__file__).parent)],
                "signature": signature,
                "pip_requirements": ["numpy", "pandas", "scipy", "scikit-learn", "pyyaml", "pyarrow"],
            }
            try:
                mlflow.pyfunc.log_model(name="model", **kwargs)
            except TypeError:
                mlflow.pyfunc.log_model(artifact_path="model", **kwargs)

    return body


def nbm_task(
    store: Store, cfg: dict[str, Any], experiment: str | None = None, job_run_id: str | None = None
) -> dict[str, Any]:
    """Per-point model challengers: every candidate fit per segment with its version history, the
    selection rule's pick tagged challenger. Writes the challenger's own scores (gold.nbm_train_scores),
    the selection table and the segment information; registers nothing. Scoring from the registered
    champion is nbm_score_task."""
    from pdm.models.nbm_pyfunc import NBMPointModels

    _no_autolog(experiment)
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
    seg_models: dict[str, dict[str, list[dict[str, Any]]]] = {c: {} for c in candidates}
    point_info: dict[str, dict[str, list[dict[str, Any]]]] = {c: {} for c in candidates}
    seg_rows, healthy = [], []
    rfx = store.read_pd("gold", "rule_floor_exceedances")
    rfx = rfx[(rfx["method"] == "iso") | (rfx["sweep"] == cfg["rules"]["vendor"]["alert_frac_zone_b_width"])]
    rfx = rfx.merge(_points(store)[["point_id", "asset_id"]], on="point_id")
    rfx["ts"] = _utc(rfx["ts"])
    for pid, rows in ind.groupby("point_id"):
        rows = rows.sort_values("hour_utc").reset_index(drop=True)
        asset = rows["asset_id"].iat[0]
        plan = nbm.segment_plan(win[win["point_id"] == pid], flagged_ids, end, rb)
        rebase = list(plan.loc[plan["reason"] != "install", "start"])
        excl = _excluded(asset, conds, wo, rebase, plan[plan["flagged"]], cfg)
        w = wo[wo["asset_id"].str.startswith(asset) & wo["wo_type"].isin(INSPECTION_OR_REPAIR)]
        open_work = [(a, b if pd.notna(b) else end) for a, b in zip(w["opened"], w["closed"], strict=True)]
        alerts = list(rfx.loc[rfx["asset_id"] == asset, "ts"])
        for cand in candidates:
            s, infos, tr, fitted = nbm.fit_score_segments(
                rows, cand, cfg, cutoff, excl, plan, end, nb["top_contributors"], open_work, alerts
            )
            seg_models[cand][pid] = fitted
            point_info[cand][pid] = infos
            if len(s):
                scores[cand].append(s.assign(asset_id=asset))
            if cand == candidates[0]:
                seg_rows += [{**i, "asset_id": asset, "excluded_intervals": len(excl)} for i in infos]
            if cand == candidates[0] and len(tr):
                healthy.append(
                    pd.concat(
                        [
                            adv.healthy_stats(g).assign(segment=k, segment_start=g["segment_start"].iat[0])
                            for k, g in tr.groupby("segment")
                        ],
                        ignore_index=True,
                    )
                )
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
    inputs = [
        ("silver", "hourly_indicators"),
        ("silver", "healthy_windows"),
        ("silver", "confirmed_conditions"),
        ("silver", "points"),
        ("gold", "baseline_sanity"),
        ("gold", "rule_floor_exceedances"),
        ("bronze", "cmms_work_orders"),
    ]
    ctx = run_context(store, cfg, inputs) if experiment else {}
    hs = pd.concat(healthy, ignore_index=True)
    model_objs = {
        c: NBMPointModels(
            c,
            nb["threshold_percentile"],
            seg_models[c],
            nb["top_contributors"],
            nb["sweep_percentiles"],
            end,
            hs,
        )
        for c in candidates
    }
    runs = {}
    for r in sel.itertuples():
        runs[r.candidate] = _mlflow_log(
            experiment,
            {
                "candidate": r.candidate,
                "basis": r.basis,
                "threshold_percentile": nb["threshold_percentile"],
                "sweep_percentiles": nb["sweep_percentiles"],
                "persistence_hours": nb["persistence_hours"],
                "rebaseline_days": nb["rebaseline_days"],
                "exclude_before_condition_days": nb["exclude_before_condition_days"],
                "min_regime_healthy_hours": nb["min_regime_healthy_hours"],
                "min_train_hours": nb["min_train_hours"],
                "champion_margin_conditions": nb["champion_margin_conditions"],
                "seed": cfg["models"]["seed"],
            },
            {
                "training_caught_share": r.training_caught,
                "training_chance_share": r.training_chance,
                "training_skill": r.skill,
                "training_false_per_asset_month": r.operating_false_per_asset_month,
                "curve_false_min": r.curve_false_min,
                "curve_false_max": r.curve_false_max,
                "vendor_training_false_per_asset_month": r.vendor_training_false_per_asset_month,
                "training_conditions": r.training_conditions,
                "segments_fitted": sum(len(v) for v in seg_models[r.candidate].values()),
            },
            {
                "name": f"nbm {r.candidate}",
                "layer": "nbm",
                "candidate": r.candidate,
                "challenger": str(r.candidate == champion).lower(),
                "model_version": version,
                **({"pdm.job_run_id": str(job_run_id)} if job_run_id else {}),
            },
            context=ctx,
            extra=_nbm_run_extra(r.candidate, model_objs[r.candidate], point_info[r.candidate], sel, ind),
            lineage=(store, inputs),
        )
    run = runs.get(champion)
    store.write_pd(sc.assign(model_version=version, candidate=champion), "gold", "nbm_train_scores")
    store.write_pd(
        sel.assign(champion=sel["candidate"] == champion, model_version=version), "gold", "nbm_candidates"
    )
    store.write_pd(info.astype({"models": str}).assign(model_version=version), "gold", "nbm_models")
    return {
        "challenger": champion,
        "selection": sel,
        "version": version,
        "mlflow_run": run,
        "mlflow_runs": runs,
        "segments": info["model"].value_counts().to_dict(),
        "model": model_objs[champion],
        "train_exceedances": len(ex_all),
    }


def nbm_score_tables(
    model: Any,
    ind: pd.DataFrame,
    pts: pd.DataFrame,
    win: pd.DataFrame,
    cfg: dict[str, Any],
    model_version: str,
) -> dict[str, pd.DataFrame]:
    """The per-point model output tables from a registered model: scores, persisted exceedances at every
    sweep percentile, the per-segment healthy stats and the exceedance diagnosis."""
    nb = cfg["models"]["nbm"]
    end = horizon(cfg)
    cutoff = splits.training_cutoff(cfg)
    assets = ind.drop_duplicates("point_id").set_index("point_id")["asset_id"]
    sc = model.score_frame(ind)
    sc["asset_id"] = sc["point_id"].map(assets)
    parts = []
    for p in nb["sweep_percentiles"]:
        ex = nbm.exceedance_hours(sc, p, nb["persistence_hours"])
        if len(ex):
            parts.append(ex.merge(pts[["point_id", "asset_id"]], on="point_id").assign(sweep=p))
    ex_all = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["point_id", "ts", "ratio", "top_contributors", "asset_id", "sweep"])
    )
    b = splits.boundaries(cfg)
    periods = {"training": (b["train"][0], cutoff), "validation": (b["validation"][0], end)}
    diag = nbm.exceedance_diagnosis(sc, win, nb["threshold_percentile"], periods, cfg)
    return {
        "nbm_scores": sc.assign(model_version=model_version, candidate=model.candidate),
        "nbm_exceedances": ex_all.assign(method="nbm", level="alarm", model_version=model_version),
        "point_healthy_stats": model.healthy_stats.assign(model_version=model_version),
        "nbm_exceedance_diagnosis": diag.assign(model_version=model_version),
    }


def nbm_score_task(store: Store, cfg: dict[str, Any], model: Any, model_version: str) -> dict[str, Any]:
    """Score every measured hour to the data horizon with the given per-point models (the registered
    champion in the job) and write gold.nbm_scores, nbm_exceedances, point_healthy_stats and
    nbm_exceedance_diagnosis with its version."""
    end = horizon(cfg)
    ind = store.read_pd("silver", "hourly_indicators")
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind = ind[ind["hour_utc"] < end]
    win = _utc_cols(store.read_pd("silver", "healthy_windows"), ["start", "end"])
    win = win[win["start"] < end]
    out = nbm_score_tables(model, ind, _points(store), win, cfg, model_version)
    for name, df in out.items():
        store.write_pd(df, "gold", name)
    return {
        "model_version": model_version,
        "candidate": model.candidate,
        **{k: len(v) for k, v in out.items()},
    }


# ---------------------------------------------------------------- controller health
def controller_health_task(
    store: Store, cfg: dict[str, Any], experiment: str | None = None
) -> dict[str, Any]:
    _no_autolog(experiment)
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
    ch_cfg = cfg["models"]["controller_health"]
    _mlflow_log(
        experiment,
        {
            "machines": len(models),
            "baseline": ch_cfg["baseline"],
            "drift_alert_frac": ch_cfg["drift_alert_frac"],
            "thermistor_residual_alert_c": ch_cfg["thermistor_residual_alert_c"],
            "sweep_scale": ch_cfg["sweep_scale"],
            "persistence_days": cfg["rules"]["persistence_consecutive"],
        },
        {f"exceedance_days_scale_{k:g}": float(v) for k, v in ex.groupby("sweep").size().items()},
        {"name": "controller_health", "layer": "controller_health", "model_version": version},
        context=run_context(store, cfg, [("silver", "controller_hourly"), ("bronze", "reg_machines")])
        if experiment
        else None,
        lineage=(store, [("silver", "controller_hourly"), ("bronze", "reg_machines")]),
        extra=lambda: _log_table(
            pd.DataFrame([{"machine_id": k, "thermistor_coef": v} for k, v in models.items()]),
            "thermistor_models",
        ),
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


def rule_floor_task(store: Store, cfg: dict[str, Any], experiment: str | None = None) -> dict[str, Any]:
    """Training-period comparison runs for the rule floor: the vendor rules at each swept offset, the
    vendor rules with the ISO zone rule (the rule floor) and the ISO rule alone, with the same metrics as
    the per-point model candidates. The rule floor is code, not a model; nothing is registered."""
    _no_autolog(experiment)
    pts = _points(store)
    conds = _conditions(store, cfg)
    cutoff = splits.training_cutoff(cfg)
    train_conds = conds[(conds["credit_ts"] < cutoff) & conds["instrumented"]]
    period = (splits.boundaries(cfg)["train"][0], cutoff)
    n_assets = pts["asset_id"].nunique()
    rf = _episodes_from(store.read_pd("gold", "rule_floor_exceedances"), pts, cfg)
    vend_op = cfg["rules"]["vendor"]["alert_frac_zone_b_width"]
    rows = []
    for (method, s), ep in rf.groupby(["method", "sweep"]):
        r, _ = _summary(method, ep, train_conds, conds, period, n_assets, cfg, store, s)
        rows.append({**r, "sweep": float(s), "operating_point": bool(method == "iso" or s == vend_op)})
    curve = pd.DataFrame(rows).sort_values(["method", "sweep"]).reset_index(drop=True)
    rf_inputs = [("gold", "rule_floor_exceedances"), ("silver", "confirmed_conditions"), ("silver", "points")]
    ctx = run_context(store, cfg, rf_inputs) if experiment else {}
    v = cfg["rules"]["vendor"]
    runs = {}
    for r in curve.itertuples():
        runs[f"{r.method} {r.sweep:g}"] = _mlflow_log(
            experiment,
            {
                "method": r.method,
                "alert_frac_zone_b_width": r.sweep if r.method != "iso" else None,
                "alarm_to_alert_ratio": v["alarm_frac_zone_b_width"] / v["alert_frac_zone_b_width"],
                "envelope_db_over_baseline": cfg["rules"]["envelope"]["db_over_baseline"],
                "persistence_consecutive": cfg["rules"]["persistence_consecutive"],
                "episode_gap_days": cfg["evaluation"]["catch"]["episode_gap_days"],
                "opened_within_days": cfg["evaluation"]["catch"]["opened_within_days"],
            },
            {
                "training_caught_share": r.caught_share,
                "training_chance_share": r.chance_share,
                "training_skill": r.caught_share - r.chance_share,
                "training_false_per_asset_month": r.false_per_asset_month,
                "training_episodes_per_asset_month": r.episodes_per_asset_month,
                "training_warning_days_median": r.warning_days_median,
                "training_conditions": r.conditions,
            },
            {
                "name": f"rule floor {r.method} {r.sweep:g}" if r.method != "iso" else "rule floor iso",
                "layer": "rule_floor",
                "method": r.method,
                "operating_point": str(r.operating_point).lower(),
            },
            context=ctx,
            lineage=(store, rf_inputs),
        )
    store.write_pd(curve, "gold", "rule_floor_training_curve")
    return {"runs": len(runs), "curve": curve}


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
    _no_autolog(experiment)
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
        close_lag_days=float(max(cfg["label_noise"]["close_lag_days"])),
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
    iv = cfg["models"]["survival"]["interval"]
    classes = survival.per_class_weibull(lives, iv)
    classes["stable"] = classes["events"] > cfg["models"]["survival"]["unstable_max_events"]
    store.write_pd(classes.assign(model_version=version), "gold", "survival_class_weibull")
    shared, shape = survival.shared_shape_weibull(lives, iv)
    store.write_pd(shared.assign(model_version=version), "gold", "survival_shared_shape_weibull")
    sv = cfg["models"]["survival"]
    _mlflow_log(
        experiment,
        {
            "lives": res["n"],
            "events": res["events"],
            "pm_review_model": sv["pm_review"],
            "covariate_models": sv["models"],
            "covariates": sv["covariates"],
            "interval": iv,
            "unstable_max_events": sv["unstable_max_events"],
            "record_end_utc": str(horizon(cfg)),
        },
        {
            "concordance_aft": res["concordance_aft"],
            "concordance_cox": res["concordance_cox"],
            "ibs_ipcw_aft": res["ibs_ipcw"],
            "ibs_ipcw_kaplan_meier": res["ibs_ipcw_km"],
            "shared_shape_rho": shape["shape_rho"],
            "shared_shape_rho_low": shape["rho_low"],
            "shared_shape_rho_high": shape["rho_high"],
            **{f"median_life_days_{r.component_class}": r.median_life_days for r in shared.itertuples()},
        },
        {"name": "survival", "layer": "survival", "model_version": version},
        context=run_context(store, cfg, SURVIVAL_INPUTS) if experiment else None,
        extra=_survival_tables(shared, classes, summary),
        lineage=(store, SURVIVAL_INPUTS),
    )
    return {
        "class_weibull": classes,
        "shared_shape_weibull": shared,
        "covariate_models_supported": bool(res["ibs_ipcw"] < res["ibs_ipcw_km"]),
        **{
            k: res[k]
            for k in ("n", "events", "concordance_aft", "concordance_cox", "ibs_ipcw", "ibs_ipcw_km")
        },
    }


def _survival_tables(
    shared: pd.DataFrame, classes: pd.DataFrame, summary: pd.DataFrame
) -> Callable[[], None]:
    def body() -> None:
        _log_table(shared, "shared_shape_weibull")
        _log_table(classes, "per_class_weibull")
        _log_table(summary, "survival_summary")

    return body


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
        d.sort_values(["ts", "rules"], kind="stable")
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
    d = _daily_first(out).assign(_pk=lambda x: x["point_id"].fillna(""))
    return (
        d.sort_values(["ts", "method", "asset_id", "_pk", "level", "rules"], kind="stable")
        .drop(columns=["_pk"])
        .reset_index(drop=True)
    )


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
    tsc = _tsc_pressure_daily(store, cfg)
    store.write_pd(tsc, "gold", "tsc_pressure_daily")
    out["tsc_pressure_days"] = len(tsc)
    return out


def _tsc_pressure_daily(store: Store, cfg: dict[str, Any]) -> pd.DataFrame:
    """Daily through-spindle coolant pressure per HMC while the pump runs (silver
    coolant_pressure_pump_on_bar) to the horizon, for the queue's controller panel."""
    hmc = set(store.read_pd("bronze", "reg_machines").query("`class` == 'hmc'")["machine_id"])
    chh = store.read_pd(
        "silver", "controller_hourly", ["machine_id", "hour_utc", "coolant_pressure_pump_on_bar"]
    )
    chh["hour_utc"] = _utc(chh["hour_utc"])
    chh = chh[chh["machine_id"].isin(hmc) & (chh["hour_utc"] < horizon(cfg))]
    a = cfg["advisory"]["tsc_pressure_reference_days"]
    return adv.tsc_pressure_daily(chh, (a[0], a[1]))


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


def _evaluation_inputs(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    """Alert episodes per method and sweep, conditions and asset counts, read once per task."""
    pts = _points(store)
    gap = cfg["evaluation"]["catch"]["episode_gap_days"]
    rf = _episodes_from(store.read_pd("gold", "rule_floor_exceedances"), pts, cfg)
    nx = store.read_pd("gold", "nbm_exceedances")
    nx["ts"] = _utc(nx["ts"])
    nx = nx[nx["ts"] < horizon(cfg)]
    nbm_ep = pd.concat(
        [ev.episodes(g[["asset_id", "point_id", "ts"]], gap).assign(sweep=p) for p, g in nx.groupby("sweep")],
        ignore_index=True,
    )
    hmc = set(store.read_pd("bronze", "reg_machines").query("`class` == 'hmc'")["machine_id"])
    cx = _controller_signals(store.read_pd("gold", "controller_health_exceedances"), hmc)
    cx["ts"] = _utc(cx["ts"])
    cx = cx[cx["ts"] < horizon(cfg)]
    ctrl_ep = pd.concat(
        [ev.episodes(g[["asset_id", "point_id", "ts"]], gap).assign(sweep=s) for s, g in cx.groupby("sweep")],
        ignore_index=True,
    )
    cs = store.read_pd("gold", "cases_sweep")
    cs = _utc_cols(cs, ["opened", "closed"])
    cs["closed"] = cs["closed"].fillna(horizon(cfg))
    cs["chain_opened"] = _utc(cs["chain_opened"])
    case_ep = cs.assign(
        point_id=cs["trigger_points"].fillna(""), last=cs["opened"], warning_from=cs["chain_opened"]
    )
    return {
        "pts": pts,
        "conds": _conditions(store, cfg),
        "rf": rf,
        "nbm_ep": nbm_ep,
        "ctrl_ep": ctrl_ep,
        "case_ep": case_ep,
        "n_instr": pts["asset_id"].nunique(),
        "n_mach": store.read_pd("bronze", "reg_machines")["machine_id"].nunique(),
        "instr_assets": set(pts["asset_id"]),
    }


BY_TYPE_COLUMNS = [
    "method",
    "sweep",
    "condition_type",
    "conditions",
    "caught",
    "warning_days_median",
    "warning_days_min",
    "warning_days_max",
]


def _operating(cfg: dict[str, Any]) -> dict[str, float]:
    vend_op = cfg["rules"]["vendor"]["alert_frac_zone_b_width"]
    op = cfg["models"]["nbm"]["threshold_percentile"]
    return {
        "vendor": vend_op,
        "rule_floor": vend_op,
        "iso": 0.0,
        "nbm": op,
        "controller": 1.0,
        "controller_uninstrumented": 1.0,
        "model_layer": op,
        "model_layer_instrumented": op,
        "model_layer_uninstrumented": op,
    }


def _evaluate(
    store: Store, cfg: dict[str, Any], period: tuple[pd.Timestamp, pd.Timestamp], inp: dict[str, Any]
) -> dict[str, Any]:
    """Section 12 for one period: operating curves with chance columns per method, the comparison at
    the vendor rules' operating point, the headline rows by condition type and diagnosis agreement."""
    conds = inp["conds"]
    conds_p = conds[(conds["credit_ts"] >= period[0]) & (conds["credit_ts"] < period[1])]
    sensor = conds_p[conds_p["instrumented"]]
    no_sensor = conds_p[~conds_p["instrumented"]]
    n_instr, n_mach, instr_assets = inp["n_instr"], inp["n_mach"], inp["instr_assets"]
    rf, nbm_ep, ctrl_ep, case_ep = inp["rf"], inp["nbm_ep"], inp["ctrl_ep"], inp["case_ep"]
    vend_op = cfg["rules"]["vendor"]["alert_frac_zone_b_width"]
    nb = cfg["models"]["nbm"]

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
    operating = _operating(cfg)
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
    if "condition_type" in caught and len(caught):
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
    else:
        by_type = pd.DataFrame(columns=BY_TYPE_COLUMNS)
    agree = _agreement(store, cfg, conds, period)
    return {
        "curves": curves,
        "matched": matched,
        "headline": headline,
        "by_type": by_type,
        "caught": caught,
        "agreement": agree,
        "conditions": len(conds_p),
        "sensor_conditions": len(sensor),
    }


def shadow_periods(cfg: dict[str, Any]) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """The periods a shadow-quarter run reports, each as of the run's horizon: the month that ends at the
    horizon, the quarter to date and, once the horizon reaches the record end, July 2025 to June 2026."""
    b = splits.boundaries(cfg)
    end = horizon(cfg)
    local_last = (end - pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ)
    month_start = pd.Timestamp(local_last.strftime("%Y-%m-01"), tz=LOCAL_TZ).tz_convert("UTC")
    q = b["shadow"][0].tz_convert(LOCAL_TZ)
    out = [
        (local_last.strftime("%Y-%m"), max(month_start, b["shadow"][0]), end),
        (f"{q.year}-Q{(q.month - 1) // 3 + 1}", b["shadow"][0], end),
    ]
    if end >= b["shadow"][1]:
        v = b["validation"][0].tz_convert(LOCAL_TZ)
        out.append((f"{v:%Y-%m}_to_{local_last:%Y-%m}", b["validation"][0], end))
    return out


def _uniform_types(df: pd.DataFrame) -> pd.DataFrame:
    """One type per column before a Delta write: columns that mix booleans and numbers (a count that is
    a boolean where a period has no rows) become numbers; other mixed columns become strings."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != object:
            continue
        kinds = {type(v) for v in out[c].dropna()}
        numeric = {bool, int, float, np.bool_, np.int64, np.int32, np.float64}
        if len(kinds) > 1 and kinds <= numeric:
            out[c] = pd.to_numeric(out[c].map(lambda v: float(v) if v is not None else np.nan))
        elif len(kinds) > 1:
            out[c] = out[c].map(
                lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            )
    return out


def _replace_periods(
    store: Store, schema: str, table: str, new: pd.DataFrame, key: str, labels: list[str]
) -> None:
    """Write the rows for these periods, keeping rows earlier runs wrote for other periods."""
    if store.exists(schema, table):
        old = store.read_pd(schema, table)
        if key in old:
            old = old[~old[key].astype(str).isin(labels)]
            new = pd.concat([old, new], ignore_index=True) if len(old) else new
    store.write_pd(_uniform_types(new.reset_index(drop=True)), schema, table)


def evaluation_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    """Section 12 evaluation. While the shadow quarter is held out: the validation period
    (gold.evaluation_*_validation). After the holdout is lifted the validation tables stay as computed,
    and each run evaluates its shadow periods as of its horizon (gold.evaluation_*_shadow with a period
    column; gold.evaluation_headline_year once the horizon reaches the record end)."""
    inp = _evaluation_inputs(store, cfg)
    if cfg["evaluation"].get("holdout_shadow", True):
        r = _evaluate(store, cfg, splits.boundaries(cfg)["validation"], inp)
        store.write_pd(r["curves"], "gold", "evaluation_curves_validation")
        store.write_pd(r["matched"], "gold", "evaluation_matched_validation")
        store.write_pd(r["headline"], "gold", "evaluation_headline_validation")
        store.write_pd(r["by_type"], "gold", "evaluation_headline_by_type_validation")
        store.write_pd(r["caught"], "gold", "evaluation_events_validation")
        store.write_pd(
            r["agreement"] if len(r["agreement"]) else pd.DataFrame({"note": ["no work requests"]}),
            "gold",
            "diagnosis_agreement_validation",
        )
        return r
    periods = shadow_periods(cfg)
    labels = [p[0] for p in periods]
    res = {label: _evaluate(store, cfg, (a, e), inp) for label, a, e in periods}
    for table, key in (
        ("evaluation_curves_shadow", "curves"),
        ("evaluation_matched_shadow", "matched"),
        ("evaluation_headline_shadow", "headline"),
        ("evaluation_headline_by_type_shadow", "by_type"),
        ("evaluation_events_shadow", "caught"),
    ):
        new = pd.concat([res[label][key].assign(period=label) for label in labels], ignore_index=True)
        _replace_periods(store, "gold", table, new, "period", labels)
    agree = pd.concat(
        [
            (
                res[label]["agreement"]
                if len(res[label]["agreement"])
                else pd.DataFrame({"note": ["no work requests"]})
            ).assign(period=label)
            for label in labels
        ],
        ignore_index=True,
    )
    _replace_periods(store, "gold", "diagnosis_agreement_shadow", agree, "period", labels)
    if len(periods) == 3:
        store.write_pd(
            res[labels[2]]["headline"].assign(period=labels[2]), "gold", "evaluation_headline_year"
        )
    return {
        "periods": labels,
        "headline": {label: res[label]["headline"] for label in labels},
        "conditions": {label: res[label]["conditions"] for label in labels},
        "sensor_conditions": {label: res[label]["sensor_conditions"] for label in labels},
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
    ev_c = _utc_cols(store.read_pd("bronze", "cmp_events"), ["ts_utc"])
    trips = ev_c[ev_c["event_type"].str.contains("trip")]
    lag = pd.Timedelta(days=cfg["advisory"]["trip_match_days"])
    trip_wos = {
        r.wo_id
        for r in conds[conds["condition_type"] == "compressor trip"].itertuples()
        if (
            (trips["compressor_id"] == r.asset_id)
            & (trips["ts_utc"] <= r.confirmed_ts)
            & (trips["ts_utc"] >= r.confirmed_ts - lag)
        ).any()
    }
    a = adv.agreement(wr, cs, conds, wo, trip_wos)
    return a.merge(wr[["request_id", "asset_id", "point_id", "created", "suspected_detail"]], on="request_id")


# ---------------------------------------------------------------- monitoring
MONITORED = ("rule_floor", "nbm", "controller", "model_layer_instrumented")
PROFILED_TABLES = (("gold", "scoring_log"), ("silver", "hourly_indicators"))


def _monitored_episodes(cfg: dict[str, Any], inp: dict[str, Any]) -> dict[str, tuple[pd.DataFrame, int]]:
    """Alert episodes at each monitored method's operating point, with the asset count it is rated on."""
    op = _operating(cfg)
    rf, nbm_ep, ctrl_ep, case_ep = inp["rf"], inp["nbm_ep"], inp["ctrl_ep"], inp["case_ep"]
    return {
        "rule_floor": (
            rf[(rf["method"] == "rule_floor") & (rf["sweep"] == op["rule_floor"])],
            inp["n_instr"],
        ),
        "nbm": (nbm_ep[nbm_ep["sweep"] == op["nbm"]], inp["n_instr"]),
        "controller": (ctrl_ep[ctrl_ep["sweep"] == op["controller"]], inp["n_mach"]),
        "model_layer_instrumented": (
            case_ep[
                (case_ep["sweep"] == op["model_layer_instrumented"])
                & case_ep["asset_id"].isin(inp["instr_assets"])
            ],
            inp["n_instr"],
        ),
    }


def alert_rows(ep: pd.DataFrame, conds: pd.DataFrame, end: pd.Timestamp, window_days: float) -> pd.DataFrame:
    """One row per alert episode (opening time) with whether a confirmed condition on the asset followed
    within the window; unknown while the window is still open at the horizon."""
    w = pd.Timedelta(days=window_days)
    if ep.empty:
        return pd.DataFrame(
            {
                "asset_id": pd.Series(dtype=object),
                "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                "confirmed_within_window": pd.Series(dtype=float),
            }
        )
    opened = pd.to_datetime(ep["opened"], utc=True)
    out = []
    for asset, t0 in zip(ep["asset_id"], opened, strict=True):
        if t0 + w > end:
            out.append(np.nan)
            continue
        c = conds.loc[conds["asset_key"] == asset, "credit_ts"]
        out.append(float(((c >= t0) & (c <= t0 + w)).any()))
    return pd.DataFrame(
        {"asset_id": ep["asset_id"].to_numpy(), "ts": opened.to_numpy(), "confirmed_within_window": out}
    )


def scoring_log(
    scores: pd.DataFrame,
    status: pd.DataFrame,
    conds: pd.DataFrame,
    threshold_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_days: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Per scored point-hour from start: score, threshold, exceedance as the prediction, the asset's
    tier at that hour, model version and the outcome (a confirmed condition on the asset within the
    window after the hour), null while the window is open at the horizon. Prediction and outcome are 0 or 1
    (doubles) for the classification inference profile."""
    w = pd.Timedelta(days=window_days)
    s = scores.copy()
    s["hour_utc"] = pd.to_datetime(s["hour_utc"], utc=True)
    s = s[(s["hour_utc"] >= start) & (s["hour_utc"] < end)].sort_values(["asset_id", "hour_utc"])
    st = status.copy()
    st["ts"] = pd.to_datetime(st["ts"], utc=True)
    st = st.sort_values("ts")
    parts = []
    for asset, g in s.groupby("asset_id", sort=False):
        g = g.sort_values("hour_utc").reset_index(drop=True)
        a = st[st["asset_id"] == asset]
        if len(a):
            idx = np.searchsorted(a["ts"].to_numpy(), g["hour_utc"].to_numpy(), side="right") - 1
            tier = np.where(idx >= 0, a["status"].to_numpy()[np.clip(idx, 0, None)], "OK")
        else:
            tier = np.full(len(g), "OK", dtype=object)
        c = np.sort(conds.loc[conds["asset_key"] == asset, "credit_ts"].to_numpy(dtype="datetime64[ns]"))
        h = g["hour_utc"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        j = np.searchsorted(c, h, side="left")
        nxt = np.where(
            j < c.size, c[np.clip(j, 0, max(c.size - 1, 0))] if c.size else h, np.datetime64("NaT")
        )
        hit = (j < c.size) & ((nxt - h) <= w.to_timedelta64())
        closed = (g["hour_utc"] + w <= end).to_numpy()
        outcome = pd.Series(np.where(closed, hit.astype(float), np.nan)).astype(object).where(closed, None)
        parts.append(
            pd.DataFrame(
                {
                    "asset_id": asset,
                    "point_id": g["point_id"].to_numpy(),
                    "hour_utc": g["hour_utc"].to_numpy(),
                    "score": g["score"].to_numpy(),
                    "threshold": g[threshold_col].to_numpy(),
                    "prediction": (g["score"] > g[threshold_col]).astype(float).to_numpy(),
                    "tier": tier,
                    "model_version": g["model_version"].to_numpy(),
                    "outcome": outcome.to_numpy(),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out):
        out["period"] = splits.chronological(out["hour_utc"], cfg).to_numpy()
    return out


def shadow_months(cfg: dict[str, Any]) -> list[str]:
    """Local months of the shadow quarter up to the run's horizon."""
    b = splits.boundaries(cfg)
    end = horizon(cfg)
    first = b["shadow"][0].tz_convert(LOCAL_TZ)
    last = (end - pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ)
    return [str(d) for d in pd.period_range(first.strftime("%Y-%m"), last.strftime("%Y-%m"), freq="M")]


def _drift_rebaselines(
    store: Store,
    cfg: dict[str, Any],
    since: pd.Timestamp,
    end: pd.Timestamp,
    scores: pd.DataFrame,
) -> dict[str, list[Any]]:
    """Per point, the drift references of its post-repair and post-remount re-baseline windows that
    started after `since`: windows usable for a refit (complete before end, not excluded for a nearby
    failure, not flagged by the post-repair sanity check), as the per-point model segments use them,
    extended by the first scored days with no open case on the point; and windows still forming at end,
    whose hours are not compared."""
    from pdm.monitoring import performance as mon

    rb = pd.Timedelta(days=cfg["models"]["nbm"]["rebaseline_days"])
    days = int(cfg["monitoring"]["drift"]["reference_scored_days"])
    win = _utc_cols(store.read_pd("silver", "healthy_windows"), ["start", "end"])
    win = win[win["start"] < end]
    sanity = store.read_pd("gold", "baseline_sanity")
    flagged_ids = set(sanity.loc[sanity["flagged"].astype(bool), "window_id"])
    cases = _utc_cols(store.read_pd("gold", "cases"), ["opened", "closed"])
    cases = cases[cases["opened"] < end]
    asset_of = _points(store).set_index("point_id")["asset_id"].to_dict()
    out: dict[str, list[Any]] = {}
    for pid, w in win.groupby("point_id"):
        plan = nbm.segment_plan(w, flagged_ids, end, rb)
        keep = plan["refit"].astype(bool) | ~plan["complete"].astype(bool)
        use = plan[(plan["reason"] != "install") & keep & (plan["start"] >= since)]
        if not len(use):
            continue
        c = cases[cases["asset_id"] == asset_of.get(pid)]
        intervals = [
            (o, min(x, end) if pd.notna(x) else end) for o, x in zip(c["opened"], c["closed"], strict=True)
        ]
        hours = scores.loc[scores["point_id"] == pid, "hour_utc"]
        wins = [(a, a + rb, bool(done)) for a, done in zip(use["start"], use["complete"], strict=True)]
        out[pid] = mon.drift_references(wins, hours, intervals, end, days)
    return out


def _refresh_profiles(store: Store) -> dict[str, str]:
    """Refresh the data profiles on the scoring log and the hourly indicators, where they exist."""
    if not store.catalog:
        return {}
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    out = {}
    for schema, table in PROFILED_TABLES:
        name = store.name(schema, table)
        try:
            r = w.quality_monitors.run_refresh(table_name=name)
            out[name] = f"refresh {r.refresh_id} {r.state}"
        except Exception as e:  # noqa: BLE001
            out[name] = f"not refreshed: {str(e)[:200]}"
        LOG.info("profile %s: %s", name, out[name])
    return out


def monitoring_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    """Monthly monitoring over the shadow quarter as of the run's horizon: gold.scoring_log;
    monitoring.event_performance and monitoring.feature_drift for every shadow month so far (recall, lead
    time and alert counts are fixed once a month has passed; alert precision fills in as the 30-day
    windows close, with the number of alerts it rests on); monitoring.retraining_decision over those
    months; then a refresh of the data profiles. Skipped while the shadow quarter is held out."""
    from pdm.monitoring import performance as mon

    if cfg["evaluation"].get("holdout_shadow", True):
        return {"skipped": "shadow quarter held out"}
    b = splits.boundaries(cfg)
    end = horizon(cfg)
    month = shadow_periods(cfg)[0][0]
    nb = cfg["models"]["nbm"]
    ev_cfg = cfg["evaluation"]
    window = ev_cfg["alert_precision_window_days"]
    inp = _evaluation_inputs(store, cfg)
    conds = inp["conds"]

    thr_col = f"thr_{nb['threshold_percentile']:g}"
    sc = store.read_pd("gold", "nbm_scores")[
        ["point_id", "asset_id", "hour_utc", "score", thr_col, "model_version"]
    ]
    log = scoring_log(
        sc,
        store.read_pd("gold", "asset_status"),
        conds,
        f"thr_{nb['threshold_percentile']:g}",
        b["validation"][0],
        end,
        window,
        cfg,
    )
    store.write_pd(log, "gold", "scoring_log")
    val = (log["hour_utc"] >= b["validation"][0]) & (log["hour_utc"] < b["validation"][1])
    store.write_pd(log[val].reset_index(drop=True), "gold", "scoring_log_baseline")

    months = shadow_months(cfg)
    events = store.read_pd("gold", "evaluation_events_shadow")
    events = events[events["period"].astype(str).isin(months)]
    op = _operating(cfg)
    eps = _monitored_episodes(cfg, inp)
    caught = pd.concat(
        [
            events[(events["method"] == m) & (events["sweep"].astype(float) == float(op[m]))]
            for m in MONITORED
        ],
        ignore_index=True,
    )
    shadow_start = b["shadow"][0]
    fa = []
    for m, (ep, _n) in eps.items():
        opened = pd.to_datetime(ep["opened"], utc=True)
        fa.append(
            alert_rows(ep[(opened >= shadow_start) & (opened < end)], conds, end, window).assign(method=m)
        )
    fa = pd.concat(fa, ignore_index=True)
    fa["ts"] = pd.to_datetime(fa["ts"], utc=True)
    credit = conds.drop_duplicates("wo_id").set_index("wo_id")["credit_ts"]
    n_assets = {m: n for m, (_e, n) in eps.items()}
    perf = (
        mon.event_performance(caught, fa, credit, n_assets)
        if len(caught) or len(fa)
        else pd.DataFrame(columns=["method", "month"])
    )
    perf = perf[perf["month"].isin(months)]
    grid = pd.DataFrame([(m, mo) for m in MONITORED for mo in months], columns=["method", "month"])
    perf = grid.merge(perf, on=["method", "month"], how="left")
    for c in ("conditions", "alerts"):
        perf[c] = perf[c].fillna(0).astype(int)
    perf["alerts_per_asset_month"] = perf["alerts_per_asset_month"].fillna(0.0)
    closed = fa.dropna(subset=["confirmed_within_window"]).assign(month=lambda d: mon.month_of(d["ts"]))
    counts = closed.groupby(["method", "month"]).size().rename("precision_alerts").reset_index()
    perf = perf.merge(counts, on=["method", "month"], how="left")
    perf["precision_alerts"] = perf["precision_alerts"].fillna(0).astype(int)
    perf = perf.assign(as_of=str(end))
    store.write_pd(perf, "monitoring", "event_performance")

    feats = list(nbm.VECTOR_SOURCE)
    ind = store.read_pd("silver", "hourly_indicators", ["point_id", "hour_utc", "implausible", *feats])
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind = ind[(ind["hour_utc"] < end) & ~ind["implausible"].fillna(False).astype(bool)]
    base_months = sorted(
        set(
            mon.month_of(
                ind.loc[
                    (ind["hour_utc"] >= b["validation"][0]) & (ind["hour_utc"] < b["validation"][1]),
                    "hour_utc",
                ]
            )
        )
    )
    sc_drift = sc.rename(columns={"score": mon.SCORE_FEATURE})
    sc_drift["hour_utc"] = _utc(sc_drift["hour_utc"])
    sc_drift = sc_drift[sc_drift["hour_utc"] < end]
    rebase = _drift_rebaselines(store, cfg, b["validation"][0], end, sc_drift)
    drift = pd.concat(
        [
            mon.feature_drift(ind, feats, base_months, months, rebase),
            mon.feature_drift(sc_drift, [mon.SCORE_FEATURE], base_months, months, rebase),
        ],
        ignore_index=True,
    ).assign(as_of=str(end))
    store.write_pd(drift, "monitoring", "feature_drift")

    curves = store.read_pd("gold", "evaluation_curves_validation")
    reference = {}
    for m in MONITORED:
        r = curves[(curves["method"] == m) & curves["operating_point"].astype(bool)]
        if len(r):
            reference[m] = {
                "recall": float(r["caught_share"].iat[0]),
                "alerts_per_asset_month": float(r["episodes_per_asset_month"].iat[0]),
            }
    decisions = pd.concat(
        [mon.retraining_decision(perf, drift, reference, cfg, m) for m in MONITORED],
        ignore_index=True,
    ).assign(as_of=str(end))
    store.write_pd(decisions, "monitoring", "retraining_decision")
    profiles = _refresh_profiles(store)
    return {
        "month": month,
        "scoring_log_rows": len(log),
        "event_performance": perf,
        "drift_rows": len(drift),
        "retraining_decision": decisions,
        "profiles": profiles,
    }
