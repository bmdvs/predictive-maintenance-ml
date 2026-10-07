"""Job task bodies shared with the ML Model's tasks (pdm.ml_tasks): controller health, the rule-floor
comparison, survival, the supervised gate and prognosis; the conditions, context and advisory helpers;
the scoring log and the data profile refresh.

Each task reads its inputs from the Store and writes its output tables back; the job's console entry
points (pdm.entrypoints) and the local parity runner call these functions. MLflow logging is optional
(tracking off in tests). While the shadow quarter is held out, no task writes anything computed on it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from pdm import log, splits
from pdm.advisory import requests as adv
from pdm.evaluation import events as ev
from pdm.features.controller import compressor_hourly
from pdm.models import controller_health as ch
from pdm.models import prognosis, supervised, survival
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
        "split_folds": "; ".join(
            f"fold {f['fold']} fit to {f['cutoff']:%Y-%m-%d}, "
            f"scored {f['score'][0]:%Y-%m-%d} to {f['score'][1]:%Y-%m-%d}"
            for f in splits.folds(cfg)
        ),
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
            "training" if f.credit_ts < cutoff else ("shadow" if f.credit_ts >= b["shadow"][0] else "gap")
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
    for per in ("training", "shadow"):
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
    name = "point_segment_stats" if store.exists("gold", "point_segment_stats") else "point_healthy_stats"
    stats = _utc_cols(store.read_pd("gold", name), ["segment_start"])
    cols = ["point_id", "hour_utc", "regime", "implausible", *adv.Z_INDICATORS.values()]
    cols += [f"ord_{o:02d}_g2" for o in range(3, 11)]
    cols = [c for c in cols if c != "ord_3_10_g2"]
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
PROFILED_TABLES = (("gold", "scoring_log"), ("silver", "hourly_indicators"))


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
