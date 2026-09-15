"""Job task bodies: train and score the per-point models and controller health, survival, the
supervised gate, prognosis, fusion, advisory, evaluation and monitoring.

Each task reads its inputs from the Store and writes its output tables back; job entry points under
jobs/ and the local parity runner call these functions. MLflow logging is optional (tracking off in
tests).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from pdm import labels as _labels  # noqa: F401
from pdm import log, splits
from pdm.advisory.requests import CONDITION_FAMILY, work_requests
from pdm.evaluation import events as ev
from pdm.fusion.status import cases as make_cases
from pdm.fusion.status import status_changes
from pdm.models import controller_health as ch
from pdm.models import nbm, prognosis, supervised, survival
from pdm.monitoring import performance as perf
from pdm.store import Store

LOG = log.get_logger("pdm.tasks")
LOCAL_TZ = "America/Chicago"


def _utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


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
            {
                k: (v if isinstance(v, (int, float, str)) else json.dumps(v, default=str))[:500]
                if isinstance(v, str)
                else v
                for k, v in params.items()
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})
        mlflow.set_tags(tags)
        return run.info.run_id


# ---------------------------------------------------------------- per-point models
def nbm_task(store: Store, cfg: dict[str, Any], experiment: str | None = None) -> dict[str, Any]:
    ind = store.read_pd("silver", "hourly_indicators")
    ind["hour_utc"] = _utc(ind["hour_utc"])
    win = store.read_pd("silver", "healthy_windows")
    for c in ("start", "end"):
        win[c] = _utc(win[c])
    flags = store.read_pd("gold", "baseline_sanity")
    flagged = set(flags.loc[flags["flagged"].astype(bool), "window_id"])
    conds = _conditions(store, cfg)
    cutoff = splits.training_cutoff(cfg)
    m = cfg["models"]["nbm"]
    rules = cfg["rules"]
    usable = win[win["usable"].astype(bool) & ~win["window_id"].isin(flagged)].sort_values("end")
    all_scores: dict[str, list[pd.DataFrame]] = {c: [] for c in nbm.CANDIDATES}
    trained = []
    for pid, rows in ind.groupby("point_id"):
        rows = rows.sort_values("hour_utc").reset_index(drop=True)
        wins = usable[usable["point_id"] == pid]
        for k, w in enumerate(wins.itertuples()):
            nxt = (
                wins["end"].iloc[k + 1]
                if k + 1 < len(wins)
                else rows["hour_utc"].max() + pd.Timedelta(hours=1)
            )
            train = rows[(rows["hour_utc"] >= w.start) & (rows["hour_utc"] < w.end)]
            score_rows = rows[(rows["hour_utc"] >= w.end) & (rows["hour_utc"] < nxt)]
            if len(train) < 50 or score_rows.empty:
                continue
            for cand in nbm.CANDIDATES:
                models = nbm.fit_point_window(train, cand, cfg, (w.start, w.end))
                s = nbm.score_rows(score_rows, models, m["top_contributors"])
                if not s.empty:
                    s["window_id"] = w.window_id
                    all_scores[cand].append(s)
                if cand == nbm.CANDIDATES[0]:
                    trained.append(
                        {
                            "point_id": pid,
                            "window_id": w.window_id,
                            "models": ",".join(models),
                            "n_train": len(train),
                        }
                    )
    sel_rows = []
    cand_alerts = {}
    for cand, parts in all_scores.items():
        s = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        a = (
            nbm.alerts_from_scores(s, rules["persistence_consecutive"], rules["suppression_days"])
            if len(s)
            else pd.DataFrame(columns=["point_id", "alert_ts"])
        )
        a["alert_ts"] = _utc(a["alert_ts"]) if len(a) else a["alert_ts"]
        cand_alerts[cand] = (s, a)
        train_conds = conds[(conds["credit_ts"] < cutoff) & conds["instrumented"]]
        alerts = _point_alerts_to_assets(a.rename(columns={"alert_ts": "ts"}), ind).assign(method=cand)
        caught = ev.caught_table(alerts, train_conds, [cand], cfg["evaluation"]["detection_lookback_days"])
        n_train_alerts = int((alerts["ts"] < cutoff).sum())
        sel_rows.append(
            {
                "candidate": cand,
                "training_recall": float(caught["caught"].mean()) if len(caught) else np.nan,
                "training_alerts": n_train_alerts,
            }
        )
    sel = (
        pd.DataFrame(sel_rows)
        .sort_values(["training_recall", "training_alerts"], ascending=[False, True])
        .reset_index(drop=True)
    )
    champion = sel["candidate"].iat[0]
    scores, alerts = cand_alerts[champion]
    version = model_version("nbm", {"champion": champion, "trained": trained})
    scores = scores.assign(model_version=version, candidate=champion)
    alerts = alerts.assign(model_version=version, candidate=champion)
    run = _mlflow_log(
        experiment,
        {"champion": champion, "windows_trained": len(trained)},
        {f"training_recall_{r.candidate}": r.training_recall for r in sel.itertuples()},
        {"name": "nbm", "model_version": version},
    )
    store.write_pd(scores, "gold", "nbm_scores")
    store.write_pd(alerts, "gold", "nbm_alerts")
    store.write_pd(
        sel.assign(champion=sel["candidate"] == champion, model_version=version), "gold", "nbm_candidates"
    )
    store.write_pd(pd.DataFrame(trained).assign(model_version=version), "gold", "nbm_models")
    return {
        "champion": champion,
        "selection": sel,
        "version": version,
        "mlflow_run": run,
        "windows_trained": len(trained),
    }


def _point_alerts_to_assets(alerts: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    pts = ind[["point_id", "asset_id", "machine_id"]].drop_duplicates()
    return alerts.merge(pts, on="point_id", how="left")


# ---------------------------------------------------------------- controller health
def controller_health_task(
    store: Store, cfg: dict[str, Any], experiment: str | None = None
) -> dict[str, Any]:
    h = store.read_pd("silver", "controller_hourly")
    machines = store.read_pd("bronze", "reg_machines")
    rated = machines.set_index("machine_id")["rated_rpm"].astype(float).to_dict()
    train_end = splits.training_cutoff(cfg).tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    scores, models = [], {}
    for mid, g in h.groupby("machine_id"):
        s, model = ch.daily_scores(g.sort_values("hour_utc").reset_index(drop=True), rated[mid], train_end)
        scores.append(s)
        models[mid] = model.coef.tolist()
    sc = pd.concat(scores, ignore_index=True)
    al = ch.alerts(sc, cfg)
    version = model_version("controller_health", models)
    _mlflow_log(
        experiment,
        {"machines": len(models)},
        {"alerts": len(al)},
        {"name": "controller_health", "model_version": version},
    )
    store.write_pd(sc.assign(model_version=version), "gold", "controller_health_daily")
    store.write_pd(al.assign(model_version=version), "gold", "controller_health_alerts")
    return {"alerts": len(al), "version": version, "machines": len(models)}


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
        (b["train"][0], b["shadow"][1]),
    )
    res = survival.fit_models(lives)
    summary = pd.DataFrame(
        [
            {"metric": "lives", "value": res["n"]},
            {"metric": "events", "value": res["events"]},
            {"metric": "concordance_weibull_aft", "value": res["concordance_aft"]},
            {"metric": "concordance_cox", "value": res["concordance_cox"]},
            {"metric": "integrated_brier_uncensored_weights", "value": res["ibs"]},
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
        {"concordance_aft": res["concordance_aft"], "concordance_cox": res["concordance_cox"]},
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
    return {
        "lives": res["n"],
        "events": res["events"],
        "c_aft": res["concordance_aft"],
        "c_cox": res["concordance_cox"],
        "ibs": res["ibs"],
    }


def supervised_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    conds = _conditions(store, cfg)
    fails = conds[conds["unplanned"]]
    gate = supervised.training_gate(fails, splits.training_cutoff(cfg), cfg)
    store.write_pd(pd.DataFrame([gate]), "gold", "supervised_gate")
    return gate


def prognosis_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["models"]["prognosis"]
    ind = store.read_pd(
        "silver",
        "hourly_indicators",
        [
            "point_id",
            "asset_id",
            "point_class",
            "hour_utc",
            "env_bpfo_1_g2",
            "env_bpfi_1_g2",
            "env_bsf_1_g2",
            "implausible",
        ],
    )
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind = ind[~ind["implausible"].fillna(False).astype(bool) & ~ind["point_class"].str.startswith("spindle")]
    ind["env_fault_g2"] = ind[["env_bpfo_1_g2", "env_bpfi_1_g2", "env_bsf_1_g2"]].fillna(0.0).sum(axis=1)
    alerts = store.read_pd("gold", "rule_floor_alerts")
    alerts["alert_ts"] = _utc(alerts["alert_ts"])
    env_alerts = alerts[(alerts["method"] == "vendor") & alerts["rules"].str.contains("envelope")]
    conds = _conditions(store, cfg)
    fails = conds[conds["unplanned"] & conds["condition_type"].str.startswith("bearing")]
    insts = []
    lookback = pd.Timedelta(days=cfg["evaluation"]["detection_lookback_days"])
    pts_by_asset = ind.groupby("asset_id")["point_id"].unique().to_dict()
    for f in fails.itertuples():
        pts = pts_by_asset.get(f.asset_id)
        if pts is None:
            continue
        a = env_alerts[
            env_alerts["point_id"].isin(pts)
            & (env_alerts["alert_ts"] < f.credit_ts)
            & (env_alerts["alert_ts"] >= f.credit_ts - lookback)
        ]
        if a.empty:
            continue
        pid = a.sort_values("alert_ts")["point_id"].iat[0]
        det = a["alert_ts"].min()
        g = ind[(ind["point_id"] == pid) & (ind["hour_utc"] >= det) & (ind["hour_utc"] < f.credit_ts)]
        daily = g.set_index("hour_utc")["env_fault_g2"].resample("1D").median().dropna()
        insts.append((f.period, prognosis.Instance(f.asset_id, det.floor("D"), f.credit_ts, daily)))
    train = [i for per, i in insts if per == "training"]
    thr = prognosis.threshold_from_training(train)
    preds = [
        prognosis.predict(i, thr, [0.25, 0.5, 0.75]).assign(period=per)
        for per, i in insts
        if not np.isnan(thr)
    ]
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if len(pred):
        pred = prognosis.alpha_lambda(pred, p["alpha"])
    store.write_pd(pred if len(pred) else pd.DataFrame({"note": ["no instances"]}), "gold", "prognosis")
    val = pred[pred["period"] != "training"] if len(pred) else pred
    return {
        "instances": len(insts),
        "training_instances": len(train),
        "threshold": thr,
        "validation_predictions": len(val),
        "alpha_lambda_share": float(val["within_alpha"].mean()) if len(val) else np.nan,
        "interval_width_median_days": float(
            val.loc[np.isfinite(val["interval_width_days"]), "interval_width_days"].median()
        )
        if len(val)
        else np.nan,
    }


# ---------------------------------------------------------------- fusion, advisory
def signals(store: Store, cfg: dict[str, Any]) -> pd.DataFrame:
    pts = store.read_pd("silver", "points", ["point_id", "asset_id", "machine_id", "point_class"])
    rf = store.read_pd("gold", "rule_floor_alerts").merge(pts, on="point_id", how="left")
    rf = rf.rename(columns={"alert_ts": "ts"})[
        ["method", "asset_id", "machine_id", "point_id", "ts", "level", "rules"]
    ]
    na = store.read_pd("gold", "nbm_alerts").merge(pts, on="point_id", how="left")
    na = na.rename(columns={"alert_ts": "ts"}).assign(
        method="nbm", level="alarm", rules=lambda d: "nbm:" + d["top_contributors"].astype(str)
    )
    na = na[["method", "asset_id", "machine_id", "point_id", "ts", "level", "rules"]]
    ca = store.read_pd("gold", "controller_health_alerts")
    hmc = set(store.read_pd("bronze", "reg_machines").query("`class` == 'hmc'")["machine_id"])
    ca_ts = pd.to_datetime(ca["alert_date"]).dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC") + pd.Timedelta(
        hours=23
    )
    ca_asset = [
        f"{m}-SP" if (r == "thermistor_residual" and m in hmc) else m
        for m, r in zip(ca["machine_id"], ca["rule"], strict=True)
    ]
    cs = pd.DataFrame(
        {
            "method": "controller",
            "asset_id": ca_asset,
            "machine_id": ca["machine_id"],
            "point_id": None,
            "ts": ca_ts,
            "level": "alarm",
            "rules": ca["rule"],
        }
    )
    for part in (rf, na, cs):
        part["ts"] = _utc(part["ts"])
    out = pd.concat([rf, na, cs], ignore_index=True)
    return out.sort_values("ts").reset_index(drop=True)


def fusion_advisory_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    sig = signals(store, cfg)
    changes = status_changes(sig, cfg)
    wo = _wo(store)
    cs = make_cases(changes, sig, wo)
    feats, ctrl = _advisory_inputs(store, cs, cfg)
    pts = (
        store.read_pd("silver", "points", ["point_id", "point_class"])
        .set_index("point_id")["point_class"]
        .to_dict()
    )
    b = splits.boundaries(cfg)
    wr = work_requests(cs, feats, pts, ctrl, b["shadow"], cfg)
    store.write_pd(sig, "gold", "signals")
    store.write_pd(changes, "gold", "asset_status")
    store.write_pd(cs, "gold", "cases")
    store.write_pd(wr, "gold", "work_requests")
    return {"signals": len(sig), "status_changes": len(changes), "cases": len(cs), "work_requests": len(wr)}


def _advisory_inputs(store: Store, cs: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict, dict]:
    roll = store.read_pd("silver", "rolling_features")
    roll["hour_utc"] = _utc(roll["hour_utc"])
    ind = store.read_pd(
        "silver", "hourly_indicators", ["point_id", "hour_utc", *[f"ord_{o:02d}_g2" for o in range(3, 11)]]
    )
    ind["hour_utc"] = _utc(ind["hour_utc"])
    ind["ord_3_10"] = ind[[f"ord_{o:02d}_g2" for o in range(3, 11)]].sum(axis=1)
    daily = store.read_pd("gold", "controller_health_daily")
    calerts = store.read_pd("gold", "controller_health_alerts")
    ch_h = store.read_pd("silver", "controller_hourly", ["machine_id", "hour_utc", "coolant_pressure_bar"])
    ch_h["hour_utc"] = _utc(ch_h["hour_utc"])
    feats, ctrl = {}, {}
    for c in cs.itertuples():
        pts = [p for p in str(c.trigger_points or "").split(",") if p]
        t = pd.Timestamp(c.opened)
        f: dict[str, float] = {}
        if pts:
            r = roll[(roll["point_id"] == pts[0]) & (roll["hour_utc"] <= t)].tail(1)
            if len(r):
                f = {k: float(v) for k, v in r.iloc[0].items() if k.endswith("_24h") and pd.notna(v)}
            g = ind[ind["point_id"] == pts[0]]
            recent = g[(g["hour_utc"] <= t) & (g["hour_utc"] > t - pd.Timedelta(hours=24))][
                "ord_3_10"
            ].median()
            ref = g[g["hour_utc"] < t - pd.Timedelta(days=30)]["ord_3_10"].median()
            f["ord_3_10_ratio"] = float(recent / ref) if ref and ref > 0 else np.nan
        mid = str(c.asset_id).split("-")[0] + "-" + str(c.asset_id).split("-")[1]
        d = daily[
            (daily["machine_id"] == mid)
            & (pd.to_datetime(daily["local_date"]) <= t.tz_convert(LOCAL_TZ).tz_localize(None))
        ].tail(1)
        a = calerts[
            (calerts["machine_id"] == mid)
            & (pd.to_datetime(calerts["alert_date"]) <= t.tz_convert(LOCAL_TZ).tz_localize(None))
            & (
                pd.to_datetime(calerts["alert_date"])
                > t.tz_convert(LOCAL_TZ).tz_localize(None) - pd.Timedelta(days=7)
            )
        ]
        cp = ch_h[ch_h["machine_id"] == mid]
        now_p = cp[(cp["hour_utc"] <= t) & (cp["hour_utc"] > t - pd.Timedelta(days=3))][
            "coolant_pressure_bar"
        ].median()
        ref_p = cp[
            (cp["hour_utc"] <= t - pd.Timedelta(days=14)) & (cp["hour_utc"] > t - pd.Timedelta(days=44))
        ]["coolant_pressure_bar"].median()
        ctrl[c.case_id] = {
            "coolant_drop": float(1 - now_p / ref_p) if ref_p and ref_p > 0 and pd.notna(now_p) else 0.0,
            "thermistor_residual_alert": bool((a["rule"] == "thermistor_residual").any()),
            "axis_drift_alert": bool((a["rule"] == "axis_drift").any()),
            "thermistor_residual_c": float(d["thermistor_residual_c"].iat[0])
            if len(d) and "thermistor_residual_c" in d
            else np.nan,
            "axis_drift_max": float(d["axis_drift_max"].iat[0])
            if len(d) and "axis_drift_max" in d
            else np.nan,
        }
        feats[c.case_id] = f
    return feats, ctrl


# ---------------------------------------------------------------- evaluation and monitoring
def _wo(store: Store) -> pd.DataFrame:
    return _utc_cols(store.read_pd("bronze", "cmms_work_orders"), ["opened", "started", "closed"])


def _utc_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = _utc(df[c])
    return df


def _conditions(store: Store, cfg: dict[str, Any]) -> pd.DataFrame:
    c = store.read_pd("silver", "confirmed_conditions")
    c = _utc_cols(c, ["confirmed_ts", "closed"])
    wo = _wo(store).set_index("wo_id")
    pm_started = c["wo_id"].map(wo["started"])
    c["credit_ts"] = c["confirmed_ts"].where(~c["pm_finding"].astype(bool), pm_started)
    c["credit_ts"] = c["credit_ts"].fillna(c["confirmed_ts"])
    mclass = store.read_pd("bronze", "reg_machines").set_index("machine_id")["class"].to_dict()
    c["machine_class"] = c["machine_id"].map(mclass).fillna("compressor")
    c["instrumented"] = c["asset_id"].str.match(r"^(HMC-\d+-(SP|TSC|CP)|CMP-\d+)$")
    return c


def evaluation_task(
    store: Store,
    cfg: dict[str, Any],
    period: str,
    period_bounds: tuple[pd.Timestamp, pd.Timestamp],
    suffix: str,
) -> dict[str, Any]:
    sig = store.read_pd("gold", "signals")
    sig["ts"] = _utc(sig["ts"])
    cs = store.read_pd("gold", "cases")
    cs["opened"] = _utc(cs["opened"])
    model_layer = pd.DataFrame(
        {
            "method": "model_layer",
            "asset_id": cs["asset_id"],
            "machine_id": cs["asset_id"].str.extract(r"^([A-Z]+-\d+)")[0],
            "ts": cs["opened"],
        }
    )
    rule_floor = sig[sig["method"].isin(["vendor", "iso"])].assign(method="rule_floor")
    alerts = pd.concat([sig, rule_floor, model_layer], ignore_index=True)
    a0, a1 = period_bounds
    alerts_p = alerts[(alerts["ts"] >= a0) & (alerts["ts"] < a1)]
    conds = _conditions(store, cfg)
    conds_p = conds[(conds["credit_ts"] >= a0) & (conds["credit_ts"] < a1)]
    methods = ["vendor", "iso", "rule_floor", "nbm", "controller", "model_layer"]
    scope_sensor = conds_p[conds_p["instrumented"]]
    lookback = cfg["evaluation"]["detection_lookback_days"]
    caught = pd.concat(
        [
            ev.caught_table(
                alerts,
                scope_sensor if m in ("vendor", "iso", "rule_floor", "nbm") else conds_p,
                [m],
                lookback,
            )
            for m in methods
        ],
        ignore_index=True,
    )
    fa = ev.false_alerts(alerts_p, conds, methods, cfg["evaluation"]["alert_precision_window_days"])
    wo = _wo(store)
    insp_h = float(wo.loc[wo["wo_type"] == "inspection", "labor_hours"].median())
    margin = (
        store.read_pd("bronze", "erp_contribution_margin")
        .set_index("machine_class")["contribution_margin_per_hour_usd"]
        .to_dict()
    )
    n_instr = 14
    n_assets = {
        "vendor": n_instr,
        "iso": n_instr,
        "rule_floor": n_instr,
        "nbm": n_instr,
        "controller": 20,
        "model_layer": n_instr + 20,
    }
    months = ev.months_between(a0, a1)
    summ = ev.summary(
        caught,
        fa,
        n_assets,
        months,
        insp_h,
        margin,
        cfg["maintenance"]["planned_repair_lead_days"][0],
        lookback / (365.25 / 12),
    ).assign(period=period)
    bt = ev.by_type(caught).assign(period=period)
    wr = store.read_pd("gold", "work_requests")
    if len(wr):
        wr["created"] = _utc(wr["created"])
        wr = wr[(wr["created"] >= a0) & (wr["created"] < a1)]
    agree = (
        ev.diagnosis_agreement(wr, conds, CONDITION_FAMILY, cfg["evaluation"]["detection_lookback_days"])
        if len(wr)
        else pd.DataFrame()
    )
    store.write_pd(summ, "gold", f"evaluation_summary_{suffix}")
    store.write_pd(bt, "gold", f"evaluation_by_type_{suffix}")
    store.write_pd(caught, "gold", f"evaluation_events_{suffix}")
    store.write_pd(fa, "gold", f"evaluation_alerts_{suffix}")
    store.write_pd(
        agree if len(agree) else pd.DataFrame({"note": ["no work requests followed by a condition"]}),
        "gold",
        f"diagnosis_agreement_{suffix}",
    )
    return {
        "summary": summ,
        "by_type": bt,
        "agreement": agree,
        "inspection_hours_each": insp_h,
        "conditions": conds_p,
    }


def monitoring_task(store: Store, cfg: dict[str, Any]) -> dict[str, Any]:
    b = splits.boundaries(cfg)
    conds = _conditions(store, cfg)
    caught = pd.concat(
        [
            store.read_pd("gold", "evaluation_events_validation"),
            store.read_pd("gold", "evaluation_events_shadow"),
        ],
        ignore_index=True,
    )
    fa = pd.concat(
        [
            store.read_pd("gold", "evaluation_alerts_validation"),
            store.read_pd("gold", "evaluation_alerts_shadow"),
        ],
        ignore_index=True,
    )
    fa["ts"] = _utc(fa["ts"])
    credit = conds.set_index("wo_id")["credit_ts"]
    n_assets = {"vendor": 14, "iso": 14, "rule_floor": 14, "nbm": 14, "controller": 20, "model_layer": 34}
    ep = perf.event_performance(caught, fa, credit, n_assets)
    ind = store.read_pd(
        "silver",
        "hourly_indicators",
        ["point_id", "hour_utc", "vel_rms_mm_s", "hf_rms_g", "kurtosis", "crest_factor", "temp_c"],
    )
    ind["hour_utc"] = _utc(ind["hour_utc"])
    val_months = sorted(
        set(
            perf.month_of(
                pd.Series(
                    pd.date_range(b["validation"][0], b["validation"][1] - pd.Timedelta(days=1), freq="D")
                )
            )
        )
    )
    shadow_months = sorted(
        set(
            perf.month_of(
                pd.Series(pd.date_range(b["shadow"][0], b["shadow"][1] - pd.Timedelta(days=1), freq="D"))
            )
        )
    )
    drift = perf.feature_drift(
        ind, ["vel_rms_mm_s", "hf_rms_g", "kurtosis", "crest_factor", "temp_c"], val_months, shadow_months
    )
    val = store.read_pd("gold", "evaluation_summary_validation")
    ref = {
        r.method: {
            "recall": r.caught_share,
            "alerts_per_asset_month": r.alerts
            / max(n_assets.get(r.method, 1), 1)
            / max(ev.months_between(*b["validation"]), 1e-9),
        }
        for r in val.itertuples()
    }
    shadow_ep = ep[ep["month"].isin(shadow_months)]
    dec = pd.concat(
        [
            perf.retraining_decision(shadow_ep, drift, ref, cfg, m)
            for m in ("nbm", "controller", "model_layer")
        ],
        ignore_index=True,
    )
    store.write_pd(ep, "monitoring", "event_performance")
    store.write_pd(drift, "monitoring", "feature_drift")
    store.write_pd(dec, "monitoring", "retraining_decision")
    return {"event_performance": len(ep), "drift_rows": len(drift), "decisions": dec}
