"""Local parity run: the pipeline's bronze, silver and gold tables and every job task, in dependency
order, on the local exports, writing Delta tables to exports/local_gold/<schema>/<table>.

python -m pdm.parity [--skip-bronze] [--only STAGE ...]

The cloud pipeline and jobs must reproduce these tables (row counts exact, numeric columns within the
parity tolerance). Writes a JSON summary of row counts, expectation results and task results.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pandas as pd

from pdm import log, paths, tasks
from pdm.config import load_config
from pdm.features import baselines, controller, indicators, points, rolling, windows
from pdm.ingest.bronze import BRONZE_TABLES, read_batch
from pdm.labels import confirmed_conditions
from pdm.quality import expectations, views
from pdm.rules.floor import rule_floor_alerts, rule_floor_exceedances
from pdm.spark import local_session
from pdm.store import Store

LOG = log.get_logger("pdm.parity")
SKIP_BRONZE_COPY = {"ctrl_hz1"}
MODEL_STAGES = ("nbm", "controller_health", "survival", "supervised", "prognosis")
STAGES = ("bronze", "silver", "gold", *MODEL_STAGES, "fusion", "evaluation", "monitoring")


def run(stages: tuple[str, ...] = STAGES) -> dict[str, Any]:
    log.configure()
    cfg = load_config()
    spark = local_session()
    store = Store(spark, local_root=paths.LOCAL_GOLD_DIR)
    root = paths.BRONZE_DIR.as_posix()
    out = paths.LOCAL_GOLD_DIR / "parity_summary.json"
    summary: dict[str, Any] = {"rows": {}, "expectations": {}, "tasks": {}, "seconds": {}}
    if out.exists():
        summary.update(json.loads(out.read_text(encoding="utf-8")))
    summary["rows"] = {}

    def stage(name: str):
        t0 = time.perf_counter()
        LOG.info("stage_start stage=%s", name)
        return t0

    def done(name: str, t0: float) -> None:
        summary["seconds"][name] = round(time.perf_counter() - t0, 1)
        LOG.info("stage_done stage=%s seconds=%.0f", name, summary["seconds"][name])

    if "bronze" in stages:
        t0 = stage("bronze")
        for t in BRONZE_TABLES:
            if t.name in SKIP_BRONZE_COPY:
                continue
            summary["rows"][f"bronze.{t.name}"] = store.write(
                read_batch(spark, root, t.name), "bronze", t.name
            )
        exp = expectations.expectations(cfg)
        for name, rules in exp.items():
            if store.exists("bronze", name):
                summary["expectations"][name] = expectations.evaluate(store.read("bronze", name), rules)
        summary["expectations"]["cm_scalars_duplicate_point_hour_axis"] = expectations.duplicate_keys(
            store.read("bronze", "cm_scalars"), ["point_id", "hour_utc", "axis"]
        )
        done("bronze", t0)

    if "silver" in stages:
        t0 = stage("silver")
        pts = points.points(store.read("bronze", "reg_sensors"))
        summary["rows"]["silver.points"] = store.write(pts, "silver", "points")
        pts = store.read("silver", "points")
        ind = indicators.hourly_indicators(store.read("bronze", "cm_scalars"), pts, cfg)
        summary["rows"]["silver.hourly_indicators"] = store.write(ind, "silver", "hourly_indicators")
        ind = store.read("silver", "hourly_indicators")
        summary["expectations"]["hourly_indicators"] = expectations.evaluate(
            ind, expectations.expectations(cfg)["hourly_indicators"]
        )
        wo = store.read("bronze", "cmms_work_orders")
        summary["rows"]["silver.healthy_windows"] = store.write(
            windows.healthy_windows(pts, wo, cfg), "silver", "healthy_windows"
        )
        summary["rows"]["silver.controller_hourly"] = store.write(
            controller.controller_hourly(store.read("bronze", "ctrl_minute")), "silver", "controller_hourly"
        )
        summary["rows"]["silver.controller_event_daily"] = store.write(
            controller.controller_event_daily(store.read("bronze", "ctrl_events")),
            "silver",
            "controller_event_daily",
        )
        summary["rows"]["silver.confirmed_conditions"] = store.write(
            confirmed_conditions(wo, cfg), "silver", "confirmed_conditions"
        )
        summary["rows"]["silver.gap_table"] = store.write(
            views.gap_table(store.read("bronze", "cm_sensor_events")), "silver", "gap_table"
        )
        summary["rows"]["silver.measurement_state_validation"] = store.write(
            views.measurement_state_validation(
                ind,
                store.read("silver", "controller_hourly"),
                cfg["measurement"]["state_rule"]["spindle_min_rpm"],
            ),
            "silver",
            "measurement_state_validation",
        )
        summary["rows"]["monitoring.data_quality_monthly"] = store.write(
            views.monthly_dq(store.read("bronze", "cm_scalars"), ind), "monitoring", "data_quality_monthly"
        )
        hz = views.hz1_to_minute(read_batch(spark, root, "ctrl_hz1"))
        summary["rows"]["silver.hz1_reconciliation"] = store.write(
            views.reconcile_minutes(hz, store.read("bronze", "ctrl_minute"), 10.0**-3),
            "silver",
            "hz1_reconciliation",
        )
        done("silver", t0)

    if "gold" in stages:
        t0 = stage("gold")
        pts = store.read("silver", "points")
        ind = store.read("silver", "hourly_indicators")
        win = store.read("silver", "healthy_windows")
        iw = windows.in_windows(ind, win)
        summary["rows"]["gold.baseline_stats"] = store.write(
            baselines.window_stats(iw), "gold", "baseline_stats"
        )
        end = tasks.horizon(cfg).tz_convert("UTC").tz_localize(None)
        flags = baselines.sanity_table(iw.where(iw.hour_utc < end), pts, win.where(win.start < end), cfg)
        summary["rows"]["gold.baseline_sanity"] = store.write(flags, "gold", "baseline_sanity")
        f0 = store.read("gold", "baseline_sanity").limit(1).toPandas()
        summary["tasks"]["sanity_factors"] = (
            f0[["sister_factor", "trend_factor", "proposed_sister", "proposed_trend"]].iloc[0].to_dict()
            if len(f0)
            else {}
        )
        base = rolling.baseline_with_ends(store.read("gold", "baseline_stats"), win)
        summary["rows"]["silver.rolling_features"] = store.write(
            rolling.rolling_features(ind, base, cfg), "silver", "rolling_features"
        )
        se = store.read("bronze", "cm_sensor_events")
        summary["rows"]["gold.rule_floor_alerts"] = store.write(
            rule_floor_alerts(store.read("bronze", "cm_scalars"), ind, pts, se, cfg),
            "gold",
            "rule_floor_alerts",
        )
        summary["rows"]["gold.rule_floor_exceedances"] = store.write(
            rule_floor_exceedances(
                store.read("bronze", "cm_scalars"),
                ind,
                pts,
                se,
                cfg,
                cfg["evaluation"]["vendor_sweep_alert_frac"],
                tasks.horizon(cfg),
            ),
            "gold",
            "rule_floor_exceedances",
        )
        done("gold", t0)

    model_tasks = {
        "nbm": tasks.nbm_task,
        "controller_health": tasks.controller_health_task,
        "survival": tasks.survival_task,
        "supervised": tasks.supervised_task,
        "prognosis": tasks.prognosis_task,
    }
    for name, fn in model_tasks.items():
        if name in stages:
            t0 = stage(name)
            summary["tasks"][name] = _jsonable(fn(store, cfg))
            done(name, t0)

    if "fusion" in stages:
        t0 = stage("fusion")
        summary["tasks"]["fusion"] = tasks.fusion_advisory_task(store, cfg)
        done("fusion", t0)

    if "evaluation" in stages:
        t0 = stage("evaluation")
        r = tasks.evaluation_task(store, cfg)
        summary["tasks"]["evaluation_validation"] = _jsonable(
            {k: r[k] for k in ("matched", "headline", "conditions", "sensor_conditions")}
        )
        done("evaluation", t0)

    if "monitoring" in stages:
        t0 = stage("monitoring")
        r = tasks.monitoring_task(store, cfg)
        summary["tasks"]["monitoring"] = _jsonable(r)
        done("monitoring", t0)

    for schema in ("bronze", "silver", "gold", "monitoring"):
        d = paths.LOCAL_GOLD_DIR / schema
        if d.exists():
            for t in sorted(p.name for p in d.iterdir() if (p / "_delta_log").exists()):
                summary["rows"].setdefault(f"{schema}.{t}", store.read(schema, t).count())
    out.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    spark.stop()
    return summary


def _jsonable(x: Any) -> Any:
    if isinstance(x, pd.DataFrame):
        return x.to_dict("records")
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m pdm.parity")
    p.add_argument("--only", nargs="*", choices=(*STAGES, "models"))
    a = p.parse_args()
    only = tuple(a.only) if a.only else STAGES
    run(tuple(s for x in only for s in (MODEL_STAGES if x == "models" else (x,))))


if __name__ == "__main__":
    main()
