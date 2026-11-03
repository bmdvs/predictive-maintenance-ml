"""Console entry points for the monthly job's Python wheel tasks: train_models, promote, score and monitor.

Each reads --catalog and --config (config.yaml in the workspace); train and promote also read
--experiment-root (MLflow experiments, by deployment target) and --job-run-id (the job run, which links
the challenger run to its promotion). Options accept hyphens or underscores, so the job's keyword-argument
form (--name=value) works. Reading stops at the shadow-quarter start while evaluation.holdout_shadow is
true in the config.
"""

from __future__ import annotations

import argparse
from typing import Any

from pdm import log, ml_tasks, registry, tasks
from pdm.config import load_config
from pdm.store import Store

LOG = log.get_logger("pdm.entrypoints")


def _flag(p: argparse.ArgumentParser, name: str, **kw: Any) -> None:
    p.add_argument(f"--{name}", f"--{name.replace('-', '_')}", dest=name.replace("-", "_"), **kw)


def _bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {v!r}")


def parse(prog: str, argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=prog)
    _flag(p, "catalog", required=True)
    _flag(p, "config", required=True, help="path to config.yaml in the workspace")
    _flag(p, "experiment-root", default="/Shared/pdm")
    _flag(p, "job-run-id", default=None)
    _flag(p, "run-id", default=None, help="promote: the challenger run, outside a job")
    _flag(p, "auto-promote", type=_bool, default=True)
    _flag(p, "as-of", default=None, help="shadow-quarter month YYYY-MM; the data horizon is its end")
    _flag(p, "train-stages", default="all", help="train: comma-separated training stages, or all")
    _flag(p, "monitor-stages", default="all", help="monitor: evaluation, monitoring, or all")
    _flag(
        p, "finalist", default="", help="0 or 1: train, register and score that finalist (empty: config grid)"
    )
    return p.parse_args(argv)


def setup(prog: str, argv: list[str] | None = None) -> tuple[Store, dict[str, Any], argparse.Namespace]:
    from pyspark.sql import SparkSession

    args = parse(prog, argv)
    log.configure()
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    cfg = load_config(args.config)
    if args.as_of:
        cfg["evaluation"]["as_of"] = args.as_of
    args.alias = registry.CHAMPION
    if str(args.finalist).strip() != "":
        args.alias = ml_tasks.apply_finalist(cfg, int(args.finalist))
        LOG.info(
            "finalist %s, alias %s, candidates %s",
            args.finalist,
            args.alias,
            cfg["models"]["nbm"]["candidates"],
        )
    LOG.info("data horizon %s (as_of %s)", tasks.horizon(cfg), args.as_of or "not set")
    return Store(spark, catalog=args.catalog), cfg, args


TRAIN_STAGES = ("rule_floor", "ml_model", "controller_health")


MONITOR_STAGES = ("evaluation", "monitoring")


def stages(value: str, known: tuple[str, ...]) -> list[str]:
    """The stages to run, in their fixed order: all, or a comma-separated subset of known."""
    if value.strip().lower() in ("", "all"):
        return list(known)
    asked = {s.strip() for s in value.split(",") if s.strip()}
    unknown = asked - set(known)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    return [s for s in known if s in asked]


def train_stages(value: str) -> list[str]:
    return stages(value, TRAIN_STAGES)


def train_models(argv: list[str] | None = None) -> None:
    """Rule-floor comparison runs, the ML Model's training run (expanding-window selection on two folds,
    the champion refit on all of training) and controller health; --train-stages runs a subset."""
    store, cfg, args = setup("pdm-train-models", argv)
    root = args.experiment_root.rstrip("/")
    stages = train_stages(args.train_stages)
    LOG.info("training stages %s", stages)
    if "rule_floor" in stages:
        tasks.rule_floor_task(store, cfg, f"{root}/rule_floor")
    if "ml_model" in stages:
        ml_tasks.train_task(store, cfg, f"{root}/ml_model", job_run_id=args.job_run_id)
    if "controller_health" in stages:
        tasks.controller_health_task(store, cfg, f"{root}/controller_health")


def promote(argv: list[str] | None = None) -> None:
    """Register the training run's champion, evaluate the gates once and move the champion alias when
    allowed."""
    store, cfg, args = setup("pdm-promote", argv)
    out = registry.promote(
        store,
        cfg,
        args.catalog,
        args.experiment_root,
        job_run_id=args.job_run_id,
        run_id=args.run_id,
        auto_promote=args.auto_promote,
        alias=args.alias,
    )
    log.get_logger("pdm.entrypoints").info("promote: %s", out)


def champion(catalog: str, alias: str = registry.CHAMPION) -> tuple[Any, str]:
    """The frozen ML Model from the Unity Catalog registry under the alias (champion, or challenger for the
    second finalist) and its version label."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    name = registry.model_name(catalog)
    v = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(name, alias).version
    model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}").unwrap_python_model()
    return model, f"{registry.MODEL}/{v}"


def score(argv: list[str] | None = None) -> None:
    """The shadow year scored with the frozen ML Model, then the levels of both methods
    (ALERT episodes and ALARM starts) with the ML Model's cases and work requests."""
    store, cfg, args = setup("pdm-score", argv)
    model, version = champion(args.catalog, args.alias)
    ml_tasks.score_task(store, cfg, model, version)
    ml_tasks.levels_task(store, cfg)


def monitor(argv: list[str] | None = None) -> None:
    """Evaluation, then monitoring over the twelve shadow-year months (skipped while the shadow year is held
    out); --monitor-stages runs one of them."""
    store, cfg, args = setup("pdm-monitor", argv)
    run = stages(args.monitor_stages, MONITOR_STAGES)
    LOG.info("monitor stages %s", run)
    if "evaluation" in run:
        ml_tasks.evaluation_task(store, cfg)
    if "monitoring" in run:
        ml_tasks.monitoring_task(store, cfg)
