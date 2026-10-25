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

from pdm import log, registry, tasks
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
    LOG.info("data horizon %s (as_of %s)", tasks.horizon(cfg), args.as_of or "not set")
    return Store(spark, catalog=args.catalog), cfg, args


def train_models(argv: list[str] | None = None) -> None:
    """Rule-floor comparison runs, per-point model challengers, controller health, survival and the
    supervised gate."""
    store, cfg, args = setup("pdm-train-models", argv)
    root = args.experiment_root.rstrip("/")
    tasks.rule_floor_task(store, cfg, f"{root}/rule_floor")
    tasks.nbm_task(store, cfg, f"{root}/nbm", job_run_id=args.job_run_id)
    tasks.controller_health_task(store, cfg, f"{root}/controller_health")
    tasks.survival_task(store, cfg, f"{root}/survival")
    tasks.supervised_task(store, cfg)


def promote(argv: list[str] | None = None) -> None:
    """Register the challenger, evaluate the gates and move the champion alias when allowed."""
    store, cfg, args = setup("pdm-promote", argv)
    out = registry.promote(
        store,
        cfg,
        args.catalog,
        args.experiment_root,
        job_run_id=args.job_run_id,
        run_id=args.run_id,
        auto_promote=args.auto_promote,
    )
    log.get_logger("pdm.entrypoints").info("promote: %s", out)


def champion(catalog: str) -> tuple[Any, str]:
    """The champion per-point models from the Unity Catalog registry and their version label."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    name = registry.model_name(catalog)
    v = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(name, registry.CHAMPION).version
    model = mlflow.pyfunc.load_model(f"models:/{name}@{registry.CHAMPION}").unwrap_python_model()
    return model, f"{registry.MODEL}/{v}"


def score(argv: list[str] | None = None) -> None:
    """Per-point model scores from the champion, then prognosis, fusion and advisory."""
    store, cfg, args = setup("pdm-score", argv)
    model, version = champion(args.catalog)
    tasks.nbm_score_task(store, cfg, model, version)
    tasks.prognosis_task(store, cfg)
    tasks.fusion_advisory_task(store, cfg)


def monitor(argv: list[str] | None = None) -> None:
    """Validation evaluation, then monitoring (skipped while the shadow quarter is held out)."""
    store, cfg, _ = setup("pdm-monitor", argv)
    tasks.evaluation_task(store, cfg)
    tasks.monitoring_task(store, cfg)
