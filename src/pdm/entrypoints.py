"""Console entry points for the monthly job's Python wheel tasks: train_models, score and monitor.

Each reads --catalog, --config (config.yaml in the workspace) and --experiment-root (MLflow experiments,
by deployment target), opens a Store on the Unity Catalog catalog and runs the task bodies in pdm.tasks.
Reading stops at the shadow-quarter start while evaluation.holdout_shadow is true in the config.
"""

from __future__ import annotations

import argparse
from typing import Any

from pdm import log, tasks
from pdm.config import load_config
from pdm.store import Store


def setup(prog: str, argv: list[str] | None = None) -> tuple[Store, dict[str, Any], argparse.Namespace]:
    from pyspark.sql import SparkSession

    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--catalog", required=True)
    p.add_argument("--config", required=True, help="path to config.yaml in the workspace")
    p.add_argument("--experiment-root", default="/Shared/pdm")
    args = p.parse_args(argv)
    log.configure()
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    return Store(spark, catalog=args.catalog), load_config(args.config), args


def train_models(argv: list[str] | None = None) -> None:
    """Rule-floor comparison runs, per-point models, controller health, survival and the supervised gate."""
    store, cfg, args = setup("pdm-train-models", argv)
    root = args.experiment_root.rstrip("/")
    tasks.rule_floor_task(store, cfg, f"{root}/rule_floor")
    tasks.nbm_task(store, cfg, f"{root}/nbm")
    tasks.controller_health_task(store, cfg, f"{root}/controller_health")
    tasks.survival_task(store, cfg, f"{root}/survival")
    tasks.supervised_task(store, cfg)


def score(argv: list[str] | None = None) -> None:
    """Prognosis, fusion and advisory over the scored record."""
    store, cfg, _ = setup("pdm-score", argv)
    tasks.prognosis_task(store, cfg)
    tasks.fusion_advisory_task(store, cfg)


def monitor(argv: list[str] | None = None) -> None:
    """Validation evaluation, then monitoring (skipped while the shadow quarter is held out)."""
    store, cfg, _ = setup("pdm-monitor", argv)
    tasks.evaluation_task(store, cfg)
    tasks.monitoring_task(store, cfg)


def register(run_id: str | None, artifact: str, name: str, alias: str = "champion") -> None:
    """Register a logged model in Unity Catalog and set its alias."""
    if run_id is None:
        return
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    mv = mlflow.register_model(f"runs:/{run_id}/{artifact}", name)
    MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(name, alias, mv.version)
