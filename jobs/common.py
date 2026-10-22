"""Shared arguments and setup for job task entry points: catalog, config path and MLflow experiment."""

from __future__ import annotations

import argparse
from typing import Any

from pyspark.sql import SparkSession

from pdm import log
from pdm.config import load_config
from pdm.store import Store


def setup(prog: str) -> tuple[Store, dict[str, Any], argparse.Namespace]:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--catalog", required=True)
    p.add_argument("--config", required=True, help="path to config.yaml in the deployed bundle files")
    p.add_argument("--experiment-root", default="/Shared/pdm")
    args = p.parse_args()
    log.configure()
    spark = SparkSession.builder.getOrCreate()
    return Store(spark, catalog=args.catalog), load_config(args.config), args


def register(run_id: str | None, artifact: str, name: str, alias: str = "champion") -> None:
    """Register a logged model in Unity Catalog and set its alias."""
    if run_id is None:
        return
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    mv = mlflow.register_model(f"runs:/{run_id}/{artifact}", name)
    MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(name, alias, mv.version)
