# Databricks notebook source
# MAGIC %md
# MAGIC # Model training
# MAGIC
# MAGIC Runs the training entry points interactively, one cell per layer. Paths are relative to this
# MAGIC notebook's folder, which holds `dist/` (the pdm wheel) and `config/config.yaml`. Each logs to MLflow under
# MAGIC `/Shared/pdm/<layer>` with the split definition, the holdout cap, the source config hash and the
# MAGIC input Delta table versions. Reading stops at the shadow-quarter start (`holdout_shadow: true`).
# MAGIC Outputs go to `pdm.gold`. Nothing is registered here: the per-point model run tagged `challenger` is
# MAGIC registered as a version of `pdm.ml.nbm_point_models` by the job's promote task (`pdm-promote`), which
# MAGIC moves the `champion` alias when its gates pass.

# COMMAND ----------

# MAGIC %pip install ./dist/pdm-0.1.13-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import mlflow

import pdm
from pdm import splits, tasks
from pdm.config import load_config
from pdm.store import Store

CONFIG = "./config/config.yaml"
dbutils.widgets.text("experiment_root", "/Shared/pdm")  # noqa: F821
ROOT = dbutils.widgets.get("experiment_root")  # noqa: F821
spark.conf.set("spark.sql.session.timeZone", "UTC")  # noqa: F821
mlflow.autolog(disable=True)
cfg = load_config(CONFIG)
store = Store(spark, catalog="pdm")  # noqa: F821
print("experiment root", ROOT)
print("pdm", pdm.__file__)
print(
    "training cutoff",
    splits.training_cutoff(cfg),
    "| reading stops at",
    tasks.horizon(cfg),
    "| holdout",
    cfg["evaluation"]["holdout_shadow"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule-floor comparison runs (`/Shared/pdm/rule_floor`)
# MAGIC One run per vendor offset, for the vendor rules alone and with the ISO zone rule, plus the ISO rule
# MAGIC alone, on the training period. Same metric names as the per-point model runs.

# COMMAND ----------

rf = tasks.rule_floor_task(store, cfg, f"{ROOT}/rule_floor")
display(
    rf["curve"][
        [
            "method",
            "sweep",
            "caught",
            "conditions",
            "caught_share",
            "chance_share",
            "false_per_asset_month",
            "operating_point",
        ]
    ]
)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-point model candidates (`/Shared/pdm/nbm`)
# MAGIC One run per candidate (PCA T-squared and SPE, Mahalanobis, isolation forest, autoencoder), each with
# MAGIC one nested run per point and the candidate's per-point models, with their version histories and
# MAGIC healthy stats, logged as `model`. The run tagged `challenger = true` is the selection rule's pick.

# COMMAND ----------

nbm = tasks.nbm_task(store, cfg, f"{ROOT}/nbm")
display(nbm["selection"])  # noqa: F821
print("challenger:", nbm["challenger"], "| run:", nbm["mlflow_run"])
print("runs:", nbm["mlflow_runs"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Controller health (`/Shared/pdm/controller_health`)

# COMMAND ----------

chr_ = tasks.controller_health_task(store, cfg, f"{ROOT}/controller_health")
print(chr_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Survival (`/Shared/pdm/survival`)

# COMMAND ----------

sv = tasks.survival_task(store, cfg, f"{ROOT}/survival")
display(sv["shared_shape_weibull"])  # noqa: F821
