"""Hourly indicator table: one row per measured point-hour with the resultant indicators, the largest
directional velocity, the implausible-row flag and the regime tag.

Inputs: bronze platform scalars, point metadata and the machine register. Output: the silver
regime-tagged hourly indicator table used by the rule floor, baselines, rolling features and models.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pdm.quality.implausible import flag_point

CORE_INDICATORS = ["accel_rms_g", "vel_rms_mm_s", "hf_rms_g", "crest_factor", "kurtosis", "temp_c"]
ORDER_COLUMNS = [f"ord_{o:02d}_g2" for o in range(1, 11)]
ENV_FIRST = ["env_bpfo_1_g2", "env_bpfi_1_g2", "env_bsf_1_g2", "env_ftf_1_g2"]


def regime_columns(cfg: dict[str, Any]) -> list[F.Column]:
    """Load band for spindle points (light, medium, heavy); a single running regime elsewhere."""
    lb = cfg["regime"]["load_bands_pct"]
    load = F.col("load_pct_at_measurement")
    band = F.when(load < lb["light"][1], "light").when(load >= lb["heavy"][0], "heavy").otherwise("medium")
    spindle = F.col("point_class").startswith("spindle")
    return [
        F.when(spindle, band).otherwise(F.lit("running")).alias("regime"),
        F.when(spindle, band).otherwise(F.lit(None)).alias("load_band"),
    ]


def implausible_flags(res: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    """Quality-layer flags per point, computed with the package's pandas rule on each point's rows."""

    def per_point(pdf: pd.DataFrame) -> pd.DataFrame:
        f = flag_point(pdf, cfg)
        return pd.DataFrame(
            {
                "point_id": pdf.loc[f.index, "point_id"],
                "hour_utc": f["hour_utc"],
                "implausible": f["implausible"],
                "implausible_rule": f["rule"],
            }
        )

    schema = "point_id string, hour_utc timestamp, implausible boolean, implausible_rule string"
    return res.groupBy("point_id").applyInPandas(per_point, schema)


def hourly_indicators(scalars: DataFrame, pts: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    measured = scalars.where(~F.col("no_measurement"))
    res = measured.where(F.col("axis") == "res")
    vel_dir = (
        measured.where(F.col("axis") != "res")
        .groupBy("point_id", "hour_utc")
        .agg(F.max("vel_rms_mm_s").alias("vel_max_dir_mm_s"))
    )
    env_cols = [c for c in res.columns if c.startswith("env_") and c.endswith("_g2")]
    toct_cols = [c for c in res.columns if c.startswith("toct_")]
    keep = [
        "point_id",
        "hour_utc",
        "speed_rpm_at_measurement",
        "load_pct_at_measurement",
        "peak_g",
        *CORE_INDICATORS,
        *ORDER_COLUMNS,
        *env_cols,
        *toct_cols,
        "run_id",
        "config_hash",
    ]
    flags = implausible_flags(res.select(*[c for c in res.columns if not c.startswith("_")]), cfg)
    return (
        res.select(*keep)
        .join(vel_dir, ["point_id", "hour_utc"], "left")
        .join(flags, ["point_id", "hour_utc"], "left")
        .join(pts.select("point_id", "asset_id", "machine_id", "point_class"), "point_id")
        .select("*", *regime_columns(cfg))
    )
