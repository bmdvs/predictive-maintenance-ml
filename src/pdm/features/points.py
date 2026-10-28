"""Measurement point metadata from the register export: asset, machine, point class and sister group.

Input: the register sensors table (bronze). Output: one row per point. Sister points share a
component model and location (the four flood pumps, the four TSC pumps, the four front and the four
rear spindle points, the two compressor motor and the two airend points).
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def point_class_expr() -> F.Column:
    loc, asset = F.col("location"), F.col("asset_id")
    return (
        F.when(loc == "front_bearing_housing", "spindle_front")
        .when(loc == "rear_bearing_housing", "spindle_rear")
        .when((loc == "motor_de") & asset.startswith("CMP"), "compressor_motor")
        .when(loc == "airend_inlet", "compressor_airend")
        .when(asset.endswith("TSC"), "tsc_pump")
        .otherwise("flood_pump")
    )


def points(reg_sensors: DataFrame) -> DataFrame:
    return reg_sensors.select(
        "point_id",
        "asset_id",
        F.regexp_extract("asset_id", r"^([A-Z]+-\d+)", 1).alias("machine_id"),
        point_class_expr().alias("point_class"),
        "location",
        F.col("bearing").cast("string").alias("bearing"),
        F.to_date("install_date").alias("install_date"),
        "remount_dates",
        "gateway",
    ).withColumn("sister_group", F.col("point_class"))
