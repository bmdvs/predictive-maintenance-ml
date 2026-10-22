"""Healthy windows per measurement point from the register and CMMS only, never from truth.

A window opens at install, at each sensor remount and at the close of each repair on the point's
asset, and lasts the configured number of days. Windows ending within the configured days before an
unplanned failure on the asset are excluded. Output: one row per window with its reason and use flag.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

LOCAL_TZ = "America/Chicago"
REPAIR_TYPES = ("unplanned_repair", "condition_based", "corrective", "pm")


def _local_midnight_utc(date_col: F.Column) -> F.Column:
    return F.to_utc_timestamp(F.to_timestamp(date_col), LOCAL_TZ)


def window_starts(pts: DataFrame, work_orders: DataFrame) -> DataFrame:
    install = pts.select(
        "point_id",
        "asset_id",
        _local_midnight_utc(F.col("install_date")).alias("start"),
        F.lit("install").alias("reason"),
    )
    remount = (
        pts.where(F.coalesce(F.col("remount_dates"), F.lit("")) != "")
        .select("point_id", "asset_id", F.explode(F.split("remount_dates", ";")).alias("d"))
        .select(
            "point_id",
            "asset_id",
            _local_midnight_utc(F.col("d")).alias("start"),
            F.lit("remount").alias("reason"),
        )
    )
    repairs = work_orders.where(F.col("wo_type").isin(*REPAIR_TYPES) & (F.col("downtime_hours") > 0)).select(
        F.col("asset_id"),
        F.col("closed").alias("start"),
        F.concat(F.lit("repair "), F.col("wo_id")).alias("reason"),
    )
    repair_rows = pts.select("point_id", "asset_id").join(repairs, "asset_id")
    return install.unionByName(remount).unionByName(
        repair_rows.select("point_id", "asset_id", "start", "reason")
    )


def healthy_windows(pts: DataFrame, work_orders: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    b = cfg["baselines"]
    days = b["healthy_after_install_days"]
    exclude_days = b["exclude_before_unplanned_failure_days"]
    starts = window_starts(pts, work_orders).withColumn(
        "end", F.col("start") + F.expr(f"INTERVAL {days} DAYS")
    )
    fails = work_orders.where(F.col("wo_type") == "unplanned_repair").select(
        F.col("asset_id").alias("f_asset"), F.col("opened").alias("failure_ts")
    )
    near = (
        starts.join(
            fails,
            (starts.asset_id == fails.f_asset)
            & (fails.failure_ts >= starts.end)
            & (fails.failure_ts <= starts.end + F.expr(f"INTERVAL {exclude_days} DAYS")),
            "left",
        )
        .groupBy("point_id", "start", "reason")
        .agg(F.min("failure_ts").alias("failure_within_exclusion"))
    )
    return (
        starts.join(near, ["point_id", "start", "reason"], "left")
        .withColumn("window_id", F.concat_ws("|", "point_id", F.date_format("start", "yyyy-MM-dd'T'HH:mm")))
        .withColumn("usable", F.col("failure_within_exclusion").isNull())
        .select(
            "window_id",
            "point_id",
            "asset_id",
            "reason",
            "start",
            "end",
            "usable",
            "failure_within_exclusion",
        )
    )


def in_windows(rows: DataFrame, windows: DataFrame, usable_only: bool = True) -> DataFrame:
    """Rows of the hourly table that fall inside a point's healthy window, tagged with the window id."""
    w = windows.where(F.col("usable")) if usable_only else windows
    w = w.select(
        F.col("point_id").alias("w_point"),
        "window_id",
        F.col("start").alias("w_start"),
        F.col("end").alias("w_end"),
    )
    j = rows.join(w, (rows.point_id == w.w_point) & (rows.hour_utc >= w.w_start) & (rows.hour_utc < w.w_end))
    return j.drop("w_point", "w_start", "w_end")
