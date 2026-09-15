"""Quality views: gap table per point, measurement-state validation against the controller, the
monthly data-quality table per point, and the 1 Hz to per-minute reconciliation for the shadow quarter.

Inputs: bronze platform scalars and sensor events, controller minute aggregates and 1 Hz rows, and the
hourly indicator table. Outputs: Spark DataFrames for silver and monitoring.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pdm.ingest.mtconnect_stream import LOAD_COLUMNS, MEAN_COLUMNS

LOCAL_TZ = "America/Chicago"


def gap_table(sensor_events: DataFrame) -> DataFrame:
    """Dropouts and battery gaps per point with start, end and hours."""
    drop = sensor_events.where(F.col("event") == "dropout").select(
        "point_id",
        F.lit("dropout").alias("kind"),
        F.col("ts").alias("start"),
        F.to_timestamp("detail").alias("end"),
    )
    bat_start = sensor_events.where(F.col("event") == "battery_gap_start").select(
        "point_id", F.col("ts").alias("start")
    )
    bat_end = sensor_events.where(F.col("event") == "battery_change").select(
        "point_id", F.col("ts").alias("end")
    )
    bat = (
        bat_start.join(bat_end, "point_id")
        .where(F.col("end") > F.col("start"))
        .groupBy("point_id", "start")
        .agg(F.min("end").alias("end"))
        .select("point_id", F.lit("battery_gap").alias("kind"), "start", "end")
    )
    return drop.unionByName(bat).withColumn(
        "hours", (F.col("end").cast("long") - F.col("start").cast("long")) / 3600.0
    )


def measurement_state_validation(indicators: DataFrame, ctrl_hourly: DataFrame, min_rpm: float) -> DataFrame:
    """Spindle point-hours measured while the controller shows no cutting above the state speed."""
    sp = indicators.where(F.col("point_class").startswith("spindle")).select(
        "point_id", "machine_id", "hour_utc"
    )
    ctrl = ctrl_hourly.groupBy("machine_id", "hour_utc").agg(F.max("spindle_speed_rpm").alias("max_speed"))
    return (
        sp.join(ctrl, ["machine_id", "hour_utc"], "left")
        .withColumn("controller_supports", F.coalesce(F.col("max_speed") >= F.lit(min_rpm), F.lit(False)))
        .groupBy("point_id")
        .agg(
            F.count("*").alias("measured_hours"),
            F.sum((~F.col("controller_supports")).cast("int")).alias("unsupported_hours"),
        )
    )


def monthly_dq(scalars: DataFrame, indicators: DataFrame) -> DataFrame:
    res = scalars.where(F.col("axis") == "res").withColumn(
        "month", F.date_format(F.from_utc_timestamp("hour_utc", LOCAL_TZ), "yyyy-MM")
    )
    sched = res.groupBy("point_id", "month").agg(
        F.count("*").alias("scheduled_hours"),
        F.sum(F.col("no_measurement").cast("int")).alias("no_measurement_hours"),
    )
    imp = (
        indicators.withColumn("month", F.date_format(F.from_utc_timestamp("hour_utc", LOCAL_TZ), "yyyy-MM"))
        .groupBy("point_id", "month")
        .agg(
            F.count("*").alias("measured_hours"),
            F.sum(F.col("implausible").cast("int")).alias("implausible_rows"),
        )
    )
    return sched.join(imp, ["point_id", "month"], "left").withColumn(
        "no_measurement_share", F.col("no_measurement_hours") / F.col("scheduled_hours")
    )


def hz1_to_minute(hz1: DataFrame) -> DataFrame:
    df = hz1.withColumn("minute_utc", F.date_trunc("minute", "ts_utc"))
    keys = ["machine_id", "minute_utc", "program", "execution"]
    aggs = [F.count("*").alias("seconds_in_state")]
    for c in LOAD_COLUMNS:
        aggs += [
            F.mean(c).alias(f"{c}_mean"),
            F.max(c).alias(f"{c}_max"),
            F.percentile_approx(c, 0.95, 10000).alias(f"{c}_p95"),
        ]
    for c in MEAN_COLUMNS:
        aggs.append(F.mean(c).alias(f"{c}_mean"))
    return df.groupBy(*keys).agg(*aggs)


def reconcile_minutes(recomputed: DataFrame, stored: DataFrame, tolerance: float) -> DataFrame:
    """Per machine: matched minutes, unmatched minutes and the largest difference in mean loads."""
    keys = ["machine_id", "minute_utc", "program", "execution"]
    bounds = recomputed.agg(F.min("minute_utc").alias("a"), F.max("minute_utc").alias("b")).first()
    s = stored.where(F.col("minute_utc").between(bounds["a"], bounds["b"]))
    s = s.where(
        F.col("machine_id").isin([r[0] for r in recomputed.select("machine_id").distinct().collect()])
    )
    j = recomputed.alias("r").join(s.alias("s"), keys, "full_outer")
    diffs = [
        F.abs(F.col(f"r.{c}_mean") - F.col(f"s.{c}_mean"))
        for c in ("spindle_load_pct", "x_load_pct", "spindle_temp_c")
    ]
    return (
        j.withColumn(
            "matched", F.col("r.seconds_in_state").isNotNull() & F.col("s.seconds_in_state").isNotNull()
        )
        .withColumn("max_diff", F.greatest(*diffs))
        .groupBy("machine_id")
        .agg(
            F.sum(F.col("matched").cast("int")).alias("minutes_matched"),
            F.sum((~F.col("matched")).cast("int")).alias("minutes_unmatched"),
            F.max("max_diff").alias("max_mean_diff"),
        )
        .withColumn("within_tolerance", F.col("max_mean_diff") <= F.lit(tolerance))
    )
