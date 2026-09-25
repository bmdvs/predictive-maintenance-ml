"""Controller features per machine, hour and program from the per-minute aggregates, and daily
controller event counts.

Inputs: bronze controller minute aggregates and controller events. Outputs: hourly active-cutting
features (seconds active, spindle and axis load statistics, speed, thermistor, coolant and hydraulic
pressure, and the coolant pressure while the through-spindle coolant pump runs) and daily event counts
by type.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

AXES = ("x", "y", "z", "b")
LOCAL_TZ = "America/Chicago"


def _wmean(col: str) -> F.Column:
    w = F.col("seconds_in_state")
    return (F.sum(F.col(col) * w) / F.sum(F.when(F.col(col).isNotNull(), w))).alias(col.replace("_mean", ""))


def controller_hourly(minute: DataFrame, pump_on_min_bar: float = 1.0) -> DataFrame:
    """Per machine, hour and program over active cutting minutes. coolant_pressure_bar is the
    time-weighted mean, pump-off time included; coolant_pressure_pump_on_bar is the median of the
    minutes' mean pressure over minutes above pump_on_min_bar (a minute's mean is 0 or the running
    pressure, so this is the pressure while the pump runs)."""
    active = minute.where((F.col("execution") == "active") & (F.col("program") != ""))
    on = F.when(
        F.col("coolant_pressure_bar_mean") > F.lit(float(pump_on_min_bar)), F.col("coolant_pressure_bar_mean")
    )
    aggs = [
        F.sum("seconds_in_state").alias("active_s"),
        _wmean("spindle_load_pct_mean").alias("spindle_load"),
        (F.sum(F.col("spindle_load_pct_p95") * F.col("seconds_in_state")) / F.sum("seconds_in_state")).alias(
            "spindle_load_p95"
        ),
        _wmean("spindle_speed_rpm_mean"),
        _wmean("spindle_temp_c_mean"),
        _wmean("coolant_pressure_bar_mean"),
        F.percentile(on, F.lit(0.5)).alias("coolant_pressure_pump_on_bar"),
        _wmean("hydraulic_pressure_bar_mean"),
    ]
    for a in AXES:
        aggs.append(
            (
                F.sum(F.col(f"{a}_load_pct_p95") * F.col("seconds_in_state"))
                / F.sum(F.when(F.col(f"{a}_load_pct_p95").isNotNull(), F.col("seconds_in_state")))
            ).alias(f"{a}_load_p95")
        )
    return (
        active.withColumn("hour_utc", F.date_trunc("hour", "minute_utc"))
        .groupBy("machine_id", "hour_utc", "program")
        .agg(*aggs)
        .withColumn("local_date", F.to_date(F.from_utc_timestamp("hour_utc", LOCAL_TZ)))
    )


def controller_event_daily(events: DataFrame) -> DataFrame:
    return (
        events.withColumn("local_date", F.to_date(F.from_utc_timestamp("ts_utc", LOCAL_TZ)))
        .groupBy("machine_id", "local_date", "event_type")
        .agg(F.count("*").alias("events"), F.sum(F.col("stopped_machine").cast("int")).alias("stops"))
    )


def compressor_hourly(minute: DataFrame, room: DataFrame) -> DataFrame:
    """Per compressor and hour: minutes loaded and running, mean discharge temperature while loaded, and
    the compressor room ambient temperature."""
    hour = F.date_trunc("hour", F.col("minute_utc")).alias("hour_utc")
    m = minute.groupBy("compressor_id", hour).agg(
        F.sum(F.when(F.col("state") == "loaded", 1).otherwise(0)).alias("load_min"),
        F.sum(F.when(F.col("state") != "off", 1).otherwise(0)).alias("run_min"),
        F.avg(F.when(F.col("state") == "loaded", F.col("discharge_temp_c"))).alias("discharge_temp_c"),
    )
    r = room.groupBy(F.date_trunc("hour", F.col("minute_utc")).alias("hour_utc")).agg(
        F.avg("ambient_c").alias("ambient_c")
    )
    return m.join(r, "hour_utc", "left")
