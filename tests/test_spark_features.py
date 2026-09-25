"""Spark feature functions on small frames: points, regime tags, healthy windows, labels, controller
hourly features and rolling slopes."""

import datetime as dt

import numpy as np
import pandas as pd

from pdm.config import load_config
from pdm.features import controller, points, rolling, windows
from pdm.labels import confirmed_conditions

CFG = load_config()


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _sensors(spark):
    return spark.createDataFrame(
        [
            ("HMC-01-CP", "HMC-01-CP", "pump_motor_de", "6205", "2023-07-05", "", "GW"),
            ("CMP-01-AE", "CMP-01", "airend_inlet", "NU2210", "2023-07-07", "2025-10-06", "GW"),
        ],
        "point_id string, asset_id string, location string, bearing string, install_date string, "
        "remount_dates string, gateway string",
    )


def test_points_classes(spark):
    p = {r.point_id: r for r in points.points(_sensors(spark)).collect()}
    assert p["HMC-01-CP"].point_class == "flood_pump" and p["HMC-01-CP"].machine_id == "HMC-01"
    assert p["CMP-01-AE"].point_class == "compressor_airend" and p["CMP-01-AE"].machine_id == "CMP-01"


def _wo(spark):
    return spark.createDataFrame(
        [
            (
                "W1",
                "HMC-01-CP",
                "unplanned_repair",
                "PMP-BRG",
                _ts("2023-08-10 10:00:00"),
                _ts("2023-08-10 20:00:00"),
                10.0,
                8.0,
                "",
                None,
            ),
            (
                "W2",
                "HMC-01-CP",
                "inspection",
                None,
                _ts("2023-08-01 10:00:00"),
                _ts("2023-08-01 11:00:00"),
                0.0,
                1.0,
                "Inspected after platform alert: found lubrication.",
                "A1",
            ),
            (
                "W3",
                "CMP-01",
                "pm",
                "oil_cooler_fouled",
                _ts("2024-08-01 10:00:00"),
                _ts("2024-08-03 10:00:00"),
                0.0,
                1.0,
                "PM: oil_cooler_cleaning.",
                None,
            ),
        ],
        "wo_id string, asset_id string, wo_type string, failure_code string, opened timestamp, "
        "closed timestamp, downtime_hours double, labor_hours double, notes string, platform_alert_id string",
    )


def test_healthy_windows_exclusion(spark):
    w = windows.healthy_windows(points.points(_sensors(spark)), _wo(spark), CFG).toPandas()
    cp = w[w["point_id"] == "HMC-01-CP"].set_index("reason")
    assert not bool(cp.loc["install", "usable"])
    assert {"install", "remount"} <= set(w.loc[w["point_id"] == "CMP-01-AE", "reason"])


def test_labels(spark):
    c = confirmed_conditions(_wo(spark), CFG).toPandas().set_index("wo_id")
    assert c.loc["W1", "condition_type"] == "bearing (unplanned)" and c.loc["W1", "period"] == "training"
    assert c.loc["W2", "condition_type"] == "lubrication"
    assert c.loc["W3", "condition_type"] == "cooler (compressor)" and bool(c.loc["W3", "pm_finding"])


def test_controller_hourly_weighting(spark):
    rows = [
        ("M", _ts("2024-01-01 10:00:00"), "P", "active", 60, 10.0, 20.0, 6000.0, 30.0),
        ("M", _ts("2024-01-01 10:01:00"), "P", "active", 30, 40.0, 50.0, 6000.0, 30.0),
    ]
    schema = (
        "machine_id string, minute_utc timestamp, program string, execution string, seconds_in_state long, "
        "spindle_load_pct_mean double, spindle_load_pct_p95 double, spindle_speed_rpm_mean double, "
        "spindle_temp_c_mean double"
    )
    df = spark.createDataFrame(rows, schema)
    for c in (
        "coolant_pressure_bar_mean",
        "hydraulic_pressure_bar_mean",
        *[f"{a}_load_pct_p95" for a in "xyzb"],
    ):
        df = df.withColumn(c, df["spindle_load_pct_mean"] * 0 + 1.0)
    r = controller.controller_hourly(df).collect()[0]
    assert r.active_s == 90 and abs(r.spindle_load - 20.0) < 1e-9


def test_controller_hourly_pump_on_pressure(spark):
    # minute means: pump off (0), running at 69, a partial minute (20), running at 68.9
    rows = [
        ("M", _ts(f"2024-01-01 10:0{i}:00"), "P", "active", 60, p)
        for i, p in enumerate([0.0, 69.0, 20.0, 68.9, 0.0])
    ]
    schema = (
        "machine_id string, minute_utc timestamp, program string, execution string, seconds_in_state long, "
        "coolant_pressure_bar_mean double"
    )
    df = spark.createDataFrame(rows, schema)
    for c in (
        "spindle_load_pct_mean",
        "spindle_load_pct_p95",
        "spindle_speed_rpm_mean",
        "spindle_temp_c_mean",
        "hydraulic_pressure_bar_mean",
        *[f"{a}_load_pct_p95" for a in "xyzb"],
    ):
        df = df.withColumn(c, df["coolant_pressure_bar_mean"] * 0 + 1.0)
    r = controller.controller_hourly(df, 1.0).collect()[0]
    assert abs(r.coolant_pressure_bar - (69.0 + 20.0 + 68.9) / 5) < 1e-9
    assert abs(r.coolant_pressure_pump_on_bar - 68.9) < 1e-9


def test_rolling_slope_linear():
    t = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    pdf = pd.DataFrame({"point_id": "P", "hour_utc": t, "regime": "running", "implausible": False})
    for col in rolling.CORE_MAP.values():
        pdf[col] = 1.0
    pdf["vel_rms_mm_s"] = np.arange(len(t)) / 24.0
    out = rolling._point_rolling(
        pdf, pd.DataFrame(columns=["point_id", "window_id", "end", "indicator", "regime", "median"]), CFG
    )
    assert abs(out["vel_rms_mm_s_slope_7d"].iloc[-1] - 1.0) < 1e-6
