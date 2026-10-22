"""Silver materialized views: points, regime-tagged hourly indicators, healthy windows, controller
features, labels and the quality views."""

from pyspark import pipelines as dp

from pdm.config import load_config
from pdm.features import controller, indicators, points, windows
from pdm.labels import confirmed_conditions as _confirmed_conditions
from pdm.quality import views
from pdm.quality.expectations import expectations

CFG = load_config(spark.conf.get("pdm.config_path"))  # noqa: F821
IND_RULES = {n: e for n, e, a in expectations(CFG)["hourly_indicators"] if a == "drop"}


CATALOG = spark.conf.get("pdm.catalog")  # noqa: F821
SCHEMA = {
    "baseline_sanity": "gold",
    "baseline_stats": "gold",
    "cm_scalars": "bronze",
    "cm_sensor_events": "bronze",
    "cmms_work_orders": "bronze",
    "confirmed_conditions": "silver",
    "controller_event_daily": "silver",
    "controller_hourly": "silver",
    "ctrl_events": "bronze",
    "ctrl_hz1": "bronze",
    "ctrl_minute": "bronze",
    "data_quality_monthly": "monitoring",
    "gap_table": "silver",
    "healthy_windows": "silver",
    "hourly_indicators": "silver",
    "hz1_reconciliation": "silver",
    "measurement_state_validation": "silver",
    "points": "silver",
    "reg_sensors": "bronze",
    "rolling_features": "silver",
    "rule_floor_alerts": "gold",
}


def _name(name: str) -> str:
    return f"{CATALOG}.{SCHEMA.get(name, 'bronze')}.{name}"


def _read(name: str):
    return spark.read.table(_name(name))  # noqa: F821


@dp.materialized_view(name=_name("points"))
def points_mv():
    return points.points(_read("reg_sensors"))


@dp.materialized_view(name=_name("hourly_indicators"))
@dp.expect_all_or_drop(IND_RULES)
def hourly_indicators():
    return indicators.hourly_indicators(_read("cm_scalars"), _read("points"), CFG)


@dp.materialized_view(name=_name("healthy_windows"))
def healthy_windows():
    return windows.healthy_windows(_read("points"), _read("cmms_work_orders"), CFG)


@dp.materialized_view(name=_name("controller_hourly"))
def controller_hourly():
    return controller.controller_hourly(_read("ctrl_minute"))


@dp.materialized_view(name=_name("controller_event_daily"))
def controller_event_daily():
    return controller.controller_event_daily(_read("ctrl_events"))


@dp.materialized_view(name=_name("confirmed_conditions"))
def confirmed_conditions():
    return _confirmed_conditions(_read("cmms_work_orders"), CFG)


@dp.materialized_view(name=_name("gap_table"))
def gap_table():
    return views.gap_table(_read("cm_sensor_events"))


@dp.materialized_view(name=_name("measurement_state_validation"))
def measurement_state_validation():
    rpm = CFG["measurement"]["state_rule"]["spindle_min_rpm"]
    return views.measurement_state_validation(_read("hourly_indicators"), _read("controller_hourly"), rpm)


@dp.materialized_view(name=_name("data_quality_monthly"))
def data_quality_monthly():
    return views.monthly_dq(_read("cm_scalars"), _read("hourly_indicators"))


@dp.materialized_view(name=_name("hz1_reconciliation"))
def hz1_reconciliation():
    return views.reconcile_minutes(views.hz1_to_minute(_read("ctrl_hz1")), _read("ctrl_minute"), 1e-3)
