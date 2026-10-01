"""Gold materialized views: baselines, the fleet baseline sanity check, rolling features and the rule
floor."""

from pyspark import pipelines as dp

from pdm import tasks
from pdm.config import load_config
from pdm.features import baselines, rolling, windows
from pdm.rules.floor import rule_floor_alerts as _rule_floor_alerts
from pdm.rules.floor import rule_floor_exceedances as _rule_floor_exceedances
from pdm.store import table_name

CFG = load_config(spark.conf.get("pdm.config_path"))  # noqa: F821


CATALOG = spark.conf.get("pdm.catalog")  # noqa: F821


def _name(name: str) -> str:
    return table_name(CATALOG, name)


def _read(name: str):
    return spark.read.table(_name(name))  # noqa: F821


def _rows_in_windows():
    return windows.in_windows(_read("hourly_indicators"), _read("healthy_windows"))


@dp.materialized_view(name=_name("baseline_stats"))
def baseline_stats():
    return baselines.window_stats(_rows_in_windows())


@dp.materialized_view(name=_name("baseline_sanity"))
def baseline_sanity():
    end = tasks.horizon(CFG).tz_convert("UTC").tz_localize(None)
    iw = _rows_in_windows()
    win = _read("healthy_windows")
    return baselines.sanity_table(
        iw.where(iw.hour_utc < end), _read("points"), win.where(win.start < end), CFG
    )


@dp.materialized_view(name=_name("rolling_features"))
def rolling_features():
    base = rolling.baseline_with_ends(_read("baseline_stats"), _read("healthy_windows"))
    return rolling.rolling_features(_read("hourly_indicators"), base, CFG)


@dp.materialized_view(name=_name("rule_floor_alerts"))
def rule_floor_alerts():
    se = _read("cm_sensor_events")
    return _rule_floor_alerts(_read("cm_scalars"), _read("hourly_indicators"), _read("points"), se, CFG)


@dp.materialized_view(name=_name("rule_floor_exceedances"))
def rule_floor_exceedances():
    se = _read("cm_sensor_events")
    return _rule_floor_exceedances(
        _read("cm_scalars"),
        _read("hourly_indicators"),
        _read("points"),
        se,
        CFG,
        CFG["evaluation"]["vendor_sweep_alert_frac"],
        tasks.horizon(CFG),
    )
