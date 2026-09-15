"""Rule floor for every instrumented point: the vendor-style baseline rules replayed from bronze and the
ISO 20816-3 zone rule on the largest directional velocity, both with persistence and suppression.

Inputs: bronze scalars (resultant rows, including hours without a measurement), sensor events and
the hourly indicator table. Output: one alert table with the method (vendor or iso), level and rule.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pdm.rules.iso_zones import zone_boundaries
from pdm.rules.persistence import ConsecutiveCounter, Suppressor
from pdm.rules.vendor_baseline import VendorRuleSettings
from pdm.rules.vendor_replay import replay_point

ISO_KEY = {
    "spindle_front": "spindle",
    "spindle_rear": "spindle",
    "tsc_pump": "tsc_pump",
    "flood_pump": "flood_pump",
    "compressor_motor": "compressor",
    "compressor_airend": "compressor",
}
ALERT_SCHEMA = "method string, point_id string, alert_ts timestamp, level string, rules string, value double"


def vendor_alerts(
    scalars: DataFrame, pts: DataFrame, sensor_events: pd.DataFrame, cfg: dict[str, Any]
) -> DataFrame:
    res = scalars.where(F.col("axis") == "res").join(pts.select("point_id", "point_class"), "point_id")
    cols = [c for c in res.columns if not c.startswith("_")]
    se = sensor_events.copy()
    se["ts"] = pd.to_datetime(se["ts"])

    def fn(pdf: pd.DataFrame) -> pd.DataFrame:
        pid = pdf["point_id"].iat[0]
        key = ISO_KEY[pdf["point_class"].iat[0]]
        settings = VendorRuleSettings.for_point_class(cfg, key, "compressor" if key == "compressor" else key)
        pdf = pdf.copy()
        pdf["hour_utc"] = pd.to_datetime(pdf["hour_utc"]).dt.tz_localize("UTC")
        ev = se.assign(ts=se["ts"].dt.tz_localize("UTC") if se["ts"].dt.tz is None else se["ts"])
        out = replay_point(pid, pdf, ev, settings, cfg)
        if not out:
            return pd.DataFrame(columns=["method", "point_id", "alert_ts", "level", "rules", "value"])
        a = pd.DataFrame(out)
        return pd.DataFrame(
            {
                "method": "vendor",
                "point_id": a["point_id"],
                "alert_ts": pd.to_datetime(a["alert_ts"]).dt.tz_convert("UTC").dt.tz_localize(None),
                "level": a["level"],
                "rules": a["rules"],
                "value": a["vel_rms_mm_s"],
            }
        )

    return res.select(*cols).groupBy("point_id").applyInPandas(fn, ALERT_SCHEMA)


def iso_alerts(indicators: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    rules = cfg["rules"]
    persist, sup_days = rules["persistence_consecutive"], rules["suppression_days"]
    zones = {
        k: zone_boundaries(v["group"], v["foundation"]) for k, v in rules["iso_class_by_point_class"].items()
    }

    def fn(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("hour_utc")
        pdf = pdf[~pdf["implausible"].fillna(False).astype(bool)]
        _, bc, cd = zones[ISO_KEY[pdf["point_class"].iat[0]]]
        c_alert, c_alarm, sup = ConsecutiveCounter(persist), ConsecutiveCounter(persist), Suppressor(sup_days)
        rows = []
        for r in pdf.itertuples():
            v = r.vel_max_dir_mm_s
            if pd.isna(v):
                continue
            alarm = c_alarm.update(bool(v >= cd))
            alert = c_alert.update(bool(v >= bc))
            if (alarm or alert) and sup.allow(pd.Timestamp(r.hour_utc)):
                rows.append(
                    (
                        "iso",
                        r.point_id,
                        r.hour_utc,
                        "alarm" if alarm else "alert",
                        "zone_D" if alarm else "zone_C",
                        float(v),
                    )
                )
        return pd.DataFrame(rows, columns=["method", "point_id", "alert_ts", "level", "rules", "value"])

    return (
        indicators.select("point_id", "hour_utc", "point_class", "vel_max_dir_mm_s", "implausible")
        .groupBy("point_id")
        .applyInPandas(fn, ALERT_SCHEMA)
    )


def rule_floor_alerts(
    scalars: DataFrame,
    indicators: DataFrame,
    pts: DataFrame,
    sensor_events: pd.DataFrame,
    cfg: dict[str, Any],
) -> DataFrame:
    return vendor_alerts(scalars, pts, sensor_events, cfg).unionByName(iso_alerts(indicators, cfg))


EXCEED_SCHEMA = "method string, sweep double, point_id string, ts timestamp, level string"


def rule_floor_exceedances(
    scalars: DataFrame,
    indicators: DataFrame,
    pts: DataFrame,
    sensor_events: pd.DataFrame,
    cfg: dict[str, Any],
    alert_fracs: list[float],
    until: pd.Timestamp | None = None,
) -> DataFrame:
    """Persisted exceedances without re-alert suppression, for alert episodes: the vendor rules at each
    alert offset (alarm offset at twice it, as configured) and the ISO zone rule, before until (UTC)."""
    res = scalars.where(F.col("axis") == "res").join(pts.select("point_id", "point_class"), "point_id")
    cols = [c for c in res.columns if not c.startswith("_")]
    se = sensor_events.copy()
    se["ts"] = pd.to_datetime(se["ts"])
    ratio = (
        cfg["rules"]["vendor"]["alarm_frac_zone_b_width"] / cfg["rules"]["vendor"]["alert_frac_zone_b_width"]
    )

    def vendor_fn(pdf: pd.DataFrame) -> pd.DataFrame:
        pid = pdf["point_id"].iat[0]
        key = ISO_KEY[pdf["point_class"].iat[0]]
        base = VendorRuleSettings.for_point_class(cfg, key, "compressor" if key == "compressor" else key)
        pdf = pdf.copy()
        pdf["hour_utc"] = pd.to_datetime(pdf["hour_utc"]).dt.tz_localize("UTC")
        ev = se.assign(ts=se["ts"].dt.tz_localize("UTC") if se["ts"].dt.tz is None else se["ts"])
        parts = []
        for f in alert_fracs:
            s = replace(base, alert_frac=f, alarm_frac=f * ratio, suppression_days=0.0)
            out = replay_point(pid, pdf, ev, s, cfg)
            if out:
                a = pd.DataFrame(out)
                parts.append(
                    pd.DataFrame(
                        {
                            "method": "vendor",
                            "sweep": f,
                            "point_id": pid,
                            "ts": pd.to_datetime(a["alert_ts"]).dt.tz_convert("UTC").dt.tz_localize(None),
                            "level": a["level"],
                        }
                    )
                )
        return (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=["method", "sweep", "point_id", "ts", "level"])
        )

    iso_cfg = {**cfg, "rules": {**cfg["rules"], "suppression_days": 0.0}}
    iso = iso_alerts(indicators, iso_cfg).select(
        "method", F.lit(0.0).alias("sweep"), "point_id", F.col("alert_ts").alias("ts"), "level"
    )
    out = res.select(*cols).groupBy("point_id").applyInPandas(vendor_fn, EXCEED_SCHEMA).unionByName(iso)
    if until is not None:
        out = out.where(F.col("ts") < F.lit(until.tz_convert("UTC").tz_localize(None)))
    return out
