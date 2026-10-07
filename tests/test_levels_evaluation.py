"""Levels for both methods (ALERT episodes per asset, ALARM criteria and the 14-day escalation), the
temperature residual, and the revision 2 evaluation pieces: groups, value and the resampled intervals."""

import numpy as np
import pandas as pd

from pdm.evaluation import rev2 as r2
from pdm.models import severity, temperature

UTC = "UTC"


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


def test_episodes_merge_points_of_one_asset_within_the_gap():
    sig = pd.DataFrame(
        {
            "asset_id": ["A", "A", "A", "B"],
            "point_id": ["A-F", "A-R", "A-F", "B-1"],
            "ts": [_ts("2025-08-01"), _ts("2025-08-05"), _ts("2025-08-20"), _ts("2025-08-02")],
            "rules": ["hf_alert", "vel_alert", "hf_alert", "zone_C"],
        }
    )
    e = severity.episodes(sig, 7)
    a = e[e["asset_id"] == "A"].reset_index(drop=True)
    assert len(a) == 2 and a.at[0, "points"] == "A-F,A-R" and a.at[0, "trigger_points"] == "A-F"
    assert a.at[0, "trigger_rules"] == "hf_alert" and a.at[0, "closed"] == _ts("2025-08-12")
    assert len(e[e["asset_id"] == "B"]) == 1


def test_escalation_day_needs_half_of_its_hours():
    opened = _ts("2025-09-01 14:00")
    hours = pd.date_range("2025-09-01 14:00", periods=24 * 30, freq="h", tz=UTC)
    half = pd.DataFrame({"ts": hours, "met": (np.arange(len(hours)) % 2 == 0)})
    third = pd.DataFrame({"ts": hours, "met": (np.arange(len(hours)) % 3 == 0)})
    assert severity.escalation_time(opened, _ts("2025-12-01"), half, 14) is not None
    assert severity.escalation_time(opened, _ts("2025-12-01"), third, 14) is None


def test_escalation_counts_operating_days_and_skips_idle_days():
    opened = _ts("2025-09-01 14:00")
    days = pd.date_range("2025-09-01 14:00", periods=30, freq="D", tz=UTC)
    op = pd.DataFrame({"ts": days, "met": True})
    idle = op[~op["ts"].dt.tz_convert("America/Chicago").dt.dayofweek.isin([5, 6])]
    t = severity.escalation_time(opened, _ts("2025-12-01"), idle, 14)
    assert t is not None and (t - opened) > pd.Timedelta(days=14)
    broken = op.copy()
    broken.loc[5, "met"] = False
    t2 = severity.escalation_time(opened, _ts("2025-12-01"), broken, 14)
    assert t2 == days[19]
    assert severity.escalation_time(opened, _ts("2025-09-10"), op, 14) is None


def test_alarm_needs_an_open_alert_and_reports_the_14_day_rule_apart():
    eps = pd.DataFrame(
        {
            "asset_id": ["A", "A"],
            "opened": [_ts("2025-09-01"), _ts("2025-11-01")],
            "last": [_ts("2025-09-03"), _ts("2025-12-10")],
            "closed": [_ts("2025-09-10"), _ts("2025-12-17")],
            "trigger_points": ["P", "P"],
            "trigger_rules": ["", ""],
        }
    )
    crit = pd.DataFrame(
        {
            "asset_id": ["A", "A"],
            "hour_utc": [_ts("2025-08-20"), _ts("2025-09-02")],
            "criterion": ["velocity zone D"] * 2,
        }
    )
    hours = pd.date_range("2025-11-01", "2025-12-16", freq="6h", tz=UTC)
    op = {"A": pd.DataFrame({"ts": hours, "point_id": "P", "met": True})}
    out = severity.alarm_starts(eps, crit, op, 14)
    assert out.at[0, "alarm_ts"] == _ts("2025-09-02") and out.at[0, "alarm_cause"] == "velocity zone D"
    assert pd.isna(out.at[1, "alarm_criterion_ts"]) and out.at[1, "alarm_cause"] == "14-day persistence"
    assert out.at[1, "alarm_ts"] < _ts("2025-11-20")


def test_vendor_escalation_follows_the_rule_that_fired():
    eps = pd.DataFrame(
        {
            "asset_id": ["A"],
            "opened": [_ts("2025-10-01")],
            "last": [_ts("2025-11-30")],
            "closed": [_ts("2025-12-07")],
            "trigger_points": ["P"],
            "trigger_rules": ["hf_alert"],
        }
    )
    hours = pd.date_range("2025-10-01", "2025-12-06", freq="6h", tz=UTC)
    other = {"A": pd.DataFrame({"ts": hours, "point_id": "P", "rules": "vel_alert"})}
    same = {"A": pd.DataFrame({"ts": hours, "point_id": "P", "rules": "hf_alert,vel_alert"})}
    none = pd.DataFrame(columns=["asset_id", "hour_utc", "criterion"])
    assert severity.alarm_starts(eps, none, other, 14)["alarm_ts"].isna().all()
    assert severity.alarm_starts(eps, none, same, 14)["alarm_ts"].notna().all()


def test_thermistor_hours_need_consecutive_minutes():
    t = pd.date_range("2025-08-01 10:00", periods=30, freq="min", tz=UTC)
    temp = np.full(30, 40.0)
    temp[5:14] = 45.0
    temp[18:30] = 45.0
    m = pd.DataFrame({"machine_id": "HMC-01", "minute_utc": t, "spindle_temp_c_mean": temp})
    h = severity.thermistor_hours(m[m["spindle_temp_c_mean"] >= 44], 44.0, 10)
    assert list(h["hour_utc"]) == [_ts("2025-08-01 10:00")]
    assert severity.thermistor_hours(m.iloc[:16][m.iloc[:16]["spindle_temp_c_mean"] >= 44], 44.0, 10).empty


def test_temperature_residual_fit_per_point():
    rng = np.random.default_rng(0)
    n = 500
    speed = rng.uniform(3000, 12000, n)
    hours = pd.date_range("2024-01-01", periods=n, freq="17h", tz=UTC)
    doy = hours.tz_convert("America/Chicago").dayofyear.to_numpy(float)
    season = 1.5 * np.sin(2 * np.pi * doy / 365.25)
    rows = pd.DataFrame(
        {
            "hour_utc": hours,
            "temp_c": 20 + 0.001 * speed + season + rng.normal(0, 0.1, n),
            "speed_rpm_at_measurement": speed,
        }
    )
    f = temperature.fit(rows, "spindle_front")
    assert abs(f.coef[1] - 0.001) < 1e-4 and abs(f.coef[2] - 1.5) < 0.05 and np.std(f.residual(rows)) < 0.15
    plain = temperature.fit(rows, "spindle_front", seasonal=False)
    assert np.std(plain.residual(rows)) > 0.8
    pump = temperature.fit(rows.assign(load_pct_at_measurement=np.nan), "flood_pump")
    assert pump.columns == [] and abs(pump.coef[0] - rows["temp_c"].mean()) < 1e-6


def _conds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wo_id": ["W1", "W2", "W3", "W4"],
            "asset_id": ["A", "B", "C", "D"],
            "asset_key": ["A", "B", "C", "D"],
            "machine_id": ["A", "B", "C", "D"],
            "condition_type": ["bearing (unplanned)", "imbalance", "lubrication", "bearing (unplanned)"],
            "failure_code": ["BRG-WEAR", "VIB-IMBAL", "BRG-LUBE", "PMP-BRG"],
            "wo_type": ["unplanned_repair", "condition_based", "inspection", "unplanned_repair"],
            "unplanned": [True, False, False, True],
            "credit_ts": [_ts("2025-10-01")] * 4,
            "confirmed_ts": [_ts("2025-10-01")] * 4,
            "downtime_hours": [96.0, 4.0, 1.0, 10.0],
            "labor_hours": [40.0, 4.0, 1.0, 8.0],
            "machine_class": ["hmc"] * 4,
            "pm_finding": [False] * 4,
            "instrumented": [True] * 4,
            "point_scope": [None] * 4,
        }
    )


def _eps(assets: list[str], opened: str) -> pd.DataFrame:
    o = _ts(opened)
    return pd.DataFrame(
        {
            "asset_id": assets,
            "opened": o,
            "last": o + pd.Timedelta(days=30),
            "closed": o + pd.Timedelta(days=37),
            "alarm_ts": pd.Series([pd.NaT] * len(assets), dtype="datetime64[ns, UTC]"),
            "alarm_criterion_ts": pd.Series([pd.NaT] * len(assets), dtype="datetime64[ns, UTC]"),
            "alarm_escalation_ts": pd.Series([pd.NaT] * len(assets), dtype="datetime64[ns, UTC]"),
        }
    )


def test_four_groups_with_controller_health_and_value():
    conds = _conds()
    ml = _eps(["A", "B"], "2025-09-20")
    vd = _eps(["B", "C"], "2025-09-25")
    ch = _eps(["D"], "2025-09-28").assign(point_id="")
    ct = r2.condition_table(conds, ml, vd, ch, 120)
    g = ct.set_index("wo_id")["group"]
    assert (
        g["W1"] == "ML Model only"
        and g["W2"] == "both"
        and g["W3"] == "Vendor Rules only"
        and g["W4"] == "neither"
    )
    assert bool(ct.set_index("wo_id").at["W4", "ch_caught"])
    assert ct.set_index("wo_id").at["W2", "ml_days_vs_vendor"] == 5.0
    wo = pd.DataFrame(
        {"wo_type": ["condition_based"], "asset_id": ["A"], "downtime_hours": [72.0], "labor_hours": [30.0]}
    )
    v = r2.value_table(ct, wo, {"hmc": 100.0}, 7)
    a = v[v["wo_id"] == "W1"].iloc[0]
    assert (
        a["plannable"]
        and a["downtime_saved_h"] == 24.0
        and a["labor_saved_h"] == 10.0
        and a["value_downtime_usd"] == 2400.0
    )
    late = r2.value_table(ct.assign(ml_alert_warning_days=3.0), wo, {"hmc": 100.0}, 7)
    assert not late["plannable"].any()


def test_bootstrap_intervals_resample_whole_conditions():
    n_all = 10
    idx = np.arange(n_all)
    vend_caught = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
    ml_caught = {99.0: np.ones(n_all, dtype=bool), 99.5: vend_caught.copy()}
    cover_v = [np.array([0]), np.array([], dtype=int), np.array([], dtype=int)]
    cover_ml = {99.0: [np.array([], dtype=int)] * 4, 99.5: [np.array([], dtype=int)]}
    out = r2.bootstrap(vend_caught, cover_v, ml_caught, cover_ml, 99.5, idx, n_all, 10.0, 200, 1)
    assert out["catch_diff_lo"] <= out["catch_diff_hi"] and out["false_diff_lo"] <= out["false_diff_hi"]
    assert out["draws"] == 200


def test_alarm_outcomes_and_false_alarms():
    t = _ts("2025-09-01")
    lev = pd.DataFrame(
        {
            "asset_id": ["A", "B", "C", "D", "E"],
            "opened": [t] * 5,
            "closed": [t + pd.Timedelta(days=20)] * 4 + [_ts("2026-07-10")],
            "alarm_ts": [t + pd.Timedelta(days=2)] * 5,
            "alarm_cause": ["temperature residual"] * 5,
        }
    )
    conf = pd.DataFrame(
        {
            "asset_key": ["A", "B"],
            "ts": [t + pd.Timedelta(days=5), t + pd.Timedelta(days=1)],
            "failure": [True, False],
            "wo_id": ["W1", "I2"],
        }
    )
    wo = pd.DataFrame({"wo_type": ["corrective"], "asset_id": ["C"], "opened": [t + pd.Timedelta(days=3)]})
    o = r2.alarm_outcomes(lev, conf, wo, (t, _ts("2026-07-01")), _ts("2026-07-01")).set_index("asset_id")[
        "outcome"
    ]
    assert list(o) == [
        "failure followed",
        "condition confirmed and repaired before failure",
        "repair or PM reached the asset first for an unrelated reason",
        "nothing found",
        "pending at the record end",
    ]
    f = r2.alarm_outcomes(lev, conf, wo, (t, _ts("2026-07-01")), _ts("2026-07-01"))
    assert list(f["false_alarm"]) == [False, False, True, True, False]
