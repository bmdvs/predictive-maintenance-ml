"""Switches for the fixes after the first evaluation-year scoring (2026-11-01): off reproduces the behavior
before; on applies the fix."""

import copy

import numpy as np
import pandas as pd

from pdm.advisory import requests as adv
from pdm.config import apply_evaluation_fixes, load_config
from pdm.evaluation import events as ev
from pdm.models import nbm, temperature


def test_switches_default_off_in_code_and_frozen_in_config():
    # code defaults: every switch off (the behavior before the fixes)
    assert not any(v for k, v in nbm.FIX_DEFAULTS.items() if isinstance(v, bool))
    # frozen A (2026-11-01): F1, F3, F4-tight, S1, C3 and C4 on; F2, T1 to T3, C1 and C2 off
    cfg = load_config()
    fx = nbm.fixes(cfg)
    on = {"f1_hold_level", "f3_spindle_temp_regime", "f4_high_ratio", "f4_exclude_pump_low_pressure"}
    on |= {"s1_on_component_window", "c3_flat_rearm", "c4_other_repair_window"}
    assert {k for k, v in fx.items() if v is True} == on
    assert fx["f4_ratio"] == 5.0
    assert cfg["evaluation"]["catch"]["episode_gap_days"] == 7
    assert cfg["evaluation"]["fixes"]["e2_scope"]
    assert not cfg["advisory"]["d1_precedence"]
    assert [f["alias"] for f in cfg["models"]["nbm"]["finalists"]] == ["champion"]


def test_e1_sets_the_close_gap():
    cfg = copy.deepcopy(load_config())
    cfg["evaluation"]["fixes"]["e1_close_gap_days"] = 30
    assert apply_evaluation_fixes(cfg)["evaluation"]["catch"]["episode_gap_days"] == 30
    assert cfg["evaluation"]["catch"]["opened_within_days"] == 120


def test_e2_scope():
    cfg = copy.deepcopy(load_config())
    c = pd.DataFrame(
        {
            "condition_type": ["cooler (compressor)", "rotary union", "rotary union", "lubrication"],
            "failure_code": ["CMP-COOLER", "SP-COOLANT-INGRESS", "RU-OTHER", "BRG-LUBE"],
        }
    )
    cfg["evaluation"]["fixes"]["e2_scope"] = False
    assert ev.in_comparison_scope(c, cfg).tolist() == [True, True, True, True]
    cfg["evaluation"]["fixes"]["e2_scope"] = True
    assert ev.in_comparison_scope(c, cfg).tolist() == [False, True, False, True]


def test_f4_high_ratio_opens_after_two_hours():
    h = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    s = pd.DataFrame(
        {
            "point_id": "P",
            "hour_utc": h,
            "state": "scored",
            "score": [1.0, 3.5, 3.5, 0.5],
            "thr": 1.0,
            "top_contributors": "",
        }
    )
    assert nbm.exceedance_hours(s, "thr", 6).empty
    ex = nbm.exceedance_hours(s, "thr", 6, (3.0, 2))
    assert list(ex["ts"]) == [h[2]]


def test_f3_regime_columns_only_when_on():
    assert temperature.covariates("spindle_front") == [
        temperature.SPEED,
        temperature.SEASON_SIN,
        temperature.SEASON_COS,
    ]
    assert temperature.REGIME_HEAVY in temperature.covariates("spindle_front", regime=True)
    assert temperature.REGIME_HEAVY not in temperature.covariates("tsc_pump", regime=True)


def test_d1_precedence():
    z = {"temp": 3.5, "hf": 3.2}
    assert adv.suspected(z, "spindle", 3.0)[0] == "lubrication"  # temperature with HF: lubrication either way
    z = {"temp": 3.5, "hf": 2.5}
    assert adv.suspected(z, "spindle", 3.0)[0] == "spindle cooling"
    assert adv.suspected(z, "spindle", 3.0, d1=True)[0] != "spindle cooling"  # HF not below 2
    z = {"thermistor": 3.5, "temp": 3.5, "hf": 3.5}
    assert adv.suspected(z, "spindle", 3.0)[0] == "spindle cooling"
    assert adv.suspected(z, "spindle", 3.0, d1=True)[0] == "lubrication"
    z = {"hf": 3.5, "ord_01": 3.5, "env_bpfo": 3.5}
    assert adv.suspected(z, "flood_pump", 3.0)[0] == "bearing"
    assert adv.suspected(z, "flood_pump", 3.0, d1=True)[0] == "cavitation or wear"
    assert np.isfinite(adv.D1_HF_QUIET)


def test_e4_lookback_credits_the_latest_open_episode():
    t = pd.Timestamp("2025-06-01", tz="UTC")
    ep = pd.DataFrame(
        {
            "asset_id": ["X", "X", "X"],
            "point_id": ["", "", ""],
            "opened": [t - pd.Timedelta(days=100), t - pd.Timedelta(days=40), t - pd.Timedelta(days=200)],
            "closed": [t - pd.Timedelta(days=50), t - pd.Timedelta(days=10), t - pd.Timedelta(days=5)],
        }
    )
    c = pd.DataFrame(
        {
            "wo_id": ["W"],
            "asset_id": ["X"],
            "asset_key": ["X"],
            "point_scope": [None],
            "condition_type": ["bearing"],
            "unplanned": [False],
            "credit_ts": [t],
            "downtime_hours": [0.0],
            "machine_class": ["hmc"],
        }
    )
    ct, credited = ev.lookback_catch_table(ep, c, 120, 30)
    assert bool(ct["caught"].iat[0]) and credited == {1}
    assert ct["warning_days"].iat[0] == 40
    assert not ev.catch_table(ep, c, 120)["caught"].iat[0]  # none open at the confirmation


def test_f1_scope_component_repairs():
    win = pd.DataFrame(
        {
            "point_id": ["HMC-01-SP-F", "HMC-01-SP-F", "HMC-01-CP", "CMP-01-AE"],
            "reason": ["repair W1", "repair W2", "repair W3", "remount"],
        }
    )
    wo = pd.DataFrame(
        {
            "wo_id": ["W1", "W2", "W3"],
            "parts": ["SPINDLE-CARTRIDGE", "SPINDLE-CHILLER-PARTS", "PUMP-UNIT;6205"],
        }
    )
    assert nbm.component_repairs(win, wo)["on_component"].tolist() == [True, False, True, False]


def test_f4_tight_skips_pump_low_pressure_hours():
    h = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    s = pd.DataFrame(
        {
            "point_id": "HMC-01-TSC",
            "hour_utc": h,
            "state": "scored",
            "score": [6.0, 6.0, 6.0],
            "thr": 1.0,
            "top_contributors": "",
            "low_pressure": [True, True, False],
        }
    )
    assert len(nbm.exceedance_hours(s, "thr", 6, (5.0, 2))) == 2
    assert nbm.exceedance_hours(s, "thr", 6, (5.0, 2), exclude_low_pressure=True).empty


def test_c1_pelt_finds_a_mean_shift_only():
    rng = np.random.default_rng(0)
    assert nbm.pelt_mean(np.r_[rng.normal(0, 0.05, 200), rng.normal(0.3, 0.05, 100)]) == [200]
    assert nbm.pelt_mean(rng.normal(0, 0.05, 300)) == []


def test_c2_season_clamp_holds_uncovered_days():
    h = pd.date_range("2024-03-01", "2024-06-30", freq="h", tz="UTC")
    rows = pd.DataFrame(
        {"hour_utc": h, "temp_c": 30.0 + np.sin(np.arange(len(h)) / 500), "speed_rpm_at_measurement": 5000.0}
    )
    tf = temperature.fit(rows, "spindle_front", season_clamp=True)
    assert tf.season_days and min(tf.season_days) >= 59 and max(tf.season_days) <= 182
    off = temperature.fit(rows, "spindle_front")
    assert off.season_days is None
    # 1 December (local day 335) lies outside the March to June coverage; its nearest covered day, circularly,
    # is the first covered day (about 1 March), so the clamped expectation equals that day's
    late = pd.DataFrame(
        {
            "hour_utc": [pd.Timestamp("2024-12-01 18:00", tz="UTC")],
            "temp_c": [30.0],
            "speed_rpm_at_measurement": [5000.0],
        }
    )
    first = min(tf.season_days)
    day0 = pd.Timestamp("2024-01-01 18:00", tz="UTC") + pd.Timedelta(days=first - 1)
    ref = pd.DataFrame({"hour_utc": [day0], "temp_c": [30.0], "speed_rpm_at_measurement": [5000.0]})
    assert abs(tf.expected(late)[0] - tf.expected(ref)[0]) < 1e-9
    assert abs(off.expected(late)[0] - off.expected(ref)[0]) > 1e-6  # without the clamp they differ


def test_c3_rearm_on_a_flat_stuck_point():
    h = pd.date_range("2025-01-01", periods=24 * 40, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    s = pd.DataFrame(
        {
            "point_id": "P",
            "asset_id": "X",
            "hour_utc": h,
            "state": "scored",
            "score": 2.0 + rng.normal(0, 0.1, len(h)),
            "threshold": 1.0,
            "thr_99": 0.9,
        }
    )
    out, log = nbm.flat_rearm(
        s,
        pd.DataFrame({"point_id": [], "segment_start": []}),
        pd.DataFrame({"asset_key": [], "credit_ts": []}),
        pd.DataFrame({"asset_id": [], "opened": [], "closed": []}),
    )
    assert len(log) == 1 and out["threshold"].iloc[-1] > 2.0
    assert out["thr_99"].iloc[-1] > 0.9
