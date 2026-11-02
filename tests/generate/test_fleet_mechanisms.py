"""Compressor N+1 dispatch, shared spare spindle, CMMS batch close, cavitation wear and vendor
temperature limits."""

import numpy as np
import pandas as pd
from generate.cmms_export import close_time
from generate.pump_wear import WearProcess
from generate.run_full import resolve_spare

from generate import plant_air, shop_schedule
from pdm.config import load_config
from pdm.rules.vendor_baseline import VendorRuleSettings, VendorRuleState

CFG = load_config()
TZ = "America/Chicago"


def _frame(days: int = 14) -> pd.DataFrame:
    start = pd.Timestamp("2023-07-03", tz=TZ).tz_convert("UTC")
    return shop_schedule.hour_frame(start, start + pd.Timedelta(days=days))


def test_dispatch_keeps_one_unit_standby_and_runs_three_at_peak():
    frame = _frame()
    plant = np.where((frame["weekday"] < 5) & (frame["local_hour"].between(6, 22)), 60, 0)
    cids = ["CMP-01", "CMP-02", "CMP-03", "CMP-04"]
    air = plant_air.compressor_hours(CFG, frame, plant, cids, np.random.default_rng(1))
    n = np.column_stack([air[f"{c}_running"] for c in cids]).sum(axis=1)
    assert n.max() == 3
    peak = (frame["weekday"] < 5) & frame["local_hour"].between(8, 10)
    assert (n[peak.to_numpy()] == 3).mean() > 0.5
    run = air[[f"{c}_running" for c in cids]].to_numpy()
    load = air[[f"{c}_load_frac" for c in cids]].to_numpy()
    shared = np.where(n > 0, (load * run).sum(axis=1) / np.maximum(n, 1), 0)
    assert np.allclose(
        load[run], np.repeat(air["demand_frac"].to_numpy() / np.maximum(n, 1), 4).reshape(-1, 4)[run]
    )
    assert (shared <= 1.0).all()


def test_spare_is_swapped_then_out_for_rebuild():
    t0 = pd.Timestamp("2023-07-03 12:00", tz="UTC")
    fails = [
        {"machine_id": "HMC-01", "ordinal": 0, "ts": t0 + pd.Timedelta(days=10)},
        {"machine_id": "HMC-05", "ordinal": 0, "ts": t0 + pd.Timedelta(days=20)},
        {"machine_id": "HMC-02", "ordinal": 0, "ts": t0 + pd.Timedelta(days=60)},
    ]
    cases, timeline = resolve_spare(CFG, fails, set(), t0)
    assert cases == {"HMC-01": {0: "swap"}, "HMC-05": {0: "no_spare"}, "HMC-02": {0: "swap"}}
    assert [r["state"] for r in timeline].count("out_for_rebuild") == 2


def test_batch_close_lands_on_friday_within_five_working_days():
    rng = np.random.default_rng(3)
    ln = CFG["label_noise"]
    closed = pd.Timestamp("2023-07-04 18:00", tz="UTC")
    out = [close_time(closed, ln, rng) for _ in range(400)]
    fri = [t for t in out if t.tz_convert(TZ).weekday() == 4 and t.tz_convert(TZ).hour == 15]
    assert 0.5 < len(fri) / len(out) < 0.8
    assert all(np.busday_count(closed.tz_convert(TZ).date(), t.tz_convert(TZ).date()) <= 5 for t in out)


def test_cavitation_extra_days_speed_wear():
    w = WearProcess("seal", pd.Timestamp("2023-07-01", tz="UTC"), 100.0)
    t = pd.Timestamp("2023-07-11", tz="UTC")
    base = w.fraction(t)
    w.extra_days = 10.0
    assert np.isclose(w.fraction(t), 2 * base)


def test_vendor_temperature_limits_persist_and_fire():
    st = VendorRuleState("CMP-01-AE", VendorRuleSettings.for_point(CFG, "compressor_airend"))
    st.baseline = {c: 1.0 for c in ["vel_rms_mm_s", "hf_rms_g", *st.settings.env_columns()]}
    row = {"vel_rms_mm_s": 1.0, "hf_rms_g": 1.0, **{c: 1.0 for c in st.settings.env_columns()}}
    t = pd.Timestamp("2023-07-01", tz="UTC")
    warn = CFG["rules"]["temperature_limits_c"]["compressor_airend"][0]
    assert st.update(t, {**row, "temp_c": warn + 1}) is None
    a = st.update(t + pd.Timedelta(hours=1), {**row, "temp_c": warn + 1})
    assert a is not None and "temp_alert" in a["rules"]
    assert st.update(t + pd.Timedelta(hours=2), {**row, "temp_c": warn - 5}) is None
