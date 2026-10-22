"""Fault conditions: level, clearing and recovery; rendered signatures by band and axis."""

import numpy as np
import pandas as pd
import pytest
from generate.vibration_synth import BurstInputs, render_burst

from generate import conditions as cnd
from pdm.config import load_config, load_register
from pdm.features.burst import ExtractionSettings, extract_burst
from pdm.features.kinematics import bearing_fault_orders, fault_frequencies_hz

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def test_ramp_clear_and_recovery():
    c = cnd.Condition("imbalance", "A", ("P",), T0, ramp_h=100.0)
    assert c.level(T0 - pd.Timedelta(hours=1)) == 0.0
    assert c.level(T0 + pd.Timedelta(hours=50)) == pytest.approx(0.5)
    c.clear(T0 + pd.Timedelta(hours=50), "repair", recovery_h=10.0)
    assert not c.active(T0 + pd.Timedelta(hours=51))
    assert c.level(T0 + pd.Timedelta(hours=55)) == pytest.approx(0.25)
    assert c.level(T0 + pd.Timedelta(hours=61)) == 0.0
    assert c.finished(T0 + pd.Timedelta(hours=60))


def test_seasonal_progress_rate():
    c = cnd.Condition("cooler_fouling", "C", (), T0, ramp_h=100.0, seasonal=True)
    c.advance(40.0, 1.5)
    assert c.level(T0) == pytest.approx(0.6)
    c.advance(100.0, 1.0)
    assert c.level(T0) == 1.0


def test_tool_assembly_high_share():
    cfg = load_config()
    rng = np.random.default_rng(3)
    draws = [cnd.tool_assembly_factor(cfg, rng) for _ in range(4000)]
    high = np.mean([h for _, h in draws])
    share = cfg["fault_conditions"]["tool_assembly_imbalance"]["high_program_share"]
    assert abs(high - share) < 0.02
    lo, hi = cfg["fault_conditions"]["tool_assembly_imbalance"]["high_factor"]
    assert all(lo <= f <= hi for f, h in draws if h)


def _bands(faults, seed=0):
    cfg = load_config()
    reg = load_register()
    s = ExtractionSettings.from_config(cfg)
    sig = dict(cfg["bearing_signature"], fault_render=cfg["fault_conditions"]["render"])
    hz = 57.5
    fhz = fault_frequencies_hz(bearing_fault_orders(reg.bearings["6205"]), hz)
    out = []
    for k in range(8):
        inp = BurstInputs(
            "flood_pump", s.fs, int(s.fs), s.band, s.hf_band, hz, hz, 0.0, 0, fhz, 4600.0, faults=faults
        )
        th = cfg["bearing_degradation"]
        b = render_burst(
            inp, sig, cfg["regime"], th["stage_thresholds_hi"], 1.0, np.random.default_rng(seed + k)
        )
        rows, diag = extract_burst(b, s, hz, fhz, band=(2000.0, 10000.0))
        out.append(
            {
                "r1": np.sqrt((rows["x"]["ord_01_g2"] + rows["y"]["ord_01_g2"]) / 2),
                "r2": np.sqrt((rows["x"]["ord_02_g2"] + rows["y"]["ord_02_g2"]) / 2),
                "z1": np.sqrt(rows["z"]["ord_01_g2"]),
                "z2": np.sqrt(rows["z"]["ord_02_g2"]),
                "r3_10": np.sqrt(
                    sum(
                        (rows["x"][f"ord_{o:02d}_g2"] + rows["y"][f"ord_{o:02d}_g2"]) / 2
                        for o in range(3, 11)
                    )
                ),
                "hf": rows["res"]["hf_rms_g"],
            }
        )
    return pd.DataFrame(out).median()


def test_imbalance_raises_radial_1x_with_small_axial():
    base = _bands([])
    f = _bands([{"kind": "imbalance", "level": 1.0, "one_x": 3.0, "axial": 0.3}])
    assert f["r1"] / base["r1"] > 2.5
    assert (f["z1"] - base["z1"]) < 0.5 * (f["r1"] - base["r1"])
    assert f["r2"] / base["r2"] < 1.2
    assert f["hf"] / base["hf"] == pytest.approx(1.0, rel=0.05)


def test_misalignment_raises_2x_with_high_axial():
    base = _bands([])
    f = _bands([{"kind": "misalignment", "level": 1.0, "one_x": 1.5, "two_x": 4.0, "axial": 0.75}])
    assert f["r2"] / base["r2"] > 2.0
    assert (f["z2"] - base["z2"]) >= 0.5 * (f["r2"] - base["r2"])


def test_looseness_raises_harmonics():
    base = _bands([])
    f = _bands([{"kind": "looseness", "level": 1.0, "harm": 3.0, "half_x": 0.35, "floor_db": 4.5}])
    assert f["r3_10"] / base["r3_10"] > 1.5
