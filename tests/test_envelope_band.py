"""Per-point demodulation band, missing out-of-band energies and inner-race modulation by bearing type."""

import numpy as np
import pytest
from generate.degradation import BearingInstance

from pdm.config import load_config
from pdm.features.burst import ExtractionSettings, coverage_width, extract_burst, select_point_band

FAULTS = {"bpfo": 600.0, "bpfi": 800.0, "bsf": 300.0, "ftf": 30.0}


@pytest.fixture(scope="module")
def s() -> ExtractionSettings:
    return ExtractionSettings.from_config(load_config())


def _impulses(s, rate, fn, seed):
    n = int(s.fs)
    t = np.arange(400) / s.fs
    x = np.zeros(n)
    x[(np.arange(0, 1, 1 / rate) * s.fs).astype(int)] = 1.0
    h = np.exp(-0.05 * 2 * np.pi * fn * t) * np.sin(2 * np.pi * fn * t)
    y = np.convolve(x, h)[:n] + np.random.default_rng(seed).normal(size=n) * 0.05
    return np.vstack([y, y, y])


def test_point_band_respects_coverage(s):
    bursts = [_impulses(s, 600.0, 7000.0, k) for k in range(5)]
    lo, hi = select_point_band(bursts, s, FAULTS, 75.0)
    assert hi - lo >= coverage_width(FAULTS, 75.0, s) or (lo, hi) == s.sk_band
    assert lo <= 7000.0 <= hi


def test_bands_outside_bandwidth_are_missing(s):
    burst = _impulses(s, 600.0, 7000.0, 1)
    rows, diags = extract_burst(burst, s, 75.0, FAULTS, band=(6500.0, 7500.0))
    assert np.isnan(rows["x"]["env_bpfi_3_g2"])
    assert np.isnan(rows["res"]["env_bpfi_3_g2"])
    assert not np.isnan(rows["x"]["env_bpfo_1_g2"])
    assert np.isnan(diags["x"]["envp_bpfi_sb_hi"]) or diags["x"]["envp_bpfi_sb_hi"] >= 0


def test_inner_race_depth_by_bearing_kind():
    cfg = load_config()
    ranges = cfg["bearing_signature"]["amplitude_modulation_depth"]["inner_race_at_shaft_by_kind"]
    for kind in ("angular_contact_ball", "deep_groove_ball"):
        b = BearingInstance(
            "c",
            "spindle_front",
            "a",
            None,
            "older",
            cfg["bearing_degradation"],
            cfg["bearing_signature"],
            np.random.default_rng(3),
            bearing_kind=kind,
        )
        b.start(0.0)
        import pandas as pd

        b.begin_onset(pd.Timestamp("2024-01-01", tz="UTC"), "test", defect="inner_race")
        lo, hi = ranges[kind]
        assert lo <= b.am_inner <= hi


def test_energy_selection_finds_housing_resonance(s):
    from generate.vibration_synth import BurstInputs, render_burst

    cfg = load_config()
    sig = cfg["bearing_signature"]
    fault = {"bpfo": 205.4, "bpfi": 312.1, "bsf": 133.5, "ftf": 22.8}
    bursts = []
    for k in range(8):
        inp = BurstInputs(
            point_class="flood_pump",
            fs=s.fs,
            n=int(s.fs),
            band=s.band,
            hf_band=s.hf_band,
            shaft_hz=57.5,
            rated_hz=57.5,
            load_pct=0.0,
            flutes=0,
            fault_hz=fault,
            resonance_hz=4600.0,
            housing_gain_db=10.0,
        )
        bursts.append(
            render_burst(
                inp,
                sig,
                cfg["regime"],
                cfg["bearing_degradation"]["stage_thresholds_hi"],
                1.0,
                np.random.default_rng(k),
            )
        )
    lo, hi = select_point_band(bursts, s, fault, 57.5)
    assert lo <= 4600.0 <= hi
    assert hi - lo < s.sk_band[1] - s.sk_band[0]
