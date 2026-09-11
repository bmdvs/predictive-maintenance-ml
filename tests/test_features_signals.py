"""Feature extraction against signals with known answers."""

import numpy as np
import pytest

from pdm.config import load_config
from pdm.constants import MM_PER_M, STANDARD_GRAVITY_M_S2
from pdm.features import envelope, filters, spectral
from pdm.features.burst import ExtractionSettings, extract_burst, indicator_columns

FAULTS = {"bpfo": 1685.3, "bpfi": 2114.7, "bsf": 843.8, "ftf": 88.7}


@pytest.fixture(scope="module")
def s() -> ExtractionSettings:
    return ExtractionSettings.from_config(load_config())


def t_axis(s):
    return np.arange(int(s.fs)) / s.fs


def tri(x):
    return np.vstack([x, x, x])


def impulse_train(s, rate, fn=6000.0, zeta=0.05, noise=0.05, seed=0):
    n = int(s.fs)
    x = np.zeros(n)
    x[(np.arange(0, 1, 1 / rate) * s.fs).astype(int)] = 1.0
    tr = np.arange(400) / s.fs
    h = np.exp(-zeta * 2 * np.pi * fn * tr) * np.sin(2 * np.pi * fn * tr)
    return np.convolve(x, h)[:n] + np.random.default_rng(seed).normal(size=n) * noise


def test_pure_tone_rms_crest_velocity(s):
    a, f = 0.5, 200.0
    rows, _ = extract_burst(tri(a * np.sin(2 * np.pi * f * t_axis(s))), s, f, FAULTS)
    assert rows["x"]["accel_rms_g"] == pytest.approx(a / np.sqrt(2), rel=0.005)
    assert rows["x"]["crest_factor"] == pytest.approx(np.sqrt(2), rel=0.01)
    v = a * STANDARD_GRAVITY_M_S2 / (2 * np.pi * f) / np.sqrt(2) * MM_PER_M
    assert rows["x"]["vel_rms_mm_s"] == pytest.approx(v, rel=0.01)


def test_velocity_excludes_tone_above_band(s):
    rows, _ = extract_burst(tri(0.5 * np.sin(2 * np.pi * 3000.0 * t_axis(s))), s, 50.0, FAULTS)
    assert rows["x"]["vel_rms_mm_s"] < 1e-6


def test_gaussian_kurtosis_and_crest(s):
    x = np.random.default_rng(3).normal(size=(3, int(s.fs))) * 0.2
    rows, _ = extract_burst(x, s, 200.0, FAULTS)
    for ax in "xyz":
        assert rows[ax]["kurtosis"] == pytest.approx(3.0, abs=0.15)
        assert 3.5 < rows[ax]["crest_factor"] < 5.5


def test_impulse_train_envelope_peak(s):
    rate = 157.0
    rows, diags = extract_burst(
        tri(impulse_train(s, rate)), s, 30.0, {"bpfo": rate, "bpfi": 230.0, "bsf": 71.0, "ftf": 9.3}
    )
    assert rows["res"]["envpk_01_hz"] == pytest.approx(rate, abs=1.0)
    assert diags["x"]["envr_bpfo"] > 3.0
    assert rows["x"]["env_bpfo_2_g2"] > 0


def test_spectral_kurtosis_finds_resonance(s):
    rng = np.random.default_rng(5)
    x = impulse_train(s, 157.0, fn=7000.0, noise=0.0) + rng.normal(size=int(s.fs)) * 0.2
    spec, freqs = filters.rfft_with_freqs(x, s.fs)
    lo, hi, _ = envelope.select_band(spec, freqs, *s.sk_band, s.sk_levels)
    assert lo <= 7000.0 <= hi


def test_inner_race_sidebands(s):
    n = int(s.fs)
    t = t_axis(s)
    rate, fr = 230.0, 30.0
    x = impulse_train(s, rate, noise=0.0) * (1 + 0.6 * np.cos(2 * np.pi * fr * t))
    x += np.random.default_rng(1).normal(size=n) * 0.02
    _, diags = extract_burst(tri(x), s, fr, {"bpfo": 157.0, "bpfi": rate, "bsf": 71.0, "ftf": 9.3})
    assert diags["x"]["envr_bpfi_sb_lo"] > 3.0 and diags["x"]["envr_bpfi_sb_hi"] > 3.0


def test_order_band_isolation(s):
    fr = 50.0
    rows, _ = extract_burst(tri(0.3 * np.sin(2 * np.pi * 3 * fr * t_axis(s))), s, fr, FAULTS)
    orders = [rows["x"][f"ord_{o:02d}_g2"] for o in s.orders]
    assert int(np.argmax(orders)) == 2
    assert orders[2] > 100 * max(orders[i] for i in range(len(orders)) if i != 2)


def test_third_octave_bands_sum_to_band_energy(s):
    x = np.random.default_rng(2).normal(size=int(s.fs))
    xb = filters.bandpass(x, s.fs, *s.band)
    spec, freqs = filters.rfft_with_freqs(xb, s.fs)
    toct = spectral.third_octave_energies(freqs, spectral.periodogram_power(spec, xb.size), s.toct_edges)
    assert len(s.toct_edges) - 1 == 32
    assert toct.sum() == pytest.approx(np.mean(xb**2), rel=0.01)


def test_bandpass_removes_dc_and_out_of_band(s):
    t = t_axis(s)
    x = 2.0 + np.sin(2 * np.pi * 5.0 * t) + np.sin(2 * np.pi * 11000.0 * t) + np.sin(2 * np.pi * 500.0 * t)
    y = filters.bandpass(x, s.fs, *s.band)
    assert np.mean(y**2) == pytest.approx(0.5, rel=0.01)


def test_top_peaks_sorted(s):
    rows, _ = extract_burst(tri(impulse_train(s, 157.0)), s, 30.0, FAULTS)
    amps = [rows["res"][f"envpk_{i:02d}_g"] for i in range(1, s.top_peaks + 1)]
    amps = [a for a in amps if not np.isnan(a)]
    assert amps == sorted(amps, reverse=True)


def test_extraction_deterministic(s):
    x = np.random.default_rng(9).normal(size=(3, int(s.fs)))
    a, _ = extract_burst(x, s, 100.0, FAULTS)
    b, _ = extract_burst(x.copy(), s, 100.0, FAULTS)
    assert a == b
    assert set(a["x"]) == set(indicator_columns(s))


def test_sk_band_respects_minimum_width(s):
    rng = np.random.default_rng(4)
    x = impulse_train(s, 1200.0, fn=7000.0, noise=0.0) + rng.normal(size=int(s.fs)) * 0.05
    spec, freqs = filters.rfft_with_freqs(x, s.fs)
    lo, hi, _ = envelope.select_band(spec, freqs, *s.sk_band, s.sk_levels, min_width=2 * 1200.0)
    assert hi - lo >= 2 * 1200.0


def test_high_rate_outer_race_peak_found(s):
    rate = 1400.0
    x = impulse_train(s, rate, fn=6800.0, zeta=0.06, noise=0.1, seed=8)
    rows, _ = extract_burst(
        tri(x), s, rate / 8.426, {"bpfo": rate, "bpfi": rate * 1.255, "bsf": rate * 0.5, "ftf": rate * 0.053}
    )
    assert rows["res"]["envpk_01_hz"] == pytest.approx(rate, rel=0.03)


def test_local_floor_excludes_fault_band(s):
    f = np.arange(0, 2000, 1.0)
    amp = np.ones_like(f)
    amp[(f >= 97) & (f <= 103)] = 50.0
    floor = envelope.local_noise_floor(f, amp, 100.0, 0.03, 0.25, 20.0)
    assert floor == pytest.approx(1.0)
