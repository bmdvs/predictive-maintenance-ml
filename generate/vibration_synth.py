"""Triaxial acceleration burst for one measurement point and hour.

Inputs: point class parameters, operating condition (speed, load, flutes), bearing state (health
index, defect location), episode state (lubrication, cavitation, wear) and the mounting transfer
function. Output: a (3, n) burst in g and the housing temperature. Defect impacts follow the
impulse-train model with slip jitter, load-zone or cage modulation and an SDOF housing resonance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_CLASS_GROUP = {
    "spindle_front": "spindle",
    "spindle_rear": "spindle",
    "tsc_pump": "pump",
    "flood_pump": "pump",
    "compressor_motor": "compressor",
    "compressor_airend": "compressor",
}
_FAULT_FOR_DEFECT = {"outer_race": "bpfo", "inner_race": "bpfi", "ball": "bsf"}


@dataclass
class BurstInputs:
    point_class: str
    fs: float
    n: int
    band: tuple[float, float]
    hf_band: tuple[float, float]
    shaft_hz: float
    rated_hz: float
    load_pct: float
    flutes: int
    fault_hz: dict[str, float]
    resonance_hz: float
    mount_gain: float = 1.0
    hi: float = 0.0
    defect: str | None = None
    lube_rise: float = 0.0
    cavitation_rise: float = 0.0
    wear_frac: float = 0.0
    lobe_hz: float = 0.0
    am_inner: float = 0.0
    housing_gain_db: float = 0.0
    contamination_rise: float = 0.0
    one_x_gain: float = 1.0
    faults: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _band_noise(
    rng: np.random.Generator, n: int, fs: float, lo: float, hi: float, rms: np.ndarray
) -> np.ndarray:
    """White Gaussian noise limited to [lo, hi] with the given RMS per axis."""
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    m = (freqs >= lo) & (freqs <= hi)
    spec = np.zeros((rms.size, freqs.size), dtype=complex)
    spec[:, m] = rng.standard_normal((rms.size, m.sum())) + 1j * rng.standard_normal((rms.size, m.sum()))
    x = np.fft.irfft(spec, n=n, axis=-1)
    x /= x.std(axis=-1, keepdims=True)
    return x * rms[:, None]


def housing_shape(x: np.ndarray, fs: float, fn: float, zeta: float, gain_db: float) -> np.ndarray:
    """Broadband floor passed through the housing response: the SDOF resonance region rises gain_db above
    the floor while the floor itself is unchanged away from the resonance."""
    if gain_db <= 0:
        return x
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    r = freqs / fn
    h_norm = 2 * zeta / np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)
    g = 10 ** (gain_db / 20.0)
    shape = np.sqrt(1 + (g**2 - 1) * h_norm**2)
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * shape, n=n, axis=-1)


def broadband_rms(sig: dict, point_class: str, load_pct: float, regime: dict) -> float:
    bb = sig["background"]["broadband_g_rms"]
    group = _CLASS_GROUP[point_class]
    if group != "spindle":
        return float(bb[group])
    light_hi = regime["load_bands_pct"]["light"][1]
    heavy_lo = regime["load_bands_pct"]["heavy"][0]
    w = np.clip((load_pct - light_hi) / (heavy_lo - light_hi), 0.0, 1.0)
    return float(bb["spindle_light"] + w * (bb["spindle_heavy"] - bb["spindle_light"]))


def _periodic(
    t: np.ndarray,
    rng: np.random.Generator,
    comps: list[tuple[float, float]],
    fmax: float,
    cap_noise_var: float,
    cap: float,
    gain: float,
    added: dict[float, float] | None = None,
) -> np.ndarray:
    """Periodic background for one axis. Healthy components are limited to the variance share cap;
    fault amplitudes in `added` (g, keyed by frequency) add in phase with the healthy component at
    that frequency, or with a fresh phase where there is none."""
    phases, var = [], 0.0
    for f, a in comps:
        if 0 < f < fmax and a > 0:
            phases.append((f, a, rng.uniform(0, 2 * np.pi)))
            var += a**2 / 2
    scale = 1.0
    if var > 0 and var / (var + cap_noise_var) > cap:
        scale = float(np.sqrt(cap / (1 - cap) * cap_noise_var / var))
    added = dict(added or {})
    x = np.zeros_like(t)
    for f, a, ph in phases:
        x += (a * scale * gain + added.pop(f, 0.0)) * np.sin(2 * np.pi * f * t + ph)
    for f, a in sorted(added.items()):
        if 0 < f < fmax and a > 0:
            x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return x


def fault_additions(
    faults: list[dict], shaft_hz: float, base: np.ndarray, axis_gain: np.ndarray, orders: int
) -> tuple[list[dict[float, float]], float]:
    """Per-axis fault amplitudes (g) by frequency and the low-frequency floor rise (dB).

    Axes are x and y radial, z axial. base holds the healthy 1x..Nx amplitudes at radial gain.
    Imbalance raises 1x radially with an axial increment at its configured share; misalignment
    raises 2x and 1x with an axial increment at its share of the radial increment; looseness sets
    3x..Nx to its gain times the healthy 3x level, adds a 0.5x subharmonic and raises the floor.
    """
    add: list[dict[float, float]] = [{}, {}, {}]
    axial_rel = float(axis_gain[2] / axis_gain[0]) if axis_gain[0] > 0 else 1.0
    floor_db = 0.0

    def put(order: float, radial: float, axial: float) -> None:
        f = order * shaft_hz
        for ax, amp in ((0, radial), (1, radial), (2, axial)):
            add[ax][f] = add[ax].get(f, 0.0) + amp

    b = np.zeros(orders)
    b[: min(base.size, orders)] = base[:orders]
    for c in faults:
        lev = float(c["level"])
        if lev <= 0:
            continue
        if c["kind"] == "imbalance":
            d1 = lev * (c["one_x"] - 1) * b[0]
            put(1, d1, c["axial"] * d1)
        elif c["kind"] == "misalignment":
            d1 = lev * (c["one_x"] - 1) * b[0]
            d2 = lev * (c["two_x"] - 1) * b[1]
            put(1, d1, c["axial"] * d1)
            put(2, d2, c["axial"] * d2)
        elif c["kind"] == "looseness":
            target = lev * c["harm"] * b[2]
            for k in range(3, orders + 1):
                d = max(target - b[k - 1], 0.0)
                put(k, d, axial_rel * d)
            h = lev * c["half_x"] * b[0]
            put(0.5, h, axial_rel * h)
            floor_db = max(floor_db, lev * c["floor_db"])
    return add, floor_db


def _impact_train(
    rng: np.random.Generator, inp: BurstInputs, sig: dict, jitter: float, amplitude: float
) -> np.ndarray:
    group = _CLASS_GROUP[inp.point_class]
    zeta = sig["damping_ratio"][group]
    wn = 2 * np.pi * inp.resonance_hz
    wd = wn * np.sqrt(1 - zeta**2)
    tau = 1.0 / (zeta * wn)
    resp_len = min(int(np.ceil(8 * tau * inp.fs)), inp.n)
    tr = np.arange(resp_len) / inp.fs
    resp = np.exp(-zeta * wn * tr) * np.sin(wd * tr)
    resp /= np.abs(resp).max()
    f_def = inp.fault_hz[_FAULT_FOR_DEFECT[inp.defect]]
    period = 1.0 / f_def
    total = inp.n + resp_len
    count = int(np.ceil(total / inp.fs / period)) + 2
    gaps = period * (1 + jitter * rng.standard_normal(count))
    times = rng.uniform(0, period) + np.concatenate([[0.0], np.cumsum(np.abs(gaps))[:-1]])
    times = times[times < total / inp.fs]
    am = sig["amplitude_modulation_depth"]
    if inp.defect == "inner_race":
        amps = 1 + inp.am_inner * np.cos(2 * np.pi * inp.shaft_hz * times + rng.uniform(0, 2 * np.pi))
    elif inp.defect == "ball":
        amps = 1 + am["ball_at_cage"] * np.cos(
            2 * np.pi * inp.fault_hz["ftf"] * times + rng.uniform(0, 2 * np.pi)
        )
    else:
        amps = np.ones_like(times)
    amps = amps / amps.max()
    sigma = sig["severity_to_impact"]["amplitude_scatter_sigma_log"]
    amps = amps * np.exp(sigma * rng.standard_normal(amps.size) - sigma**2 / 2)
    pulses = np.zeros(total)
    np.add.at(pulses, np.minimum((times * inp.fs).astype(int), total - 1), amps * amplitude)
    size = 1 << int(np.ceil(np.log2(total + resp_len)))
    out = np.fft.irfft(np.fft.rfft(pulses, size) * np.fft.rfft(resp, size), size)[:total]
    return out[resp_len:]


def render_burst(
    inp: BurstInputs, sig: dict, regime: dict, stages: dict, hi_fail: float, rng: np.random.Generator
) -> np.ndarray:
    """Acceleration burst in g, shape (3, n). stages holds the health-index stage thresholds."""
    stage4_lo = stages["stage4"]
    t = np.arange(inp.n) / inp.fs
    group = _CLASS_GROUP[inp.point_class]
    gains_noise = np.asarray(sig["axis_gain"]["noise"], dtype=float)
    gains_imp = np.asarray(sig["axis_gain"]["impacts"], dtype=float)
    s4 = float(np.clip((inp.hi - stage4_lo) / (hi_fail - stage4_lo), 0.0, 1.0)) if inp.defect else 0.0
    st4 = sig["stage4"]
    wear = sig.get("_wear", {})

    bb = broadband_rms(sig, inp.point_class, inp.load_pct, regime)
    bb *= 1 + s4 * (st4["broadband_gain_at_failure"] - 1)
    bb *= 1 + inp.wear_frac * (wear.get("broadband_gain", 1.0) - 1)
    x = _band_noise(rng, inp.n, inp.fs, *inp.band, bb * gains_noise * inp.mount_gain)
    x = housing_shape(x, inp.fs, inp.resonance_hz, sig["damping_ratio"][group], inp.housing_gain_db)

    hf_share = (inp.hf_band[1] - inp.hf_band[0]) / (inp.band[1] - inp.band[0])
    stage1 = (
        sig["stage1_hf_rise_at_stage2"] * float(np.clip(inp.hi / stages["stage2"], 0.0, 1.0))
        if inp.defect
        else 0.0
    )
    rise = (1 + inp.lube_rise) * (1 + inp.cavitation_rise) * (1 + stage1) * (1 + inp.contamination_rise)
    if rise > 1:
        extra_rms = bb * np.sqrt(hf_share * (rise**2 - 1))
        x += _band_noise(rng, inp.n, inp.fs, *inp.hf_band, extra_rms * gains_noise * inp.mount_gain)

    bg = sig["background"]
    speed_ratio = inp.shaft_hz / inp.rated_hz if inp.rated_hz > 0 else 1.0
    harm_key = {"spindle": "spindle", "pump": "pump"}.get(group, inp.point_class)
    harm = np.asarray(bg["shaft_harmonics_g"][harm_key], dtype=float) * speed_ratio**2
    harm = harm * (1 + s4 * (st4["shaft_harmonic_gain_at_failure"] - 1))
    harm[0] *= 1 + inp.wear_frac * (wear.get("shaft_1x_gain", 1.0) - 1)
    harm[0] *= inp.one_x_gain
    comps = [((k + 1) * inp.shaft_hz, a) for k, a in enumerate(harm)]
    if group == "spindle" and inp.flutes > 0:
        tp = inp.flutes * inp.shaft_hz
        comps += [
            ((k + 1) * tp, c * inp.load_pct) for k, c in enumerate(bg["tooth_pass_harmonics_g_per_load_pct"])
        ]
    if inp.point_class == "compressor_airend" and inp.lobe_hz > 0:
        comps += [((k + 1) * inp.lobe_hz, a) for k, a in enumerate(bg["lobe_pass_harmonics_g"])]
    added, floor_db = (
        fault_additions(inp.faults, inp.shaft_hz, harm, gains_noise, int(sig["fault_render"]["max_order"]))
        if inp.faults
        else ([{}, {}, {}], 0.0)
    )
    cap = bg["periodic_variance_share_max"]
    for axis in range(3):
        noise_var = (bb * gains_noise[axis]) ** 2
        per = _periodic(t, rng, comps, inp.band[1], noise_var, cap, gains_noise[axis], added[axis])
        x[axis] += per * inp.mount_gain
    if floor_db > 0:
        lo, hi = sig["fault_render"]["floor_band_hz"]
        share = (hi - lo) / (inp.band[1] - inp.band[0])
        floor_rms = bb * np.sqrt(share * (10 ** (floor_db / 10) - 1))
        x += _band_noise(rng, inp.n, inp.fs, lo, hi, floor_rms * gains_noise * inp.mount_gain)

    if inp.defect and inp.hi > 0:
        imp = sig["severity_to_impact"]
        amplitude = imp["impact_peak_g_at_hi_1"][group] * inp.hi ** imp["exponent"]
        jitter = sig["slip_jitter_frac"][group] + s4 * (
            st4["jitter_frac_at_failure"] - sig["slip_jitter_frac"][group]
        )
        train = _impact_train(rng, inp, sig, jitter, amplitude)
        train = housing_shape(
            train, inp.fs, inp.resonance_hz, sig["damping_ratio"][group], inp.housing_gain_db
        )
        x += train[None, :] * gains_imp[:, None] * inp.mount_gain

    snr = sig["measurement_noise_snr_db"][group]
    noise_rms = x.std(axis=-1) / 10 ** (snr / 20)
    x += rng.standard_normal(x.shape) * noise_rms[:, None]
    return x
