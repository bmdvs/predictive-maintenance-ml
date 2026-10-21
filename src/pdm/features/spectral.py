"""Averaged spectrum, shaft-order band energies and one-third-octave band energies.

Inputs: band-limited acceleration in g with time on the last axis. Outputs: power per bin in g^2
(bins sum to the mean square), order-band energies and 32 fixed log-spaced band energies.
"""

from __future__ import annotations

import numpy as np


def averaged_spectrum(
    x: np.ndarray, fs: float, segment_s: float, averages: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed spectrum averaged over consecutive non-overlapping segments."""
    seg = int(round(segment_s * fs))
    usable = x[..., : seg * averages]
    segs = usable.reshape(*x.shape[:-1], averages, seg)
    win = np.hanning(seg)
    spec = np.fft.rfft(segs * win, axis=-1)
    power = 2.0 * np.abs(spec) ** 2 / (seg * np.sum(win**2))
    power[..., 0] /= 2.0
    return np.fft.rfftfreq(seg, 1.0 / fs), power.mean(axis=-2)


def periodogram_power(spec: np.ndarray, n: int) -> np.ndarray:
    """Power per bin (g^2) of an unwindowed one-sided FFT."""
    power = 2.0 * np.abs(spec) ** 2 / n**2
    power[..., 0] /= 2.0
    return power


def band_energy(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Sum of power in [lo, hi]; when no bin center falls inside, the nearest bin is used."""
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        idx = int(np.argmin(np.abs(freqs - 0.5 * (lo + hi))))
        return power[..., idx]
    return power[..., m].sum(axis=-1)


def order_band_energies(
    freqs: np.ndarray, power: np.ndarray, shaft_hz: float, orders: list[int], half_width_frac: float
) -> np.ndarray:
    out = [
        band_energy(freqs, power, o * shaft_hz * (1 - half_width_frac), o * shaft_hz * (1 + half_width_frac))
        for o in orders
    ]
    return np.stack(out, axis=-1)


def third_octave_edges(n_bands: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    return lo_hz * (hi_hz / lo_hz) ** (np.arange(n_bands + 1) / n_bands)


def third_octave_energies(freqs: np.ndarray, power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, freqs, side="right") - 1
    n_bands = len(edges) - 1
    valid = (idx >= 0) & (idx < n_bands)
    out = np.zeros((*power.shape[:-1], n_bands))
    for b in range(n_bands):
        sel = valid & (idx == b)
        if sel.any():
            out[..., b] = power[..., sel].sum(axis=-1)
    return out
