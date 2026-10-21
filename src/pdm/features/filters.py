"""Frequency-domain filtering and integration on uniformly sampled acceleration bursts.

Inputs are arrays with time on the last axis, in g. Outputs are band-limited signals, band mean
squares and velocity RMS in mm/s. All operations use one real FFT per signal.
"""

from __future__ import annotations

import numpy as np

from pdm.constants import MM_PER_M, STANDARD_GRAVITY_M_S2


def rfft_with_freqs(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[-1]
    return np.fft.rfft(x, axis=-1), np.fft.rfftfreq(n, 1.0 / fs)


def band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs <= hi)


def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    spec, freqs = rfft_with_freqs(x, fs)
    spec = spec * band_mask(freqs, lo, hi)
    return np.fft.irfft(spec, n=x.shape[-1], axis=-1)


def band_mean_square(spec: np.ndarray, freqs: np.ndarray, n: int, lo: float, hi: float) -> np.ndarray:
    """Mean square of the time signal inside [lo, hi] from its one-sided FFT (Parseval)."""
    m = band_mask(freqs, lo, hi) & (freqs > 0)
    return 2.0 * np.sum(np.abs(spec[..., m]) ** 2, axis=-1) / n**2


def velocity_rms_mm_s(spec: np.ndarray, freqs: np.ndarray, n: int, lo: float, hi: float) -> np.ndarray:
    """Velocity RMS in mm/s from an acceleration spectrum in g, integrated in the frequency domain."""
    m = band_mask(freqs, lo, hi) & (freqs > 0)
    vel = spec[..., m] * STANDARD_GRAVITY_M_S2 / (2.0 * np.pi * freqs[m])
    return np.sqrt(2.0 * np.sum(np.abs(vel) ** 2, axis=-1) / n**2) * MM_PER_M
