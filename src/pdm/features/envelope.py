"""Envelope analysis: spectral-kurtosis band selection, Hilbert envelope and envelope spectrum.

Input: the one-sided FFT of a band-limited acceleration burst. Outputs: the selected demodulation
band, the envelope amplitude spectrum, fault-band energies, sideband energies, the noise floor and
the largest envelope peaks.
"""

from __future__ import annotations

import numpy as np


def candidate_bands(lo: float, hi: float, levels: int) -> list[tuple[float, float]]:
    """Dyadic bands of the kurtogram tree plus half-shifted bands at each level."""
    bands: list[tuple[float, float]] = []
    for level in range(levels + 1):
        count = 2**level
        width = (hi - lo) / count
        bands += [(lo + i * width, lo + (i + 1) * width) for i in range(count)]
        if level >= 1:
            bands += [(lo + (i + 0.5) * width, lo + (i + 1.5) * width) for i in range(count - 1)]
    return bands


def spectral_kurtosis(spec_band: np.ndarray) -> float:
    c = np.fft.ifft(spec_band)
    p = np.abs(c) ** 2
    mean_p = p.mean()
    if mean_p <= 0:
        return 0.0
    return float((p**2).mean() / mean_p**2 - 2.0)


def select_band(
    spec: np.ndarray, freqs: np.ndarray, lo: float, hi: float, levels: int, min_width: float = 0.0
) -> tuple[float, float, float]:
    """Band in [lo, hi] with the largest spectral kurtosis of its complex envelope.

    Candidate bands narrower than min_width are skipped; the full band is always a candidate.
    """
    best = (lo, hi, -np.inf)
    for b_lo, b_hi in candidate_bands(lo, hi, levels):
        if b_hi - b_lo < min_width and (b_lo, b_hi) != (lo, hi):
            continue
        m = (freqs >= b_lo) & (freqs < b_hi)
        if m.sum() < 2:
            continue
        sk = spectral_kurtosis(spec[m])
        if sk > best[2]:
            best = (b_lo, b_hi, sk)
    return best


def envelope_spectrum(
    spec: np.ndarray, freqs: np.ndarray, n: int, fs: float, b_lo: float, b_hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum (g) of the Hilbert envelope of the signal inside [b_lo, b_hi)."""
    m = (freqs >= b_lo) & (freqs < b_hi)
    nb = int(m.sum())
    size = 2 * nb
    z_spec = np.zeros(size, dtype=complex)
    z_spec[:nb] = spec[m]
    env = np.abs(np.fft.ifft(z_spec)) * size * 2.0 / n
    e_spec = np.fft.rfft(env - env.mean())
    amp = 2.0 * np.abs(e_spec) / size
    amp[0] = 0.0
    return np.arange(amp.size) * (fs / n), amp


def band_values(
    env_freqs: np.ndarray, amp: np.ndarray, center: float, half_width_frac: float
) -> tuple[float, float]:
    """Energy (g^2) and RMS amplitude (g) of the envelope spectrum within center +/- half width."""
    half = max(center * half_width_frac, env_freqs[1] - env_freqs[0])
    m = (env_freqs >= center - half) & (env_freqs <= center + half)
    if not m.any():
        return 0.0, 0.0
    a2 = amp[m] ** 2
    return float(a2.sum() / 2.0), float(np.sqrt(a2.mean()))


def noise_floor(env_freqs: np.ndarray, amp: np.ndarray, bandwidth: float) -> float:
    m = (env_freqs >= 1.0) & (env_freqs <= bandwidth)
    return float(np.median(amp[m])) if m.any() else 0.0


def top_peaks(
    env_freqs: np.ndarray, amp: np.ndarray, count: int, min_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    inner = amp[1:-1]
    is_peak = (inner > amp[:-2]) & (inner >= amp[2:])
    idx = np.nonzero(is_peak)[0] + 1
    idx = idx[env_freqs[idx] >= min_hz]
    order = idx[np.argsort(amp[idx])[::-1][:count]]
    f = np.full(count, np.nan)
    a = np.full(count, np.nan)
    f[: order.size] = env_freqs[order]
    a[: order.size] = amp[order]
    return f, a


def local_noise_floor(
    env_freqs: np.ndarray,
    amp: np.ndarray,
    center: float,
    band_half_width_frac: float,
    half_width_frac: float,
    min_half_width_hz: float,
) -> float:
    """Median envelope amplitude around center, excluding the fault band itself."""
    half = max(center * half_width_frac, min_half_width_hz)
    band = max(center * band_half_width_frac, env_freqs[1] - env_freqs[0])
    near = np.abs(env_freqs - center)
    m = (near <= half) & (near > band) & (env_freqs >= 1.0)
    return float(np.median(amp[m])) if m.any() else 0.0


def band_peak(env_freqs: np.ndarray, amp: np.ndarray, center: float, half_width_frac: float) -> float:
    """Largest envelope amplitude (g) within center +/- half width."""
    half = max(center * half_width_frac, env_freqs[1] - env_freqs[0])
    m = (env_freqs >= center - half) & (env_freqs <= center + half)
    return float(amp[m].max()) if m.any() else 0.0
