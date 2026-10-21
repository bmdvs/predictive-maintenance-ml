"""Burst to scalar indicators for one triaxial measurement.

Input: a (3, n) acceleration burst in g, the shaft speed and the bearing fault frequencies.
Output: one indicator row per axis and for the resultant, plus envelope detection diagnostics.
Used by the sensor-platform export and by the pipeline when retained bursts are re-read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pdm.features import envelope, filters, spectral
from pdm.features.kinematics import FAULT_KEYS

AXES = ("x", "y", "z")
RESULTANT = "res"


@dataclass(frozen=True)
class ExtractionSettings:
    fs: float
    band: tuple[float, float]
    vel_band: tuple[float, float]
    hf_band: tuple[float, float]
    segment_s: float
    averages: int
    orders: tuple[int, ...]
    order_half_width: float
    sk_band: tuple[float, float]
    sk_levels: int
    cover_bpfi_harmonics: int
    cover_shaft_sidebands: int
    band_tie_tolerance: float
    harmonics: int
    fault_half_width: float
    toct_edges: np.ndarray
    top_peaks: int
    local_floor_half_width_frac: float
    local_floor_min_half_width_hz: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ExtractionSettings:
        meas, ind = cfg["measurement"], cfg["indicators"]
        toct = ind["third_octave"]
        return cls(
            fs=float(meas["fs_hz"]),
            band=tuple(meas["band_hz"]),
            vel_band=tuple(ind["velocity_band_hz"]),
            hf_band=tuple(ind["hf_band_hz"]),
            segment_s=float(ind["spectrum"]["segment_s"]),
            averages=int(ind["spectrum"]["averages"]),
            orders=tuple(ind["order_bands"]["orders"]),
            order_half_width=float(ind["order_bands"]["half_width_frac"]),
            sk_band=tuple(ind["envelope"]["sk_band_hz"]),
            sk_levels=int(ind["envelope"]["sk_levels"]),
            cover_bpfi_harmonics=int(ind["envelope"]["band_min_coverage"]["bpfi_harmonics"]),
            cover_shaft_sidebands=int(ind["envelope"]["band_min_coverage"]["shaft_sidebands"]),
            band_tie_tolerance=float(ind["envelope"]["band_selection_tie_tolerance"]),
            harmonics=int(ind["envelope"]["harmonics"]),
            fault_half_width=float(ind["envelope"]["band_half_width_frac"]),
            toct_edges=spectral.third_octave_edges(int(toct["bands"]), *toct["range_hz"]),
            top_peaks=int(ind["top_peaks"]),
            local_floor_half_width_frac=float(ind["envelope"]["local_floor"]["half_width_frac"]),
            local_floor_min_half_width_hz=float(ind["envelope"]["local_floor"]["min_half_width_hz"]),
        )


def indicator_columns(settings: ExtractionSettings) -> list[str]:
    cols = ["accel_rms_g", "peak_g", "crest_factor", "kurtosis", "vel_rms_mm_s", "hf_rms_g"]
    cols += [f"ord_{o:02d}_g2" for o in settings.orders]
    cols += [f"env_{k}_{h}_g2" for k in FAULT_KEYS for h in range(1, settings.harmonics + 1)]
    cols += ["env_bpfi_sb_lo_g2", "env_bpfi_sb_hi_g2", "env_bsf_sb_lo_g2", "env_bsf_sb_hi_g2"]
    cols += [f"toct_{b + 1:02d}_g2" for b in range(len(settings.toct_edges) - 1)]
    cols += [f"envpk_{i + 1:02d}_hz" for i in range(settings.top_peaks)]
    cols += [f"envpk_{i + 1:02d}_g" for i in range(settings.top_peaks)]
    return cols


def _local(env_f: np.ndarray, amp: np.ndarray, center: float, s: ExtractionSettings) -> float:
    return envelope.local_noise_floor(
        env_f, amp, center, s.fault_half_width, s.local_floor_half_width_frac, s.local_floor_min_half_width_hz
    )


def coverage_width(fault_hz: dict[str, float], shaft_hz: float, s: ExtractionSettings) -> float:
    """Demodulation bandwidth needed to cover the BPFI harmonics and their shaft-speed sidebands."""
    return s.cover_bpfi_harmonics * fault_hz["bpfi"] + s.cover_shaft_sidebands * shaft_hz


def select_point_band(
    bursts: list[np.ndarray], s: ExtractionSettings, fault_hz_rated: dict[str, float], shaft_hz_rated: float
) -> tuple[float, float]:
    """Demodulation band for a measurement point, from its baseline bursts.

    Among candidate bands at least as wide as the coverage width at rated speed, the band with the
    highest spectral energy concentration (mean power per Hz relative to the whole demodulation range)
    is selected; bands within the tie tolerance of the best are separated by spectral kurtosis.
    """
    lo, hi = s.sk_band
    min_width = coverage_width(fault_hz_rated, shaft_hz_rated, s)
    cands = [
        b for b in envelope.candidate_bands(lo, hi, s.sk_levels) if b[1] - b[0] >= min_width or b == (lo, hi)
    ]
    if not bursts:
        return lo, hi
    conc = np.zeros(len(cands))
    sk = np.zeros(len(cands))
    for burst in bursts:
        x = filters.remove_dc(np.asarray(burst, dtype=float))
        spec, freqs = filters.rfft_with_freqs(x, s.fs)
        power = np.abs(spec) ** 2
        full = (freqs >= lo) & (freqs < hi)
        full_density = power[:, full].mean(axis=-1)
        for j, (b_lo, b_hi) in enumerate(cands):
            m = (freqs >= b_lo) & (freqs < b_hi)
            conc[j] += float(np.mean(power[:, m].mean(axis=-1) / full_density))
            sk[j] += np.mean([envelope.spectral_kurtosis(spec[a][m]) for a in range(spec.shape[0])])
    best = conc.max()
    tied = np.nonzero(conc >= best * (1 - s.band_tie_tolerance))[0]
    return cands[int(tied[np.argmax(sk[tied])])]


def _envelope_block(
    env_f: np.ndarray,
    amp: np.ndarray,
    bandwidth: float,
    shaft_hz: float,
    fault_hz: dict[str, float],
    s: ExtractionSettings,
) -> tuple[dict[str, float], dict[str, float]]:
    row: dict[str, float] = {}
    diag: dict[str, float] = {}
    floor = envelope.noise_floor(env_f, amp, bandwidth)
    diag["env_floor_g"] = floor

    def covered(center: float) -> bool:
        return center * (1 + s.fault_half_width) <= bandwidth

    for k in FAULT_KEYS:
        for h in range(1, s.harmonics + 1):
            center = h * fault_hz[k]
            if not covered(center):
                row[f"env_{k}_{h}_g2"] = np.nan
                if h == 1:
                    diag[f"envr_{k}"] = np.nan
                    diag[f"envp_{k}"] = np.nan
                continue
            energy, rms_amp = envelope.band_values(env_f, amp, center, s.fault_half_width)
            row[f"env_{k}_{h}_g2"] = energy
            if h == 1:
                lf = _local(env_f, amp, center, s)
                diag[f"envr_{k}"] = rms_amp / lf if lf > 0 else 0.0
                peak = envelope.band_peak(env_f, amp, center, s.fault_half_width)
                diag[f"envp_{k}"] = peak / lf if lf > 0 else 0.0
    sidebands = {
        "bpfi_sb_lo": fault_hz["bpfi"] - shaft_hz,
        "bpfi_sb_hi": fault_hz["bpfi"] + shaft_hz,
        "bsf_sb_lo": fault_hz["bsf"] - fault_hz["ftf"],
        "bsf_sb_hi": fault_hz["bsf"] + fault_hz["ftf"],
    }
    for name, f0 in sidebands.items():
        if not covered(f0):
            row[f"env_{name}_g2"] = np.nan
            diag[f"envr_{name}"] = np.nan
            diag[f"envp_{name}"] = np.nan
            continue
        energy, rms_amp = envelope.band_values(env_f, amp, f0, s.fault_half_width)
        row[f"env_{name}_g2"] = energy
        lf = _local(env_f, amp, f0, s)
        diag[f"envr_{name}"] = rms_amp / lf if lf > 0 else 0.0
        diag[f"envp_{name}"] = envelope.band_peak(env_f, amp, f0, s.fault_half_width) / lf if lf > 0 else 0.0
    pk_f, pk_a = envelope.top_peaks(env_f, amp, s.top_peaks, min_hz=1.0)
    for i in range(s.top_peaks):
        row[f"envpk_{i + 1:02d}_hz"] = float(pk_f[i])
        row[f"envpk_{i + 1:02d}_g"] = float(pk_a[i])
    return row, diag


def extract_burst(
    burst: np.ndarray,
    settings: ExtractionSettings,
    shaft_hz: float,
    fault_hz: dict[str, float],
    band: tuple[float, float] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Indicator rows and envelope diagnostics for axes x, y, z and the resultant.

    Args:
        burst: acceleration in g, shape (3, n).
        settings: extraction settings from config.
        shaft_hz: shaft speed of the monitored component.
        fault_hz: BPFO, BPFI, BSF and FTF in Hz at that speed.
        band: the point's demodulation band; when None it is selected for this burst.
    Returns:
        (rows, diagnostics), each keyed by axis name.
    """
    s = settings
    n = burst.shape[-1]
    x = filters.remove_dc(np.asarray(burst, dtype=float))
    spec, freqs = filters.rfft_with_freqs(x, s.fs)
    spec = spec * filters.band_mask(freqs, *s.band)
    xb = np.fft.irfft(spec, n=n, axis=-1)

    ms = np.mean(xb**2, axis=-1)
    rms = np.sqrt(ms)
    peak = np.max(np.abs(xb), axis=-1)
    crest = np.where(rms > 0, peak / np.where(rms > 0, rms, 1.0), 0.0)
    kurt = np.where(ms > 0, np.mean(xb**4, axis=-1) / np.where(ms > 0, ms, 1.0) ** 2, 0.0)
    vel = filters.velocity_rms_mm_s(spec, freqs, n, *s.vel_band)
    hf_ms = filters.band_mean_square(spec, freqs, n, *s.hf_band)

    w_freqs, w_power = spectral.averaged_spectrum(xb, s.fs, s.segment_s, s.averages)
    orders = spectral.order_band_energies(w_freqs, w_power, shaft_hz, list(s.orders), s.order_half_width)
    toct = spectral.third_octave_energies(freqs, spectral.periodogram_power(spec, n), s.toct_edges)

    min_width = coverage_width(fault_hz, shaft_hz, s)
    rows: dict[str, dict[str, float]] = {}
    diags: dict[str, dict[str, float]] = {}
    env_power_sum = None
    env_f_common = None
    bw_max = 0.0
    for i, axis in enumerate(AXES):
        if band is None:
            b_lo, b_hi, _ = envelope.select_band(spec[i], freqs, *s.sk_band, s.sk_levels, min_width)
        else:
            b_lo, b_hi = band
        env_f, amp = envelope.envelope_spectrum(spec[i], freqs, n, s.fs, b_lo, b_hi)
        row = {
            "accel_rms_g": float(rms[i]),
            "peak_g": float(peak[i]),
            "crest_factor": float(crest[i]),
            "kurtosis": float(kurt[i]),
            "vel_rms_mm_s": float(vel[i]),
            "hf_rms_g": float(np.sqrt(hf_ms[i])),
        }
        row.update({f"ord_{o:02d}_g2": float(orders[i, j]) for j, o in enumerate(s.orders)})
        row.update({f"toct_{b + 1:02d}_g2": float(toct[i, b]) for b in range(toct.shape[-1])})
        env_row, diag = _envelope_block(env_f, amp, b_hi - b_lo, shaft_hz, fault_hz, s)
        diag.update({"sk_band_lo_hz": b_lo, "sk_band_hi_hz": b_hi})
        row.update(env_row)
        rows[axis], diags[axis] = row, diag
        if env_power_sum is None or amp.size > env_power_sum.size:
            grown = np.zeros(amp.size)
            if env_power_sum is not None:
                grown[: env_power_sum.size] = env_power_sum
            env_power_sum, env_f_common = grown, env_f
        env_power_sum[: amp.size] += amp**2
        bw_max = max(bw_max, b_hi - b_lo)

    res = {
        "accel_rms_g": float(np.sqrt(ms.sum())),
        "peak_g": float(peak.max()),
        "crest_factor": float(crest.max()),
        "kurtosis": float(kurt.max()),
        "vel_rms_mm_s": float(np.sqrt(np.sum(vel**2))),
        "hf_rms_g": float(np.sqrt(hf_ms.sum())),
    }
    for col in rows["x"]:
        if col.startswith(("ord_", "toct_", "env_")):
            res[col] = float(sum(rows[a][col] for a in AXES))
    res_amp = np.sqrt(env_power_sum)
    env_row, res_diag = _envelope_block(env_f_common, res_amp, bw_max, shaft_hz, fault_hz, s)
    res.update({k: v for k, v in env_row.items() if k.startswith("envpk_")})
    rows[RESULTANT], diags[RESULTANT] = res, res_diag
    return rows, diags
