"""Pilot figures: spectrum and envelope spectrum of retained bursts at stage 0, 2 and 3, and the
four-stage indicator trends for the forced instance. PNG files under runs/<run_id>/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm.config import Register
from pdm.features import envelope, filters, spectral
from pdm.features.burst import ExtractionSettings
from pdm.features.kinematics import bearing_fault_orders

_STAGE_COLORS = {1: "#fde7c4", 2: "#fbc98e", 3: "#f59e5b", 4: "#e0643a"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _burst_panels(
    path: Path, burst: np.ndarray, s: ExtractionSettings, axis: int, faults: dict[str, float], title: str
) -> None:
    x = filters.remove_dc(burst[axis].astype(float))
    spec, freqs = filters.rfft_with_freqs(x, s.fs)
    spec = spec * filters.band_mask(freqs, *s.band)
    xb = np.fft.irfft(spec, n=x.size)
    wf, wp = spectral.averaged_spectrum(xb, s.fs, s.segment_s, s.averages)
    min_width = s.cover_bpfi_harmonics * faults["bpfi"]
    b_lo, b_hi, sk = envelope.select_band(spec, freqs, *s.sk_band, s.sk_levels, min_width)
    ef, ea = envelope.envelope_spectrum(spec, freqs, x.size, s.fs, b_lo, b_hi)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5))
    inband = (wf >= s.band[0]) & (wf <= s.band[1])
    a1.semilogy(wf[inband], np.sqrt(wp[inband]), lw=0.7, color="#33415c")
    a1.axvspan(b_lo, b_hi, color="#9db4c0", alpha=0.25, label=f"SK band {b_lo:.0f}-{b_hi:.0f} Hz")
    a1.set_xlim(0, s.band[1])
    a1.set_xlabel("Hz")
    a1.set_ylabel("g RMS per bin")
    a1.legend(loc="upper right", fontsize=8)
    a1.set_title(title, fontsize=10)
    a2.plot(ef, ea * 1000, lw=0.7, color="#33415c")
    colors = {"bpfo": "#c0392b", "bpfi": "#2874a6", "bsf": "#7d3c98", "ftf": "#1e8449"}
    top = min(b_hi - b_lo, 3.5 * faults["bpfi"])
    for k, f0 in faults.items():
        for h in range(1, 4):
            if h * f0 < top:
                a2.axvline(
                    h * f0, color=colors[k], lw=0.8, ls="--", alpha=0.7, label=k.upper() if h == 1 else None
                )
    a2.set_xlim(0, top)
    a2.set_xlabel("Hz")
    a2.set_ylabel("envelope amplitude (mg)")
    a2.legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def burst_figures(
    cfg: dict[str, Any], reg: Register, bronze_root: Path, truth: pd.DataFrame, point_id: str, out: Path
) -> list[Path]:
    s = ExtractionSettings.from_config(cfg)
    idx_path = bronze_root / "cm_platform" / "burst_index.parquet"
    if not idx_path.exists():
        return []
    idx = pd.read_parquet(idx_path)
    idx = idx[idx["point_id"] == point_id].merge(
        truth[truth["point_id"] == point_id][["hour_utc", "stage", "lube_severity"]], on="hour_utc"
    )
    orders = bearing_fault_orders(reg.bearings[reg.sensors[point_id].bearing]).as_dict()
    axis = _AXIS_INDEX[reg.sensors[point_id].radial_axis or "y"]
    files = []
    for stage, label in ((0, "healthy"), (2, "stage2"), (3, "stage3")):
        cand = idx[(idx["stage"] == stage) & (idx["lube_severity"] == 0)]
        if cand.empty:
            continue
        row = cand.iloc[len(cand) // 2]
        data = np.load(bronze_root / "cm_platform" / row["path"])
        shaft = row["speed_rpm"] / 60.0
        faults = {k: v * shaft for k, v in orders.items()}
        path = out / f"burst_{label}_{point_id}.png"
        title = (
            f"{point_id} {label}, {row['hour_utc']:%Y-%m-%d %H:00} UTC, "
            f"{row['speed_rpm']:.0f} rpm, radial axis"
        )
        _burst_panels(path, data["accel_g"], s, axis, faults, title)
        files.append(path)
    return files


def trend_figure(m: pd.DataFrame, detail: pd.DataFrame, point_id: str, out: Path) -> Path | None:
    g = m[(m["point_id"] == point_id) & m["radio_noise"].isna()]
    if g.empty or detail is None or detail.empty or point_id not in set(detail["component_id"]):
        return None
    d = detail[detail["component_id"] == point_id].iloc[0]
    t0 = d["onset"] - pd.Timedelta(days=21)
    t1 = d["failure"] + pd.Timedelta(days=2)
    res = g[(g["axis"] == "res") & g["hour_utc"].between(t0, t1)].set_index("hour_utc").sort_index()
    rad = (
        g[(g["axis"] != "res") & g["hour_utc"].between(t0, t1)]
        .groupby("hour_utc")[["vel_rms_mm_s", "kurtosis"]]
        .max()
    )
    panels = [
        (res["hf_rms_g"], "HF accel RMS 2-10 kHz (g)", "t_hf"),
        (
            res[["res_envr_bpfo", "res_envr_bpfi", "res_envr_bsf"]].max(axis=1),
            "max fault-band envelope / floor",
            "t_env",
        ),
        (res["crest_factor"], "crest factor (max axis)", "t_crest_peak"),
        (rad["kurtosis"], "kurtosis (radial)", None),
        (rad["vel_rms_mm_s"], "velocity RMS (mm/s, radial)", "t_zone_c"),
        (res["temp_c"], "sensor temperature (C)", None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 13), sharex=True)
    stage = res["stage"]
    for ax, (series, label, mark) in zip(axes, panels, strict=False):
        for st, col in _STAGE_COLORS.items():
            sel = stage[stage == st]
            if not sel.empty:
                ax.axvspan(sel.index.min(), sel.index.max(), color=col, alpha=0.6, lw=0)
        ax.plot(series.index, series.values, ".", ms=2, color="#33415c")
        roll = series.rolling("24h").median()
        ax.plot(roll.index, roll.values, lw=1.2, color="#111111")
        if mark and pd.notna(d.get(mark)):
            ax.axvline(d[mark], color="#c0392b", lw=1.0, ls="--")
        ax.axvline(d["failure"], color="black", lw=1.2)
        ax.set_ylabel(label, fontsize=8)
    axes[0].set_title(
        f"{point_id} {d['instance_id']} ({d['defect']}, onset cause {d['cause']}): stage shading 1 to 4, "
        "dashed = detection time, solid = failure",
        fontsize=9,
    )
    fig.tight_layout()
    path = out / f"trends_{point_id}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
