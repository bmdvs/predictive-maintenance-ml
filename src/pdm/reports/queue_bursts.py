"""Bursts for the maintenance queue's spectrum panels: loading retained and rendered bursts, the resultant
envelope spectrum shown on the page, and the two checks on rendered bursts (byte identity against retained
copies; every burst-derived hourly value against the stored row within 1e-9).

Inputs: build/queue_cache (bursts, burst plan, stored rows, sensor events). Outputs: spectra and check rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdm.config import load_config
from pdm.features import envelope, filters, kinematics
from pdm.features.burst import AXES, ExtractionSettings, extract_burst
from pdm.reports.queue_data import CACHE

TOL = 1e-9
SKIP = {"temp_c", "speed_rpm_at_measurement", "load_pct_at_measurement"}


def _rendered_path(pid: str, hour: pd.Timestamp, cache: Path) -> Path:
    return cache / "bursts" / "rendered" / pid / f"{hour:%Y%m%dT%H}.npz"


def load(pid: str, hour: pd.Timestamp, retained: str | None, cache: Path = CACHE) -> tuple[np.ndarray, str]:
    """The burst (3, n) in g as float32, from the retained copy when one exists, else the rendered copy."""
    if retained:
        with np.load(cache / "bursts" / "retained" / retained) as z:
            return np.asarray(z["accel_g"]), "retained"
    with np.load(_rendered_path(pid, hour, cache)) as z:
        return np.asarray(z["accel_g"]), "rendered"


def band_at(events: pd.DataFrame, pid: str, hour: pd.Timestamp) -> tuple[float, float]:
    """The point's demodulation band in force at the hour (the last band selection before it)."""
    e = events[(events["point_id"] == pid) & (events["event"] == "envelope_band") & (events["ts"] <= hour)]
    lo, hi = str(e.sort_values("ts")["detail"].iloc[-1]).split("-")
    return float(lo), float(hi)


def envelope_resultant(
    burst: np.ndarray, fs: float, band: tuple[float, float], s: ExtractionSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope amplitude spectrum (g) of the three axes combined (root of the summed powers), as the resultant."""
    x = filters.remove_dc(np.asarray(burst, dtype=float))
    n = x.shape[-1]
    spec, freqs = filters.rfft_with_freqs(x, fs)
    spec = spec * filters.band_mask(freqs, *s.band)
    power = None
    for i in range(len(AXES)):
        f, a = envelope.envelope_spectrum(spec[i], freqs, n, fs, *band)
        power = a**2 if power is None else power + a**2
    return f, np.sqrt(power)


def _plan(cache: Path) -> pd.DataFrame:
    p = pd.DataFrame(json.loads((cache / "burst_plan.json").read_text(encoding="utf-8")))
    p["hour_utc"] = pd.to_datetime(p["hour_utc"], utc=True)
    return p


def checks(state: Any, cache: Path = CACHE) -> dict[str, Any]:
    """Byte identity of rendered copies of retained bursts, and every rendered burst's hourly values against the
    stored rows (relative 1e-9, absolute floor 1e-12)."""
    cfg = load_config()
    s = ExtractionSettings.from_config(cfg)
    rendered = json.loads((cache / "bursts" / "rendered" / "rendered.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(cache / "burst_rows.parquet")
    rows["hour_utc"] = pd.to_datetime(rows["hour_utc"], utc=True)
    plan = _plan(cache)
    ident, values = [], []
    for r in rendered:
        pid, hour = r["point_id"], pd.Timestamp(r["hour_utc"])
        with np.load(r["path"]) as z:
            a32, a64, shaft = np.asarray(z["accel_g"]), np.asarray(z["accel_g64"]), float(z["shaft_hz"])
        ret = plan[(plan["point_id"] == pid) & (plan["hour_utc"] == hour) & plan["retained"].notna()]
        if len(ret):
            with np.load(cache / "bursts" / "retained" / ret["retained"].iloc[0]) as z:
                orig = np.asarray(z["accel_g"])
            ident.append(
                {
                    "point_id": pid,
                    "hour_utc": hour,
                    "retained": ret["retained"].iloc[0],
                    "dtype": str(orig.dtype),
                    "shape": list(orig.shape),
                    "byte_identical": orig.dtype == a32.dtype
                    and orig.shape == a32.shape
                    and orig.tobytes() == a32.tobytes(),
                }
            )
            continue
        st = rows[(rows["point_id"] == pid) & (rows["hour_utc"] == hour)]
        speed = float(st["speed_rpm_at_measurement"].dropna().iloc[0]) / 60.0
        fault_hz = kinematics.fault_frequencies_hz(state.fault_orders(pid), speed)
        band = band_at(state.sensor_events, pid, hour)
        got, _ = extract_burst(a64, s, speed, fault_hz, band=band)
        worst, n, beyond = 0.0, 0, []
        for _, sr in st.iterrows():
            g = got[sr["axis"]]
            for k, v in g.items():
                if k in SKIP or k not in sr.index:
                    continue
                want = float(sr[k]) if pd.notna(sr[k]) else np.nan
                if np.isnan(v) and np.isnan(want):
                    n += 1
                    continue
                d = abs(v - want)
                rel = d / max(abs(want), abs(v), 1e-300)
                n += 1
                if not (d <= max(TOL * max(abs(want), abs(v)), 1e-12)):
                    beyond.append(f"{sr['axis']}:{k}")
                worst = max(worst, rel if np.isfinite(rel) else np.inf)
        values.append(
            {
                "point_id": pid,
                "hour_utc": hour,
                "axes": sorted(st["axis"].unique().tolist()),
                "values_compared": n,
                "max_relative_difference": worst,
                "beyond_1e-9": beyond,
                "shaft_hz_matches_stored_speed": abs(shaft - speed) <= 1e-9 * max(speed, 1.0),
                "pass": not beyond and n > 0,
            }
        )
    return {"identity": ident, "values": values}
