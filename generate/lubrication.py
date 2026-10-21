"""Lubrication-loss episodes on spindles and compressor airends.

Inputs: lubrication configuration and episode start times. Outputs: per-hour severity (0 to 1),
the HF rise and temperature rise it produces, clearing and recovery, and conversion to a bearing
onset when an episode is not cleared within its window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LubeEpisode:
    asset_id: str
    start: pd.Timestamp
    ramp_days: float
    hf_rise: float
    temp_rise_c: float
    window_days: float
    recovery_days: float
    cleared_at: pd.Timestamp | None = None
    cleared_by: str | None = None
    converted: bool = False

    @classmethod
    def draw(
        cls, cfg: dict[str, Any], asset_id: str, start: pd.Timestamp, rng: np.random.Generator
    ) -> LubeEpisode:
        lub = cfg["lubrication"]
        return cls(
            asset_id=asset_id,
            start=start,
            ramp_days=float(rng.uniform(*lub["ramp_days"])),
            hf_rise=float(rng.uniform(*lub["hf_rise_frac"])),
            temp_rise_c=float(rng.uniform(*lub["temp_rise_c"])),
            window_days=float(rng.uniform(*lub["conversion_window_days"])),
            recovery_days=float(rng.uniform(*lub["recovery_days"])),
        )

    def _ramp(self, ts: pd.Timestamp) -> float:
        return float(np.clip((ts - self.start) / pd.Timedelta(days=self.ramp_days), 0.0, 1.0))

    def severity(self, ts: pd.Timestamp) -> float:
        if ts < self.start:
            return 0.0
        if self.cleared_at is None or ts < self.cleared_at:
            return self._ramp(ts)
        at_clear = self._ramp(self.cleared_at)
        frac = (ts - self.cleared_at) / pd.Timedelta(days=self.recovery_days)
        return float(at_clear * max(0.0, 1.0 - frac))

    def active(self, ts: pd.Timestamp) -> bool:
        return self.start <= ts and self.cleared_at is None

    def finished(self, ts: pd.Timestamp) -> bool:
        return self.cleared_at is not None and ts >= self.cleared_at + pd.Timedelta(days=self.recovery_days)

    def clear(self, ts: pd.Timestamp, by: str) -> None:
        if self.cleared_at is None:
            self.cleared_at = ts
            self.cleared_by = by

    def conversion_due(self, ts: pd.Timestamp) -> bool:
        return (
            (not self.converted)
            and self.active(ts)
            and ts >= self.start + pd.Timedelta(days=self.window_days)
        )
