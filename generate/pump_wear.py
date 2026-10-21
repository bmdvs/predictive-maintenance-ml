"""Coolant pump seal and impeller wear, cavitation episodes and seal failures.

Inputs: pump configuration and calendar time. Outputs: wear fraction (0 to 1) per pump, cavitation
state and HF rise, TSC pressure drop, and seal-failure hazard per hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class WearProcess:
    kind: str
    start: pd.Timestamp
    days_to_end: float

    def fraction(self, ts: pd.Timestamp) -> float:
        return float(np.clip((ts - self.start) / pd.Timedelta(days=self.days_to_end), 0.0, 1.0))


@dataclass
class Cavitation:
    start: pd.Timestamp
    hf_rise: float
    cleared_at: pd.Timestamp | None = None
    cleared_by: str | None = None

    def active(self, ts: pd.Timestamp) -> bool:
        return self.start <= ts and (self.cleared_at is None or ts < self.cleared_at)

    def clear(self, ts: pd.Timestamp, by: str) -> None:
        if self.cleared_at is None:
            self.cleared_at = ts
            self.cleared_by = by


@dataclass
class PumpState:
    asset_id: str
    kind: str
    wear: list[WearProcess]
    cavitation: Cavitation | None = None

    def wear_fraction(self, ts: pd.Timestamp) -> float:
        return max((w.fraction(ts) for w in self.wear if w.start <= ts), default=0.0)

    def wear_active(self, ts: pd.Timestamp) -> bool:
        return any(w.start <= ts for w in self.wear)

    def reset(self) -> None:
        self.wear = []
        if self.cavitation is not None and self.cavitation.cleared_at is None:
            self.cavitation = None

    def cavitation_rise(self, ts: pd.Timestamp) -> float:
        return self.cavitation.hf_rise if self.cavitation is not None and self.cavitation.active(ts) else 0.0


def hourly_probability(rate_per_year: float, hours: float, hours_per_year: float) -> float:
    return float(1.0 - np.exp(-rate_per_year * hours / hours_per_year))


def maybe_start_wear(
    cfg: dict[str, Any], pump: PumpState, ts: pd.Timestamp, p_hour: dict[str, float], rng: np.random.Generator
) -> list[str]:
    started = []
    for kind, key in (("seal", "seal_wear"), ("impeller", "impeller_wear")):
        if any(w.kind == kind for w in pump.wear):
            continue
        if rng.random() < p_hour[kind]:
            days = float(rng.uniform(*cfg["pump"][key]["growth_days_to_end"]))
            pump.wear.append(WearProcess(kind, ts, days))
            started.append(kind)
    return started
