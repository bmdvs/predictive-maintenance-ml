"""Persistence and re-alert suppression shared by every alerting layer.

A condition must hold on a set number of consecutive measurements; hours without a measurement
neither count toward nor break the run. A point that has alerted is silent for a set number of days.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ConsecutiveCounter:
    required: int
    count: int = 0

    def update(self, condition: bool | None) -> bool:
        """Advance with one hour; None means no measurement in that hour."""
        if condition is None:
            return self.count >= self.required
        self.count = self.count + 1 if condition else 0
        return self.count >= self.required


@dataclass
class Suppressor:
    days: float
    last: pd.Timestamp | None = None

    def allow(self, ts: pd.Timestamp) -> bool:
        if self.last is not None and ts - self.last < pd.Timedelta(days=self.days):
            return False
        self.last = ts
        return True


def persisted(conditions: list[bool | None], required: int) -> list[bool]:
    counter = ConsecutiveCounter(required)
    return [counter.update(c) if c is not None else False for c in conditions]
