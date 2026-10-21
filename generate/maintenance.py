"""Maintenance team behavior: PM execution with compliance, alert response timing and repair times.

Inputs: maintenance configuration and the shop calendar. Outputs: PM completion dates and
outcomes, inspection and planned-repair times, and repair durations by component.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from generate.shop_schedule import LOCAL_TZ


@dataclass
class PmTask:
    asset_id: str
    task: str
    basis: str
    interval: float
    next_due: pd.Timestamp | None = None
    next_due_h: float | None = None


def pm_outcome(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[str, float]:
    """Outcome (on_time, late, skipped) and days late."""
    m = cfg["maintenance"]
    shares = m["pm_outcome_shares"]
    keys = list(shares)
    w = np.array([shares[k] for k in keys], dtype=float)
    outcome = str(rng.choice(keys, p=w / w.sum()))
    if outcome == "late":
        late = m["pm_late_days_lognormal"]
        return outcome, float(late["median"] * np.exp(late["sigma"] * rng.standard_normal()))
    return outcome, 0.0


def add_working_days(ts: pd.Timestamp, days: int, holidays: set[date], at_local_hour: int) -> pd.Timestamp:
    local = ts.tz_convert(LOCAL_TZ).normalize()
    added = 0
    while added < days:
        local += pd.Timedelta(days=1)
        if local.weekday() < 5 and local.date() not in holidays:
            added += 1
    return (local + pd.Timedelta(hours=at_local_hour)).tz_convert("UTC")


def next_saturday_on_or_after(ts: pd.Timestamp, at_local_hour: int) -> pd.Timestamp:
    local = ts.tz_convert(LOCAL_TZ).normalize()
    while local.weekday() != 5:
        local += pd.Timedelta(days=1)
    return (local + pd.Timedelta(hours=at_local_hour)).tz_convert("UTC")


def repair_duration(cfg: dict[str, Any], component_class: str, rng: np.random.Generator) -> pd.Timedelta:
    rt = cfg["maintenance"]["repair_time"]
    if component_class.startswith("spindle"):
        return pd.Timedelta(days=float(rng.uniform(*rt["spindle_rebuild_days"])))
    if component_class.startswith("compressor"):
        return pd.Timedelta(days=float(rng.uniform(*rt["compressor_bearing_days"])))
    return pd.Timedelta(hours=float(rng.uniform(*rt["pump_hours"])))


def detection_probability(cfg: dict[str, Any], stage: int) -> float:
    if stage <= 0:
        return 0.0
    return float(cfg["maintenance"]["inspection_detection_prob_by_stage"][f"stage{min(stage, 4)}"])


def wear_stage(cfg: dict[str, Any], wear_frac: float) -> int:
    if wear_frac <= 0:
        return 0
    edges = cfg["maintenance"]["wear_stage_by_fraction"]
    return 1 + int(np.searchsorted(edges, wear_frac, side="right"))
