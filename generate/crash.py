"""Collision events from the controller event log and their outcomes.

Inputs: crash configuration, cutting minutes, shift and program age per hour. Outputs: collision
events with severity flag and outcome (no lasting effect, brinelling, immediate failure).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pdm.constants import SECONDS_PER_MINUTE


def rate_per_cutting_hour(cfg: dict[str, Any], annual_cutting_h: float) -> float:
    return cfg["crash"]["rate_per_machine_year"] / annual_cutting_h


def hour_probability(
    cfg: dict[str, Any], base_rate_h: float, cutting_min: float, shift: str, program_age: pd.Timedelta | None
) -> float:
    c = cfg["crash"]
    mult = c["second_shift_multiplier"] if shift == "second" else 1.0
    if program_age is not None and program_age < pd.Timedelta(weeks=1):
        mult *= c["new_program_first_week_multiplier"]
    lam = base_rate_h * mult * cutting_min / SECONDS_PER_MINUTE
    return float(1.0 - np.exp(-lam))


def draw_outcome(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[str, bool]:
    """Outcome and whether severity is above the register threshold."""
    c = cfg["crash"]
    props = c["outcome_proportions"]
    keys = list(props)
    w = np.array([props[k] for k in keys], dtype=float)
    outcome = str(rng.choice(keys, p=w / w.sum()))
    if outcome != "no_effect":
        return outcome, True
    damaging = 1.0 - props["no_effect"] / w.sum()
    p_above = (c["severity_above_threshold_share"] - damaging) / (1.0 - damaging)
    return outcome, bool(rng.random() < max(p_above, 0.0))
