"""Machinery fault conditions and auxiliary-system degradation.

Imbalance, misalignment and looseness on pumps and compressors; rotary union leaks and coolant
ingress on HMC spindles; compressor oil cooler fouling; spindle cooling unit degradation. Inputs:
condition parameters from config and time. Outputs: per-hour level (0 to 1) of each condition,
clearing with optional recovery, and the drawn parameters of each episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
GRADUAL_FAULTS = ("imbalance", "misalignment", "looseness")
HAZARD_FAULTS = ("imbalance", "misalignment")


@dataclass
class Condition:
    kind: str
    asset_id: str
    points: tuple[str, ...]
    start: pd.Timestamp
    ramp_h: float
    params: dict[str, Any] = field(default_factory=dict)
    cause: str = "gradual"
    seasonal: bool = False
    progress: float = 0.0
    cleared_at: pd.Timestamp | None = None
    cleared_by: str | None = None
    recovery_h: float = 0.0
    level_at_clear: float = 0.0

    def ramp(self, ts: pd.Timestamp) -> float:
        if self.seasonal:
            return float(min(self.progress, 1.0))
        if self.ramp_h <= 0:
            return 1.0
        return float(np.clip((ts - self.start) / HOUR / self.ramp_h, 0.0, 1.0))

    def level(self, ts: pd.Timestamp) -> float:
        if ts < self.start:
            return 0.0
        if self.cleared_at is None or ts < self.cleared_at:
            return self.ramp(ts)
        if self.recovery_h <= 0:
            return 0.0
        frac = (ts - self.cleared_at) / HOUR / self.recovery_h
        return float(self.level_at_clear * max(0.0, 1.0 - frac))

    def active(self, ts: pd.Timestamp) -> bool:
        return self.start <= ts and self.cleared_at is None

    def finished(self, ts: pd.Timestamp) -> bool:
        return self.cleared_at is not None and ts >= self.cleared_at + self.recovery_h * HOUR

    def clear(self, ts: pd.Timestamp, by: str, recovery_h: float = 0.0) -> None:
        if self.cleared_at is None:
            self.level_at_clear = self.ramp(ts)
            self.cleared_at = ts
            self.cleared_by = by
            self.recovery_h = recovery_h

    def advance(self, hours: float, rate_multiplier: float) -> None:
        """Seasonal progression: add hours at the given rate multiplier to the ramp."""
        if self.cleared_at is None and self.ramp_h > 0:
            self.progress += hours * rate_multiplier / self.ramp_h


def _u(rng: np.random.Generator, lo_hi: list[float]) -> float:
    return float(rng.uniform(*lo_hi))


def draw_fault(
    cfg: dict[str, Any],
    kind: str,
    asset_id: str,
    points: list[str],
    ts: pd.Timestamp,
    rng: np.random.Generator,
    cause: str = "gradual",
) -> Condition:
    """Imbalance, misalignment or looseness with its drawn growth and signature parameters."""
    fc = cfg["fault_conditions"][kind]
    week_h = pd.Timedelta(weeks=1) / HOUR
    if kind == "imbalance":
        params = {
            "one_x": _u(rng, fc["one_x_growth_factor"]),
            "axial": _u(rng, fc["axial_share_of_radial"]),
            "hazard": _u(rng, fc["bearing_hazard_multiplier"]),
        }
        ramp = _u(rng, fc["growth_weeks"]) * week_h
    elif kind == "misalignment":
        params = {
            "one_x": _u(rng, fc["one_x_growth_factor"]),
            "two_x": _u(rng, fc["two_x_growth_factor"]),
            "axial": _u(rng, fc["axial_share_of_radial_at_end"]),
            "hazard": _u(rng, fc["bearing_hazard_multiplier"]),
        }
        ramp = 0.0 if cause == "post_reinstallation" else _u(rng, fc["growth_weeks"]) * week_h
    else:
        params = {
            "harm": _u(rng, fc["harmonic_gain_3x_10x"]),
            "half_x": _u(rng, fc["half_x_share_of_1x"]),
            "floor_db": _u(rng, fc["noise_floor_rise_db"]),
        }
        ramp = 0.0
    return Condition(kind, asset_id, tuple(points), ts, ramp, params, cause)


def rotary_union_remaining_h(cfg: dict[str, Any], age_h: float, rng: np.random.Generator) -> float:
    """Remaining TSC running hours to a leak, Weibull conditional on the union's age."""
    ci = cfg["coolant_ingress"]
    eta, k = float(ci["rotary_union_life_tsc_running_h"]), float(ci["rotary_union_weibull_shape"])
    total = eta * ((age_h / eta) ** k - np.log(rng.random())) ** (1.0 / k)
    return float(total - age_h)


def draw_leak(cfg: dict[str, Any], asset_id: str, ts: pd.Timestamp, rng: np.random.Generator) -> Condition:
    ci = cfg["coolant_ingress"]
    ingress = bool(rng.random() < ci["ingress_fraction_of_leaks"])
    delay = pd.Timedelta(days=_u(rng, ci["leak_to_ingress_days"]))
    params = {
        "drop": _u(rng, ci["tsc_pressure_drop_frac"]),
        "ingress": ingress,
        "ingress_at": ts + delay,
    }
    return Condition("rotary_union_leak", asset_id, (), ts, 0.0, params)


def draw_ingress(
    cfg: dict[str, Any], asset_id: str, point: str, ts: pd.Timestamp, rng: np.random.Generator
) -> Condition:
    ci = cfg["coolant_ingress"]
    params = {
        "hf_rise": _u(rng, ci["front_hf_rise_frac"]),
        "hazard": _u(rng, ci["front_bearing_hazard_multiplier"]),
    }
    return Condition("coolant_ingress", asset_id, (point,), ts, 0.0, params)


def draw_fouling(cfg: dict[str, Any], asset_id: str, ts: pd.Timestamp, rng: np.random.Generator) -> Condition:
    oc = cfg["oil_cooler_fouling"]
    week_h = pd.Timedelta(weeks=1) / HOUR
    params = {"rise_c": _u(rng, oc["discharge_temp_rise_c"])}
    return Condition(
        "cooler_fouling", asset_id, (), ts, _u(rng, oc["rise_weeks"]) * week_h, params, seasonal=True
    )


def draw_spindle_cooling(
    cfg: dict[str, Any], asset_id: str, ts: pd.Timestamp, rng: np.random.Generator
) -> Condition:
    sc = cfg["spindle_cooling"]
    week_h = pd.Timedelta(weeks=1) / HOUR
    params = {"rise_c": _u(rng, sc["thermistor_residual_rise_c"])}
    return Condition("spindle_cooling", asset_id, (), ts, _u(rng, sc["rise_weeks"]) * week_h, params)


def tool_assembly_factor(cfg: dict[str, Any], rng: np.random.Generator) -> tuple[float, bool]:
    """Tool-assembly 1x level of a part program, relative to the spindle's healthy 1x."""
    ta = cfg["fault_conditions"]["tool_assembly_imbalance"]
    if rng.random() < ta["high_program_share"]:
        return _u(rng, ta["high_factor"]), True
    return float(np.exp(ta["baseline_spread_sigma_log"] * rng.standard_normal())), False


def fault_terms(c: Condition, ts: pd.Timestamp) -> dict[str, Any]:
    """Render-side description of a fault condition at an hour."""
    return {"kind": c.kind, "level": c.level(ts), **{k: v for k, v in c.params.items() if k != "hazard"}}
