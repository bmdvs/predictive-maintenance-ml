"""Rolling-bearing degradation per component instance (exponential health-index model).

Inputs: component class parameters from config and the running hours accrued each hour.
Outputs: onset, health index, stage and the failure due time for each bearing instance; renewal
on replacement. Onset is Weibull in running hours with crash and lubrication multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

_LIFE_KEY = {
    "spindle_front": "spindle_front",
    "spindle_rear": "spindle_rear",
    "tsc_pump": "tsc_pump_motor",
    "flood_pump": "flood_pump_motor",
    "compressor_motor": "compressor_motor",
    "compressor_airend": "compressor_airend",
}
_BETA_KEY = {
    "spindle_front": "spindle",
    "spindle_rear": "spindle",
    "tsc_pump": "tsc_pump",
    "flood_pump": "flood_pump",
    "compressor_motor": "compressor",
    "compressor_airend": "compressor",
}


@dataclass
class BearingInstance:
    component_id: str
    component_class: str
    asset_id: str
    point_id: str | None
    age_class: str
    params: dict[str, Any] = field(repr=False)
    sig: dict[str, Any] = field(repr=False)
    rng: np.random.Generator = field(repr=False)
    age_h: float = 0.0
    instance_no: int = 0
    remaining_h: float = np.inf
    onset_ts: pd.Timestamp | None = None
    t_run: float = 0.0
    theta: float = 0.0
    beta: float = 0.0
    eps: float = 0.0
    h: float = 0.0
    defect: str | None = None
    failure_due: pd.Timestamp | None = None
    onset_cause: str | None = None
    forced: bool = False
    detectable: bool = True
    beta_quantile: float | None = None
    bearing_kind: str = "angular_contact_ball"
    am_inner: float = 0.0

    @property
    def eta(self) -> float:
        return float(
            self.params["characteristic_life_running_h"][_LIFE_KEY[self.component_class]][self.age_class]
        )

    @property
    def shape(self) -> float:
        return float(self.params["onset_weibull_shape"])

    @property
    def instance_id(self) -> str:
        return f"{self.component_id}#{self.instance_no}"

    def draw_remaining(self) -> None:
        u = self.rng.random()
        total = self.eta * ((self.age_h / self.eta) ** self.shape - np.log(u)) ** (1.0 / self.shape)
        self.remaining_h = total - self.age_h

    def start(self, age_h: float) -> None:
        self.age_h = age_h
        self.draw_remaining()

    def renew(self) -> None:
        self.instance_no += 1
        self.age_h = 0.0
        self.onset_ts = None
        self.t_run = self.theta = self.beta = self.eps = self.h = 0.0
        self.defect = self.failure_due = self.onset_cause = None
        self.am_inner = 0.0
        self.forced = False
        self.detectable = True
        self.beta_quantile = None
        self.draw_remaining()

    def scale_remaining(self, factor: float) -> None:
        if self.onset_ts is None:
            self.remaining_h *= factor

    def begin_onset(
        self, ts: pd.Timestamp, cause: str, theta: float | None = None, defect: str | None = None
    ) -> None:
        hi = self.params["health_index"]
        th = hi["theta_lognormal"]
        b_mean, b_sd = hi["beta_per_running_h"][_BETA_KEY[self.component_class]]
        self.onset_ts = ts
        self.onset_cause = cause
        self.theta = (
            theta
            if theta is not None
            else float(th["median"] * np.exp(th["sigma_log"] * self.rng.standard_normal()))
        )
        z = norm.ppf(self.beta_quantile) if self.beta_quantile is not None else self.rng.standard_normal()
        self.beta = float(max(b_mean + b_sd * z, np.finfo(float).eps))
        if defect is None:
            shares = self.sig["defect_location_shares"]
            keys = list(shares)
            w = np.array([shares[k] for k in keys], dtype=float)
            defect = str(self.rng.choice(keys, p=w / w.sum()))
        self.defect = defect
        rng_am = self.sig["amplitude_modulation_depth"]["inner_race_at_shaft_by_kind"][self.bearing_kind]
        self.am_inner = float(self.rng.uniform(*rng_am)) if defect == "inner_race" else 0.0
        self.t_run = 0.0
        self.eps = 0.0
        self.h = self.params["health_index"]["phi"] + self.theta

    def step(self, ts: pd.Timestamp, run_h: float, life_rate: float = 1.0) -> list[dict[str, Any]]:
        """Advance by run_h running hours; life_rate > 1 consumes pre-onset life faster."""
        events: list[dict[str, Any]] = []
        if run_h <= 0:
            return events
        self.age_h += run_h
        if self.onset_ts is None:
            self.remaining_h -= run_h * life_rate
            if self.remaining_h <= 0:
                self.begin_onset(ts, "wear_out")
                events.append(
                    {
                        "event": "bearing_onset",
                        "cause": "wear_out",
                        "defect": self.defect,
                        "am_inner": self.am_inner,
                    }
                )
            return events
        hi = self.params["health_index"]
        self.t_run += run_h
        rho = hi["noise"]["ar1_per_h"] ** run_h
        self.eps = rho * self.eps + np.sqrt(1 - rho**2) * hi["noise"]["sd"] * self.rng.standard_normal()
        self.h = hi["phi"] + self.theta * np.exp(self.beta * self.t_run + self.eps)
        if self.failure_due is None and self.h >= self.params["failure"]["hi_threshold"]:
            lo, up = self.params["failure"]["offset_days"]
            self.failure_due = ts + pd.Timedelta(days=float(self.rng.uniform(lo, up)))
            events.append({"event": "failure_threshold", "failure_due": self.failure_due})
        return events

    @property
    def stage(self) -> int:
        if self.onset_ts is None:
            return 0
        th = self.params["stage_thresholds_hi"]
        if self.h < th["stage2"]:
            return 1
        if self.h < th["stage3"]:
            return 2
        if self.h < th["stage4"]:
            return 3
        return 4

    def signature_hi(self) -> float:
        return (
            float(min(self.h, self.params["failure"]["hi_threshold"])) if self.onset_ts is not None else 0.0
        )
