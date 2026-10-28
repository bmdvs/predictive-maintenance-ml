"""Vendor-style per-point baseline alerts on velocity RMS, HF acceleration RMS and envelope fault bands.

Input: resultant indicator rows in time order and the point's baseline. Output: alert records with
level (alert or alarm), triggering rule and values against baseline. Applies persistence and
re-alert suppression from pdm.rules.persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pdm.features.kinematics import FAULT_KEYS
from pdm.rules.iso_zones import zone_b_width
from pdm.rules.persistence import ConsecutiveCounter, Suppressor


@dataclass(frozen=True)
class VendorRuleSettings:
    alert_frac: float
    alarm_frac: float
    hf_reference_g: float
    env_db: float
    env_consecutive: int
    persistence: int
    suppression_days: float
    group: int
    foundation: str
    harmonics: int

    @classmethod
    def for_point_class(
        cls, cfg: dict[str, Any], iso_class_key: str, hf_class_key: str
    ) -> VendorRuleSettings:
        rules = cfg["rules"]
        iso = rules["iso_class_by_point_class"][iso_class_key]
        return cls(
            alert_frac=rules["vendor"]["alert_frac_zone_b_width"],
            alarm_frac=rules["vendor"]["alarm_frac_zone_b_width"],
            hf_reference_g=rules["vendor"]["hf_reference_width_g"][hf_class_key],
            env_db=rules["envelope"]["db_over_baseline"],
            env_consecutive=rules["envelope"]["consecutive"],
            persistence=rules["persistence_consecutive"],
            suppression_days=rules["suppression_days"],
            group=iso["group"],
            foundation=iso["foundation"],
            harmonics=cfg["indicators"]["envelope"]["harmonics"],
        )

    def env_columns(self) -> list[str]:
        return [f"env_{k}_{h}_g2" for k in FAULT_KEYS for h in range(1, self.harmonics + 1)]


def compute_baseline(
    rows: pd.DataFrame, settings: VendorRuleSettings, statistic: str = "median"
) -> dict[str, float]:
    """Baseline values from the resultant rows of a baselining window."""
    agg = getattr(rows, statistic)
    cols = ["vel_rms_mm_s", "hf_rms_g", *settings.env_columns()]
    return {c: float(agg(numeric_only=True)[c]) for c in cols}


@dataclass
class VendorRuleState:
    point_id: str
    settings: VendorRuleSettings
    baseline: dict[str, float] | None = None
    vel_alert: ConsecutiveCounter = field(init=False)
    vel_alarm: ConsecutiveCounter = field(init=False)
    hf_alert: ConsecutiveCounter = field(init=False)
    hf_alarm: ConsecutiveCounter = field(init=False)
    env: ConsecutiveCounter = field(init=False)
    suppressor: Suppressor = field(init=False)

    def __post_init__(self) -> None:
        s = self.settings
        self.vel_alert = ConsecutiveCounter(s.persistence)
        self.vel_alarm = ConsecutiveCounter(s.persistence)
        self.hf_alert = ConsecutiveCounter(s.persistence)
        self.hf_alarm = ConsecutiveCounter(s.persistence)
        self.env = ConsecutiveCounter(s.env_consecutive)
        self.suppressor = Suppressor(s.suppression_days)

    def reset_counters(self) -> None:
        for c in (self.vel_alert, self.vel_alarm, self.hf_alert, self.hf_alarm, self.env):
            c.count = 0

    def thresholds(self) -> dict[str, float]:
        s, b = self.settings, self.baseline or {}
        width = zone_b_width(s.group, s.foundation)
        return {
            "vel_alert": b.get("vel_rms_mm_s", np.inf) + s.alert_frac * width,
            "vel_alarm": b.get("vel_rms_mm_s", np.inf) + s.alarm_frac * width,
            "hf_alert": b.get("hf_rms_g", np.inf) + s.alert_frac * s.hf_reference_g,
            "hf_alarm": b.get("hf_rms_g", np.inf) + s.alarm_frac * s.hf_reference_g,
        }

    def update(self, ts: pd.Timestamp, row: dict[str, float] | None) -> dict[str, Any] | None:
        """Advance one hour. row is the resultant indicator row, or None without a measurement."""
        if self.baseline is None:
            return None
        if row is None:
            for c in (self.vel_alert, self.vel_alarm, self.hf_alert, self.hf_alarm, self.env):
                c.update(None)
            return None
        th = self.thresholds()
        ratio = 10 ** (self.settings.env_db / 10.0)
        env_hits = [
            c
            for c in self.settings.env_columns()
            if self.baseline.get(c, 0.0) > 0 and row[c] >= ratio * self.baseline[c]
        ]
        fired = {
            "vel_alarm": self.vel_alarm.update(row["vel_rms_mm_s"] >= th["vel_alarm"]),
            "vel_alert": self.vel_alert.update(row["vel_rms_mm_s"] >= th["vel_alert"]),
            "hf_alarm": self.hf_alarm.update(row["hf_rms_g"] >= th["hf_alarm"]),
            "hf_alert": self.hf_alert.update(row["hf_rms_g"] >= th["hf_alert"]),
            "envelope": self.env.update(bool(env_hits)),
        }
        triggered = [k for k, v in fired.items() if v]
        if not triggered or not self.suppressor.allow(ts):
            return None
        level = "alarm" if any(k.endswith("alarm") for k in triggered) else "alert"
        return {
            "point_id": self.point_id,
            "alert_ts": ts,
            "level": level,
            "rules": ",".join(triggered),
            "vel_rms_mm_s": row["vel_rms_mm_s"],
            "vel_baseline_mm_s": self.baseline["vel_rms_mm_s"],
            "hf_rms_g": row["hf_rms_g"],
            "hf_baseline_g": self.baseline["hf_rms_g"],
            "env_bands": ",".join(env_hits),
        }
