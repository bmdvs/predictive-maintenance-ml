"""Loads config.yaml and register.yaml into plain mappings and a typed asset register.

Inputs: the YAML files under config/. Outputs: a configuration dict and a Register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pdm import paths


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or paths.CONFIG_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class BearingGeometry:
    part: str
    kind: str
    n: int
    d_mm: float
    pitch_mm: float
    contact_deg: float


@dataclass(frozen=True)
class SensorPoint:
    point_id: str
    asset_id: str
    location: str
    bearing: str
    install_date: str
    mounting: str
    gateway: str
    radial_axis: str | None = None


@dataclass
class Register:
    bearings: dict[str, BearingGeometry]
    machines: dict[str, dict[str, Any]]
    hmc_spindles: dict[str, dict[str, Any]]
    pumps: dict[str, dict[str, Any]]
    compressors: dict[str, dict[str, Any]]
    sensors: dict[str, SensorPoint]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def point_class(self, point_id: str) -> str:
        """Point class used to index per-class parameters."""
        loc = self.sensors[point_id].location
        asset = self.sensors[point_id].asset_id
        if loc == "front_bearing_housing":
            return "spindle_front"
        if loc == "rear_bearing_housing":
            return "spindle_rear"
        if loc == "motor_de" and asset.startswith("CMP"):
            return "compressor_motor"
        if loc == "airend_inlet":
            return "compressor_airend"
        return "tsc_pump" if asset.endswith("TSC") else "flood_pump"


def load_register(path: Path | None = None) -> Register:
    with open(path or paths.REGISTER_FILE, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    bearings = {str(k): BearingGeometry(part=str(k), **v) for k, v in raw["bearing_types"].items()}
    sensors = {
        s["point_id"]: SensorPoint(
            point_id=s["point_id"],
            asset_id=s["asset_id"],
            location=s["location"],
            bearing=str(s["bearing"]),
            install_date=str(s["install_date"]),
            mounting=s["mounting"],
            gateway=s["gateway"],
            radial_axis=s.get("radial_axis"),
        )
        for s in raw["sensors"]
    }
    return Register(
        bearings=bearings,
        machines={m["machine_id"]: m for m in raw["machines"]},
        hmc_spindles={m["machine_id"]: m for m in raw["hmc_spindles"]},
        pumps={p["asset_id"]: p for p in raw["pumps"]},
        compressors={c["asset_id"]: c for c in raw["compressors"]},
        sensors=sensors,
        raw=raw,
    )
