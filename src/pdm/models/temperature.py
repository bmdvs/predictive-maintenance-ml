"""Temperature residual per sensor point: housing temperature less its expected value.

The expected value is a linear fit on the point's own healthy training hours: spindle points on spindle
speed; pump points on load where the record carries it (otherwise a constant for the running state);
compressor points on load and the compressor-room ambient. No cross-machine correction.
Inputs: a point's hourly rows with temp_c, speed and load, and the room ambient by hour for compressor
points. Output: fit coefficients per point and the residual column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TEMP = "temp_c"
SPEED = "speed_rpm_at_measurement"
LOAD = "load_pct_at_measurement"
AMBIENT = "room_ambient_c"
MIN_FIT_HOURS = 24


def covariates(point_class: str) -> list[str]:
    if point_class.startswith("spindle"):
        return [SPEED]
    if point_class.startswith("compressor"):
        return [LOAD, AMBIENT]
    return [LOAD]


@dataclass
class TempFit:
    columns: list[str]
    coef: np.ndarray = field(default_factory=lambda: np.zeros(1))
    fill: dict[str, float] = field(default_factory=dict)

    def expected(self, rows: pd.DataFrame) -> np.ndarray:
        x = [np.ones(len(rows))]
        for c in self.columns:
            v = rows[c].to_numpy(dtype=float) if c in rows else np.full(len(rows), np.nan)
            x.append(np.where(np.isfinite(v), v, self.fill.get(c, 0.0)))
        return np.column_stack(x) @ self.coef

    def residual(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[TEMP].to_numpy(dtype=float) - self.expected(rows)

    def to_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "coef": [float(v) for v in self.coef], "fill": self.fill}


def fit(rows: pd.DataFrame, point_class: str) -> TempFit:
    """Least-squares fit of temp_c on the point's covariates; a covariate with no finite value in the
    rows is dropped, and missing values are filled with the training median."""
    t = rows[TEMP].to_numpy(dtype=float)
    ok = np.isfinite(t)
    cols = [
        c for c in covariates(point_class) if c in rows and np.isfinite(rows[c].to_numpy(dtype=float)).any()
    ]
    fill = {c: float(np.nanmedian(rows[c].to_numpy(dtype=float))) for c in cols}
    tf = TempFit(cols, np.zeros(len(cols) + 1), fill)
    if ok.sum() < MIN_FIT_HOURS:
        tf.coef[0] = float(np.nanmedian(t)) if ok.any() else 0.0
        return tf
    x = np.column_stack(
        [np.ones(len(rows))]
        + [np.where(np.isfinite(v := rows[c].to_numpy(dtype=float)), v, fill[c]) for c in cols]
    )
    tf.coef = np.linalg.lstsq(x[ok], t[ok], rcond=None)[0]
    return tf


def hourly_ambient(room: pd.DataFrame) -> pd.Series:
    """Compressor-room ambient per UTC hour from the minute export (columns minute_utc, ambient_c)."""
    t = pd.to_datetime(room["minute_utc"], utc=True).dt.floor("h")
    return room.groupby(t)["ambient_c"].mean().rename(AMBIENT)
