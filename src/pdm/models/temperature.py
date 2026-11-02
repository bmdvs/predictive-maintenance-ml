"""Temperature residual per sensor point: housing temperature less its expected value.

The expected value is a linear fit on the point's own healthy training hours: spindle points on spindle
speed and an annual harmonic of the local day of year (the seasonal term, as Controller Health's
thermistor residual; the exports carry no shop-floor ambient); pump points on load where the record
carries it (otherwise a constant for the running state) and the annual harmonic; compressor points on
load and the compressor-room ambient. No cross-machine correction. (Step 6 design freeze 2026-10-30:
the seasonal term added for spindle points; originally spindle speed alone. Checkpoint (b) diagnosis
decisions 2026-10-30: the same seasonal term added for pump points; originally load alone.)
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
SEASON_SIN = "_season_sin"
SEASON_COS = "_season_cos"
# F3 (2026-11-01, switch models.nbm.fixes.f3_spindle_temp_regime): the load regime on spindle points, as
# indicators against the medium regime
REGIME_LIGHT = "_regime_light"
REGIME_HEAVY = "_regime_heavy"
DAYS_PER_YEAR = 365.25
LOCAL_TZ = "America/Chicago"
MIN_FIT_HOURS = 24


def _column(rows: pd.DataFrame, c: str) -> np.ndarray:
    if c in (SEASON_SIN, SEASON_COS):
        doy = pd.to_datetime(rows["hour_utc"], utc=True).dt.tz_convert(LOCAL_TZ).dt.dayofyear.to_numpy(float)
        ang = 2 * np.pi * doy / DAYS_PER_YEAR
        return np.sin(ang) if c == SEASON_SIN else np.cos(ang)
    if c in (REGIME_LIGHT, REGIME_HEAVY):
        if "regime" not in rows:
            return np.full(len(rows), np.nan)
        return (rows["regime"].astype(str) == ("light" if c == REGIME_LIGHT else "heavy")).to_numpy(float)
    return rows[c].to_numpy(dtype=float) if c in rows else np.full(len(rows), np.nan)


def covariates(point_class: str, seasonal: bool = True, regime: bool = False) -> list[str]:
    if point_class.startswith("spindle"):
        base = [SPEED, SEASON_SIN, SEASON_COS] if seasonal else [SPEED]
        return base + [REGIME_LIGHT, REGIME_HEAVY] if regime else base
    if point_class.startswith("compressor"):
        return [LOAD, AMBIENT]
    # pumps (checkpoint (b) diagnosis decisions 2026-10-30, item 3.3): load and the annual harmonic, as on
    # spindles (was load alone)
    return [LOAD, SEASON_SIN, SEASON_COS] if seasonal else [LOAD]


def _clamped_season(rows: pd.DataFrame, c: str, days: list[int]) -> np.ndarray:
    """C2: the seasonal term at the nearest covered day of year (circular) for days outside the training
    coverage."""
    doy = pd.to_datetime(rows["hour_utc"], utc=True).dt.tz_convert(LOCAL_TZ).dt.dayofyear.to_numpy(int)
    cov = np.asarray(sorted(days), dtype=int)
    d = np.abs(doy[:, None] - cov[None, :])
    d = np.minimum(d, 366 - d)
    near = cov[np.argmin(d, axis=1)].astype(float)
    ang = 2 * np.pi * near / DAYS_PER_YEAR
    return np.sin(ang) if c == SEASON_SIN else np.cos(ang)


@dataclass
class TempFit:
    columns: list[str]
    coef: np.ndarray = field(default_factory=lambda: np.zeros(1))
    fill: dict[str, float] = field(default_factory=dict)
    # C2 (2026-11-01, switch models.nbm.fixes.c2_season_clamp): the days of year the training hours cover;
    # None = no clamp (the behavior before)
    season_days: list[int] | None = None

    def expected(self, rows: pd.DataFrame) -> np.ndarray:
        x = [np.ones(len(rows))]
        for c in self.columns:
            if self.season_days and c in (SEASON_SIN, SEASON_COS) and len(rows):
                v = _clamped_season(rows, c, self.season_days)
            else:
                v = _column(rows, c)
            x.append(np.where(np.isfinite(v), v, self.fill.get(c, 0.0)))
        return np.column_stack(x) @ self.coef

    def residual(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[TEMP].to_numpy(dtype=float) - self.expected(rows)

    def to_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "coef": [float(v) for v in self.coef], "fill": self.fill}


def fit(
    rows: pd.DataFrame,
    point_class: str,
    seasonal: bool = True,
    regime: bool = False,
    season_min_months: int | None = None,
    season_clamp: bool = False,
) -> TempFit:
    """Least-squares fit of temp_c on the point's covariates; a covariate with no finite value in the
    rows is dropped, and missing values are filled with the training median."""
    t = rows[TEMP].to_numpy(dtype=float)
    ok = np.isfinite(t)
    if season_min_months and point_class.startswith("spindle") and len(rows):
        # T2 (2026-11-01, switch models.nbm.fixes.t2_season_min_months): the seasonal term only where the
        # training hours cover at least that many distinct calendar months; otherwise speed and load only
        hours = pd.to_datetime(rows.loc[ok, "hour_utc"], utc=True)
        seasonal = seasonal and hours.dt.tz_convert(LOCAL_TZ).dt.month.nunique() >= int(season_min_months)
    vals = {c: _column(rows, c) for c in covariates(point_class, seasonal, regime)}
    cols = [c for c, v in vals.items() if np.isfinite(v).any()]
    fill = {c: float(np.nanmedian(vals[c])) for c in cols}
    tf = TempFit(cols, np.zeros(len(cols) + 1), fill)
    if season_clamp and point_class.startswith("spindle") and SEASON_SIN in cols and ok.any():
        hrs = pd.to_datetime(rows.loc[ok, "hour_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
        tf.season_days = sorted({int(d) for d in hrs.dt.dayofyear})
    if ok.sum() < MIN_FIT_HOURS:
        tf.coef[0] = float(np.nanmedian(t)) if ok.any() else 0.0
        return tf
    x = np.column_stack(
        [np.ones(len(rows))] + [np.where(np.isfinite(vals[c]), vals[c], fill[c]) for c in cols]
    )
    tf.coef = np.linalg.lstsq(x[ok], t[ok], rcond=None)[0]
    return tf


def hourly_ambient(room: pd.DataFrame) -> pd.Series:
    """Compressor-room ambient per UTC hour from the minute export (columns minute_utc, ambient_c)."""
    t = pd.to_datetime(room["minute_utc"], utc=True).dt.floor("h")
    return room.groupby(t)["ambient_c"].mean().rename(AMBIENT)
