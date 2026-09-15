"""Controller-only health for all 20 machines: per-program axis-load and spindle-load drift against
the program's first-month baseline, and the spindle thermistor residual against its expected value
for speed and season.

Input: hourly controller features (pandas, one machine). Outputs: daily drift and residual scores per
machine and the persisted alerts. The thermistor model is fitted on the training period only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pdm.rules.persistence import ConsecutiveCounter, Suppressor

AXES = ("x", "y", "z", "b")
FIRST_MONTH = pd.Timedelta(days=30)
DAYS_PER_YEAR = 365.25


def program_baselines(h: pd.DataFrame) -> pd.DataFrame:
    """Median axis and spindle load per program over its first month of cutting in the record."""
    first = h.groupby("program")["hour_utc"].transform("min")
    fm = h[h["hour_utc"] < first + FIRST_MONTH]
    cols = [f"{a}_load_p95" for a in AXES] + ["spindle_load"]
    return fm.groupby("program")[cols].median().add_suffix("_base")


def drift_daily(h: pd.DataFrame) -> pd.DataFrame:
    """Active-time-weighted daily drift per axis and for the spindle load, across programs."""
    base = program_baselines(h)
    d = h.join(base, on="program")
    out = d[["machine_id", "local_date", "active_s"]].copy()
    for c in [f"{a}_load_p95" for a in AXES] + ["spindle_load"]:
        out[f"{c}_drift"] = d[c] / d[f"{c}_base"] - 1.0
    w = out["active_s"]

    def wavg(g: pd.DataFrame) -> pd.Series:
        r = {}
        for c in [k for k in g.columns if k.endswith("_drift")]:
            m = g[c].notna()
            r[c] = (
                float(np.average(g.loc[m, c], weights=g.loc[m, "active_s"]))
                if m.any() and g.loc[m, "active_s"].sum() > 0
                else np.nan
            )
        return pd.Series(r)

    del w
    daily = out.groupby(["machine_id", "local_date"]).apply(wavg, include_groups=False).reset_index()
    axis_cols = [f"{a}_load_p95_drift" for a in AXES]
    daily["axis_drift_max"] = daily[axis_cols].max(axis=1)
    return daily


@dataclass
class ThermistorModel:
    coef: np.ndarray
    rated_rpm: float

    @staticmethod
    def features(h: pd.DataFrame, rated_rpm: float) -> np.ndarray:
        doy = pd.to_datetime(h["local_date"]).dt.dayofyear.to_numpy(float)
        ang = 2 * np.pi * doy / DAYS_PER_YEAR
        return np.column_stack(
            [np.ones(len(h)), h["spindle_speed_rpm"].to_numpy(float) / rated_rpm, np.sin(ang), np.cos(ang)]
        )

    @classmethod
    def fit(cls, h: pd.DataFrame, rated_rpm: float) -> ThermistorModel:
        x = cls.features(h, rated_rpm)
        y = h["spindle_temp_c"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        coef = np.linalg.lstsq(x[ok], y[ok], rcond=None)[0]
        resid = y[ok] - x[ok] @ coef
        keep = np.abs(resid - np.median(resid)) < 3 * 1.4826 * np.median(np.abs(resid - np.median(resid)))
        coef = np.linalg.lstsq(x[ok][keep], y[ok][keep], rcond=None)[0]
        return cls(coef, rated_rpm)

    def residual(self, h: pd.DataFrame) -> np.ndarray:
        return h["spindle_temp_c"].to_numpy(float) - self.features(h, self.rated_rpm) @ self.coef


def residual_daily(h: pd.DataFrame, model: ThermistorModel) -> pd.DataFrame:
    d = h[["machine_id", "local_date"]].copy()
    d["thermistor_residual_c"] = model.residual(h)
    return d.groupby(["machine_id", "local_date"])["thermistor_residual_c"].median().reset_index()


def daily_scores(
    h: pd.DataFrame, rated_rpm: float, train_end: pd.Timestamp
) -> tuple[pd.DataFrame, ThermistorModel]:
    """Daily drift and thermistor residual for one machine; the thermistor model uses training days only."""
    train = h[pd.to_datetime(h["local_date"]) <= train_end]
    model = ThermistorModel.fit(train, rated_rpm)
    scores = drift_daily(h).merge(residual_daily(h, model), on=["machine_id", "local_date"], how="outer")
    return scores.sort_values("local_date").reset_index(drop=True), model


def alerts(scores: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Persisted daily alerts on axis drift, spindle-load drift and thermistor residual."""
    m = cfg["models"]["controller_health"]
    persist = cfg["rules"]["persistence_consecutive"]
    sup_days = cfg["rules"]["suppression_days"]
    rules = {
        "axis_drift": ("axis_drift_max", m["drift_alert_frac"]),
        "spindle_load_drift": ("spindle_load_drift", m["drift_alert_frac"]),
        "thermistor_residual": ("thermistor_residual_c", m["thermistor_residual_alert_c"]),
    }
    rows = []
    for mid, g in scores.groupby("machine_id"):
        g = g.sort_values("local_date")
        for name, (col, thr) in rules.items():
            if col not in g:
                continue
            counter, sup = ConsecutiveCounter(persist), Suppressor(sup_days)
            for r in g.itertuples():
                v = getattr(r, col)
                cond = None if pd.isna(v) else bool(v >= thr)
                ts = pd.Timestamp(r.local_date)
                if counter.update(cond) and cond and sup.allow(ts):
                    rows.append(
                        {
                            "machine_id": mid,
                            "alert_date": r.local_date,
                            "rule": name,
                            "value": float(v),
                            "threshold": thr,
                        }
                    )
    return pd.DataFrame(rows, columns=["machine_id", "alert_date", "rule", "value", "threshold"])
