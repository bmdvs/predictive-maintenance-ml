"""Failure-within-horizon classifier for the rotating auxiliaries (TSC pumps, flood pumps, compressors),
trained only when the training period holds the configured minimum number of failure events.

Inputs: rolling features per point-hour with hours since last repair, and the auxiliaries' unplanned
failures from bronze CMMS. Outputs: the training gate decision with the event count, and, when
trained, calibrated logistic and gradient-boosting candidates with event-wise evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

AUX_PATTERN = r"^(HMC-\d+-(TSC|CP)|CMP-\d+)$"


def training_gate(failures: pd.DataFrame, train_end: pd.Timestamp, cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg["models"]["supervised"]
    aux = failures[failures["asset_id"].str.match(AUX_PATTERN)]
    n = int((aux["confirmed_ts"] < train_end).sum())
    return {"events_in_training": n, "minimum": s["min_train_events"], "train": n >= s["min_train_events"]}


def labels(rows: pd.DataFrame, failures: pd.DataFrame, horizon_days: float) -> pd.Series:
    """1 when an unplanned failure on the row's asset follows within the horizon."""
    h = pd.Timedelta(days=horizon_days)
    y = np.zeros(len(rows), dtype=int)
    for asset, g in rows.groupby("asset_id"):
        f = failures.loc[failures["asset_id"] == asset, "confirmed_ts"].sort_values().to_numpy()
        if f.size == 0:
            continue
        t = g["hour_utc"].to_numpy()
        idx = np.searchsorted(f, t, side="left")
        nxt = np.where(idx < f.size, f[np.minimum(idx, f.size - 1)], np.datetime64("NaT"))
        y[g.index.to_numpy()] = ((nxt - t) <= h.to_timedelta64()).astype(int)
    return pd.Series(y, index=rows.index)


def fit_candidates(x: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    seed = cfg["models"]["seed"]
    method = cfg["models"]["supervised"]["calibration"]
    models = {
        "logistic_l2": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        ),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    return {
        k: CalibratedClassifierCV(m, method=method, cv=3).fit(x.fillna(0.0), y) for k, m in models.items()
    }
