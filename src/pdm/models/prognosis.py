"""Prognosis for one failure mode only: pump and compressor bearings from the first envelope
detection onward, with the exponential degradation model fitted to the envelope fault-band energy.

Inputs: per instance, the daily median envelope fault-band energy from the first envelope alert to the
failure work order (bronze only). Output: per prediction time the remaining-time estimate to the
failure threshold with an interval, and the alpha-lambda accuracy at the configured alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Instance:
    asset_id: str
    detected: pd.Timestamp
    failure: pd.Timestamp
    daily: pd.Series


def fit_exponential(t_days: np.ndarray, energy: np.ndarray) -> tuple[float, float, float]:
    """log(energy) = a + b t; returns a, b and the residual standard deviation."""
    y = np.log(np.maximum(energy, 1e-12))
    x = np.column_stack([np.ones_like(t_days), t_days])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    sd = float(np.std(resid, ddof=2)) if len(y) > 2 else np.nan
    return float(coef[0]), float(coef[1]), sd


def threshold_from_training(instances: list[Instance], last_days: int = 3) -> float:
    """Failure threshold: median of the energy over the last days before failure on training instances."""
    vals = [float(np.median(i.daily.iloc[-last_days:])) for i in instances if len(i.daily) >= last_days]
    return float(np.median(vals)) if vals else np.nan


def predict(inst: Instance, threshold: float, at_fracs: list[float], min_points: int = 5) -> pd.DataFrame:
    rows = []
    t_all = (inst.daily.index - inst.detected) / pd.Timedelta(days=1)
    life = (inst.failure - inst.detected) / pd.Timedelta(days=1)
    for lam in at_fracs:
        t_now = lam * life
        sel = t_all <= t_now
        if sel.sum() < min_points:
            continue
        a, b, sd = fit_exponential(t_all[sel].to_numpy(float), inst.daily[sel].to_numpy(float))
        if b <= 0:
            rul, lo, hi = np.inf, np.inf, np.inf
        else:
            t_hit = (np.log(threshold) - a) / b
            rul = max(t_hit - t_now, 0.0)
            span = 1.645 * sd / b if pd.notna(sd) else np.nan
            lo, hi = max(rul - span, 0.0), rul + span
        rows.append(
            {
                "asset_id": inst.asset_id,
                "lambda": lam,
                "t_days": t_now,
                "true_rul_days": life - t_now,
                "pred_rul_days": rul,
                "pi_low": lo,
                "pi_high": hi,
            }
        )
    return pd.DataFrame(rows)


def alpha_lambda(pred: pd.DataFrame, alpha: float) -> pd.DataFrame:
    p = pred.copy()
    p["within_alpha"] = (p["pred_rul_days"] >= (1 - alpha) * p["true_rul_days"]) & (
        p["pred_rul_days"] <= (1 + alpha) * p["true_rul_days"]
    )
    p["interval_width_days"] = p["pi_high"] - p["pi_low"]
    return p
