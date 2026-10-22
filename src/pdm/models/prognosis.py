"""Prognosis for one failure mode only: pump and compressor bearings from the first rule-floor
detection onward, with the exponential degradation model fitted to velocity RMS against the ISO
20816-3 zone D boundary of the asset's machine group.

Model: v(t) = phi + exp(a + b t + e), with phi the velocity before detection. A normal prior on (a, b)
comes from the training instances; each instance updates it with its own observations (conjugate
Bayesian linear regression on log(v - phi)), and the remaining-time distribution follows from
posterior draws. Output per prediction time: the median remaining time, the interval, alpha-lambda
accuracy and interval coverage.
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
    phi: float
    threshold: float


def design(inst: Instance, t_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    t = ((inst.daily.index - inst.detected) / pd.Timedelta(days=1)).to_numpy(float)
    v = inst.daily.to_numpy(float) - inst.phi
    ok = (v > 0) & np.isfinite(v)
    if t_max is not None:
        ok &= t <= t_max
    return np.column_stack([np.ones(ok.sum()), t[ok]]), np.log(v[ok])


@dataclass
class Prior:
    mean: np.ndarray
    cov: np.ndarray
    noise_var: float


def prior_from_training(instances: list[Instance]) -> Prior:
    coefs, resid = [], []
    for i in instances:
        x, y = design(i)
        if len(y) < 3:
            continue
        c, *_ = np.linalg.lstsq(x, y, rcond=None)
        coefs.append(c)
        resid.append(y - x @ c)
    c = np.array(coefs)
    mean = c.mean(axis=0)
    spread = np.cov(c.T) if len(c) > 2 else np.diag(np.maximum(c.var(axis=0), 1e-6))
    cov = spread + np.diag(np.maximum(np.abs(mean) * 0.5, 1e-3) ** 2)
    noise = float(np.var(np.concatenate(resid))) if resid else 0.1
    return Prior(mean, cov, max(noise, 1e-4))


def posterior(prior: Prior, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.linalg.inv(prior.cov)
    pn = p0 + x.T @ x / prior.noise_var
    cov = np.linalg.inv(pn)
    mean = cov @ (p0 @ prior.mean + x.T @ y / prior.noise_var)
    return mean, cov


def predict(
    inst: Instance, prior: Prior, at_fracs: list[float], interval: float, draws: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    life = (inst.failure - inst.detected) / pd.Timedelta(days=1)
    lo_q, hi_q = (1 - interval) / 2, 1 - (1 - interval) / 2
    rows = []
    for lam in at_fracs:
        t_now = lam * life
        x, y = design(inst, t_now)
        mean, cov = posterior(prior, x, y) if len(y) else (prior.mean, prior.cov)
        ab = rng.multivariate_normal(mean, cov, size=draws)
        target = np.log(max(inst.threshold - inst.phi, 1e-6))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hit = np.where(ab[:, 1] > 0, (target - ab[:, 0]) / ab[:, 1], np.inf)
        rul = np.maximum(t_hit - t_now, 0.0)
        finite = rul[np.isfinite(rul)]
        rows.append(
            {
                "asset_id": inst.asset_id,
                "lambda": lam,
                "t_days": t_now,
                "observations": len(y),
                "true_rul_days": life - t_now,
                "pred_rul_days": float(np.median(rul)) if np.isfinite(np.median(rul)) else np.inf,
                "pi_low": float(np.quantile(finite, lo_q)) if finite.size else np.inf,
                "pi_high": float(np.quantile(finite, hi_q)) if finite.size else np.inf,
                "share_never_reaching": float(np.mean(~np.isfinite(rul))),
            }
        )
    return pd.DataFrame(rows)


def scores(pred: pd.DataFrame, alpha: float) -> pd.DataFrame:
    p = pred.copy()
    p["within_alpha"] = (p["pred_rul_days"] >= (1 - alpha) * p["true_rul_days"]) & (
        p["pred_rul_days"] <= (1 + alpha) * p["true_rul_days"]
    )
    p["covered"] = (p["true_rul_days"] >= p["pi_low"]) & (p["true_rul_days"] <= p["pi_high"])
    p["interval_width_days"] = p["pi_high"] - p["pi_low"]
    return p
