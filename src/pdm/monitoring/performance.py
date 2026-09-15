"""Monthly event performance per method, feature drift against the validation baseline, and the
retraining decision.

Inputs: caught-condition rows and alert-precision rows from the evaluation, the regime-tagged hourly
indicators, and the configured tolerances. Outputs: monthly event_performance, monthly drift per point
and indicator (Wasserstein distance scaled by the baseline spread) and retraining_decision. Profiles
on the platform come from Databricks data profiling; the local run computes the same drift statistic
as the reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

LOCAL_TZ = "America/Chicago"


def month_of(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts)
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t
    return t.dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m")


def event_performance(
    caught: pd.DataFrame, fa: pd.DataFrame, credit: pd.Series, n_assets: dict[str, int]
) -> pd.DataFrame:
    c = caught.assign(month=month_of(credit.reindex(caught["wo_id"]).reset_index(drop=True)).to_numpy())
    f = fa.assign(month=month_of(fa["ts"]))
    rows = []
    for (method, month), g in c.groupby(["method", "month"]):
        fm = f[(f["method"] == method) & (f["month"] == month)]
        rows.append(
            {
                "method": method,
                "month": month,
                "conditions": len(g),
                "recall": float(g["caught"].mean()) if len(g) else np.nan,
                "lead_days_median": float(g.loc[g["caught"], "warning_days"].median())
                if g["caught"].any()
                else np.nan,
                "alerts": len(fm),
                "alerts_per_asset_month": len(fm) / max(n_assets.get(method, 1), 1),
                "alert_precision": float(fm["confirmed_within_window"].mean()) if len(fm) else np.nan,
            }
        )
    for (method, month), fm in f.groupby(["method", "month"]):
        if not ((c["method"] == method) & (c["month"] == month)).any():
            rows.append(
                {
                    "method": method,
                    "month": month,
                    "conditions": 0,
                    "recall": np.nan,
                    "lead_days_median": np.nan,
                    "alerts": len(fm),
                    "alerts_per_asset_month": len(fm) / max(n_assets.get(method, 1), 1),
                    "alert_precision": float(fm["confirmed_within_window"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "month"]).reset_index(drop=True)


def feature_drift(
    ind: pd.DataFrame, features: list[str], baseline_months: list[str], months: list[str]
) -> pd.DataFrame:
    d = ind.assign(month=month_of(ind["hour_utc"]))
    rows = []
    for pid, g in d.groupby("point_id"):
        base = g[g["month"].isin(baseline_months)]
        for f in features:
            b = base[f].dropna().to_numpy(float)
            if b.size < 10:
                continue
            spread = float(np.std(b)) or 1.0
            for m in months:
                x = g.loc[g["month"] == m, f].dropna().to_numpy(float)
                if x.size < 10:
                    continue
                rows.append(
                    {
                        "point_id": pid,
                        "feature": f,
                        "month": m,
                        "wasserstein_scaled": wasserstein_distance(b, x) / spread,
                    }
                )
    return pd.DataFrame(rows)


def retraining_decision(
    perf: pd.DataFrame,
    drift: pd.DataFrame,
    reference: dict[str, dict[str, float]],
    cfg: dict[str, Any],
    method: str,
) -> pd.DataFrame:
    mon = cfg["monitoring"]
    tol = mon["tolerance"]
    lo, hi = tol["alerts_per_asset_month_ratio"]
    need = mon["consecutive_months"]
    share = mon["drift"]["feature_share_drifted"]
    thr = mon["drift"]["feature_threshold"]
    p = perf[perf["method"] == method].sort_values("month")
    ref = reference.get(method, {})
    rows, streak = [], 0
    for r in p.itertuples():
        recall_bad = (
            pd.notna(r.recall)
            and pd.notna(ref.get("recall"))
            and (ref["recall"] - r.recall) > tol["recall_drop"]
        )
        rate_ratio = (
            r.alerts_per_asset_month / ref["alerts_per_asset_month"]
            if ref.get("alerts_per_asset_month")
            else np.nan
        )
        rate_bad = pd.notna(rate_ratio) and not (lo <= rate_ratio <= hi)
        streak = streak + 1 if (recall_bad or rate_bad) else 0
        dm = drift[drift["month"] == r.month] if len(drift) else drift
        drifted = float((dm["wasserstein_scaled"] > thr).mean()) if len(dm) else np.nan
        rows.append(
            {
                "method": method,
                "month": r.month,
                "recall": r.recall,
                "recall_reference": ref.get("recall"),
                "alerts_rate_ratio": rate_ratio,
                "primary_trigger": bool(streak >= need),
                "drift_share": drifted,
                "secondary_trigger": bool(pd.notna(drifted) and drifted > share),
                "retrain": bool(streak >= need),
            }
        )
    return pd.DataFrame(rows)
