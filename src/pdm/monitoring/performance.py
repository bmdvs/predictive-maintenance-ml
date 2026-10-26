"""Monthly event performance per method, feature and score drift, and the retraining decision.

Inputs: caught-condition rows and alert-precision rows from the evaluation, the regime-tagged hourly
indicators, the per-point model scores, each point's post-repair re-baseline windows and the configured
tolerances. Drift is the Wasserstein distance scaled by the reference spread; the reference is the
validation period, or for a point re-baselined after the validation start its latest completed
post-repair re-baseline window, as the baselines are. Outputs: monthly event_performance, monthly
drift per point and feature (the per-point model score as feature "nbm_score") and
retraining_decision with the action a retrain verdict calls for. Profiles on the platform come from
Databricks data profiling; the local run computes the same drift statistic as the reference.
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


SCORE_FEATURE = "nbm_score"
RETRAIN_ACTION = (
    "review and full refit: training cutoff moved forward, healthy windows and thresholds re-derived, "
    "registered as a challenger with auto_promote=false for manual review"
)


def month_end_utc(month: str) -> pd.Timestamp:
    """The end of a local month (YYYY-MM) as a UTC timestamp."""
    start = pd.Timestamp(f"{month}-01").tz_localize(LOCAL_TZ)
    return (start + pd.offsets.MonthBegin(1)).tz_convert("UTC")


def feature_drift(
    ind: pd.DataFrame,
    features: list[str],
    baseline_months: list[str],
    months: list[str],
    rebaselines: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] | None = None,
) -> pd.DataFrame:
    """Per point, feature and month: the month's values against the reference.

    Args:
        rebaselines: per point, its post-repair re-baseline windows (start, end) that started after the
            validation start and passed the post-repair check. Once a window has completed by a month's
            end, the month is compared with that window, from the window's end on; hours inside any
            re-baseline window are not compared.

    Returns:
        point_id, feature, month, wasserstein_scaled and reference ("validation" or "re-baseline
        <start date>").
    """
    rebaselines = rebaselines or {}
    d = ind.assign(month=month_of(ind["hour_utc"]))
    rows = []
    for pid, g in d.groupby("point_id"):
        t = pd.to_datetime(g["hour_utc"], utc=True)
        wins = sorted(rebaselines.get(pid, []))
        inside = np.zeros(len(g), dtype=bool)
        for a, b in wins:
            inside |= ((t >= a) & (t < b)).to_numpy()
        for m in months:
            done = [w for w in wins if w[1] <= month_end_utc(m)]
            in_month = (g["month"] == m).to_numpy() & ~inside
            if done:
                a, b = done[-1]
                base_mask = ((t >= a) & (t < b)).to_numpy()
                cur_mask = in_month & (t >= b).to_numpy()
                ref = f"re-baseline {a.tz_convert(LOCAL_TZ):%Y-%m-%d}"
            else:
                base_mask = g["month"].isin(baseline_months).to_numpy()
                cur_mask = in_month
                ref = "validation"
            for f in features:
                b_ = g.loc[base_mask, f].dropna().to_numpy(float)
                x = g.loc[cur_mask, f].dropna().to_numpy(float)
                if b_.size < 10 or x.size < 10:
                    continue
                spread = float(np.std(b_)) or 1.0
                rows.append(
                    {
                        "point_id": pid,
                        "feature": f,
                        "month": m,
                        "wasserstein_scaled": wasserstein_distance(b_, x) / spread,
                        "reference": ref,
                    }
                )
    return pd.DataFrame(rows, columns=["point_id", "feature", "month", "wasserstein_scaled", "reference"])


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
        fm = dm[dm["feature"] != SCORE_FEATURE] if len(dm) else dm
        sm = dm[dm["feature"] == SCORE_FEATURE] if len(dm) else dm
        drifted = float((fm["wasserstein_scaled"] > thr).mean()) if len(fm) else np.nan
        score_drifted = float((sm["wasserstein_scaled"] > thr).mean()) if len(sm) else np.nan
        retrain = bool(streak >= need)
        rows.append(
            {
                "method": method,
                "month": r.month,
                "recall": r.recall,
                "recall_reference": ref.get("recall"),
                "alerts_rate_ratio": rate_ratio,
                "primary_trigger": retrain,
                "drift_share": drifted,
                "score_drift_share": score_drifted,
                "secondary_trigger": bool(
                    (pd.notna(drifted) and drifted > share)
                    or (pd.notna(score_drifted) and score_drifted > share)
                ),
                "retrain": retrain,
                "action": RETRAIN_ACTION if retrain else "none",
            }
        )
    return pd.DataFrame(rows)
