"""Monthly event performance per method, feature and score drift, and the retraining decision.

Inputs: caught-condition rows and alert-precision rows from the evaluation, the regime-tagged hourly
indicators, the per-point model scores, each point's post-repair re-baseline windows and the configured
tolerances. Drift is the Wasserstein distance scaled by the reference spread; the reference is the
validation period, or for a point re-baselined after the validation start its latest completed
post-repair re-baseline window plus the first scored days after it with no open case on the point (the
window alone, marked, when a case opens within those days). Outputs: monthly event_performance, monthly
drift per point and feature (the per-point model score as feature "nbm_score") and
retraining_decision with the action a retrain verdict calls for. Profiles on the platform come from
Databricks data profiling; the local run computes the same drift statistic as the reference.
"""

from __future__ import annotations

from typing import Any, NamedTuple

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


class Reference(NamedTuple):
    """A post-repair drift reference: the re-baseline window [start, window_end) extended to
    reference_end; reference_end is None while the reference is still forming at the horizon."""

    start: pd.Timestamp
    window_end: pd.Timestamp
    reference_end: pd.Timestamp | None
    window_only: bool


def drift_references(
    windows: list[tuple[pd.Timestamp, pd.Timestamp, bool]],
    scored_hours: pd.Series,
    cases: list[tuple[pd.Timestamp, pd.Timestamp]],
    end: pd.Timestamp,
    days: int,
) -> list[Reference]:
    """Per re-baseline window of a point: the window plus the first `days` scored local days after it
    with no case open on the point, or the window alone (window_only) when a case opens within them.

    Args:
        windows: (start, window_end, complete) per window; an incomplete window is still forming.
        scored_hours: the point's per-point model score hours (UTC) before end.
        cases: the point's asset's cases (opened, closed), opened before end; closed is cut at end.
        end: the run's horizon.
        days: the scored days that extend the window.
    """
    hours = pd.to_datetime(pd.Series(scored_hours), utc=True).sort_values()
    out = []
    for a, b, complete in sorted(windows):
        if not complete:
            out.append(Reference(a, b, None, False))
            continue
        after = hours[(hours >= b) & (hours < end)]
        open_ = np.zeros(len(after), dtype=bool)
        for o, c in cases:
            open_ |= ((after >= o) & (after < c)).to_numpy()
        day = after[~open_].dt.tz_convert(LOCAL_TZ).dt.normalize()
        first = list(dict.fromkeys(day))[:days]
        span_end = (first[-1] + pd.Timedelta(days=1)).tz_convert("UTC") if len(first) == days else end
        opened = any(b <= o < span_end for o, _ in cases)
        if opened:
            out.append(Reference(a, b, b, True))
        elif len(first) == days:
            out.append(Reference(a, b, span_end, False))
        else:
            out.append(Reference(a, b, None, False))
    return out


def feature_drift(
    ind: pd.DataFrame,
    features: list[str],
    baseline_months: list[str],
    months: list[str],
    references: dict[str, list[Reference]] | None = None,
) -> pd.DataFrame:
    """Per point, feature and month: the month's values against the reference.

    Args:
        references: per point, its post-repair drift references (drift_references). Once a reference
            has formed by a month's end, the month is compared with it, from its end on; hours inside a
            reference, or after the start of one still forming, are not compared.

    Returns:
        point_id, feature, month, wasserstein_scaled and reference ("validation", "re-baseline <start
        date> plus scored days" or "re-baseline <start date>, window only (case within the scored
        days)").
    """
    references = references or {}
    d = ind.assign(month=month_of(ind["hour_utc"]))
    rows = []
    for pid, g in d.groupby("point_id"):
        t = pd.to_datetime(g["hour_utc"], utc=True)
        refs = sorted(references.get(pid, []))
        inside = np.zeros(len(g), dtype=bool)
        for r in refs:
            stop = r.reference_end
            inside |= ((t >= r.start) & ((t < stop) if stop is not None else True)).to_numpy()
        for m in months:
            done = [r for r in refs if r.reference_end is not None and r.reference_end <= month_end_utc(m)]
            in_month = (g["month"] == m).to_numpy() & ~inside
            if done:
                r = done[-1]
                base_mask = ((t >= r.start) & (t < r.reference_end)).to_numpy()
                cur_mask = in_month & (t >= r.reference_end).to_numpy()
                day = f"{r.start.tz_convert(LOCAL_TZ):%Y-%m-%d}"
                ref = (
                    f"re-baseline {day}, window only (case within the scored days)"
                    if r.window_only
                    else f"re-baseline {day} plus scored days"
                )
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
