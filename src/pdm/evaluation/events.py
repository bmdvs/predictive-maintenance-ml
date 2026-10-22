"""Event-based evaluation of every alerting method against confirmed conditions from bronze CMMS.

Inputs: a unified alert table (method, asset, time), confirmed conditions, CMMS inspection labor,
work requests and contribution margin by machine class. Outputs: per method and condition type the
conditions caught, days of warning, false alerts per asset-month and the inspection hours they
represent, alert-level precision, diagnosis agreement, and the headline comparison with the downtime
and cost of unplanned repairs that a method's warning would have allowed to be planned.

A condition is caught when the method alerted on the condition's asset (or, for controller-only
alerts, on the condition's machine) within the lookback window before the condition was confirmed.
A PM finding is credited only when the alert precedes the PM date.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MACHINE_LEVEL = {"controller"}


def _match(alerts: pd.DataFrame, cond: pd.Series, method: str) -> pd.DataFrame:
    if method in MACHINE_LEVEL:
        return alerts[alerts["machine_id"] == cond["machine_id"]]
    return alerts[alerts["asset_id"] == cond["asset_id"]]


def caught_table(
    alerts: pd.DataFrame, conditions: pd.DataFrame, methods: list[str], lookback_days: float
) -> pd.DataFrame:
    rows = []
    lb = pd.Timedelta(days=lookback_days)
    for method in methods:
        a = alerts[alerts["method"] == method]
        for _, c in conditions.iterrows():
            m = _match(a, c, method)
            t = c["credit_ts"]
            prior = m[(m["ts"] < t) & (m["ts"] >= t - lb)]
            first = prior["ts"].min() if len(prior) else pd.NaT
            rows.append(
                {
                    "method": method,
                    "wo_id": c["wo_id"],
                    "asset_id": c["asset_id"],
                    "condition_type": c["condition_type"],
                    "unplanned": bool(c["unplanned"]),
                    "period": c["period"],
                    "caught": bool(len(prior)),
                    "warning_days": (t - first) / pd.Timedelta(days=1) if len(prior) else np.nan,
                    "downtime_hours": c["downtime_hours"],
                    "machine_class": c.get("machine_class"),
                }
            )
    return pd.DataFrame(rows)


def false_alerts(
    alerts: pd.DataFrame, conditions: pd.DataFrame, methods: list[str], window_days: float
) -> pd.DataFrame:
    """Alerts not followed by a confirmed condition on the matched asset within the window."""
    w = pd.Timedelta(days=window_days)
    rows = []
    for method in methods:
        a = alerts[alerts["method"] == method]
        for _, r in a.iterrows():
            key = "machine_id" if method in MACHINE_LEVEL else "asset_id"
            after = conditions[
                (conditions[key] == r[key])
                & (conditions["credit_ts"] >= r["ts"])
                & (conditions["credit_ts"] <= r["ts"] + w)
            ]
            rows.append(
                {
                    "method": method,
                    "asset_id": r["asset_id"],
                    "ts": r["ts"],
                    "confirmed_within_window": bool(len(after)),
                }
            )
    return pd.DataFrame(rows, columns=["method", "asset_id", "ts", "confirmed_within_window"])


def summary(
    caught: pd.DataFrame,
    fa: pd.DataFrame,
    n_assets: dict[str, int],
    months: float,
    inspection_hours_each: float,
    margin_by_class: dict[str, float],
    planned_lead_days: float,
    lookback_months: float = 0.0,
) -> pd.DataFrame:
    """chance_catch_share: the share of conditions an asset's alerts would precede by chance within the
    lookback, at the method's alert rate per asset-month (Poisson)."""
    rows = []
    for method, g in caught.groupby("method"):
        f = fa[fa["method"] == method]
        n_false = int((~f["confirmed_within_window"]).sum()) if len(f) else 0
        warn = g.loc[g["caught"], "warning_days"]
        avoid = g[g["caught"] & g["unplanned"] & (g["warning_days"] >= planned_lead_days)]
        cost = float(
            sum(r.downtime_hours * margin_by_class.get(r.machine_class, 0.0) for r in avoid.itertuples())
        )
        rows.append(
            {
                "method": method,
                "conditions": len(g),
                "caught": int(g["caught"].sum()),
                "caught_share": float(g["caught"].mean()) if len(g) else np.nan,
                "warning_days_median": float(warn.median()) if len(warn) else np.nan,
                "warning_days_min": float(warn.min()) if len(warn) else np.nan,
                "warning_days_max": float(warn.max()) if len(warn) else np.nan,
                "alerts": len(f),
                "alert_precision": float(f["confirmed_within_window"].mean()) if len(f) else np.nan,
                "false_alerts": n_false,
                "false_alerts_per_asset_month": n_false / (n_assets.get(method, 1) * months)
                if months > 0
                else np.nan,
                "false_alert_inspection_hours": n_false * inspection_hours_each,
                "unplanned_plannable": len(avoid),
                "unplanned_downtime_plannable_h": float(avoid["downtime_hours"].sum()),
                "unplanned_cost_plannable_usd": cost,
                "chance_catch_share": 1.0
                - float(np.exp(-(len(f) / (n_assets.get(method, 1) * months)) * lookback_months))
                if months > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def by_type(caught: pd.DataFrame) -> pd.DataFrame:
    return (
        caught.groupby(["method", "condition_type"])
        .agg(
            conditions=("wo_id", "count"),
            caught=("caught", "sum"),
            warning_days_median=("warning_days", "median"),
        )
        .reset_index()
    )


def diagnosis_agreement(
    requests: pd.DataFrame, conditions: pd.DataFrame, family: dict[str, str], window_days: float
) -> pd.DataFrame:
    """For work requests followed by a confirmed condition on the asset, whether the suspected
    condition matches the condition family from the finding code."""
    rows = []
    w = pd.Timedelta(days=window_days)
    for r in requests.itertuples():
        after = conditions[
            (conditions["asset_id"] == r.asset_id)
            & (conditions["credit_ts"] >= r.created)
            & (conditions["credit_ts"] <= r.created + w)
        ]
        if after.empty:
            continue
        c = after.sort_values("credit_ts").iloc[0]
        fam = family.get(c["condition_type"])
        rows.append(
            {
                "request_id": r.request_id,
                "asset_id": r.asset_id,
                "suspected": r.suspected_condition,
                "confirmed": c["condition_type"],
                "agree": fam == r.suspected_condition,
            }
        )
    return pd.DataFrame(rows, columns=["request_id", "asset_id", "suspected", "confirmed", "agree"])


def credit_time(conditions: pd.DataFrame, pm_started: pd.Series | None = None) -> pd.Series:
    """The time a method must alert before: the work-order open time, or the PM date for PM findings."""
    t = conditions["confirmed_ts"].copy()
    if pm_started is not None:
        t = t.where(~conditions["pm_finding"], pm_started.reindex(conditions.index).fillna(t))
    return t


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return (b - a) / pd.Timedelta(days=365.25 / 12)


def config_values(cfg: dict[str, Any]) -> dict[str, float]:
    e = cfg["evaluation"]
    return {"lookback_days": e["detection_lookback_days"], "precision_days": e["alert_precision_window_days"]}
