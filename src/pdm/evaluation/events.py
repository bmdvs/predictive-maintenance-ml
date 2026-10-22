"""Event-based evaluation against confirmed conditions from bronze CMMS.

A method catches a confirmed condition if it had an alert episode or an open case on the same asset, at
the affected point (or any point for asset-level causes), at the moment of confirmation, opened within
the configured days before it. An alert episode runs from its first persisted exceedance until the
configured gap after its last; a fused case runs over its open interval. Warning days run from the
episode opening, or for fused cases from the first opening of the case's chain (warning_from).
The chance share is the share of time the asset (at the condition's scope)
has an episode or case open that was opened within the window. False episodes are those whose open
interval holds no confirmation on the asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)
AFFECTED_POINT = {"CMP-BRG-MOTOR": "-MDE", "CMP-BRG-AIREND": "-AE", "coolant_ingress": "-F"}
INSTRUMENTED = r"^(HMC-\d+-(SP|TSC|CP)|CMP-\d+)$"


def condition_scope(conds: pd.DataFrame, points_by_asset: dict[str, list[str]]) -> pd.DataFrame:
    """Asset key for matching and the affected point (None for asset-level causes)."""
    c = conds.copy()
    inst = c["asset_id"].str.match(INSTRUMENTED)
    c["asset_key"] = np.where(inst, c["asset_id"], c["asset_id"].str.extract(r"^([A-Z]+-\d+)")[0])
    pts = []
    for r in c.itertuples():
        cand = points_by_asset.get(r.asset_id, [])
        suffix = AFFECTED_POINT.get(r.failure_code) if isinstance(r.failure_code, str) else None
        if len(cand) == 1:
            pts.append(cand[0])
        elif suffix:
            hit = [p for p in cand if p.endswith(suffix)]
            pts.append(hit[0] if hit else None)
        else:
            pts.append(None)
    c["point_scope"] = pts
    return c


def episodes(exceed: pd.DataFrame, gap_days: float) -> pd.DataFrame:
    """Episodes per asset and point from persisted exceedance or alert times (columns asset_id,
    point_id, ts). An episode closes gap_days after its last exceedance."""
    gap = pd.Timedelta(days=gap_days)
    if exceed.empty:
        return pd.DataFrame(columns=["asset_id", "point_id", "opened", "last", "closed"])
    rows = []
    e = exceed.assign(point_id=exceed["point_id"].fillna(""))
    for (asset, pid), g in e.sort_values("ts").groupby(["asset_id", "point_id"]):
        t = pd.to_datetime(g["ts"], utc=True).to_list()
        start = last = t[0]
        for x in t[1:]:
            if x - last > gap:
                rows.append((asset, pid or None, start, last))
                start = x
            last = x
        rows.append((asset, pid or None, start, last))
    out = pd.DataFrame(rows, columns=["asset_id", "point_id", "opened", "last"])
    out["closed"] = out["last"] + gap
    return out


def _scope_mask(ep: pd.DataFrame, asset_key: str, point: str | None) -> pd.Series:
    m = ep["asset_id"] == asset_key
    if point is not None:
        pts = ep["point_id"].fillna("")
        m &= (pts == "") | pts.str.contains(point, regex=False)
    return m


def catch_table(ep: pd.DataFrame, conds: pd.DataFrame, window_days: float) -> pd.DataFrame:
    w = pd.Timedelta(days=window_days)
    rows = []
    for c in conds.itertuples():
        t = c.credit_ts
        m = (
            _scope_mask(ep, c.asset_key, c.point_scope)
            & (ep["opened"] <= t)
            & (ep["closed"] > t)
            & (ep["opened"] >= t - w)
        )
        hit = ep[m]
        col = "warning_from" if "warning_from" in ep else "opened"
        first = hit[col].min() if len(hit) else pd.NaT
        rows.append(
            {
                "wo_id": c.wo_id,
                "asset_id": c.asset_id,
                "point_scope": c.point_scope,
                "condition_type": c.condition_type,
                "unplanned": bool(c.unplanned),
                "caught": bool(len(hit)),
                "warning_days": (t - first) / DAY if len(hit) else np.nan,
                "downtime_hours": c.downtime_hours,
                "machine_class": c.machine_class,
            }
        )
    return pd.DataFrame(rows)


def chance_share(
    ep: pd.DataFrame, conds: pd.DataFrame, period: tuple[pd.Timestamp, pd.Timestamp], window_days: float
) -> float:
    """Mean over conditions of the time share in the period when an episode or case at the condition's
    scope is open and was opened within the window (hourly grid)."""
    g = pd.date_range(period[0], period[1], freq="h", inclusive="left")
    w = pd.Timedelta(days=window_days)
    cache: dict[tuple[str, str | None], float] = {}
    vals = []
    for c in conds.itertuples():
        key = (c.asset_key, c.point_scope)
        if key not in cache:
            e = ep[_scope_mask(ep, c.asset_key, c.point_scope)]
            on = np.zeros(len(g), dtype=bool)
            for r in e.itertuples():
                on |= np.asarray((g >= r.opened) & (g < r.closed) & (g - r.opened <= w))
            cache[key] = float(on.mean()) if len(on) else np.nan
        vals.append(cache[key])
    return float(np.nanmean(vals)) if vals else np.nan


def false_episodes(ep: pd.DataFrame, conds_all: pd.DataFrame) -> pd.Series:
    """True where no confirmation on the asset falls inside the episode's open interval."""
    out = []
    for r in ep.itertuples():
        hit = conds_all[
            (conds_all["asset_key"] == r.asset_id)
            & (conds_all["credit_ts"] >= r.opened)
            & (conds_all["credit_ts"] < r.closed)
        ]
        out.append(hit.empty)
    return pd.Series(out, index=ep.index, dtype=bool)


def summarize(
    method: str,
    level: str,
    ep: pd.DataFrame,
    conds: pd.DataFrame,
    conds_all: pd.DataFrame,
    period: tuple[pd.Timestamp, pd.Timestamp],
    n_assets: int,
    window_days: float,
    inspection_hours_each: float,
    margin_by_class: dict[str, float],
    planned_lead_days: float,
) -> tuple[dict, pd.DataFrame]:
    a0, a1 = period
    ep_p = ep[(ep["opened"] >= a0) & (ep["opened"] < a1)]
    caught = catch_table(ep, conds, window_days).assign(method=method, level=level)
    months = (a1 - a0) / pd.Timedelta(days=365.25 / 12)
    fe = false_episodes(ep_p, conds_all) if len(ep_p) else pd.Series(dtype=bool)
    n_false = int(fe.sum())
    warn = caught.loc[caught["caught"], "warning_days"] if len(caught) else pd.Series(dtype=float)
    plan = (
        caught[caught["caught"] & caught["unplanned"] & (caught["warning_days"] >= planned_lead_days)]
        if len(caught)
        else caught
    )
    cost = float(sum(r.downtime_hours * margin_by_class.get(r.machine_class, 0.0) for r in plan.itertuples()))
    row = {
        "method": method,
        "level": level,
        "conditions": len(caught),
        "caught": int(caught["caught"].sum()) if len(caught) else 0,
        "caught_share": float(caught["caught"].mean()) if len(caught) else np.nan,
        "chance_share": chance_share(ep, conds, period, window_days),
        "warning_days_median": float(warn.median()) if len(warn) else np.nan,
        "warning_days_min": float(warn.min()) if len(warn) else np.nan,
        "warning_days_max": float(warn.max()) if len(warn) else np.nan,
        "episodes": len(ep_p),
        "episodes_per_asset_month": len(ep_p) / (n_assets * months),
        "false_episodes": n_false,
        "false_per_asset_month": n_false / (n_assets * months),
        "false_inspection_hours": n_false * inspection_hours_each,
        "unplanned_plannable": len(plan),
        "unplanned_downtime_plannable_h": float(plan["downtime_hours"].sum()) if len(plan) else 0.0,
        "unplanned_cost_plannable_usd": cost,
    }
    return row, caught


def at_budget(curve: pd.DataFrame, budget: float) -> dict[str, float]:
    """Catch share interpolated at a false-alarm budget along a method's operating curve; outside the
    curve's range the nearest end is returned and flagged."""
    c = curve.sort_values("false_per_asset_month")
    x = c["false_per_asset_month"].to_numpy(dtype=float)
    y = c["caught_share"].to_numpy(dtype=float)
    ch = c["chance_share"].to_numpy(dtype=float)
    inside = bool(len(x) and x.min() <= budget <= x.max())
    return {
        "caught_share": float(np.interp(budget, x, y)) if len(x) else np.nan,
        "chance_share": float(np.interp(budget, x, ch)) if len(x) else np.nan,
        "within_curve": inside,
    }


def by_type(caught: pd.DataFrame) -> pd.DataFrame:
    return (
        caught.groupby(["method", "level", "condition_type"])
        .agg(
            conditions=("wo_id", "count"),
            caught=("caught", "sum"),
            warning_days_median=("warning_days", "median"),
        )
        .reset_index()
    )
