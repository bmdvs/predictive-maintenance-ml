"""Evaluation of the ML Model against the Vendor Rules, with Controller Health beside them (revision 2).

Detection is compared at ALERT or higher: a condition is caught when an ALERT episode was open on its asset
at confirmation and opened within the catch window. Prioritization is compared at ALARM: the share of
failures that reached ALARM before failure, days from ALARM to failure and the false ALARM rate. False
episodes are counted per asset for every method. Conditions fall into four groups by the ML Model and the
Vendor Rules, with Controller Health as a column; conditions only the ML Model caught, and that Controller
Health missed, are valued by the R3.6 rule.
Inputs: level episodes per method, confirmed conditions, work orders and controller exceedances. Outputs:
the evaluation tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pdm.evaluation import events as ev
from pdm.models import severity

DAY = pd.Timedelta(days=1)
MONTH = pd.Timedelta(days=365.25 / 12)
LOCAL_TZ = "America/Chicago"
REPAIRS = ("unplanned_repair", "condition_based", "corrective")


def asset_months(period: tuple[pd.Timestamp, pd.Timestamp], n_assets: int) -> float:
    return n_assets * (period[1] - period[0]) / MONTH


def conditions_in(
    conds: pd.DataFrame, period: tuple[pd.Timestamp, pd.Timestamp], instrumented: bool = True
) -> pd.DataFrame:
    c = conds[(conds["credit_ts"] >= period[0]) & (conds["credit_ts"] < period[1])]
    c = c[c["instrumented"] == instrumented].copy()
    c["point_scope"] = None
    return c.reset_index(drop=True)


def alert_view(lev: pd.DataFrame) -> pd.DataFrame:
    return lev.assign(point_id="")[["asset_id", "point_id", "opened", "last", "closed"]].reset_index(
        drop=True
    )


def alarm_view(lev: pd.DataFrame, escalation: bool = True, sensor_only: bool = False) -> pd.DataFrame:
    """ALARM periods: from the ALARM start to the episode's close, with or without the 14-day rule, on the
    full criteria or on the sensor criteria only."""
    pre = "alarm_sensor_" if sensor_only else "alarm_"
    start = lev[f"{pre}ts"] if escalation else lev[f"{pre}criterion_ts"]
    a = lev[start.notna()].assign(opened=start[start.notna()], point_id="")
    return a[["asset_id", "point_id", "opened", "last", "closed"]].reset_index(drop=True)


def false_mask(eps: pd.DataFrame, conds_all: pd.DataFrame) -> pd.Series:
    if eps.empty:
        return pd.Series(dtype=bool)
    return ev.false_episodes(eps, conds_all)


def method_row(
    method: str,
    level: str,
    eps: pd.DataFrame,
    conds: pd.DataFrame,
    conds_all: pd.DataFrame,
    period: tuple[pd.Timestamp, pd.Timestamp],
    n_assets: int,
    window_days: float,
    ctx: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    row, caught = ev.summarize(
        method,
        level,
        eps,
        conds,
        conds_all,
        period,
        n_assets,
        window_days,
        ctx["inspection_hours_each"],
        ctx["margin"],
        cfg["evaluation"]["plannable_min_warning_days"],
    )
    row["asset_months"] = asset_months(period, n_assets)
    return row, caught


def at_budget(curve: pd.DataFrame, budget: float) -> dict[str, Any]:
    return ev.at_budget(curve, budget)


def _cover(eps: pd.DataFrame, conds_all: pd.DataFrame) -> list[np.ndarray]:
    """Indices of conditions (rows of conds_all) confirmed inside each episode's open interval on its
    asset."""
    out = []
    ak = conds_all["asset_key"].to_numpy()
    ct = conds_all["credit_ts"].to_numpy()
    for r in eps.itertuples():
        out.append(
            np.flatnonzero(
                (ak == r.asset_id)
                & (ct >= np.datetime64(r.opened.tz_convert(None)))
                & (ct < np.datetime64(r.closed.tz_convert(None)))
            )
        )
    return out


def bootstrap(
    vendor_caught: np.ndarray,
    vendor_eps_cover: list[np.ndarray],
    ml_caught: dict[float, np.ndarray],
    ml_eps_cover: dict[float, list[np.ndarray]],
    ml_op: float,
    eval_index: np.ndarray,
    n_all: int,
    asset_months_: float,
    draws: int,
    seed: int,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, float]:
    """90% intervals on (a) the ML Model's catch share at the Vendor Rules' false-episode budget less the
    Vendor Rules' catch share and (b) the ML Model's false-episode rate at its operating threshold less the
    Vendor Rules', resampling whole conditions with replacement. Conditions outside the resample count as
    unconfirmed, so an episode they covered becomes false. eval_index: positions of the evaluated
    conditions among all conditions (conds_all rows)."""
    rng = np.random.default_rng(seed)
    n = len(eval_index)
    if n == 0:
        return {}
    d_catch, d_false = [], []
    pcts = sorted(ml_caught)
    for _ in range(draws):
        pick = rng.integers(0, n, n)
        present = np.zeros(n_all, dtype=bool)
        present[eval_index[pick]] = True
        present_other = np.ones(n_all, dtype=bool)
        present_other[eval_index] = False
        present |= present_other

        def false_rate(cover: list[np.ndarray], present: np.ndarray = present) -> float:
            return sum(1 for c in cover if not present[c].any()) / asset_months_

        v_false = false_rate(vendor_eps_cover)
        v_catch = float(vendor_caught[pick].mean())
        xs = np.array([false_rate(ml_eps_cover[p]) for p in pcts])
        ys = np.array([float(ml_caught[p][pick].mean()) for p in pcts])
        o = np.argsort(xs)
        d_catch.append(float(np.interp(v_false, xs[o], ys[o])) - v_catch)
        d_false.append(false_rate(ml_eps_cover[ml_op]) - v_false)
    q = np.quantile(d_catch, quantiles), np.quantile(d_false, quantiles)
    return {
        "catch_diff_lo": float(q[0][0]),
        "catch_diff_hi": float(q[0][1]),
        "false_diff_lo": float(q[1][0]),
        "false_diff_hi": float(q[1][1]),
        "draws": draws,
    }


def controller_signals(ex: pd.DataFrame, hmc: set[str], scale: float = 1.0) -> pd.DataFrame:
    """Controller Health exceedances as signals per asset: thermistor residual and spindle-load drift on an
    HMC go to its spindle asset, everything else to the machine."""
    e = ex[np.isclose(ex["sweep"], scale)].reset_index(drop=True)
    ts = pd.to_datetime(e["alert_date"]).dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC") + pd.Timedelta(
        hours=23
    )
    asset = [
        f"{m}-SP" if (r in ("thermistor_residual", "spindle_load_drift") and m in hmc) else m
        for m, r in zip(e["machine_id"], e["rule"], strict=True)
    ]
    return pd.DataFrame({"asset_id": asset, "point_id": "", "ts": ts, "rules": e["rule"].to_numpy()})


def controller_episodes(sig: pd.DataFrame, gap_days: float) -> pd.DataFrame:
    return severity.episodes(sig, gap_days)


def condition_table(
    conds: pd.DataFrame,
    ml: pd.DataFrame,
    vendor: pd.DataFrame,
    ch: pd.DataFrame,
    window_days: float,
) -> pd.DataFrame:
    """Per confirmed condition: caught at ALERT and at ALARM by each method (with warning days), Controller
    Health caught, the four groups and the timing difference on shared catches."""

    def catch(eps: pd.DataFrame) -> pd.DataFrame:
        if eps.empty:
            return pd.DataFrame({"wo_id": conds["wo_id"], "caught": False, "warning_days": np.nan})
        return ev.catch_table(eps, conds, window_days)[["wo_id", "caught", "warning_days"]]

    out = conds[
        [
            "wo_id",
            "asset_id",
            "asset_key",
            "machine_id",
            "condition_type",
            "failure_code",
            "wo_type",
            "unplanned",
            "credit_ts",
            "confirmed_ts",
            "downtime_hours",
            "labor_hours",
            "machine_class",
            "pm_finding",
        ]
    ].copy()
    for name, lev in (("ml", ml), ("vendor", vendor)):
        a = (
            catch(alert_view(lev))
            .add_prefix(f"{name}_alert_")
            .rename(columns={f"{name}_alert_wo_id": "wo_id"})
        )
        b = (
            catch(alarm_view(lev, True))
            .add_prefix(f"{name}_alarm_")
            .rename(columns={f"{name}_alarm_wo_id": "wo_id"})
        )
        c = (
            catch(alarm_view(lev, False))
            .add_prefix(f"{name}_alarm_no14_")
            .rename(columns={f"{name}_alarm_no14_wo_id": "wo_id"})
        )
        out = out.merge(a, on="wo_id").merge(b, on="wo_id").merge(c, on="wo_id")
    chc = catch(ch.assign(point_id="") if len(ch) else ch).rename(
        columns={"caught": "ch_caught", "warning_days": "ch_warning_days"}
    )
    out = out.merge(chc, on="wo_id")
    m, v = out["ml_alert_caught"], out["vendor_alert_caught"]
    out["group"] = np.select(
        [m & v, m & ~v, ~m & v], ["both", "ML Model only", "Vendor Rules only"], "neither"
    )
    out["ml_days_vs_vendor"] = np.where(
        m & v, out["ml_alert_warning_days"] - out["vendor_alert_warning_days"], np.nan
    )
    return out


def value_table(
    ct: pd.DataFrame,
    wo: pd.DataFrame,
    margin: dict[str, float],
    min_warning_days: float,
) -> pd.DataFrame:
    """R3.6 value of conditions only the ML Model caught: an unplanned failure the ML Model warned of at
    least min_warning_days ahead is plannable; value is (unplanned less planned downtime) priced at the
    machine class's contribution margin per hour, plus the repair labor difference. Planned downtime and
    labor are the medians of the record's condition-based repairs of the same component class; HMC spindle
    failures keep the downtime that actually applied (spare available or out). Two views: the ML Model
    caught and both the Vendor Rules and Controller Health missed; and the Vendor Rules alone missed."""
    planned = wo[wo["wo_type"] == "condition_based"].assign(
        cls=lambda d: d["asset_id"].map(severity.asset_class)
    )
    med_dt = planned.groupby("cls")["downtime_hours"].median().to_dict()
    med_lab = planned.groupby("cls")["labor_hours"].median().to_dict()
    rows = []
    for view, mask in (
        (
            "ML Model only, Vendor Rules and Controller Health missed",
            ct["ml_alert_caught"] & ~ct["vendor_alert_caught"] & ~ct["ch_caught"],
        ),
        ("ML Model only, Vendor Rules missed", ct["ml_alert_caught"] & ~ct["vendor_alert_caught"]),
    ):
        for r in ct[mask].itertuples():
            cls = severity.asset_class(r.asset_id)
            plannable = bool(r.unplanned and r.ml_alert_warning_days >= min_warning_days)
            dt_p = med_dt.get(cls, np.nan)
            lab_p = med_lab.get(cls, np.nan)
            saved_h = max(r.downtime_hours - dt_p, 0.0) if plannable and np.isfinite(dt_p) else 0.0
            labor_h = max(r.labor_hours - lab_p, 0.0) if plannable and np.isfinite(lab_p) else 0.0
            rate = margin.get(r.machine_class, margin.get("hmc", 0.0))
            rows.append(
                {
                    "view": view,
                    "wo_id": r.wo_id,
                    "asset_id": r.asset_id,
                    "condition_type": r.condition_type,
                    "unplanned": bool(r.unplanned),
                    "ml_warning_days": r.ml_alert_warning_days,
                    "plannable": plannable,
                    "downtime_actual_h": r.downtime_hours,
                    "downtime_planned_h": dt_p,
                    "downtime_saved_h": saved_h,
                    "margin_per_h": rate,
                    "labor_actual_h": r.labor_hours,
                    "labor_planned_h": lab_p,
                    "labor_saved_h": labor_h,
                    "value_downtime_usd": saved_h * rate,
                }
            )
    return pd.DataFrame(rows)


def merged_false_rate(
    eps_list: list[pd.DataFrame],
    conds_all: pd.DataFrame,
    gap_days: float,
    period: tuple[pd.Timestamp, pd.Timestamp],
    n_assets: int,
) -> tuple[int, float]:
    """Per-asset false episodes from any of several methods, merged when they overlap or fall within the
    gap: the inspection workload of running them together."""
    sig = []
    for e in eps_list:
        if len(e):
            for r in e.itertuples():
                sig.append((r.asset_id, r.opened))
                sig.append((r.asset_id, r.last))
    if not sig:
        return 0, 0.0
    s = pd.DataFrame(sig, columns=["asset_id", "ts"]).assign(point_id="")
    merged = severity.episodes(s, gap_days)
    m = merged[(merged["opened"] >= period[0]) & (merged["opened"] < period[1])]
    n_false = int(false_mask(m, conds_all).sum()) if len(m) else 0
    return n_false, n_false / asset_months(period, n_assets)


def miss_reason(
    r: pd.Series,
    method: str,
    scores: pd.DataFrame,
    window_days: float,
    gap_hours: pd.DataFrame,
) -> str:
    """Why a method missed a condition: point re-centering (the ML Model scored nothing on the asset in the
    window because a level was forming or flagged), dropout (no measured hours in the last 7 days), no
    precursor in the inputs (scores or indicators inside the healthy range throughout the window),
    precursor under the threshold (exceedances that did not persist), or signature outside the inputs
    (controller-only conditions on an instrumented asset)."""
    t = r["credit_ts"]
    a = t - pd.Timedelta(days=window_days)
    s = scores[(scores["asset_id"] == r["asset_key"]) & (scores["hour_utc"] >= a) & (scores["hour_utc"] < t)]
    if r["condition_type"] in ("spindle cooling", "rotary union", "crash", "compressor trip"):
        return "signature outside the inputs"
    if s.empty:
        g = gap_hours[
            (gap_hours["asset_id"] == r["asset_key"])
            & (gap_hours["hour_utc"] >= t - 7 * DAY)
            & (gap_hours["hour_utc"] < t)
        ]
        return "dropout" if g.empty else "no precursor in the inputs"
    if method == "ml_model":
        if (s["state"] != "scored").all():
            return "point re-centering"
        over = (s["state"] == "scored") & (s["score"] > s["threshold"])
        return "precursor under the threshold" if over.any() else "no precursor in the inputs"
    return (
        "precursor under the threshold"
        if bool(s.get("vendor_over", pd.Series(False)).any())
        else "no precursor in the inputs"
    )
