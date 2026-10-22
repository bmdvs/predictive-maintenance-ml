"""Component lifetimes for the fleet survival model: Weibull AFT and Cox with left truncation.

Inputs: register install and replacement dates (pumps, spindles, compressor bearings), CMMS work
orders, controller collision events and the PM schedule. Output: one row per component life with its
start, entry into the record, end, event flag (unplanned failure) or censoring (preventive replacement
or record end), and covariates (component class, machine age class, crash count, lube PM compliance).
Times are calendar days.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BEARING_CODES = {
    "BRG-WEAR",
    "BRG-SPALL",
    "BRG-LUBE",
    "PMP-BRG",
    "CMP-BRG-MOTOR",
    "CMP-BRG-AIREND",
    "PMP-SEAL",
    "PMP-IMP",
}
DAY = pd.Timedelta(days=1)


def _dates(s: Any) -> list[pd.Timestamp]:
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return []
    return [pd.Timestamp(x, tz="UTC") for x in str(s).split(";") if x]


def lifetimes(
    machines: pd.DataFrame,
    pumps: pd.DataFrame,
    spindles: pd.DataFrame,
    compressors: pd.DataFrame,
    work_orders: pd.DataFrame,
    collisions: pd.DataFrame,
    pm: pd.DataFrame,
    record: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    r0, r1 = record
    age = machines.set_index("machine_id")["age_class"].to_dict()
    comps = []
    for p in pumps.itertuples():
        comps.append(
            (
                p.asset_id,
                "tsc_pump" if p.kind == "tsc" else "flood_pump",
                p.machine_id,
                _dates(p.replacement_dates),
            )
        )
    for s in spindles.itertuples():
        comps.append((f"{s.machine_id}-SP", "spindle", s.machine_id, _dates(s.rebuild_dates)))
    for c in compressors.itertuples():
        starts = sorted(set(_dates(c.motor_bearing_replaced) + _dates(c.airend_rebuilt)))
        comps.append((c.asset_id, "compressor", c.asset_id, starts))
    wo = work_orders.copy()
    fail = wo[(wo["wo_type"] == "unplanned_repair") & wo["failure_code"].isin(BEARING_CODES)]
    prevent = wo[wo["wo_type"].isin(["pm", "condition_based"]) & (wo["downtime_hours"] > 0)]
    lube = pm[pm["pm_task"] == "spindle_lube_check"]
    rows = []
    for asset, cls, machine, starts in comps:
        starts = sorted(starts) or [r0 - 365 * DAY]
        ev_f = fail[fail["asset_id"] == asset].sort_values("opened")
        ev_p = prevent[prevent["asset_id"] == asset].sort_values("opened")
        bounds = sorted(set([s for s in starts]) | set(ev_f["closed"]) | set(ev_p["closed"]))
        bounds = [b for b in bounds if b < r1]
        for k, start in enumerate(bounds):
            end_limit = bounds[k + 1] if k + 1 < len(bounds) else r1
            f = ev_f[(ev_f["opened"] > start) & (ev_f["opened"] <= end_limit)]
            pv = ev_p[(ev_p["opened"] > start) & (ev_p["opened"] <= end_limit)]
            if len(f):
                end, event = f["opened"].iat[0], 1
            elif len(pv):
                end, event = pv["opened"].iat[0], 0
            else:
                end, event = end_limit, 0
            if end <= r0:
                continue
            entry = max(start, r0)
            mid = machine
            col = collisions[
                (collisions["machine_id"] == mid)
                & (collisions["ts_utc"] > entry)
                & (collisions["ts_utc"] <= end)
            ]
            lp = lube[(lube["asset_id"] == f"{mid}-SP") & (lube["due"] > entry) & (lube["due"] <= end)]
            rows.append(
                {
                    "asset_id": asset,
                    "component_class": cls,
                    "machine_age_class": age.get(mid, "older"),
                    "start": start,
                    "entry_days": (entry - start) / DAY,
                    "duration_days": (end - start) / DAY,
                    "event": event,
                    "crash_count": len(col),
                    "lube_pm_compliance": float((lp["outcome"] == "on_time").mean()) if len(lp) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fit_models(lives: pd.DataFrame) -> dict[str, Any]:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.utils import concordance_index

    d = lives.copy()
    d["lube_pm_compliance"] = d["lube_pm_compliance"].fillna(d["lube_pm_compliance"].median())
    x = pd.get_dummies(d[["component_class", "machine_age_class"]], drop_first=True, dtype=float)
    data = pd.concat(
        [d[["duration_days", "event", "entry_days", "crash_count", "lube_pm_compliance"]], x], axis=1
    )
    data = data[data["duration_days"] > data["entry_days"]]
    out: dict[str, Any] = {"n": len(data), "events": int(data["event"].sum())}
    aft = WeibullAFTFitter(penalizer=0.1).fit(data, "duration_days", "event", entry_col="entry_days")
    cox = CoxPHFitter(penalizer=0.1).fit(data, "duration_days", "event", entry_col="entry_days")
    out["aft"], out["cox"] = aft, cox
    out["concordance_aft"] = float(
        concordance_index(data["duration_days"], aft.predict_median(data), data["event"])
    )
    out["concordance_cox"] = float(cox.concordance_index_)
    med = aft.predict_median(data)
    out["median_life_days_by_class"] = (
        pd.DataFrame({"component_class": d.loc[data.index, "component_class"], "median_days": med.to_numpy()})
        .groupby("component_class")["median_days"]
        .median()
        .to_dict()
    )
    ibs = integrated_brier_ipcw(aft, data)
    out["ibs_ipcw"], out["ibs_ipcw_km"], out["ibs_times"] = ibs["model"], ibs["kaplan_meier"], ibs["times"]
    return out


def integrated_brier_ipcw(model: Any, data: pd.DataFrame, n_times: int = 20) -> dict[str, Any]:
    """Integrated Brier score with inverse-probability-of-censoring weights (scikit-survival), on the
    fitting data, over the 10th to 90th percentile of observed times; the Kaplan-Meier curve with no
    covariates is the reference score."""
    from sksurv.metrics import integrated_brier_score
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.util import Surv

    y = Surv.from_arrays(data["event"].astype(bool).to_numpy(), data["duration_days"].to_numpy(float))
    obs = data["duration_days"]
    hi = min(obs.quantile(0.9), obs[data["event"] == 0].max() - 1e-6, obs.max() - 1e-6)
    times = np.linspace(obs.quantile(0.1), hi, n_times)
    sf = model.predict_survival_function(data, times=times).T.to_numpy()
    km_t, km_s = kaplan_meier_estimator(data["event"].astype(bool).to_numpy(), obs.to_numpy(float))
    km = np.interp(times, km_t, km_s)
    return {
        "model": float(integrated_brier_score(y, y, sf, times)),
        "kaplan_meier": float(integrated_brier_score(y, y, np.tile(km, (len(data), 1)), times)),
        "times": (float(times[0]), float(times[-1])),
    }


def _log_ci(summary: pd.DataFrame, name: str, z: float) -> tuple[float, float]:
    v, se = float(summary.at[name, "coef"]), float(summary.at[name, "se(coef)"])
    return v * float(np.exp(-z * se / v)), v * float(np.exp(z * se / v))


def per_class_weibull(lives: pd.DataFrame, interval: float) -> pd.DataFrame:
    """Weibull lifetime per component class with no other covariates, with left truncation at entry
    into the record and parameter intervals on the log scale; a class with no event gets no fit."""
    from lifelines import WeibullFitter
    from scipy.stats import norm

    rows = []
    for cls, g in lives.groupby("component_class"):
        d = g[g["duration_days"] > g["entry_days"]]
        row = {"component_class": cls, "lives": len(d), "events": int(d["event"].sum())}
        if row["events"] == 0:
            rows.append({**row, "note": "no events in the record"})
            continue
        f = WeibullFitter(alpha=1 - interval).fit(d["duration_days"], d["event"], entry=d["entry_days"])
        sm = f.summary
        z = float(norm.ppf(0.5 + interval / 2))
        (l_lo, l_hi), (r_lo, r_hi) = _log_ci(sm, "lambda_", z), _log_ci(sm, "rho_", z)
        rows.append(
            {
                **row,
                "shape_rho": float(f.rho_),
                "scale_lambda_days": float(f.lambda_),
                "median_life_days": float(f.median_survival_time_),
                "lambda_low": l_lo,
                "lambda_high": l_hi,
                "rho_low": r_lo,
                "rho_high": r_hi,
                "note": "",
            }
        )
    return pd.DataFrame(rows)
