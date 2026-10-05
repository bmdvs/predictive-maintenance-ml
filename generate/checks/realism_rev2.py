"""Realism checks 23 to 29 and the temperature reports added with the fleet expansion.

Spindle housing response to cooling faults, vendor temperature alerts, the shared spare spindle,
compressor duty, technician labor, the lubrication trend, and maintenance independence with the
crash share of HMC spindle failures. Inputs: the bronze export and the truth archive of one run.
Outputs: check results and report tables.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from generate.checks.realism import CheckResult
from pdm.config import Register

TZ = "America/Chicago"
HOUR = pd.Timedelta(hours=1)
_TEMP_CLASS = {
    "spindle_front": "spindle",
    "spindle_rear": "spindle",
    "tsc_pump": "tsc_pump",
    "flood_pump": "flood_pump",
    "compressor_motor": "compressor_motor",
    "compressor_airend": "compressor_airend",
}
_BANNED_IMPORTS = (
    "pdm.models",
    "pdm.fusion",
    "pdm.advisory",
    "pdm.monitoring",
    "pdm.evaluation",
    "pdm.tasks",
)


def _res(scalars: pd.DataFrame) -> pd.DataFrame:
    r = scalars[(scalars["axis"] == "res") & ~scalars["no_measurement"].astype(bool)].copy()
    r["hour_utc"] = pd.to_datetime(r["hour_utc"], utc=True)
    return r


def _cond_starts(events: pd.DataFrame, kind: str) -> pd.DataFrame:
    cs = events[(events["event"] == "condition_start") & (events["kind"] == kind)].copy()
    cl = events[(events["event"] == "condition_cleared") & (events["kind"] == kind)][["asset_id", "ts"]]
    ends = []
    for r in cs.itertuples():
        later = cl[(cl["asset_id"] == r.asset_id) & (cl["ts"] >= r.ts)]["ts"]
        ends.append(later.min() if len(later) else pd.NaT)
    cs["cleared"] = ends
    return cs


def check_cooling_housing(
    cfg: dict[str, Any],
    reg: Register,
    events: pd.DataFrame,
    scalars: pd.DataFrame,
    machine_hours: pd.DataFrame,
    ctrl_events: pd.DataFrame,
) -> CheckResult:
    """Check 23: housing rise during gradual cooling faults at 0.5 to 1.0 times the thermistor residual,
    speed-normalized and lagged; abrupt faults reach the overheat limit within hours while cutting."""
    lo, hi = cfg["checks"]["housing_share_of_residual_check"]
    within_h = cfg["checks"]["abrupt_overheat_within_h"]
    base_days = cfg["checks"]["thermal_baseline_days"]
    res = _res(scalars)
    res = res[res["point_id"].str.contains("-SP-")]
    mh = machine_hours.copy()
    mh["hour_utc"] = pd.to_datetime(mh["hour_utc"], utc=True)
    cs = _cond_starts(events, "spindle_cooling")
    rows = []
    for c in cs.itertuples():
        mid = c.asset_id.removesuffix("-SP")
        rated = float(reg.machines[mid]["rated_rpm"])
        abrupt = bool(getattr(c, "p_abrupt", False)) if pd.notna(getattr(c, "p_abrupt", np.nan)) else False
        end = c.cleared if pd.notna(c.cleared) else res["hour_utc"].max()
        if abrupt:
            stops = ctrl_events[
                (ctrl_events["machine_id"] == mid)
                & (ctrl_events["event_type"] == "spindle_overheat")
                & (ctrl_events["ts_utc"] >= c.ts)
            ]
            cut = mh[(mh["machine_id"] == mid) & (mh["hour_utc"] >= c.ts) & (mh["cut_min"] > 0)]
            first_cut = cut["hour_utc"].min() if len(cut) else pd.NaT
            first_stop = stops["ts_utc"].min() if len(stops) else pd.NaT
            hours = (
                (first_stop - first_cut) / HOUR if pd.notna(first_stop) and pd.notna(first_cut) else np.nan
            )
            rows.append(
                {
                    "asset_id": c.asset_id,
                    "start": c.ts,
                    "abrupt": True,
                    "measure": "hours from first cutting to overheat",
                    "value": hours,
                    "pass": bool(pd.notna(hours) and hours <= within_h),
                }
            )
            continue
        for pid in (f"{mid}-SP-F", f"{mid}-SP-R"):
            p = res[res["point_id"] == pid].copy()
            p["band"] = (p["speed_rpm_at_measurement"] / rated * 4).round()
            pre = p[(p["hour_utc"] >= c.ts - pd.Timedelta(days=base_days)) & (p["hour_utc"] < c.ts)]
            during = p[(p["hour_utc"] >= c.ts) & (p["hour_utc"] < end)].copy()
            if pre.empty or during.empty:
                continue
            base = pre.groupby("band")["temp_c"].median()
            during = during[during["band"].isin(base.index)]
            m = mh[mh["machine_id"] == mid].set_index("hour_utc")
            during["resid"] = m["cooler_residual_c"].reindex(during["hour_utc"]).to_numpy()
            during["cut"] = (m["cut_min"].reindex(during["hour_utc"]).to_numpy() / 60.0).clip(0, 1)
            during["speed"] = during["speed_rpm_at_measurement"] / rated
            plateau = during[during["resid"] >= 0.5 * during["resid"].max()]
            if len(plateau) < 12:
                continue
            excess = plateau["temp_c"].to_numpy() - base.reindex(plateau["band"]).to_numpy()
            ratio = float(
                np.median(excess / (plateau["resid"] * plateau["speed"] * plateau["cut"]).clip(lower=0.1))
            )
            rows.append(
                {
                    "asset_id": pid,
                    "start": c.ts,
                    "abrupt": False,
                    "measure": "housing excess over residual x speed x cutting share (median)",
                    "value": ratio,
                    "pass": bool(lo * 0.8 <= ratio <= hi * 1.2),
                }
            )
    tab = pd.DataFrame(rows)
    if tab.empty:
        return CheckResult(
            "23", "Spindle cooling and housing temperature", "episodes", "no measurable episodes", None, tab
        )
    return CheckResult(
        "23",
        "Spindle cooling and housing temperature",
        f"gradual: housing rise {lo}-{hi} x thermistor residual "
        "(speed-normalized, 20% tolerance for lag and noise); "
        f"abrupt: overheat stop within {within_h} h of cutting",
        "; ".join(
            f"{r.asset_id} {'abrupt' if r.abrupt else 'gradual'} {r.value:.2f}" for r in tab.itertuples()
        ),
        bool(tab["pass"].all()),
        tab,
    )


def _healthy_temp(
    cfg: dict[str, Any],
    reg: Register,
    scalars: pd.DataFrame,
    truth: pd.DataFrame,
    machine_hours: pd.DataFrame,
    air: pd.DataFrame | None,
) -> pd.DataFrame:
    res = _res(scalars)[["point_id", "hour_utc", "temp_c"]]
    t = truth.drop(columns=[c for c in ("temp_c",) if c in truth]).copy()
    t["hour_utc"] = pd.to_datetime(t["hour_utc"], utc=True)
    df = res.merge(t, on=["point_id", "hour_utc"], how="left")
    healthy = (
        (df["stage"].fillna(0) < 3)
        & (df["lube_severity"].fillna(0) == 0)
        & ~df["cavitation"].fillna(False).astype(bool)
        & (df["wear_frac"].fillna(0) == 0)
        & ~df["radio_noise"].notna()
    )
    df["machine_id"] = df["point_id"].str.extract(r"^([A-Z]+-\d+)")[0]
    mh = machine_hours.copy()
    mh["hour_utc"] = pd.to_datetime(mh["hour_utc"], utc=True)
    cool = mh.set_index(["machine_id", "hour_utc"])["cooler_residual_c"]
    key = pd.MultiIndex.from_frame(df[["machine_id", "hour_utc"]])
    df["cooler"] = cool.reindex(key).fillna(0).to_numpy()
    healthy &= df["cooler"] == 0
    if air is not None:
        a = air.copy()
        a["hour_utc"] = pd.to_datetime(a["hour_utc"], utc=True)
        foul = a.set_index("hour_utc")
        f = np.zeros(len(df))
        for cid in reg.compressors:
            col = f"{cid}_fouling_rise_c"
            if col in foul:
                sel = (df["machine_id"] == cid).to_numpy()
                f[sel] = foul[col].reindex(df.loc[sel, "hour_utc"]).fillna(0).to_numpy()
        healthy &= f == 0
    df["point_class"] = df["point_id"].map(reg.point_class)
    lim = cfg["rules"]["temperature_limits_c"]
    df["warning_c"] = df["point_class"].map(lambda c: lim[_TEMP_CLASS[c]][0])
    df["healthy"] = healthy
    return df


def check_temperature_alerts(
    cfg: dict[str, Any],
    reg: Register,
    scalars: pd.DataFrame,
    truth: pd.DataFrame,
    machine_hours: pd.DataFrame,
    air: pd.DataFrame | None,
    alerts: pd.DataFrame,
) -> tuple[CheckResult, pd.DataFrame]:
    """Check 24: temperature alerts only above the limits; no healthy point crosses a warning."""
    df = _healthy_temp(cfg, reg, scalars, truth, machine_hours, air)
    res = _res(scalars).set_index(["point_id", "hour_utc"])["temp_c"]
    ta = alerts[alerts["rules"].astype(str).str.contains("temp_")] if len(alerts) else alerts
    bad_alerts = 0
    for a in ta.itertuples():
        t = res.get((a.point_id, pd.Timestamp(a.alert_ts)), np.nan)
        lim = cfg["rules"]["temperature_limits_c"][_TEMP_CLASS[reg.point_class(a.point_id)]][0]
        bad_alerts += int(not (t >= lim))
    h = df[df["healthy"]]
    crossings = int((h["temp_c"] >= h["warning_c"]).sum())
    h = h.assign(local=h["hour_utc"].dt.tz_convert(TZ))
    summary = (
        h.groupby("point_class")
        .agg(max_healthy_c=("temp_c", "max"), warning_c=("warning_c", "first"), hours=("temp_c", "size"))
        .reset_index()
    )
    summary["margin_c"] = summary["warning_c"] - summary["max_healthy_c"]
    july = h[(h["local"].dt.month == 7) & h["point_class"].str.startswith("compressor")]
    if len(july):
        top = july.loc[july["temp_c"].idxmax()]
        when = f"{top['point_id']} {top['local']:%Y-%m-%d %H:00}"
        hottest = f"hottest July compressor reading {top['temp_c']:.1f} C ({when})"
    else:
        hottest = "no July compressor hours"
    ok = bad_alerts == 0 and crossings == 0
    return (
        CheckResult(
            "24",
            "Vendor temperature alerts",
            "temperature alerts only at or above the limits; no healthy point-hour at or above a warning",
            f"temperature alerts {len(ta)}, below limit {bad_alerts}; "
            f"healthy hours at or above warning {crossings}; {hottest}",
            bool(ok),
            summary,
        ),
        summary,
    )


def check_spare_spindle(cfg: dict[str, Any], truth_dir: Path, wo_engine: pd.DataFrame) -> CheckResult:
    """Check 25: no two unplanned swaps overlap while the spare is out; longer repairs only when it is out."""
    tl = (
        pd.read_parquet(truth_dir / "spare_spindle.parquet")
        if (truth_dir / "spare_spindle.parquet").exists()
        else pd.DataFrame()
    )
    passes = json.loads((truth_dir / "spare_spindle_passes.json").read_text(encoding="utf-8"))
    cols = ["type", "spindle_repair_case", "opened", "asset_id", "downtime_h"]
    w = wo_engine.reindex(columns=sorted(set(wo_engine.columns) | set(cols)))
    u = w[(w["type"] == "unplanned_repair") & w["spindle_repair_case"].notna()]
    u = u.sort_values("opened")
    ss = cfg["maintenance"]["spare_spindle"]
    out_until = None
    bad = []
    for r in u.itertuples():
        out = out_until is not None and r.opened < out_until
        if r.spindle_repair_case == "swap" and out:
            bad.append(f"{r.asset_id} swap at {r.opened:%Y-%m-%d} while spare out")
        if r.spindle_repair_case == "no_spare" and not out:
            bad.append(f"{r.asset_id} no_spare at {r.opened:%Y-%m-%d} while spare available")
        if r.spindle_repair_case == "swap":
            ret = tl[(tl["state"] == "available") & (tl["ts"] > r.opened)]["ts"]
            out_until = ret.min() if len(ret) else None
    days = u.assign(days=pd.to_numeric(u["downtime_h"]) / 24)
    by_case = days.groupby("spindle_repair_case")["days"].agg(["count", "min", "max"]).round(1)
    measured = (
        f"unplanned spindle failures {len(u)}; "
        + "; ".join(f"{k} {int(v['count'])} ({v['min']}-{v['max']} d)" for k, v in by_case.iterrows())
        + f"; passes {passes['passes']}, rerun {passes['reruns']}"
    )
    return CheckResult(
        "25",
        "Spare spindle",
        f"swaps never overlap while the spare is out ({ss['rebuild_out_working_days']} working days); "
        "no_spare only while out",
        measured + (f"; violations: {bad}" if bad else ""),
        bool(not bad),
        by_case.reset_index(),
    )


def check_compressor_duty(
    cfg: dict[str, Any], reg: Register, air: pd.DataFrame | None
) -> tuple[CheckResult, pd.DataFrame]:
    """Check 26: up to three units at first-shift peak, one always available as standby."""
    if air is None:
        return CheckResult("26", "Compressor duty", "", "no compressor hours", None, None), pd.DataFrame()
    cids = list(reg.compressors)
    run = np.column_stack([air[f"{c}_running"].astype(bool).to_numpy() for c in cids])
    n = run.sum(axis=1)
    local = pd.to_datetime(air["hour_utc"], utc=True).dt.tz_convert(TZ)
    p0, p1 = cfg["compressor"]["first_shift_peak_hours_local"]
    peak = (local.dt.weekday < 5) & (local.dt.hour >= p0) & (local.dt.hour < p1) & (n > 0)
    split = pd.DataFrame({"compressor_id": cids, "running_h": run.sum(axis=0)})
    years = len(air) / (365.25 * 24)
    split["running_h_per_year"] = (split["running_h"] / years).round(0)
    split["share"] = (split["running_h"] / split["running_h"].sum()).round(3)
    max_units = int(n.max()) if len(n) else 0
    three_at_peak = float((n[peak.to_numpy()] == 3).mean()) if peak.any() else 0.0
    standby_ok = bool(max_units <= len(cids) - 1)
    ok = standby_ok and max_units <= cfg["compressor"]["max_running_units"] and three_at_peak > 0
    return (
        CheckResult(
            "26",
            "Compressor duty",
            "at most three units running, one always standby; three units at first-shift peak",
            f"max running {max_units}; peak hours with three units {three_at_peak:.0%}; running h per year "
            + ", ".join(f"{r.compressor_id} {int(r.running_h_per_year)}" for r in split.itertuples()),
            bool(ok),
            split,
        ),
        split,
    )


def check_technician_labor(cfg: dict[str, Any], wo: pd.DataFrame) -> tuple[CheckResult, pd.DataFrame]:
    """Check 27: maintenance labor under 80% of the technicians' available hours every month."""
    n_tech = len(cfg["cmms"]["technicians"])
    avail = n_tech * 2080.0 / 12.0
    w = wo.copy()
    w["month"] = (
        pd.to_datetime(w["started"].fillna(w["opened"]), utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m")
    )
    m = w.groupby("month")["labor_hours"].sum().reset_index()
    m["share"] = (m["labor_hours"] / avail).round(3)
    lim = cfg["checks"]["technician_labor_share_max"]
    peak = m.loc[m["share"].idxmax()] if len(m) else None
    return (
        CheckResult(
            "27",
            "Technician labor",
            f"monthly labor below {lim:.0%} of {n_tech} technicians' hours ({avail:.0f} h)",
            f"peak {peak['month']} {peak['labor_hours']:.0f} h ({peak['share']:.0%}); "
            f"mean {m['share'].mean():.0%}"
            if peak is not None
            else "no work orders",
            bool(peak is not None and peak["share"] < lim),
            m,
        ),
        m,
    )


def check_lube_trend(
    cfg: dict[str, Any], wo: pd.DataFrame, events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> CheckResult:
    """Check 28: confirmed lubrication findings on HMC spindles show no trend across record years."""
    alpha = cfg["checks"]["lube_trend_alpha"]
    notes = wo["notes"].fillna("")
    found = wo[
        wo["asset_id"].str.match(r"HMC-\d+-SP$")
        & (
            ((wo["wo_type"] == "inspection") & notes.str.contains("lubrication"))
            | ((wo["wo_type"] == "pm") & notes.str.contains("lubrication degraded"))
        )
    ]
    t = pd.to_datetime(found["opened"], utc=True)
    yrs = int(round((end - start) / pd.Timedelta(days=365.25)))
    if yrs < 2:
        return CheckResult(
            "28",
            "Lubrication trend",
            "record years",
            f"confirmed {len(found)} (window under two years)",
            None,
            None,
        )
    k = ((t - start) / pd.Timedelta(days=365.25)).astype(int).clip(upper=yrs - 1)
    conf = np.bincount(k, minlength=yrs)
    on = events[
        (events["event"] == "lube_episode_start") & events["asset_id"].astype(str).str.endswith("-SP")
    ]
    ko = (
        ((pd.to_datetime(on["ts"], utc=True) - start) / pd.Timedelta(days=365.25))
        .astype(int)
        .clip(upper=yrs - 1)
    )
    onsets = np.bincount(ko, minlength=yrs)
    p = float(chisquare(conf).pvalue) if conf.sum() > 0 else 1.0
    return CheckResult(
        "28",
        "Lubrication trend",
        f"confirmed findings by record year consistent with a flat rate (chi-square p > {alpha})",
        f"confirmed {conf.tolist()} (p = {p:.2f}); onsets {onsets.tolist()}",
        bool(p > alpha),
        None,
    )


def check_independence_and_crash(cfg: dict[str, Any], events: pd.DataFrame) -> CheckResult:
    """Check 29: the generator imports nothing derived from the vibration model, and the crash share
    of HMC spindle failures against the published 30 to 60 percent (reported)."""
    root = Path(__file__).resolve().parents[1]
    hits = []
    for f in root.rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            hits += [f"{f.name}:{m}" for m in mods if m.startswith(_BANNED_IMPORTS)]
        if f.resolve() == Path(__file__).resolve():
            continue
        text = f.read_text(encoding="utf-8")
        hits += [f"{f.name}:gold" for tok in ("local_gold", "gold_snapshot") if tok in text]
    f = events[
        (events["event"] == "unplanned_failure")
        & events["asset_id"].astype(str).str.match(r"HMC-\d+-SP$")
        & events["mode"].isin(["bearing", "crash"])
    ]
    n_crash, n_all = int((f["mode"] == "crash").sum()), len(f)
    share = n_crash / n_all if n_all else np.nan
    lo, hi = cfg["checks"]["crash_share_hmc_spindle_published"]
    return CheckResult(
        "29",
        "Maintenance independence; crash share",
        "no generator import of model, fusion, advisory, monitoring or evaluation code; "
        f"crash share reported against {lo:.0%}-{hi:.0%}",
        f"imports {hits or 'none'}; crash share {n_crash}/{n_all}"
        + (
            f" = {share:.0%} ({'inside' if lo <= share <= hi else 'outside'} the published range)"
            if n_all
            else ""
        ),
        bool(not hits),
        None,
    )


def thermistor_report(cfg: dict[str, Any], bronze_root: Path, machine_hours: pd.DataFrame) -> pd.DataFrame:
    """Healthy spindle thermistor peak in summer (July and August) per HMC, against 44 C and 46 C."""
    root = bronze_root / "controller" / "minute_agg"
    mh = machine_hours.copy()
    mh["hour_utc"] = pd.to_datetime(mh["hour_utc"], utc=True)
    rows = []
    for f in sorted(root.glob("month=*/HMC-*.parquet")):
        month = int(f.parent.name.split("-")[1])
        if month not in (7, 8):
            continue
        mid = f.stem
        a = pd.read_parquet(f, columns=["minute_utc", "spindle_temp_c_mean"])
        a["hour_utc"] = pd.to_datetime(a["minute_utc"], utc=True).dt.floor("h")
        m = mh[mh["machine_id"] == mid].set_index("hour_utc")
        sick = (m["cooler_residual_c"] > 0) | (m["lube_temp_c"] > 0) | (m["spindle_s4"] > 0)
        a = a[~sick.reindex(a["hour_utc"]).fillna(False).astype(bool).to_numpy()]
        if a["spindle_temp_c_mean"].notna().sum() == 0:
            continue
        rows.append(
            {
                "machine_id": mid,
                "month": f.parent.name,
                "max_c": float(a["spindle_temp_c_mean"].max()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.groupby("machine_id")["max_c"].max().reset_index()
    out["critical_c"] = cfg["models"]["severity"]["thermistor_critical_c"]
    out["limit_c"] = cfg["spindle_cooling"]["overheat_limit_c"]
    out["margin_to_critical_c"] = (out["critical_c"] - out["max_c"]).round(2)
    return out


def overheat_report(events: pd.DataFrame, machine_hours: pd.DataFrame) -> dict[str, int]:
    """Overheat alarms by cause: stage-4 bearing heat, lubrication, cooler."""
    oh = events[events["event"] == "overheat_alarm"].copy()
    mh = machine_hours.copy()
    mh["hour_utc"] = pd.to_datetime(mh["hour_utc"], utc=True)
    m = mh.set_index(["machine_id", "hour_utc"])
    out = {"stage4": 0, "lubrication": 0, "both": 0, "other": 0}
    for r in oh.itertuples():
        key = (str(r.asset_id).removesuffix("-SP"), pd.Timestamp(r.ts).floor("h"))
        row = m.loc[key] if key in m.index else None
        s4 = row is not None and float(row["spindle_s4"]) > 0
        lube = row is not None and float(row["lube_temp_c"]) > 0
        out["both" if s4 and lube else "stage4" if s4 else "lubrication" if lube else "other"] += 1
    stops = events[(events["event"] == "unplanned_failure") & (events["mode"] == "spindle_cooler")]
    out["cooler_stops"] = len(stops)
    return out


_EVENT_COLS = ["ts", "event", "asset_id", "mode", "kind", "cause", "p_abrupt", "p_rise_c", "ramp_h"]


def run_rev2_checks(
    cfg: dict[str, Any],
    reg: Register,
    bronze_root: Path,
    truth_dir: Path,
    scalars: pd.DataFrame,
    truth: pd.DataFrame,
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[CheckResult], dict[str, Any]]:
    events = events.reindex(columns=sorted(set(events.columns) | set(_EVENT_COLS)))
    machine_hours = pd.read_parquet(truth_dir / "machine_hours.parquet")
    air_path = truth_dir / "parts" / "compressor_hours.parquet"
    air = pd.read_parquet(air_path) if air_path.exists() else None
    ce_dir = bronze_root / "controller" / "events"
    ctrl = (
        pd.concat([pd.read_parquet(f) for f in ce_dir.glob("*.parquet")], ignore_index=True)
        if ce_dir.exists()
        else pd.DataFrame(columns=["machine_id", "event_type", "ts_utc"])
    )
    if len(ctrl):
        ctrl["ts_utc"] = pd.to_datetime(ctrl["ts_utc"], utc=True)
    alerts_path = bronze_root / "cm_platform" / "alerts" / "alerts.parquet"
    alerts = (
        pd.read_parquet(alerts_path)
        if alerts_path.exists()
        else pd.DataFrame(columns=["point_id", "alert_ts", "rules"])
    )
    wo = pd.read_parquet(bronze_root / "cmms" / "work_orders.parquet")
    wo_engine = pd.read_parquet(truth_dir / "work_orders.parquet")
    c24, temps = check_temperature_alerts(cfg, reg, scalars, truth, machine_hours, air, alerts)
    c26, split = check_compressor_duty(cfg, reg, air)
    c27, labor = check_technician_labor(cfg, wo)
    results = [
        check_cooling_housing(cfg, reg, events, scalars, machine_hours, ctrl),
        c24,
        check_spare_spindle(cfg, truth_dir, wo_engine),
        c26,
        c27,
        check_lube_trend(cfg, wo, events, start, end),
        check_independence_and_crash(cfg, events),
    ]
    return results, {
        "temperature_healthy": temps,
        "compressor_split": split,
        "technician_labor": labor,
        "thermistor_summer": thermistor_report(cfg, bronze_root, machine_hours),
        "overheat_alarms": overheat_report(events, machine_hours),
    }
