"""CMMS export: work orders with label noise, PM schedule, drawbar force log and oil samples.

Inputs: work orders, PM rows and events from the asset-group engines, axis-wear events, the
truth stage timeline for compressor bearings and the register. Outputs: bronze cmms tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate.provenance import RunContext, rng_for
from generate.shop_schedule import LOCAL_TZ
from pdm.constants import DAYS_PER_YEAR, HOURS_PER_DAY, ISO_4406_LOWER_FRAC_OF_UPPER, iso_4406_upper_per_ml

_MODE_COMPONENT = {
    "atc_fault": "ATC",
    "pallet_changer_hydraulic": "HYDRAULIC",
    "servo_drive": "SERVO",
    "electrical": "ELECTRICAL",
    "lube_system": "LUBE",
    "way_cover": "WAY_COVER",
    "chip_conveyor": "CHIP_CONVEYOR",
    "coolant_pump_non_hmc": "COOLANT_PUMP",
    "ball_screw_or_guide": "AXIS",
    "trip": "COMPRESSOR",
    "high_temp_trip": "COMPRESSOR",
}
_CODE_NOTES = {
    "imbalance": "Imbalance found; impeller cleaned and rotor balanced.",
    "misalignment": "Misalignment found; laser alignment.",
    "looseness": "Looseness found; base and mounting bolts retightened.",
    "rotary_union_leak": "Through-spindle coolant pressure low; rotary union leaking; rotary union replaced.",
    "coolant_ingress": "Coolant in front bearing area; rotary union leaking; rotary union replaced.",
    "oil_cooler_fouled": "High discharge temperature; oil cooler fouled; cooler cleaned.",
    "spindle_cooler_fault": "Spindle overheat; spindle chiller repaired.",
}
_CODE_PARTS = {
    "rotary_union_leak": "ROTARY-UNION",
    "coolant_ingress": "ROTARY-UNION",
    "spindle_cooler_fault": "SPINDLE-CHILLER-PARTS",
}
_CONDITION_CODES = set(_CODE_NOTES)
_NOTES = {
    "SPINDLE": "Spindle noisy and running hot; spindle removed and exchanged.",
    "TSC_PUMP": "Through-spindle coolant pressure low; pump unit replaced.",
    "COOLANT_PUMP": "Coolant pump leaking and noisy; pump replaced.",
    "COMPRESSOR": "Compressor tripped; repaired and returned to service.",
    "ATC": "Tool changer fault; arm adjusted.",
    "HYDRAULIC": "Pallet changer hydraulic pressure low; repaired leak.",
    "SERVO": "Servo drive alarm; drive repaired.",
    "ELECTRICAL": "Electrical fault in cabinet; repaired.",
    "LUBE": "Way lube alarm; lube unit repaired.",
    "WAY_COVER": "Way cover damaged; replaced.",
    "CHIP_CONVEYOR": "Chip conveyor jammed; cleared and repaired.",
    "AXIS": "Axis accuracy out on parts; ball screw replaced.",
}
_ENGINE_PM_TASKS = {"tsc_pump_unit_replacement", "spindle_rebuild_older"}
_FREE_TEXT = [
    "machine down, fixed",
    "repaired, back running",
    "down most of shift, ok now",
    "see operator note",
]


def component_of(asset_id: str, mode: str | None) -> str:
    if mode in _MODE_COMPONENT:
        return _MODE_COMPONENT[mode]
    if asset_id.endswith("-SP"):
        return "SPINDLE"
    if asset_id.endswith("-TSC"):
        return "TSC_PUMP"
    if asset_id.endswith("-CP"):
        return "COOLANT_PUMP"
    if asset_id.startswith("CMP"):
        return "COMPRESSOR"
    return "MACHINE"


def true_code(cfg: dict[str, Any], comp: str, wo: dict[str, Any], rng: np.random.Generator) -> str | None:
    codes = cfg["cmms"]["failure_codes"].get(comp)
    if not codes:
        return None
    if isinstance(wo.get("code_override"), str):
        return str(wo["code_override"])
    mode = wo.get("failure_mode")
    comps = str(wo.get("components") or "")
    findings = str(wo.get("findings") or "")
    if mode == "spindle_cooler":
        return "spindle_cooler_fault"
    if mode == "high_temp_trip":
        return "oil_cooler_fouled"
    if comp == "SPINDLE":
        if mode == "crash":
            return "CRASH-DMG"
        return (
            "BRG-LUBE"
            if wo.get("onset_cause") == "lubrication"
            else ("BRG-SPALL" if mode == "bearing" else "BRG-WEAR")
        )
    if comp in ("TSC_PUMP", "COOLANT_PUMP"):
        if mode == "seal_failure":
            return "PMP-SEAL"
        if "wear" in findings:
            return str(rng.choice(["PMP-SEAL", "PMP-IMP"]))
        return "PMP-BRG"
    if comp == "COMPRESSOR":
        if mode == "trip":
            return str(rng.choice(["CMP-TRIP", "CMP-VALVE"]))
        return "CMP-BRG-AIREND" if "-AE" in comps + findings else "CMP-BRG-MOTOR"
    return str(rng.choice(codes))


def build_work_orders(
    cfg: dict[str, Any],
    reg: Any,
    ctx: RunContext,
    work_orders: list[dict[str, Any]],
    pm_rows: list[dict[str, Any]],
    events: pd.DataFrame,
    axis_events: list[dict[str, Any]],
    sensor_events: pd.DataFrame,
) -> pd.DataFrame:
    cm, ln = cfg["cmms"], cfg["label_noise"]
    rng = rng_for(ctx.run_seed, "cmms")
    onsets: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    if not events.empty:
        for _, e in events[events["event"] == "bearing_onset"].iterrows():
            cid = str(e["component_id"])
            asset = cid[:-2] if cid.endswith(("-F", "-R")) and "-SP-" in cid else cid
            onsets.setdefault(asset, []).append((e["ts"], str(e.get("cause"))))
    rows = []
    all_components = list(cm["failure_codes"])
    for wo in work_orders:
        if str(wo.get("findings")) == "superseded":
            continue
        asset = str(wo["asset_id"])
        mode = wo.get("failure_mode")
        override = wo.get("component_override")
        comp = override if isinstance(override, str) else component_of(asset, mode)
        wtype = wo["type"]
        code = None
        notes = ""
        if wtype in ("unplanned_repair", "condition_based", "corrective"):
            prior = [c for t, c in onsets.get(asset, []) if t <= wo["opened"]]
            code = true_code(cfg, comp, {**wo, "onset_cause": prior[-1] if prior else None}, rng)
            notes = _CODE_NOTES.get(code or "", _NOTES.get(comp, "Repaired."))
            if wo.get("post_repair_finding") == "misalignment":
                notes += (
                    " Post-repair check: misalignment found; laser alignment, corrected "
                    f"{pd.Timestamp(wo['post_repair_corrected']).tz_convert(LOCAL_TZ):%Y-%m-%d}."
                )
        elif wtype == "inspection":
            notes = "Inspected after platform alert: " + (
                "no fault found." if wo.get("no_fault_found") else f"found {wo.get('findings')}."
            )
        elif wtype == "pm":
            notes = f"PM: {wo.get('task')}."
        down_h = float(wo.get("downtime_h") or 0.0)
        started = wo.get("started", wo["opened"])
        closed = wo.get("closed", started)
        lag = pd.Timedelta(0)
        if wtype in ("unplanned_repair", "condition_based", "corrective"):
            lag = pd.Timedelta(days=float(rng.uniform(*ln["close_lag_days"])))
            labor = float(rng.uniform(*cm["labor_frac_of_repair_time"])) * min(
                down_h, cm["labor_h_max_per_day"] * max(np.ceil(down_h / HOURS_PER_DAY), 1.0)
            )
        elif wtype == "inspection":
            labor = float(rng.uniform(*cm["inspection_labor_h"]))
        else:
            labor = float(rng.uniform(*cm["pm_labor_h"]))
        rows.append(
            {
                "wo_id": wo["wo_id"],
                "asset_id": asset,
                "component": comp if comp != "MACHINE" else None,
                "wo_type": wtype,
                "failure_code": code,
                "opened": wo["opened"],
                "started": started,
                "closed": closed + lag,
                "downtime_hours": round(down_h, 2),
                "technician": str(rng.choice(cm["technicians"])),
                "parts": _CODE_PARTS.get(code or "")
                if code in _CONDITION_CODES
                else _parts(reg, asset, comp, wtype),
                "labor_hours": round(labor, 2),
                "notes": notes,
                "no_fault_found": bool(wo.get("no_fault_found", False)) if wtype == "inspection" else False,
                "platform_alert_id": wo.get("alert_id"),
                "true_component": comp,
                "true_failure_code": code,
            }
        )
    for i, p in enumerate(pm_rows):
        if p["outcome"] == "skipped" or p["task"] in _ENGINE_PM_TASKS:
            continue
        note = f"PM: {p['task']}."
        if p["task"] == "backlash_check":
            note += _backlash_note(p, axis_events)
        finding = p.get("finding")
        finding = finding if isinstance(finding, str) else None
        pm_code, pm_comp = None, None
        if finding == "oil_cooler_fouled":
            pm_code, pm_comp = finding, "COMPRESSOR"
            note += " Cooler fouled; cleaned."
        elif finding == "rotary_union_leak":
            note += " TSC pressure low; rotary union leaking; replacement planned."
        rows.append(
            {
                "wo_id": f"PM-{i:06d}",
                "asset_id": p["asset_id"],
                "component": pm_comp,
                "wo_type": "pm",
                "failure_code": pm_code,
                "opened": p["due"],
                "started": p["completed"],
                "closed": p["completed"],
                "downtime_hours": 0.0,
                "technician": str(rng.choice(cm["technicians"])),
                "parts": None,
                "labor_hours": round(float(rng.uniform(*cm["pm_labor_h"])), 2),
                "notes": note,
                "no_fault_found": False,
                "platform_alert_id": None,
                "true_component": pm_comp,
                "true_failure_code": pm_code,
            }
        )
    for _, se in sensor_events[sensor_events["event"].isin(["install", "remount"])].iterrows():
        rows.append(
            {
                "wo_id": f"PRJ-{se['point_id']}-{se['ts']:%Y%m%d}",
                "asset_id": reg.sensors[se["point_id"]].asset_id,
                "component": None,
                "wo_type": "project",
                "failure_code": None,
                "opened": se["ts"],
                "started": se["ts"],
                "closed": se["ts"],
                "downtime_hours": 0.0,
                "technician": str(rng.choice(cm["technicians"])),
                "parts": "SENSOR",
                "labor_hours": round(float(rng.uniform(*cm["pm_labor_h"])), 2),
                "notes": f"Wireless vibration sensor {se['event']} on {se['point_id']}.",
                "no_fault_found": False,
                "platform_alert_id": None,
                "true_component": None,
                "true_failure_code": None,
            }
        )
    df = pd.DataFrame(rows).sort_values("opened").reset_index(drop=True)
    unplanned = df.index[df["wo_type"] == "unplanned_repair"].to_numpy()
    draw = rng.random(unplanned.size)
    wrong = unplanned[draw < ln["wrong_code_frac"]]
    missing = unplanned[
        (draw >= ln["wrong_code_frac"]) & (draw < ln["wrong_code_frac"] + ln["missing_code_frac"])
    ]
    for k in wrong:
        others = [c for c in all_components if c != df.at[k, "true_component"]]
        comp = str(rng.choice(others))
        df.at[k, "component"] = comp
        df.at[k, "failure_code"] = str(rng.choice(cm["failure_codes"][comp]))
    for k in missing:
        df.at[k, "failure_code"] = None
        df.at[k, "notes"] = str(rng.choice(_FREE_TEXT))
    df["label_noise"] = None
    df.loc[wrong, "label_noise"] = "wrong_code"
    df.loc[missing, "label_noise"] = "missing_code"
    return df.assign(**ctx.stamp())


def _parts(reg: Any, asset: str, comp: str, wtype: str) -> str | None:
    if wtype not in ("unplanned_repair", "condition_based", "pm"):
        return None
    if comp == "SPINDLE":
        return "SPINDLE-CARTRIDGE"
    if comp in ("TSC_PUMP", "COOLANT_PUMP"):
        p = reg.pumps.get(asset)
        return f"PUMP-UNIT;{p['motor_de_bearing']}" if p else "PUMP-UNIT"
    if comp == "COMPRESSOR":
        c = reg.compressors.get(asset)
        return f"{c['motor_de_bearing']};{c['airend_inlet_bearing']}" if c else None
    if comp == "AXIS":
        return "BALL-SCREW"
    return comp


def _backlash_note(p: dict[str, Any], axis_events: list[dict[str, Any]]) -> str:
    for e in axis_events:
        if e["machine_id"] == p["asset_id"] and e["start"] <= p["completed"] <= e["end"]:
            frac = (p["completed"] - e["start"]) / (e["end"] - e["start"])
            return f" Backlash comp {e['axis'].upper()} axis +{frac * e['backlash_increase_um']:.0f} um."
    return " Backlash comp unchanged."


def pm_schedule(pm_rows: list[dict[str, Any]], ctx: RunContext) -> pd.DataFrame:
    df = pd.DataFrame(pm_rows)
    df = df.rename(columns={"task": "pm_task", "basis": "interval_basis"})
    return df[["asset_id", "pm_task", "interval_basis", "due", "completed", "days_late", "outcome"]].assign(
        **ctx.stamp()
    )


def drawbar_log(
    cfg: dict[str, Any], reg: Any, ctx: RunContext, pm_rows: list[dict[str, Any]], events: pd.DataFrame
) -> pd.DataFrame:
    db = cfg["drawbar_force"]
    rng = rng_for(ctx.run_seed, "drawbar")
    rebuilds: dict[str, list[pd.Timestamp]] = {}
    for mid, s in reg.hmc_spindles.items():
        start = s["last_rebuild"] or f"{reg.machines[mid]['install_year']}-01-01"
        rebuilds[mid] = [pd.Timestamp(str(start), tz=LOCAL_TZ).tz_convert("UTC")]
    if not events.empty:
        rep = events[
            (events["event"] == "replacement") & events["component_id"].astype(str).str.endswith("SP-F")
        ]
        for _, e in rep.iterrows():
            rebuilds[str(e["component_id"])[:6]].append(e["ts"])
    rows = []
    for p in pm_rows:
        if p["task"] != "drawbar_force" or p["outcome"] == "skipped":
            continue
        mid = p["asset_id"][:6]
        last = max(t for t in rebuilds[mid] if t <= p["completed"])
        years = (p["completed"] - last) / pd.Timedelta(days=DAYS_PER_YEAR)
        force = (
            db["nominal_kn"] * (1 - db["spring_fatigue_frac_per_year"] * years)
            + db["gauge_sd_kn"] * rng.standard_normal()
        )
        rows.append(
            {
                "machine_id": mid,
                "date": p["completed"].tz_convert(LOCAL_TZ).date(),
                "clamp_force_kn": round(force, 2),
                "gauge_id": str(rng.choice(db["gauge_ids"])),
            }
        )
    return pd.DataFrame(rows).assign(**ctx.stamp())


def oil_samples(
    cfg: dict[str, Any], ctx: RunContext, pm_rows: list[dict[str, Any]], truth: pd.DataFrame
) -> pd.DataFrame:
    os_ = cfg["compressor"]["oil_sample"]
    rng = rng_for(ctx.run_seed, "oil")
    code = os_["particle_count_iso_code_healthy"][0]
    stage3 = {}
    t = truth[truth["point_id"].str.startswith("CMP") & (truth["stage"] >= 3)]
    for (pid, _inst), g in t.groupby(["point_id", "instance_id"]):
        stage3.setdefault(pid[:6], []).append(g["hour_utc"].min())
    quarter = pd.Timedelta(days=DAYS_PER_YEAR / 4)
    rows = []
    for p in pm_rows:
        if p["task"] != "oil_sample" or p["outcome"] == "skipped":
            continue
        cid, ts = p["asset_id"], p["completed"]
        upper = iso_4406_upper_per_ml(code)
        count = float(rng.uniform(ISO_4406_LOWER_FRAC_OF_UPPER * upper, upper))
        if any(ts <= s3 <= ts + quarter for s3 in stage3.get(cid, [])):
            count *= float(rng.uniform(*os_["particle_count_multiplier_before_stage3"]))
        rows.append(
            {
                "compressor_id": cid,
                "date": ts.tz_convert(LOCAL_TZ).date(),
                "particle_count_4um_per_ml": round(count),
                "viscosity_cst": round(
                    os_["viscosity_cst"]["nominal"] + os_["viscosity_cst"]["sd"] * rng.standard_normal(), 1
                ),
                "water_ppm": round(
                    os_["water_ppm"]["median"] * np.exp(os_["water_ppm"]["sigma_log"] * rng.standard_normal())
                ),
            }
        )
    return pd.DataFrame(rows).assign(**ctx.stamp())


def write_cmms(tables: dict[str, pd.DataFrame], bronze_root: Path, truth_dir: Path) -> dict[str, int]:
    root = bronze_root / "cmms"
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in tables.items():
        out = df
        if name == "work_orders":
            df.to_parquet(truth_dir / "work_orders_cmms_truth.parquet", index=False)
            out = df.drop(columns=["true_component", "true_failure_code", "label_noise"])
        out.to_parquet(root / f"{name}.parquet", index=False)
        counts[name] = len(out)
    return counts
