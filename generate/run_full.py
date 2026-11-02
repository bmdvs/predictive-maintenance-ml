"""Full-record export replay across the fleet in parallel worker processes.

Each worker runs one asset group (an HMC with its sensors, the compressor pair, or one
controller-only machine), renders its controller or compressor exports and writes its parts.
The parent merges parts into bronze tables and the truth archive and builds the CMMS, ERP and
register exports.
"""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate import maintenance, plant_air, shop_schedule
from generate.asset_group import CompressorGroup, HmcGroup
from generate.cmms_export import build_work_orders, drawbar_log, oil_samples, pm_schedule, write_cmms
from generate.compressor_controller_export import render_compressors
from generate.controller_export import render_machine
from generate.erp_export import jobs_table, write_erp, write_register
from generate.fleet import FleetMachineGroup
from generate.provenance import RunContext, rng_for
from generate.shared import build_shared
from generate.truth_archive import _frame
from pdm import log
from pdm.config import load_config, load_register

LOG = log.get_logger("generate.full")
PART_TABLES = (
    "truth_hours",
    "events",
    "work_orders",
    "pm_rows",
    "sensor_events",
    "alerts",
    "machine_hours",
    "jobs",
    "burst_index",
)


def _sink(bronze_root: Path, group: str):
    def write(local_month: str, rows: list[dict[str, Any]]) -> None:
        df = pd.DataFrame(rows)
        for m, part in df.groupby(df["hour_utc"].dt.strftime("%Y-%m")):
            d = bronze_root / "cm_platform" / "scalars" / f"month={m}"
            d.mkdir(parents=True, exist_ok=True)
            part.to_parquet(d / f"{group}_{local_month}.parquet", index=False)

    return write


def _write_bursts(bursts: list[tuple[dict[str, Any], np.ndarray]], bronze_root: Path) -> list[dict[str, Any]]:
    index = []
    for meta, arr in bursts:
        rel = Path("bursts") / meta["point_id"] / f"{meta['hour_utc']:%Y%m%dT%H}.npz"
        path = bronze_root / "cm_platform" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, accel_g=arr, fs_hz=meta["fs_hz"])
        index.append({**meta, "path": rel.as_posix()})
    return index


def run_group(spec: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    log.configure()
    cfg, reg = load_config(), load_register()
    ctx = RunContext(
        spec["run_id"], spec["config_hash"], spec["generator_version"], spec["run_seed"], spec["label"]
    )
    start, end = spec["start"], spec["end"]
    bronze_root, truth_dir = Path(spec["bronze_root"]), Path(spec["truth_dir"])
    rendered = set(spec["rendered"]) if spec.get("rendered") is not None else set(reg.sensors)
    shared, sats = build_shared(cfg, reg, ctx, start, end, rendered)
    group = spec["group"]
    tables: dict[str, list[dict[str, Any]]] = {k: [] for k in PART_TABLES}
    counts: dict[str, int] = {}
    if spec["kind"] in ("hmc", "fleet"):
        mid = group
        machine = reg.machines[mid]
        sat_key = "hmc" if machine["class"] == "hmc" else "other"
        sched, _, ben = shop_schedule.machine_schedule(
            cfg,
            machine,
            shared.frame,
            shared.holidays,
            sats[sat_key],
            rng_for(ctx.run_seed, "schedule", mid),
            spec["heavy"].get(mid, []),
            spec["titanium"] if mid in spec["titanium_machines"] else None,
            spec["drift"].get(mid, []),
        )
        if spec["kind"] == "hmc":
            eng = HmcGroup(
                ctx,
                cfg,
                reg,
                shared,
                mid,
                sched,
                spec.get("overrides"),
                spec["axis_events"],
                spec.get("spindle_cases", {}).get(mid, {}),
            )
        else:
            eng = FleetMachineGroup(ctx, cfg, reg, shared, mid, sched, spec["axis_events"])
        spindle_failures = getattr(eng, "spindle_failures", [])
        eng.scalar_sink = _sink(bronze_root, mid)
        out = eng.run()
        mh = pd.DataFrame(eng.machine_hours)
        counts.update(
            render_machine(
                cfg,
                reg,
                ctx,
                mid,
                sched,
                mh,
                out.events,
                shared.shop_ambient,
                spec["shadow"],
                bronze_root,
                truth_dir,
            )
        )
        tables["machine_hours"] = eng.machine_hours
        tables["jobs"] = jobs_table(cfg, ctx, mid, sched, mh).to_dict("records")
        sched.assign(**ctx.stamp()).to_parquet(truth_dir / "parts" / f"schedule_{mid}.parquet", index=False)
        benign = ben + [
            {"event": "coolant_drift", "asset_id": mid, "ts": a, "end": b, "load_gain": g}
            for a, b, g in spec["drift"].get(mid, [])
        ]
        if mid in spec["titanium_machines"]:
            benign.append(
                {
                    "event": "titanium_job",
                    "asset_id": mid,
                    "ts": spec["titanium"][0],
                    "end": spec["titanium"][1],
                }
            )
        pd.DataFrame(benign).assign(**ctx.stamp()).to_parquet(
            truth_dir / "parts" / f"benign_{mid}.parquet", index=False
        )
    else:
        air = plant_air.compressor_hours(
            cfg, shared.frame, shared.plant_minutes, list(reg.compressors), rng_for(ctx.run_seed, "air")
        )
        air["room_c"] = shared.room_ambient
        eng = CompressorGroup(ctx, cfg, reg, shared, air, spec.get("overrides"))
        spindle_failures = []
        eng.scalar_sink = _sink(bronze_root, group)
        out = eng.run()
        counts.update(
            render_compressors(cfg, reg, ctx, air, eng.ops, bronze_root, eng.fouling_rise, eng.ctrl_events)
        )
        for cid, rise in eng.fouling_rise.items():
            air[f"{cid}_fouling_rise_c"] = rise
        air.assign(**ctx.stamp()).to_parquet(truth_dir / "parts" / "compressor_hours.parquet", index=False)
        pd.DataFrame(shared.heat_episodes).assign(**ctx.stamp()).to_parquet(
            truth_dir / "parts" / "benign_CMP.parquet", index=False
        )
    tables["truth_hours"] = out.truth_hours
    tables["events"] = out.events
    tables["work_orders"] = out.work_orders
    tables["pm_rows"] = out.pm_rows
    tables["sensor_events"] = out.sensor_events
    tables["alerts"] = out.alerts
    tables["burst_index"] = _write_bursts(out.bursts, bronze_root)
    for name, rows in tables.items():
        if rows:
            _frame(rows, ctx).to_parquet(truth_dir / "parts" / f"{name}_{group}.parquet", index=False)
    return {
        "group": group,
        "bursts_rendered": out.bursts_rendered,
        "render_s": out.render_seconds,
        "elapsed_s": time.perf_counter() - t0,
        "spindle_failures": spindle_failures,
        **counts,
    }


def merged(truth_dir: Path, name: str) -> pd.DataFrame:
    files = sorted((truth_dir / "parts").glob(f"{name}_*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _execute(specs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results = []
    if workers == 1 or len(specs) == 1:
        done = [run_group(sp) for sp in specs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
            futs = [pool.submit(run_group, sp) for sp in specs]
            done = [f.result() for f in as_completed(futs)]
    for r in done:
        LOG.info(
            "group_done group=%s elapsed_s=%.0f bursts=%d", r["group"], r["elapsed_s"], r["bursts_rendered"]
        )
        results.append(r)
    return results


def resolve_spare(
    cfg: dict[str, Any], failures: list[dict[str, Any]], holidays: set, start: pd.Timestamp
) -> tuple[dict[str, dict[int, str]], list[dict[str, Any]]]:
    """Assign the cell's spare spindle in time order. A failure with the spare available is a swap and
    sends its spindle out for rebuild, returning as the spare after the rebuild time; a failure while the
    spare is out waits for its own spindle (no_spare). Returns the case per machine and ordinal and the
    spare's state timeline."""
    ss = cfg["maintenance"]["spare_spindle"]
    free_at = start
    cases: dict[str, dict[int, str]] = {}
    timeline = [{"ts": start, "state": "available", "machine_id": None, "ordinal": None}]
    for f in sorted(failures, key=lambda r: (r["ts"], r["machine_id"], r["ordinal"])):
        ts = f["ts"]
        row = {"machine_id": f["machine_id"], "ordinal": int(f["ordinal"])}
        if ts >= free_at:
            case = "swap"
            local_h = int(ts.tz_convert(shop_schedule.LOCAL_TZ).hour)
            free_at = maintenance.add_working_days(ts, int(ss["rebuild_out_working_days"]), holidays, local_h)
            timeline.append({"ts": ts, "state": "out_for_rebuild", **row})
            timeline.append({"ts": free_at, "state": "available", **row})
        else:
            case = "no_spare"
            timeline.append({"ts": ts, "state": "unavailable_at_failure", **row})
        cases.setdefault(f["machine_id"], {})[int(f["ordinal"])] = case
    return cases, timeline


def _clear_group_outputs(group: str, reg: Any, bronze_root: Path, truth_dir: Path) -> None:
    """Remove a group's per-group files before it is rerun, so a rerun leaves no stale rows or bursts."""
    for f in (truth_dir / "parts").glob(f"*_{group}.parquet"):
        f.unlink()
    for pid, sensor in reg.sensors.items():
        if sensor.asset_id.startswith(group + "-") or sensor.asset_id == group:
            d = bronze_root / "cm_platform" / "bursts" / pid
            if d.exists():
                shutil.rmtree(d)
    d = bronze_root / "controller" / "khz" / group
    if d.exists():
        shutil.rmtree(d)
    for f in (bronze_root / "cm_platform" / "scalars").glob(f"month=*/{group}_*.parquet"):
        f.unlink()


def run_fleet(
    cfg: dict[str, Any],
    reg: Any,
    ctx: RunContext,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bronze_root: Path,
    truth_dir: Path,
    workers: int,
    groups: list[str] | None = None,
    rendered: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every asset group (or the listed ones), then resolve the HMC cell's shared spare spindle:
    rerun only the HMC groups whose spindle repair case changed, until no assignment changes."""
    (truth_dir / "parts").mkdir(parents=True, exist_ok=True)
    from generate.machine_behavior import draw_axis_events

    hmc_ids = [m for m, v in reg.machines.items() if v["class"] == "hmc"]
    heavy, drift, titanium = shop_schedule.cell_benign_draws(
        cfg, hmc_ids, start, end, rng_for(ctx.run_seed, "cell_benign")
    )
    axis_events = draw_axis_events(cfg, reg, start, end, rng_for(ctx.run_seed, "axis_wear"))
    pd.DataFrame(axis_events).assign(**ctx.stamp()).to_parquet(truth_dir / "axis_events.parquet", index=False)
    hz = cfg["meta"]["hz1_window"]
    hz1 = (
        pd.Timestamp(str(hz["start"]), tz=shop_schedule.LOCAL_TZ).tz_convert("UTC"),
        pd.Timestamp(str(hz["end"]), tz=shop_schedule.LOCAL_TZ).tz_convert("UTC") + pd.Timedelta(days=1),
    )
    base = {
        "run_id": ctx.run_id,
        "config_hash": ctx.config_hash,
        "generator_version": ctx.generator_version,
        "run_seed": ctx.run_seed,
        "label": ctx.label,
        "start": start,
        "end": end,
        "bronze_root": str(bronze_root),
        "truth_dir": str(truth_dir),
        "axis_events": axis_events,
        "heavy": heavy,
        "drift": drift,
        "titanium": titanium,
        "titanium_machines": set(cfg["benign"]["titanium_job"]["machines"]),
        "shadow": hz1,
        "rendered": rendered,
        "overrides": overrides,
        "spindle_cases": {},
    }
    specs = [{**base, "kind": "hmc", "group": m} for m in hmc_ids] + [{**base, "kind": "cmp", "group": "CMP"}]
    specs += [{**base, "kind": "fleet", "group": m} for m, v in reg.machines.items() if v["class"] != "hmc"]
    if groups is not None:
        specs = [sp for sp in specs if sp["group"] in groups]
    results = {r["group"]: r for r in _execute(specs, workers)}
    local0, local1 = start.tz_convert(shop_schedule.LOCAL_TZ), end.tz_convert(shop_schedule.LOCAL_TZ)
    holidays = shop_schedule.holiday_dates(
        cfg["shop"]["calendar"]["holidays"], range(local0.year, local1.year + 2)
    )
    hmc_specs = {sp["group"]: sp for sp in specs if sp["kind"] == "hmc"}
    passes, rerun_log, timeline, failures = 1, [], [], []
    max_passes = int(cfg["maintenance"]["spare_spindle"]["resolve_max_passes"])
    while True:
        failures = [f for g in hmc_specs for f in results[g]["spindle_failures"]]
        cases, timeline = resolve_spare(cfg, failures, holidays, start)
        used = {g: {int(f["ordinal"]): f["case"] for f in results[g]["spindle_failures"]} for g in hmc_specs}
        changed = [
            g for g in hmc_specs if any(cases.get(g, {}).get(n, "swap") != c for n, c in used[g].items())
        ]
        if not changed:
            break
        if passes >= max_passes:
            raise RuntimeError(f"spare spindle assignment not stable after {passes} passes: {changed}")
        passes += 1
        LOG.info("spare_spindle_pass pass=%d rerun=%s", passes, ",".join(changed))
        rerun_log.append({"pass": passes, "groups": changed})
        for g in changed:
            _clear_group_outputs(g, reg, bronze_root, truth_dir)
            hmc_specs[g] = {**hmc_specs[g], "spindle_cases": {g: cases.get(g, {})}}
        for r in _execute([hmc_specs[g] for g in changed], workers):
            results[r["group"]] = r
    pd.DataFrame(timeline).assign(**ctx.stamp()).to_parquet(truth_dir / "spare_spindle.parquet", index=False)
    spare = {"passes": passes, "reruns": rerun_log, "unplanned_spindle_failures": len(failures)}
    with open(truth_dir / "spare_spindle_passes.json", "w", encoding="utf-8") as fh:
        json.dump(spare, fh, indent=1, default=str)
    return {"results": list(results.values()), "axis_events": axis_events, "shadow": hz1, "spare": spare}


def merge_and_export(
    cfg: dict[str, Any],
    reg: Any,
    ctx: RunContext,
    bronze_root: Path,
    truth_dir: Path,
    axis_events: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    tables = {name: merged(truth_dir, name) for name in PART_TABLES}
    tables["truth_hours"].to_parquet(truth_dir / "point_hours.parquet", index=False)
    tables["events"].to_parquet(truth_dir / "events.parquet", index=False)
    tables["work_orders"].to_parquet(truth_dir / "work_orders.parquet", index=False)
    tables["pm_rows"].to_parquet(truth_dir / "pm.parquet", index=False)
    tables["machine_hours"].to_parquet(truth_dir / "machine_hours.parquet", index=False)
    benign = merged(truth_dir, "benign")
    benign.to_parquet(truth_dir / "benign.parquet", index=False)
    cm = bronze_root / "cm_platform"
    for name in ("alerts", "sensor_events"):
        (cm / name).mkdir(parents=True, exist_ok=True)
        tables[name].to_parquet(cm / name / f"{name}.parquet", index=False)
        counts[name] = len(tables[name])
    tables["burst_index"].to_parquet(cm / "burst_index.parquet", index=False)
    counts["bursts"] = len(tables["burst_index"])
    wo_rows = tables["work_orders"].replace({np.nan: None}).to_dict("records")
    pm_rows = tables["pm_rows"].to_dict("records")
    wo = build_work_orders(
        cfg, reg, ctx, wo_rows, pm_rows, tables["events"], axis_events, tables["sensor_events"]
    )
    counts.update(
        write_cmms(
            {
                "work_orders": wo,
                "pm_schedule": pm_schedule(pm_rows, ctx),
                "drawbar_force_log": drawbar_log(cfg, reg, ctx, pm_rows, tables["events"]),
                "oil_samples": oil_samples(cfg, ctx, pm_rows, tables["truth_hours"]),
            },
            bronze_root,
            truth_dir,
        )
    )
    jobs = tables["jobs"]
    counts.update(write_erp(cfg, ctx, [jobs], bronze_root, truth_dir))
    counts.update(
        {
            f"register_{k}": v
            for k, v in write_register(
                reg, ctx, tables["events"], tables["sensor_events"], bronze_root
            ).items()
        }
    )
    return counts


def run_full_record(
    ctx: RunContext,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bronze_root: Path,
    workers: int,
    groups: list[str] | None = None,
    rendered: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg, reg = load_config(), load_register()
    truth_dir = ctx.run_dir / "truth"
    fleet = run_fleet(cfg, reg, ctx, start, end, bronze_root, truth_dir, workers, groups, rendered, overrides)
    counts = merge_and_export(cfg, reg, ctx, bronze_root, truth_dir, fleet["axis_events"])
    return {
        "counts": counts,
        "results": fleet["results"],
        "shadow": fleet["shadow"],
        "spare": fleet["spare"],
        "truth_dir": truth_dir,
    }
