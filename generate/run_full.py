"""Full-record export replay across the fleet in parallel worker processes.

Each worker runs one asset group (an HMC with its sensors, the compressor pair, or one
controller-only machine), renders its controller or compressor exports and writes its parts.
The parent merges parts into bronze tables and the truth archive and builds the CMMS, ERP and
register exports.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate import plant_air, shop_schedule
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
    shared, sats = build_shared(cfg, reg, ctx, start, end, set(reg.sensors))
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
            eng = HmcGroup(ctx, cfg, reg, shared, mid, sched, None, spec["axis_events"])
        else:
            eng = FleetMachineGroup(ctx, cfg, reg, shared, mid, sched, spec["axis_events"])
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
        eng = CompressorGroup(ctx, cfg, reg, shared, air, None)
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
        **counts,
    }


def merged(truth_dir: Path, name: str) -> pd.DataFrame:
    files = sorted((truth_dir / "parts").glob(f"{name}_*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_fleet(
    cfg: dict[str, Any],
    reg: Any,
    ctx: RunContext,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bronze_root: Path,
    truth_dir: Path,
    workers: int,
) -> dict[str, Any]:
    (truth_dir / "parts").mkdir(parents=True, exist_ok=True)
    from generate.machine_behavior import draw_axis_events

    hmc_ids = [m for m, v in reg.machines.items() if v["class"] == "hmc"]
    heavy, drift, titanium = shop_schedule.cell_benign_draws(
        cfg, hmc_ids, start, end, rng_for(ctx.run_seed, "cell_benign")
    )
    axis_events = draw_axis_events(cfg, reg, start, end, rng_for(ctx.run_seed, "axis_wear"))
    pd.DataFrame(axis_events).assign(**ctx.stamp()).to_parquet(truth_dir / "axis_events.parquet", index=False)
    shadow = (
        pd.Timestamp(str(cfg["meta"]["shadow"]["start"]), tz=shop_schedule.LOCAL_TZ).tz_convert("UTC"),
        pd.Timestamp(str(cfg["meta"]["shadow"]["end"]), tz=shop_schedule.LOCAL_TZ).tz_convert("UTC")
        + pd.Timedelta(days=1),
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
        "shadow": shadow,
    }
    specs = [{**base, "kind": "hmc", "group": m} for m in hmc_ids] + [{**base, "kind": "cmp", "group": "CMP"}]
    specs += [{**base, "kind": "fleet", "group": m} for m, v in reg.machines.items() if v["class"] != "hmc"]
    results = []
    if workers == 1:
        for s in specs:
            r = run_group(s)
            LOG.info(
                "group_done group=%s elapsed_s=%.0f bursts=%d",
                r["group"],
                r["elapsed_s"],
                r["bursts_rendered"],
            )
            results.append(r)
        return {"results": results, "axis_events": axis_events, "shadow": shadow}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_group, s): s["group"] for s in specs}
        for f in as_completed(futs):
            r = f.result()
            LOG.info(
                "group_done group=%s elapsed_s=%.0f bursts=%d",
                r["group"],
                r["elapsed_s"],
                r["bursts_rendered"],
            )
            results.append(r)
    return {"results": results, "axis_events": axis_events, "shadow": shadow}


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
    ctx: RunContext, start: pd.Timestamp, end: pd.Timestamp, bronze_root: Path, workers: int
) -> dict[str, Any]:
    cfg, reg = load_config(), load_register()
    truth_dir = ctx.run_dir / "truth"
    fleet = run_fleet(cfg, reg, ctx, start, end, bronze_root, truth_dir, workers)
    counts = merge_and_export(cfg, reg, ctx, bronze_root, truth_dir, fleet["axis_events"])
    return {"counts": counts, "results": fleet["results"], "shadow": fleet["shadow"], "truth_dir": truth_dir}
