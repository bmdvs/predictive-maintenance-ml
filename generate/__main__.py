"""Command line entry for the export replay: python -m generate [--pilot].

Builds the shop calendar and schedules, runs each asset group, writes bronze exports and the truth
archive, runs the realism checks and writes the run report under runs/<run_id>/.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from generate import plant_air, shop_schedule
from generate.asset_group import CompressorGroup, GroupOutput, HmcGroup
from generate.checks import realism, run_report
from generate.provenance import RUNS_DIR, RunContext, new_run, rng_for
from generate.provenance import config_hash as config_hash_of
from generate.run_full import run_full_record
from generate.sensor_platform_export import write_cm_platform
from generate.shared import build_shared
from generate.truth_archive import write_truth
from pdm import log, paths
from pdm.config import load_config, load_register

LOG = log.get_logger("generate")


def next_label(prefix: str) -> str:
    existing = [p.name for p in RUNS_DIR.glob(f"{prefix}*")] if RUNS_DIR.exists() else []
    nums = [
        int(n[len(prefix) : len(prefix) + 2]) for n in existing if n[len(prefix) : len(prefix) + 2].isdigit()
    ]
    return f"{prefix}{(max(nums) + 1 if nums else 1):02d}"


def generator_commit() -> tuple[str, list[str]]:
    """HEAD commit and the tracked generator, package and config files that differ from it."""
    root = paths.REPO_ROOT
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "generate",
                "src",
                "config",
                "pyproject.toml",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", []
    return head, [line[3:] for line in dirty]


def local_ts(d: Any) -> pd.Timestamp:
    return pd.Timestamp(str(d), tz=shop_schedule.LOCAL_TZ).tz_convert("UTC")


def run(pilot: bool, window_days: int | None = None, workers: int | None = None) -> RunContext:
    """Generate one run. window_days truncates the window (used by tests only)."""
    log.configure()
    cfg = load_config()
    reg = load_register()
    pilot_cfg = None
    if pilot:
        with open(paths.CONFIG_DIR / "pilot.yaml", encoding="utf-8") as fh:
            pilot_cfg = yaml.safe_load(fh)
    resolved = {"config": copy.deepcopy(cfg), "register": reg.raw, "pilot": pilot_cfg}
    head, dirty = generator_commit()
    if not pilot and window_days is None and dirty:
        raise SystemExit(
            f"uncommitted generator, package or config changes: {dirty}; commit before a numbered run"
        )
    ctx = new_run(resolved, next_label("pilot" if pilot else "run"))
    ctx.run_dir.mkdir(parents=True, exist_ok=False)
    (ctx.run_dir / "generator_commit.txt").write_text(
        head + (f" (uncommitted: {', '.join(dirty)})" if dirty else "") + "\n", encoding="utf-8"
    )
    LOG.info("run_start run_id=%s config_hash=%s", ctx.run_id, ctx.config_hash)
    t_start = time.perf_counter()

    if pilot:
        start, end = (
            local_ts(pilot_cfg["window"]["start"]),
            local_ts(pilot_cfg["window"]["end"]) + pd.Timedelta(days=1),
        )
        rendered = set(pilot_cfg["points"])
        machines = [m for m in pilot_cfg["simulated_assets"] if m.startswith("HMC")]
        overrides = pilot_cfg.get("override", {})
    else:
        start, end = (
            local_ts(cfg["meta"]["record"]["start"]),
            local_ts(cfg["meta"]["record"]["end"]) + pd.Timedelta(days=1),
        )
        rendered = set(reg.sensors)
        machines = [m for m, v in reg.machines.items() if v["class"] == "hmc"]
        overrides = {}

    if window_days is not None:
        end = start + pd.Timedelta(days=window_days)
    if not pilot:
        return run_full(ctx, cfg, reg, resolved, start, end, t_start, workers)
    shared, sats = build_shared(cfg, reg, ctx, start, end, rendered)
    hmc_ids = [m for m, v in reg.machines.items() if v["class"] == "hmc"]
    heavy, drift, titanium = shop_schedule.cell_benign_draws(
        cfg, hmc_ids, start, end, rng_for(ctx.run_seed, "cell_benign")
    )
    ti_machines = set(cfg["benign"]["titanium_job"]["machines"])

    outputs: list[GroupOutput] = []
    benign: list[dict[str, Any]] = []
    schedules = {}
    for mid in machines:
        sched, _, ben = shop_schedule.machine_schedule(
            cfg,
            reg.machines[mid],
            shared.frame,
            shared.holidays,
            sats["hmc"],
            rng_for(ctx.run_seed, "schedule", mid),
            heavy[mid],
            titanium if mid in ti_machines else None,
            drift[mid],
        )
        schedules[mid] = sched
        benign += ben + [
            {"event": "coolant_drift", "asset_id": mid, "ts": a, "end": b, "load_gain": g}
            for a, b, g in drift[mid]
        ]
        LOG.info("group_start group=%s hours=%d", mid, len(sched))
        outputs.append(HmcGroup(ctx, cfg, reg, shared, mid, sched, overrides).run())
    air = plant_air.compressor_hours(
        cfg, shared.frame, shared.plant_minutes, list(reg.compressors), rng_for(ctx.run_seed, "air")
    )
    LOG.info("group_start group=CMP hours=%d", len(air))
    outputs.append(CompressorGroup(ctx, cfg, reg, shared, air, overrides).run())

    bronze_root = paths.DATA_DIR / "pilot" / ctx.run_id / "bronze" if pilot else paths.BRONZE_DIR
    counts = write_cm_platform(outputs, bronze_root, ctx)
    truth_dir = write_truth(outputs, ctx, resolved)
    pd.DataFrame(benign).assign(**ctx.stamp()).to_parquet(truth_dir / "benign.parquet", index=False)
    for mid, sched in schedules.items():
        sched.assign(**ctx.stamp()).to_parquet(truth_dir / f"schedule_{mid}.parquet", index=False)
    air.assign(**ctx.stamp()).to_parquet(truth_dir / "compressor_hours.parquet", index=False)
    elapsed = time.perf_counter() - t_start
    timing = {
        "elapsed_s": elapsed,
        "bursts_rendered": sum(o.bursts_rendered for o in outputs),
        "render_extract_s": sum(o.render_seconds for o in outputs),
    }
    LOG.info("generation_done elapsed_s=%.1f bursts=%d", elapsed, timing["bursts_rendered"])
    with open(ctx.run_dir / "generation.json", "w", encoding="utf-8") as fh:
        json.dump({"counts": counts, "timing": timing, "pilot": pilot}, fh, indent=1)
    report(ctx, cfg, reg, bronze_root, truth_dir, counts, timing, pilot)
    LOG.info("run_done run_id=%s", ctx.run_id)
    return ctx


def run_full(
    ctx: RunContext,
    cfg: dict[str, Any],
    reg: Any,
    resolved: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    t_start: float,
    workers: int | None,
) -> RunContext:
    bronze_root = paths.BRONZE_DIR
    if bronze_root.exists() and any(bronze_root.iterdir()):
        raise SystemExit(f"{bronze_root} is not empty; a full run writes a fresh bronze export")
    truth_dir = ctx.run_dir / "truth"
    truth_dir.mkdir(parents=True)
    with open(ctx.run_dir / "resolved_config.json", "w", encoding="utf-8") as fh:
        json.dump(resolved, fh, indent=1, default=str)
    with open(truth_dir / "seeds.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_seed": ctx.run_seed,
                "run_id": ctx.run_id,
                "config_hash": ctx.config_hash,
                "generator_version": ctx.generator_version,
            },
            fh,
            indent=1,
        )
    res = run_full_record(ctx, start, end, bronze_root, workers or max((os.cpu_count() or 2) - 1, 1))
    elapsed = time.perf_counter() - t_start
    timing = {
        "elapsed_s": elapsed,
        "bursts_rendered": sum(r["bursts_rendered"] for r in res["results"]),
        "render_extract_s": sum(r["render_s"] for r in res["results"]),
        "groups": {r["group"]: round(r["elapsed_s"], 1) for r in res["results"]},
    }
    counts = {
        **res["counts"],
        **{
            k: sum(r.get(k, 0) for r in res["results"])
            for k in (
                "hz1_rows",
                "minute_rows",
                "events",
                "khz_windows",
                "compressor_minutes",
                "room_minutes",
                "compressor_events",
            )
        },
    }
    LOG.info("generation_done elapsed_s=%.1f bursts=%d", elapsed, timing["bursts_rendered"])
    with open(ctx.run_dir / "generation.json", "w", encoding="utf-8") as fh:
        json.dump({"counts": counts, "timing": timing, "pilot": False}, fh, indent=1)
    report(ctx, cfg, reg, bronze_root, truth_dir, counts, timing, False)
    LOG.info("run_done run_id=%s", ctx.run_id)
    return ctx


def report(
    ctx: RunContext,
    cfg: dict[str, Any],
    reg: Any,
    bronze_root: Path,
    truth_dir: Path,
    counts: dict[str, int],
    timing: dict[str, float],
    pilot: bool,
) -> None:
    results = realism.run_checks(cfg, reg, ctx, bronze_root, truth_dir, pilot=pilot)
    run_report.write(ctx, cfg, results, counts, timing, bronze_root, truth_dir, pilot=pilot)


def recheck(run_id: str) -> None:
    """Re-run checks and the report on an existing run's outputs; nothing is regenerated."""
    log.configure()
    run_dir = RUNS_DIR / run_id
    with open(run_dir / "resolved_config.json", encoding="utf-8") as fh:
        resolved = json.load(fh)
    with open(run_dir / "generation.json", encoding="utf-8") as fh:
        gen = json.load(fh)
    with open(run_dir / "truth" / "seeds.json", encoding="utf-8") as fh:
        seeds = json.load(fh)
    cfg = load_config()
    current = {"config": copy.deepcopy(cfg), "register": load_register().raw, "pilot": resolved.get("pilot")}
    diff_file = run_dir / "recheck_checks_diff.txt"
    if config_hash_of(current) != seeds["config_hash"]:
        before = run_report._flatten(json.loads(json.dumps(resolved, default=str)))
        after = run_report._flatten(json.loads(json.dumps(current, default=str)))
        changed = sorted(k for k in set(before) | set(after) if str(before.get(k)) != str(after.get(k)))
        outside = [k for k in changed if not k.startswith("config.checks.")]
        if outside:
            raise SystemExit(f"recheck refused: changes outside the checks block: {outside}")
        diff_file.write_text(
            "\n".join(f"{k}: {before.get(k)} -> {after.get(k)}" for k in changed) + "\n", encoding="utf-8"
        )
    elif diff_file.exists():
        diff_file.unlink()
    ctx = RunContext(
        run_id, seeds["config_hash"], seeds["generator_version"], seeds["run_seed"], run_id.split("_")[0]
    )
    pilot = gen["pilot"]
    bronze_root = paths.DATA_DIR / "pilot" / run_id / "bronze" if pilot else paths.BRONZE_DIR
    report(ctx, cfg, load_register(), bronze_root, run_dir / "truth", gen["counts"], gen["timing"], pilot)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m generate")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--recheck", metavar="RUN_ID")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    np.seterr(all="ignore")
    if args.recheck:
        recheck(args.recheck)
    else:
        run(args.pilot, workers=args.workers)


if __name__ == "__main__":
    logging.captureWarnings(True)
    main()
