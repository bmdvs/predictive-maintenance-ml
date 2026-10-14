"""Renders the bursts the maintenance queue's spectrum panels need from the frozen generator with run09's
seeds, for hours without a retained burst, plus retained hours used to check the rendering.

Each listed asset group is replayed in memory with run09's run context (and its final spare-spindle
assignment); the burst of every requested point-hour is captured where the engine renders it. Nothing is
written under data/ or runs/: the scalar writer is disabled and no export or truth file is produced.

python tools/render_queue_bursts.py build/queue_cache/burst_targets.json build/queue_cache/bursts/rendered
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from generate.__main__ import local_ts  # noqa: E402
from generate.asset_group import CompressorGroup, HmcGroup  # noqa: E402
from generate.machine_behavior import draw_axis_events  # noqa: E402
from generate.provenance import RunContext, rng_for  # noqa: E402
from generate.shared import build_shared  # noqa: E402

from generate import plant_air, shop_schedule  # noqa: E402
from pdm.config import load_config, load_register  # noqa: E402

RUN = ROOT / "runs" / "run09_20261030T090522Z_3eced8fb"


def _context() -> RunContext:
    s = json.loads((RUN / "truth" / "seeds.json").read_text(encoding="utf-8"))
    return RunContext(s["run_id"], s["config_hash"], s["generator_version"], int(s["run_seed"]), "run09")


def _spindle_cases(machine: str) -> dict[int, str]:
    """run09's final spare-spindle assignment for the machine (the second pass reran HMC-03 with it)."""
    t = pd.read_parquet(RUN / "truth" / "spare_spindle.parquet")
    t = t[t["machine_id"] == machine]
    out = {}
    for r in t.itertuples():
        if r.state == "unavailable_at_failure":
            out[int(r.ordinal)] = "no_spare"
        elif r.state == "out_for_rebuild":
            out[int(r.ordinal)] = "swap"
    return out


def render_group(group: str, targets: list[tuple[str, str]], out_dir: str) -> list[dict]:
    cfg, reg = load_config(), load_register()
    ctx = _context()
    start = local_ts(cfg["meta"]["record"]["start"])
    end = local_ts(cfg["meta"]["record"]["end"]) + pd.Timedelta(days=1)
    want = {(p, pd.Timestamp(h)) for p, h in targets}
    shared, sats = build_shared(cfg, reg, ctx, start, end, set(reg.sensors))
    if group == "CMP":
        air = plant_air.compressor_hours(
            cfg, shared.frame, shared.plant_minutes, list(reg.compressors), rng_for(ctx.run_seed, "air")
        )
        air["room_c"] = shared.room_ambient
        eng = CompressorGroup(ctx, cfg, reg, shared, air, None)
    else:
        hmc_ids = [m for m, v in reg.machines.items() if v["class"] == "hmc"]
        heavy, drift, titanium = shop_schedule.cell_benign_draws(
            cfg, hmc_ids, start, end, rng_for(ctx.run_seed, "cell_benign")
        )
        axis_events = draw_axis_events(cfg, reg, start, end, rng_for(ctx.run_seed, "axis_wear"))
        tit_machines = set(cfg["benign"]["titanium_job"]["machines"])
        sched, _, _ = shop_schedule.machine_schedule(
            cfg,
            reg.machines[group],
            shared.frame,
            shared.holidays,
            sats["hmc"],
            rng_for(ctx.run_seed, "schedule", group),
            heavy.get(group, []),
            titanium if group in tit_machines else None,
            drift.get(group, []),
        )
        eng = HmcGroup(ctx, cfg, reg, shared, group, sched, None, axis_events, _spindle_cases(group))
    eng.scalar_sink = None
    captured: list[dict] = []
    original = eng.render_point_hour

    def hook(p, ts, cond):
        burst, fault_hz = original(p, ts, cond)
        if (p.point_id, ts) in want:
            path = Path(out_dir) / p.point_id / f"{ts:%Y%m%dT%H}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                accel_g=burst.astype(np.float32),
                accel_g64=burst,
                fs_hz=eng.settings.fs,
                shaft_hz=cond["shaft_hz"],
            )
            captured.append({"point_id": p.point_id, "hour_utc": ts.isoformat(), "path": str(path)})
        return burst, fault_hz

    eng.render_point_hour = hook
    eng.run()
    return captured


def main(targets_file: str, out_dir: str) -> None:
    t = json.loads(Path(targets_file).read_text(encoding="utf-8"))
    by_group: dict[str, list[tuple[str, str]]] = {}
    for r in t:
        g = "CMP" if r["point_id"].startswith("CMP") else "-".join(r["point_id"].split("-")[:2])
        by_group.setdefault(g, []).append((r["point_id"], r["hour_utc"]))
    # each group replays its machine's full record in memory; three at a time fit in memory (eight did not)
    with ProcessPoolExecutor(max_workers=min(3, len(by_group))) as pool:
        futs = {g: pool.submit(render_group, g, v, out_dir) for g, v in by_group.items()}
        for f in futs.values():
            f.result()
    rows = []
    for f in sorted(Path(out_dir).glob("*/*.npz")):
        ts = pd.Timestamp(f.stem.replace("T", " ") + ":00", tz="UTC")
        rows.append({"point_id": f.parent.name, "hour_utc": ts.isoformat(), "path": str(f)})
    Path(out_dir, "rendered.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    missing = {(p, pd.Timestamp(h)) for v in by_group.values() for p, h in v} - {
        (r["point_id"], pd.Timestamp(r["hour_utc"])) for r in rows
    }
    print(json.dumps({"rendered": len(rows), "missing": sorted(map(str, missing))}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
