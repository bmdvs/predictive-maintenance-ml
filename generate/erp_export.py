"""ERP/MES export and asset-register export.

Inputs: per-machine hourly schedules, per-machine hourly state (downtime), the register and the
event timeline. Outputs: bronze erp tables (jobs, contribution margin) and register tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate.provenance import RunContext, rng_for
from generate.shop_schedule import LOCAL_TZ
from pdm.constants import SECONDS_PER_MINUTE


def jobs_table(
    cfg: dict[str, Any], ctx: RunContext, machine_id: str, sched: pd.DataFrame, machine_hours: pd.DataFrame
) -> pd.DataFrame:
    s = sched.reset_index(drop=True).copy()
    s["actual_cut_min"] = machine_hours["cut_min"].to_numpy()
    s["setup_min"] = np.where(s["setup"], s["scheduled_min"], 0.0)
    s = s[s["job_id"].notna()]
    g = s.groupby("job_id", sort=False)
    jobs = g.agg(
        start=("hour_utc", "min"),
        program=("program_id", "first"),
        material=("material", "first"),
        cutting_min=("actual_cut_min", "sum"),
        planned_cutting_min=("cutting_min", "sum"),
        setup_min=("setup_min", "sum"),
    )
    jobs = jobs.reset_index()
    cyc = cfg["erp"]["cycle_time_min_by_material"]
    qty = []
    for _, j in jobs.iterrows():
        r = rng_for(ctx.run_seed, "cycle", j["program"])
        cycle = float(r.uniform(*cyc[j["material"]]))
        qty.append(int(np.floor(j["cutting_min"] / cycle)))
    return pd.DataFrame(
        {
            "job_id": jobs["job_id"],
            "machine_id": machine_id,
            "date": jobs["start"].dt.tz_convert(LOCAL_TZ).dt.date,
            "part_number": "PN-" + jobs["program"].str.split("-P").str[-1],
            "material": jobs["material"],
            "program": jobs["program"],
            "cutting_hours": (jobs["cutting_min"] / SECONDS_PER_MINUTE).round(2),
            "setup_hours": (jobs["setup_min"] / SECONDS_PER_MINUTE).round(2),
            "quantity": qty,
            "planned_cutting_hours": (jobs["planned_cutting_min"] / SECONDS_PER_MINUTE).round(2),
        }
    ).assign(**ctx.stamp())


def write_erp(
    cfg: dict[str, Any], ctx: RunContext, jobs: list[pd.DataFrame], bronze_root: Path, truth_dir: Path
) -> dict[str, int]:
    root = bronze_root / "erp"
    root.mkdir(parents=True, exist_ok=True)
    allj = pd.concat(jobs, ignore_index=True)
    allj.to_parquet(truth_dir / "jobs_truth.parquet", index=False)
    allj.drop(columns=["planned_cutting_hours"]).to_parquet(root / "jobs.parquet", index=False)
    margin = pd.DataFrame(
        [
            {"machine_class": k, "contribution_margin_per_hour_usd": v}
            for k, v in cfg["economics"]["contribution_margin_per_hour"].items()
        ]
    ).assign(**ctx.stamp())
    margin.to_parquet(root / "contribution_margin.parquet", index=False)
    return {"jobs": len(allj), "contribution_margin": len(margin)}


def write_register(
    reg: Any, ctx: RunContext, events: pd.DataFrame, sensor_events: pd.DataFrame, bronze_root: Path
) -> dict[str, int]:
    root = bronze_root / "register"
    root.mkdir(parents=True, exist_ok=True)
    stamp = ctx.stamp()
    tables = {
        "machines": pd.DataFrame(list(reg.machines.values())),
        "bearings": pd.DataFrame([vars(b) for b in reg.bearings.values()]),
        "compressors": pd.DataFrame(list(reg.compressors.values())),
    }
    rep = (
        events[events["event"] == "replacement"]
        if not events.empty
        else pd.DataFrame(columns=["component_id", "ts"])
    )
    spind = []
    for mid, s in reg.hmc_spindles.items():
        dates = [str(s["last_rebuild"])] if s["last_rebuild"] else []
        dates += [f"{t:%Y-%m-%d}" for t in rep[rep["component_id"] == f"{mid}-SP-F"]["ts"]]
        spind.append(
            {**{k: v for k, v in s.items() if k != "last_rebuild"}, "rebuild_dates": ";".join(dates)}
        )
    tables["hmc_spindles"] = pd.DataFrame(spind)
    pumps = []
    for aid, p in reg.pumps.items():
        dates = [str(p["last_replacement"])] if p["last_replacement"] else []
        dates += [f"{t:%Y-%m-%d}" for t in rep[rep["component_id"] == aid]["ts"]]
        pumps.append(
            {**{k: v for k, v in p.items() if k != "last_replacement"}, "replacement_dates": ";".join(dates)}
        )
    tables["pumps"] = pd.DataFrame(pumps)
    sens = []
    for pid, s in reg.sensors.items():
        se = sensor_events[sensor_events["point_id"] == pid]
        sens.append(
            {
                **vars(s),
                "remount_dates": ";".join(f"{t:%Y-%m-%d}" for t in se[se["event"] == "remount"]["ts"]),
                "battery_change_dates": ";".join(
                    f"{t:%Y-%m-%d}" for t in se[se["event"] == "battery_change"]["ts"]
                ),
            }
        )
    tables["sensors"] = pd.DataFrame(sens)
    counts = {}
    for name, df in tables.items():
        df.astype({c: str for c in df.columns if df[c].dtype == object}).assign(**stamp).to_parquet(
            root / f"{name}.parquet", index=False
        )
        counts[name] = len(df)
    return counts
