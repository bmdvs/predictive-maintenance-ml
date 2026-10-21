"""Sensor-platform export: hourly scalars, alert log, sensor events and retained bursts.

Input: group outputs from the asset-group engine. Output: bronze cm_platform tables as Parquet
partitioned by month, and retained bursts as .npz files with an index table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from generate.asset_group import GroupOutput
from generate.provenance import RunContext


def _write_partitioned(df: pd.DataFrame, root: Path, ts_col: str, name: str) -> int:
    if df.empty:
        return 0
    months = df[ts_col].dt.strftime("%Y-%m")
    for m, part in df.groupby(months):
        d = root / f"month={m}"
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / f"{name}.parquet", index=False)
    return len(df)


def write_cm_platform(outputs: list[GroupOutput], bronze_root: Path, ctx: RunContext) -> dict[str, int]:
    root = bronze_root / "cm_platform"
    counts: dict[str, int] = {}
    scalars = pd.DataFrame([r for o in outputs for r in o.scalars])
    if scalars.empty:
        raise ValueError("no sensor rows in the window")
    scalars = scalars.sort_values(["point_id", "hour_utc", "axis"]).reset_index(drop=True)
    counts["scalars"] = _write_partitioned(scalars, root / "scalars", "hour_utc", "scalars")
    alerts = pd.DataFrame([a for o in outputs for a in o.alerts])
    if not alerts.empty:
        (root / "alerts").mkdir(parents=True, exist_ok=True)
        alerts.to_parquet(root / "alerts" / "alerts.parquet", index=False)
    counts["alerts"] = len(alerts)
    events = pd.DataFrame([e for o in outputs for e in o.sensor_events])
    events = events.assign(**ctx.stamp())
    (root / "sensor_events").mkdir(parents=True, exist_ok=True)
    events.to_parquet(root / "sensor_events" / "sensor_events.parquet", index=False)
    counts["sensor_events"] = len(events)
    index = []
    for o in outputs:
        for meta, arr in o.bursts:
            rel = Path("bursts") / meta["point_id"] / f"{meta['hour_utc']:%Y%m%dT%H}.npz"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, accel_g=arr, fs_hz=meta["fs_hz"])
            index.append({**meta, "path": rel.as_posix()})
    if index:
        pd.DataFrame(index).to_parquet(root / "burst_index.parquet", index=False)
    counts["bursts"] = len(index)
    return counts
