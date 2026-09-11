"""Compressor controller export (Modbus, one-minute grain) and compressor-room conditions.

Inputs: hourly compressor running state and load from the compressor engine and the room ambient.
Outputs: bronze compressor_ctrl tables: per compressor per minute load/unload state, discharge
pressure, discharge temperature and run hours; per minute room ambient and dryer dew point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate.provenance import RunContext, rng_for
from pdm.constants import SECONDS_PER_MINUTE

MINUTES_PER_HOUR = int(SECONDS_PER_MINUTE)


def render_compressors(
    cfg: dict[str, Any],
    reg: Any,
    ctx: RunContext,
    air: pd.DataFrame,
    running: dict[int, dict[str, bool]],
    bronze_root: Path,
) -> dict[str, int]:
    cc, comp = cfg["compressor_ctrl"], cfg["compressor"]
    lo, hi = comp["load_unload_band_bar"]
    dt = comp["discharge_temp_c"]
    amb_mean = cfg["benign"]["ambient_room_c"]["annual_mean"]
    stamp = ctx.stamp()
    root = bronze_root / "compressor_ctrl"
    hours = air["hour_utc"]
    naive = hours.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    minute_ts = (naive[:, None] + (np.arange(MINUTES_PER_HOUR) * np.timedelta64(1, "m"))[None, :]).ravel()
    room_h = air["room_c"].to_numpy()
    room_m = np.interp(np.arange(minute_ts.size) / MINUTES_PER_HOUR, np.arange(room_h.size), room_h)
    counts: dict[str, int] = {}
    frames = []
    for cid, c in reg.compressors.items():
        rng = rng_for(ctx.run_seed, "compressor_ctrl", cid)
        run_h = np.array([running[i][cid] for i in range(len(air))], dtype=bool)
        load = air[f"{cid}_load_frac"].to_numpy()
        run_m = np.repeat(run_h, MINUTES_PER_HOUR)
        load_m = np.repeat(
            np.where(run_h, np.where(load > 0, load, air["demand_frac"].to_numpy()), 0.0), MINUTES_PER_HOUR
        )
        cyc = cc["load_cycle_minutes"]
        pos = np.arange(minute_ts.size) % cyc
        loaded = run_m & (pos < np.round(load_m * cyc))
        phase = np.where(
            loaded,
            pos / np.maximum(np.round(load_m * cyc), 1),
            (pos - np.round(load_m * cyc)) / np.maximum(cyc - np.round(load_m * cyc), 1),
        )
        pressure = np.where(loaded, lo + (hi - lo) * phase, hi - (hi - lo) * phase)
        pressure = np.where(
            run_m, pressure + cc["discharge_pressure_noise_bar"] * rng.standard_normal(pressure.size), 0.0
        )
        temp = np.where(
            run_m,
            dt["base"]
            + dt["per_c_ambient"] * (room_m - amb_mean)
            + dt["per_load_frac"] * load_m
            + cc["discharge_temp_noise_c"] * rng.standard_normal(pressure.size),
            room_m,
        )
        run_hours = c["airend_running_h_at_record_start"] + np.cumsum(run_m) / MINUTES_PER_HOUR
        state = np.where(~run_m, "off", np.where(loaded, "loaded", "unloaded"))
        frames.append(
            pd.DataFrame(
                {
                    "compressor_id": cid,
                    "minute_utc": pd.DatetimeIndex(minute_ts).tz_localize("UTC"),
                    "state": state,
                    "discharge_pressure_bar": np.round(pressure, 2),
                    "discharge_temp_c": np.round(temp, 1),
                    "run_hours": np.round(run_hours, 2),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True).assign(**stamp)
    _write_monthly(df, root / "minute", "minute_utc", "compressor_minute")
    counts["compressor_minutes"] = len(df)
    dew = cc["dryer_dew_point_c"]
    rng = rng_for(ctx.run_seed, "dryer")
    room = pd.DataFrame(
        {
            "minute_utc": frames[0]["minute_utc"],
            "ambient_c": np.round(room_m, 1),
            "dryer_dew_point_c": np.round(
                dew["nominal"]
                + dew["sd"] * rng.standard_normal(room_m.size)
                + dew["per_c_ambient_above_threshold"]
                * np.clip(room_m - dew["ambient_threshold_c"], 0, None),
                1,
            ),
        }
    ).assign(**stamp)
    _write_monthly(room, root / "room", "minute_utc", "room")
    counts["room_minutes"] = len(room)
    return counts


def _write_monthly(df: pd.DataFrame, root: Path, ts_col: str, name: str) -> None:
    months = df[ts_col].dt.strftime("%Y-%m")
    for m, part in df.groupby(months):
        d = root / f"month={m}"
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / f"{name}.parquet", index=False)
