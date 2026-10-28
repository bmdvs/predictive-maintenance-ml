"""MTConnect-style 1 Hz controller stream to per-minute aggregates by machine, program and
execution state.

Input: 1 Hz rows (machine_id, ts_utc, execution, program and signal columns). Output: one row per
(machine, minute, program, execution) with seconds in state, mean/max/p95 of each load and the
mean of the other signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOAD_COLUMNS = ["spindle_load_pct", "x_load_pct", "y_load_pct", "z_load_pct", "b_load_pct"]
MEAN_COLUMNS = [
    "spindle_speed_rpm",
    "spindle_temp_c",
    "feed_override_pct",
    "hydraulic_pressure_bar",
    "coolant_pressure_bar",
]
EXECUTION_STATES = ["active", "ready", "stopped", "interrupted"]
HZ1_COLUMNS = [
    "machine_id",
    "ts_utc",
    "execution",
    "program",
    *MEAN_COLUMNS[:1],
    *LOAD_COLUMNS,
    *MEAN_COLUMNS[1:],
    "alarm_code",
]
AGG_DECIMALS = 3
P95 = 0.95


def aggregate_minutes(hz: pd.DataFrame) -> pd.DataFrame:
    """Per-minute aggregates; minutes split by program and execution state."""
    df = hz.assign(minute_utc=hz["ts_utc"].dt.floor("min"))
    keys = ["machine_id", "minute_utc", "program", "execution"]
    g = df.groupby(keys, sort=True, observed=True, dropna=False)
    out = g.size().rename("seconds_in_state").to_frame()
    for c in LOAD_COLUMNS:
        out[f"{c}_mean"] = g[c].mean()
        out[f"{c}_max"] = g[c].max()
        out[f"{c}_p95"] = g[c].quantile(P95)
    for c in MEAN_COLUMNS:
        out[f"{c}_mean"] = g[c].mean()
    out = out.reset_index()
    num = out.select_dtypes(include=[np.floating]).columns
    out[num] = out[num].round(AGG_DECIMALS)
    return out


def compare_aggregates(stored: pd.DataFrame, recomputed: pd.DataFrame, tolerance: float) -> dict[str, float]:
    """Row match and largest absolute difference between stored and recomputed aggregates."""
    keys = ["machine_id", "minute_utc", "program", "execution"]
    m = stored.merge(recomputed, on=keys, how="outer", suffixes=("_s", "_r"), indicator=True)
    both = m[m["_merge"] == "both"]
    cols = [c[:-2] for c in m.columns if c.endswith("_s")]
    diffs = [
        float(np.nanmax(np.abs(both[f"{c}_s"] - both[f"{c}_r"])))
        for c in cols
        if len(both) and both[f"{c}_s"].notna().any()
    ]
    worst = max(diffs) if diffs else 0.0
    return {
        "rows_stored": float(len(stored)),
        "rows_recomputed": float(len(recomputed)),
        "rows_unmatched": float((m["_merge"] != "both").sum()),
        "max_abs_diff": worst,
        "within_tolerance": float(worst <= tolerance),
    }
