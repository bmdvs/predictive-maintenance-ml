"""Implausible-row flags on hourly sensor scalars: temperature jumps, spikes and stuck rows.

Input: resultant scalar rows for one point in time order (measured rows only). Output: a frame
with a boolean flag and the rule that fired, aligned to the input index.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SPIKE_COLUMNS = ["accel_rms_g", "vel_rms_mm_s", "hf_rms_g"]


def indicator_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "point_id",
        "hour_utc",
        "axis",
        "measurement_state",
        "no_measurement",
        "run_id",
        "config_hash",
        "speed_rpm_at_measurement",
        "load_pct_at_measurement",
    }
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def flag_point(rows: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    q = cfg["quality"]
    r = rows.sort_values("hour_utc").set_index("hour_utc")
    window = f"{q['spike_reference_window_h']}h"
    prev_temp = r["temp_c"].shift(1).rolling(window, min_periods=1).median()
    temp_jump = (r["temp_c"] - prev_temp).abs() > q["temp_jump_c"]
    spike = pd.Series(False, index=r.index)
    for c in SPIKE_COLUMNS:
        ref = r[c].shift(1).rolling(window, min_periods=1).median()
        spike |= r[c] > q["spike_ratio"] * ref
    cols = indicator_columns(r.reset_index())
    vals = r[cols].to_numpy()
    same = np.zeros(len(r), dtype=bool)
    if len(r) > 1:
        same[1:] = np.all((vals[1:] == vals[:-1]) | (np.isnan(vals[1:]) & np.isnan(vals[:-1])), axis=1)
    rule = np.where(temp_jump, "temp_jump", np.where(spike, "spike", np.where(same, "stuck", "")))
    out = pd.DataFrame({"hour_utc": r.index, "implausible": rule != "", "rule": rule})
    out.index = rows.sort_values("hour_utc").index
    return out
