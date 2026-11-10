"""Unit tests for the export comparator on small in-memory frames.

These tests never touch the real exports. They establish that the comparator accepts exactly
equal frames and rejects each kind of difference the chain could introduce: a changed value,
a dropped or duplicated row, a reordered column, a float that differs in the last bit, a
timestamp stored at a different unit, and a different row order (which must be accepted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idc.checks import compare_exports as ce


def _hz1_frame() -> pd.DataFrame:
    # The bronze exports store nanosecond timestamps; pandas 3 defaults to microseconds, so
    # the fixture sets the unit explicitly, exactly as the landing writers must.
    ts = pd.date_range("2026-06-09 21:31:07", periods=6, freq="s", tz="UTC").as_unit("ns")
    return pd.DataFrame(
        {
            "machine_id": ["HMC-01"] * 6,
            "ts_utc": ts,
            "execution": ["active"] * 3 + ["interrupted"] * 3,
            "program": ["HMC-01-P0137"] * 6,
            "spindle_speed_rpm": [8710.0, 8675.0, 8723.0, 0.0, 0.0, 0.0],
            "spindle_load_pct": [4.5, 5.6, 4.8, 2.0, 2.0, 2.0],
            "y_load_pct": [np.nan] * 6,
            "alarm_code": ["", "", "", "atc_fault", "", ""],
            "run_id": ["run09"] * 6,
            "config_hash": ["abc"] * 6,
        }
    )


def test_equal_frames_pass():
    a = _hz1_frame()
    assert ce.compare_frames(a, a.copy()) is None


def test_row_order_is_normalized(tmp_path):
    a = _hz1_frame()
    shuffled = a.sample(frac=1.0, random_state=3)
    p_expected, p_actual = tmp_path / "e.parquet", tmp_path / "a.parquet"
    a.to_parquet(p_expected, index=False)
    shuffled.to_parquet(p_actual, index=False)
    result = ce.compare_files("ctrl_hz1", p_expected, p_actual)
    assert result.ok, result.detail


def test_changed_value_fails():
    a = _hz1_frame()
    b = a.copy()
    b.loc[2, "spindle_load_pct"] = 4.9
    detail = ce.compare_frames(a, b)
    assert detail is not None and "spindle_load_pct" in detail and "row 2" in detail


def test_last_bit_float_difference_fails():
    a = _hz1_frame()
    b = a.copy()
    # 696 * 0.1 is 69.60000000000001, not the float64 that float("69.6") or 696 / 10 gives;
    # a writer that scales by multiplying would land a value the comparator must reject.
    a.loc[0, "spindle_speed_rpm"] = 69.6
    b.loc[0, "spindle_speed_rpm"] = 696 * 0.1
    assert ce.compare_frames(a, b) is not None


def test_nan_equals_nan():
    a = _hz1_frame()
    assert ce.compare_frames(a, a.copy()) is None
    b = a.copy()
    b.loc[0, "y_load_pct"] = 3.0
    detail = ce.compare_frames(a, b)
    assert detail is not None and "y_load_pct" in detail


def test_missing_row_fails():
    a = _hz1_frame()
    b = a.iloc[:-1].reset_index(drop=True)
    detail = ce.compare_frames(a, b)
    assert detail is not None and "row count" in detail


def test_column_order_fails():
    a = _hz1_frame()
    b = a[[c for c in a.columns if c != "program"] + ["program"]]
    detail = ce.compare_frames(a, b)
    assert detail is not None and "column order" in detail


def test_timestamp_unit_fails():
    a = _hz1_frame()
    b = a.copy()
    b["ts_utc"] = b["ts_utc"].astype("datetime64[us, UTC]")
    detail = ce.compare_frames(a, b)
    assert detail is not None and "dtypes differ" in detail


def test_empty_string_is_not_nan():
    a = _hz1_frame()
    b = a.copy()
    b.loc[0, "alarm_code"] = None
    assert ce.compare_frames(a, b) is not None


def test_recompute_minutes_matches_pipeline_aggregation(tmp_path):
    pdm = pytest.importorskip("pdm.ingest.mtconnect_stream")
    # Build an hz1 frame with the full column set the pipeline expects.
    ts = pd.date_range("2026-06-09 21:30:00", periods=120, freq="s", tz="UTC").as_unit("ns")
    n = len(ts)
    hz = pd.DataFrame(
        {
            "machine_id": ["HMC-01"] * n,
            "ts_utc": ts,
            "execution": ["active"] * 90 + ["ready"] * 30,
            "program": ["HMC-01-P0137"] * n,
            "spindle_speed_rpm": [8700.0] * 90 + [0.0] * 30,
            "spindle_load_pct": np.round(np.linspace(4.0, 6.0, n), 1),
            "x_load_pct": [12.0] * n,
            "y_load_pct": [7.0] * n,
            "z_load_pct": [19.0] * n,
            "b_load_pct": [6.0] * n,
            "spindle_temp_c": [35.0] * n,
            "feed_override_pct": [80.0] * n,
            "hydraulic_pressure_bar": [70.0] * n,
            "coolant_pressure_bar": [68.0] * 90 + [0.0] * 30,
            "alarm_code": [""] * n,
            "run_id": ["run09"] * n,
            "config_hash": ["abc"] * n,
        }
    )
    hz = hz[pdm.HZ1_COLUMNS + ["run_id", "config_hash"]]
    path = tmp_path / "HMC-01.parquet"
    hz.to_parquet(path, index=False)
    recomputed = ce.recompute_minutes(path)
    expected = pdm.aggregate_minutes(hz[pdm.HZ1_COLUMNS])
    expected["run_id"] = "run09"
    expected["config_hash"] = "abc"
    assert ce.compare_frames(expected, recomputed) is None
    assert set(recomputed["execution"]) == {"active", "ready"}
