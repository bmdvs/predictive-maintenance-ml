"""SHDR log reading, the minute writer and the acceptance table."""

from __future__ import annotations

import gzip
from datetime import datetime

import pandas as pd
import pytest

from idc.checks.acceptance import format_table
from idc.replay.shdr_adapter import log_lines, shdr_lines


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2026-06-09T00:00:00Z", periods=4, freq="1s").as_unit("ns")
    return pd.DataFrame({
        "machine_id": "HMC-01", "ts_utc": ts, "execution": "active", "program": "O1001",
        "spindle_speed_rpm": [1200.0, 1200.0, 1250.5, 1250.5],
        "spindle_load_pct": 30.0, "x_load_pct": 1.0, "y_load_pct": 2.0, "z_load_pct": 3.0,
        "b_load_pct": 4.0, "spindle_temp_c": 35.25, "feed_override_pct": 100.0,
        "hydraulic_pressure_bar": 70.0, "coolant_pressure_bar": 5.5,
        "alarm_code": ["", "", "", ""],
    })


def test_log_lines_replays_the_adapter_stream(tmp_path):
    lines = [line for _, line in shdr_lines(_frame())]
    path = tmp_path / "2026-06-09.shdr.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as out:
        out.write("\n".join(lines) + "\n")
    replayed = list(log_lines(path))
    assert [line for _, line in replayed] == lines
    assert replayed[0][0] == datetime.fromisoformat("2026-06-09T00:00:00.000Z")
    assert all(a[0] <= b[0] for a, b in zip(replayed, replayed[1:], strict=False))


def test_minute_writer_matches_the_pipeline_aggregation(tmp_path):
    pytest.importorskip("pdm.ingest.mtconnect_stream")
    from pdm.ingest.mtconnect_stream import HZ1_COLUMNS, aggregate_minutes

    from idc.landing.minute_writer import build_minutes, write_minutes

    hz1 = _frame()
    hz1 = hz1[[c for c in HZ1_COLUMNS if c in hz1.columns]].copy()
    for column in HZ1_COLUMNS:
        if column not in hz1.columns:
            hz1[column] = float("nan")
    hz1 = hz1[HZ1_COLUMNS]
    hz1["run_id"] = "run-x"
    hz1["config_hash"] = "hash-y"
    expected = aggregate_minutes(hz1[HZ1_COLUMNS])
    expected["run_id"] = "run-x"
    expected["config_hash"] = "hash-y"
    pd.testing.assert_frame_equal(build_minutes(hz1), expected)

    source = tmp_path / "controller" / "hz1" / "month=2026-06" / "HMC-01.parquet"
    source.parent.mkdir(parents=True)
    hz1.to_parquet(source, index=False)
    path, rows = write_minutes(tmp_path, "HMC-01", "2026-06")
    assert rows == len(expected)
    pd.testing.assert_frame_equal(pd.read_parquet(path), expected)


def test_acceptance_table_reports_failures():
    results = [
        {"table": "ctrl_hz1", "path": "a", "ok": True, "rows_expected": 10, "rows_actual": 10,
         "detail": ""},
        {"table": "cmp_room", "path": "b", "ok": False, "rows_expected": None,
         "rows_actual": None, "detail": "landed file missing"},
    ]
    text = format_table(results)
    assert "PASS" in text and "FAIL" in text and "landed file missing" in text
    assert text.endswith("1 of 2 targets pass")
