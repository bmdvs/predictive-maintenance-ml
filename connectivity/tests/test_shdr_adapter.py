"""Unit tests for the SHDR replay adapter on small in-memory frames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from idc.replay import shdr_adapter as sa


def _frame(seconds, **overrides) -> pd.DataFrame:
    ts = pd.to_datetime(seconds, utc=True).as_unit("ns")
    n = len(ts)
    base = {
        "machine_id": ["HMC-01"] * n,
        "ts_utc": ts,
        "execution": ["active"] * n,
        "program": ["HMC-01-P0137"] * n,
        "spindle_speed_rpm": [8723.0] * n,
        "spindle_load_pct": [4.8] * n,
        "x_load_pct": [13.8] * n,
        "y_load_pct": [6.3] * n,
        "z_load_pct": [21.6] * n,
        "b_load_pct": [6.4] * n,
        "spindle_temp_c": [35.0] * n,
        "feed_override_pct": [80.0] * n,
        "hydraulic_pressure_bar": [69.7] * n,
        "coolant_pressure_bar": [68.7] * n,
        "alarm_code": [""] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_format_value_round_trips_floats():
    for v in [4.8, 69.6, 6.85, 8723.0, 0.0, 70.3, 33.2, 2.0]:
        text = sa.format_value("spindle_load_pct", v)
        assert float(text) == v, (v, text)
    assert sa.format_value("spindle_speed_rpm", 8723.0) == "8723"
    assert sa.format_value("spindle_load_pct", 2.0) == "2"
    assert sa.format_value("y_load_pct", float("nan")) is None
    assert sa.format_value("execution", "interrupted") == "INTERRUPTED"


def test_first_line_sends_everything_then_only_changes():
    f = _frame(["2026-06-09 21:31:08", "2026-06-09 21:31:09", "2026-06-09 21:31:10"])
    f.loc[2, "spindle_load_pct"] = 5.1
    lines = [line for _, line in sa.shdr_lines(f)]
    assert lines[0].endswith("|avail|AVAILABLE")
    assert lines[1].count("|") == 2 + 2 * 12  # record_ts plus 12 signals (no NaN here)
    assert lines[2] == "2026-06-09T21:31:09.000Z|record_ts|2026-06-09T21:31:09.000Z"
    assert lines[3] == "2026-06-09T21:31:10.000Z|record_ts|2026-06-09T21:31:10.000Z|Sload|5.1"
    assert lines[-1] == "2026-06-09T21:31:11.000Z|avail|UNAVAILABLE"


def test_nan_signals_are_omitted():
    f = _frame(["2026-06-09 21:31:08"], y_load_pct=[np.nan], b_load_pct=[np.nan],
               hydraulic_pressure_bar=[np.nan], coolant_pressure_bar=[np.nan])
    lines = [line for _, line in sa.shdr_lines(f)]
    assert "Yload" not in lines[1] and "Bload" not in lines[1]
    assert "Hpres" not in lines[1] and "Cpres" not in lines[1]
    assert "Xload|13.8" in lines[1]


def test_segment_break_emits_unavailable_and_resends_state():
    f = _frame(["2026-06-09 03:59:59", "2026-06-09 11:00:00", "2026-06-09 11:00:01"])
    stats = sa.ReplayStats()
    lines = [line for _, line in sa.shdr_lines(f, stats)]
    assert stats.segments == 2
    assert lines[2] == "2026-06-09T04:00:00.000Z|avail|UNAVAILABLE"
    assert lines[3] == "2026-06-09T11:00:00.000Z|avail|AVAILABLE"
    # Segment start resends every value even though none changed.
    assert lines[4].count("|") == lines[1].count("|")


def test_alarm_is_fault_then_normal():
    f = _frame(["2026-06-09 21:31:09", "2026-06-09 21:31:10", "2026-06-09 21:31:11"],
               alarm_code=["", "atc_fault", ""])
    stats = sa.ReplayStats()
    lines = [line for _, line in sa.shdr_lines(f, stats)]
    assert "2026-06-09T21:31:10.000Z|alarm|FAULT|atc_fault|||atc_fault" in lines
    assert "2026-06-09T21:31:11.000Z|alarm|NORMAL||||" in lines
    assert stats.alarms == 1
    fault_i = lines.index("2026-06-09T21:31:10.000Z|alarm|FAULT|atc_fault|||atc_fault")
    normal_i = lines.index("2026-06-09T21:31:11.000Z|alarm|NORMAL||||")
    assert fault_i < normal_i


def test_record_count_and_timestamp_format():
    f = _frame(["2026-06-09 21:31:08", "2026-06-09 21:31:09"])
    stats = sa.ReplayStats()
    list(sa.shdr_lines(f, stats))
    assert stats.records == 2
    assert sa.format_timestamp(f["ts_utc"].iloc[0]) == "2026-06-09T21:31:08.000Z"


class _Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


def _current(sequence: int, timestamp: str) -> str:
    return ('<MTConnectStreams><Header nextSequence="900"/><Availability dataItemId="a" '
            f'name="avail" sequence="{sequence}" timestamp="{timestamp}">UNAVAILABLE'
            '</Availability></MTConnectStreams>')


def test_agent_has_matches_the_final_line_and_ignores_older_observations(monkeypatch):
    final = "2026-05-01T04:00:00.000Z|avail|UNAVAILABLE"
    monkeypatch.setattr(sa.requests, "get",
                        lambda *a, **k: _Response(_current(500, "2026-05-01T04:00:00Z")))
    assert sa.agent_has("http://agent", "HMC-01", final)
    assert not sa.agent_has("http://agent", "HMC-01", final, min_sequence=501)
    assert not sa.agent_has("http://agent", "HMC-01",
                            "2026-05-01T03:59:59.000Z|avail|UNAVAILABLE")
    assert sa.agent_next_sequence("http://agent", "HMC-01") == 900


def _serve_once(confirm, timeout):
    import asyncio
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    lines = [(None, "2026-06-09T11:00:00.000Z|avail|AVAILABLE"),
             (None, "2026-06-09T11:00:01.000Z|avail|UNAVAILABLE")]

    async def scenario():
        task = asyncio.create_task(sa.serve(lines, port, 0, host="127.0.0.1", confirm=confirm,
                                            confirm_timeout=timeout))
        for _ in range(50):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                break
            except OSError:
                await asyncio.sleep(0.05)
        received = b""
        while chunk := await reader.read(4096):
            received += chunk
        writer.close()
        return await task, received

    return asyncio.run(scenario())


def test_serve_closes_only_after_the_agent_confirms():
    calls = []

    def confirm(final_line):
        calls.append(final_line)
        return len(calls) >= 2

    stats, received = _serve_once(confirm, timeout=10)
    assert stats.confirmed is True
    assert calls[0] == "2026-06-09T11:00:01.000Z|avail|UNAVAILABLE" and len(calls) == 2
    assert received.endswith(b"2026-06-09T11:00:01.000Z|avail|UNAVAILABLE\n")


def test_serve_reports_an_unconfirmed_stream():
    stats, _ = _serve_once(lambda final_line: False, timeout=0.5)
    assert stats.confirmed is False
