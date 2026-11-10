"""Unit tests for the controller landing writer, including an in-memory run of the whole
chain: hz1 rows -> SHDR adapter -> agent observations -> bridge groups -> writer -> hz1 rows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from idc.bridge import mtconnect_to_mqtt as br
from idc.landing import controller_writer as cw
from idc.replay import shdr_adapter as sa

RUN_ID = "run09_20261030T090522Z_3eced8fb"
CONFIG_HASH = "3eced8fb"
TOPIC = "shop/main/machining/cell-hmc/HMC-01/controller"


def _hz1(seconds, **overrides) -> pd.DataFrame:
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
    frame = pd.DataFrame(base)
    for c in cw.STRING_COLUMNS - {"run_id", "config_hash"}:
        frame[c] = frame[c].astype("str")
    frame["run_id"] = pd.array([RUN_ID] * n, dtype="str")
    frame["config_hash"] = pd.array([CONFIG_HASH] * n, dtype="str")
    return frame


def _record(ts: str, **fields) -> dict:
    body = {"machine_id": "HMC-01", "ts_utc": ts}
    body.update(fields)
    body.update({"agent_seq": [1, 1], "instance_id": 7})
    return body


def _landing(tmp_path) -> cw.Landing:
    return cw.Landing(tmp_path, RUN_ID, CONFIG_HASH)


def test_unchanged_values_are_carried_forward_and_alarm_is_not():
    land = _landing(None)
    land.handle(TOPIC, _record("2026-06-09T21:31:08.000Z", avail="AVAILABLE", execution="active",
                               program="P1", spindle_load_pct=4.8, x_load_pct=13.8))
    land.handle(TOPIC, _record("2026-06-09T21:31:09.000Z", spindle_load_pct=5.1,
                               alarm_code="atc_fault"))
    land.handle(TOPIC, _record("2026-06-09T21:31:10.000Z", alarm_code=""))
    land.handle(TOPIC, _record("2026-06-09T21:31:11.000Z"))
    rows = list(land.rows.values())
    assert [r["spindle_load_pct"] for r in rows] == [4.8, 5.1, 5.1, 5.1]
    assert [r["x_load_pct"] for r in rows] == [13.8] * 4
    assert [r["alarm_code"] for r in rows] == ["", "atc_fault", "", ""]
    assert all(np.isnan(r["y_load_pct"]) for r in rows)  # never sent, so NaN
    assert land.stats.gaps == []


def test_values_persist_across_a_segment_break():
    # The agent filters duplicates, so a value unchanged across a shutdown is never re-sent:
    # execution stays "ready" and spindle_load 2.0 here without appearing in the 11:00 record.
    land = _landing(None)
    land.handle(TOPIC, _record("2026-06-09T03:59:58.000Z", avail="AVAILABLE", execution="ready",
                               program="P1", spindle_load_pct=2.0))
    land.handle(TOPIC, _record("2026-06-09T03:59:59.000Z"))
    land.handle(TOPIC + "/status", {"machine_id": "HMC-01", "ts_utc": "2026-06-09T04:00:00.000Z",
                                    "avail": "UNAVAILABLE", "source": "adapter",
                                    "agent_seq": [9, 9], "instance_id": 7})
    land.handle(TOPIC, _record("2026-06-09T11:00:00.000Z", avail="AVAILABLE", program="P2"))
    assert land.stats.segments == 2
    assert land.stats.status == 1
    assert land.stats.gaps == []  # the seven-hour break is a segment boundary, not a gap
    row = list(land.rows.values())[-1]
    assert row["program"] == "P2"
    assert row["execution"] == "ready" and row["spindle_load_pct"] == 2.0


def test_missing_second_inside_segment_is_a_gap():
    land = _landing(None)
    land.handle(TOPIC, _record("2026-06-09T21:31:08.000Z", avail="AVAILABLE", execution="active",
                               program="P1"))
    land.handle(TOPIC, _record("2026-06-09T21:31:11.000Z"))
    assert len(land.stats.gaps) == 1
    assert land.stats.gaps[0].missing_seconds == 2


def test_agent_link_loss_resets_state_without_a_gap():
    land = _landing(None)
    land.handle(TOPIC, _record("2026-06-09T21:31:08.000Z", avail="AVAILABLE", execution="active",
                               program="P1", spindle_load_pct=4.8))
    land.handle(TOPIC + "/status", {"machine_id": "HMC-01", "ts_utc": "2026-11-08T22:38:30.0223Z",
                                    "avail": "UNAVAILABLE", "source": "agent",
                                    "agent_seq": [20, 33], "instance_id": 7})
    assert land.stats.link_losses == 1 and land.stats.gaps == []
    assert land.states["HMC-01"].values == {}


def test_duplicate_message_lands_once():
    land = _landing(None)
    msg = _record("2026-06-09T21:31:08.000Z", avail="AVAILABLE", execution="active", program="P1")
    land.handle(TOPIC, msg)
    land.handle(TOPIC, msg)
    assert len(land.rows) == 1 and land.stats.duplicates == 1


def test_frame_matches_export_dtypes_through_parquet(tmp_path, export_root):
    land = _landing(tmp_path)
    land.handle(TOPIC, _record("2026-06-09T21:31:08.000Z", avail="AVAILABLE", execution="active",
                               program="P1", spindle_load_pct=4.8))
    (path,) = land.write_all()
    back = pd.read_parquet(path)
    assert list(back.columns) == cw.FILE_COLUMNS
    export = pd.read_parquet(export_root / "controller" / "hz1" / "month=2026-06" /
                             "HMC-01.parquet")
    assert back.dtypes.astype(str).to_dict() == export.dtypes.astype(str).to_dict()
    assert str(back["y_load_pct"].dtype) == "float64" and back["y_load_pct"].isna().all()
    assert path.name == "HMC-01.parquet" and path.parent.name == "month=2026-06"


def test_rewrite_merges_with_existing_file_on_key(tmp_path):
    first = _landing(tmp_path)
    first.handle(TOPIC, _record("2026-06-09T21:31:08.000Z", avail="AVAILABLE", execution="active",
                                program="P1", spindle_load_pct=4.8))
    first.write_all()
    second = _landing(tmp_path)
    second.handle(TOPIC, _record("2026-06-09T21:31:08.000Z", avail="AVAILABLE",
                                 execution="active", program="P1", spindle_load_pct=4.9))
    second.handle(TOPIC, _record("2026-06-09T21:31:09.000Z"))
    (path,) = second.write_all()
    back = pd.read_parquet(path)
    assert len(back) == 2
    assert back["spindle_load_pct"].tolist() == [4.9, 4.9]  # latest write wins on the key


# --- in-memory chain -------------------------------------------------------------------------

def _agent(lines: list[str]) -> list[br.Observation]:
    """Turn SHDR lines into the observations an agent would buffer, in arrival order.

    Like cppagent, a value equal to the data item's current value is filtered out, so the
    adapter's full resend at a segment start yields observations only for values that changed
    during the break.
    """
    obs: list[br.Observation] = []
    current: dict[str, str] = {}
    seq = 100
    for line in lines:
        parts = line.split("|")
        ts = parts[0]
        if parts[1] == "alarm":
            level = parts[2]
            obs.append(br.Observation("HMC-01", "alarm", seq, ts, level, "condition",
                                      level=level, native_code=parts[3] or None))
            seq += 1
            continue
        for key, value in zip(parts[1::2], parts[2::2], strict=True):
            if key != "record_ts" and current.get(key) == value:
                continue
            current[key] = value
            kind = "sample" if key in {"Srpm", "Sload", "Xload", "Yload", "Zload", "Bload",
                                       "Stemp", "Hpres", "Cpres"} else "event"
            obs.append(br.Observation("HMC-01", key, seq, ts, value, kind))
            seq += 1
    return obs


def _chain(frame: pd.DataFrame, tmp_path) -> pd.DataFrame:
    lines = [line for _, line in sa.shdr_lines(frame)]
    groups = br.collapse_status(br.group_by_second(_agent(lines)))
    land = _landing(tmp_path)
    for g in groups:
        if g.is_record:
            land.handle(TOPIC, g.payload(7))
        else:
            land.handle(TOPIC + "/status", g.status_payload(7))
    (path,) = land.write_all()
    assert land.stats.gaps == []
    return pd.read_parquet(path)


def test_chain_reproduces_rows_exactly(tmp_path):
    frame = _hz1(["2026-06-09 21:31:08", "2026-06-09 21:31:09", "2026-06-09 21:31:10",
                  "2026-06-09 21:31:11"])
    frame.loc[1, "spindle_load_pct"] = 5.1
    frame.loc[2, "alarm_code"] = "atc_fault"
    frame.loc[2, "execution"] = "interrupted"
    frame.loc[3, "hydraulic_pressure_bar"] = 69.6   # 696 * 0.1 != 69.6: must round-trip
    frame.loc[3, "coolant_pressure_bar"] = 6.85
    landed = _chain(frame, tmp_path)
    assert_frame_equal(frame, landed, check_exact=True)


def test_chain_reproduces_segment_break_and_nan_columns(tmp_path):
    frame = _hz1(["2026-06-09 03:59:58", "2026-06-09 03:59:59", "2026-06-09 11:00:00",
                  "2026-06-09 11:00:01"],
                 y_load_pct=[np.nan] * 4, b_load_pct=[np.nan] * 4,
                 hydraulic_pressure_bar=[np.nan] * 4, coolant_pressure_bar=[np.nan] * 4)
    frame.loc[2, "program"] = "HMC-01-P0140"
    frame.loc[2, "execution"] = "ready"
    frame.loc[3, "execution"] = "active"
    landed = _chain(frame, tmp_path)
    assert_frame_equal(frame, landed, check_exact=True)


def test_hz1_columns_match_pipeline_contract():
    pytest.importorskip("pdm")
    from pdm.ingest.mtconnect_stream import HZ1_COLUMNS
    assert list(HZ1_COLUMNS) == cw.HZ1_COLUMNS


def test_subscribe_patterns():
    config = {"mqtt": {"topic_root": "shop/main",
                       "controller_topic": "{root}/machining/{cell}/{machine_id}/controller"}}
    assert cw.subscribe_topics(config) == ["shop/main/machining/+/+/controller",
                                           "shop/main/machining/+/+/controller/status"]


def test_report_is_json(tmp_path):
    stats = cw.WriterStats(messages=3, records=2, status=1)
    path = cw.write_report(tmp_path, stats, [])
    body = json.loads(path.read_text())
    assert body["records"] == 2 and body["gaps"] == []


def test_partition_month_uses_the_plant_time_zone():
    late = pd.Timestamp("2026-05-01T03:59:59Z")
    assert cw.partition_month(late, "America/Chicago") == "2026-04"
    assert cw.partition_month(late, "UTC") == "2026-05"
    assert cw.partition_month(pd.Timestamp("2026-05-01T11:00:00Z"), "America/Chicago") == \
        "2026-05"


def test_rows_after_utc_midnight_land_in_the_plant_local_month(tmp_path):
    landing = cw.Landing(tmp_path, RUN_ID, CONFIG_HASH, "America/Chicago")
    landing.handle(TOPIC, _record("2026-04-30T23:59:59Z", avail="AVAILABLE",
                                  execution="active", program="P1"))
    landing.handle(TOPIC, _record("2026-05-01T00:00:00Z"))
    landing.handle(TOPIC, _record("2026-05-01T11:00:00Z", avail="AVAILABLE"))
    assert landing.partitions() == [("HMC-01", "2026-04"), ("HMC-01", "2026-05")]
    assert len(landing.frame("HMC-01", "2026-04")) == 2
    assert len(landing.frame("HMC-01", "2026-05")) == 1


def test_unreadable_record_is_rejected_without_changing_state(tmp_path):
    landing = _landing(tmp_path)
    landing.handle(TOPIC, _record("2026-06-09T11:00:00Z", avail="AVAILABLE",
                                  execution="active", program="P1"))
    landing.handle(TOPIC, _record("2026-04* ADAPTERVERSION: 2.7"))
    landing.handle_raw(TOPIC, b"not json")
    landing.handle(TOPIC, _record("2026-06-09T11:00:01Z"))
    s = landing.stats
    assert s.rejected == 2 and len(s.rejected_samples) == 2
    assert s.gaps == [] and len(landing.rows) == 2
