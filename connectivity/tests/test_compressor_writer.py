"""Compressor writer: slot reconstruction from Sparkplug messages, dedupe and landing."""

from __future__ import annotations

import pandas as pd

from idc.landing import sparkplug as sp
from idc.landing.compressor_writer import CompressorLanding, expected_gaps
from idc.replay import register_map as rm

MINUTE = pd.Timestamp("2026-06-01T00:07Z")


def _cmp_words(minute: pd.Timestamp, pressure: float = 6.85) -> list[int]:
    return rm.encode_block({"record_ts": minute, "state": "loaded",
                            "discharge_pressure_bar": pressure, "discharge_temp_c": -0.2,
                            "run_hours": 21101.25,
                            "snapshot": rm.minute_index(minute) & 0xFFFF}, rm.COMPRESSOR_MAP)


def _room_words(minute: pd.Timestamp) -> list[int]:
    return rm.encode_block({"record_ts": minute, "ambient_c": 30.1, "dryer_dew_point_c": -3.4,
                            "snapshot": rm.minute_index(minute) & 0xFFFF}, rm.ROOM_MAP) + \
        [0, 0, 0]


def _metrics(slot: int, words: list[int]) -> list[sp.Metric]:
    return [sp.Metric(f"s{slot:02d}_r{k}", None, None, sp.DATATYPE_INT32, w)
            for k, w in enumerate(words)]


def _birth(device: str) -> bytes:
    metrics = [sp.Metric(f"s{s:02d}_r{k}", None, None, sp.DATATYPE_INT32, 0)
               for s in range(rm.RING_SLOTS) for k in range(rm.SLOT_WIDTH)]
    return sp.encode(metrics, 1)


def _landing(tmp_path) -> CompressorLanding:
    return CompressorLanding(tmp_path, "run-x", "hash-y")


def test_slot_is_accepted_only_after_a_confirming_message(tmp_path):
    landing = _landing(tmp_path)
    slot = rm.ring_slot(MINUTE)
    landing.handle("spBv1.0/shop/DBIRTH/main/CMP-01", _birth("CMP-01"))
    landing.handle("spBv1.0/shop/DDATA/main/CMP-01",
                   sp.encode(_metrics(slot, _cmp_words(MINUTE)), 2))
    assert not landing.cmp_rows
    other = MINUTE + pd.Timedelta(minutes=1)
    landing.handle("spBv1.0/shop/DDATA/main/CMP-01",
                   sp.encode(_metrics(rm.ring_slot(other), _cmp_words(other)), 3))
    row = landing.cmp_rows[("CMP-01", MINUTE)]
    assert row == {"state": "loaded", "discharge_pressure_bar": 6.85,
                   "discharge_temp_c": -0.2, "run_hours": 21101.25}


def test_partial_slot_update_is_not_accepted(tmp_path):
    landing = _landing(tmp_path)
    slot = rm.ring_slot(MINUTE)
    landing.handle("spBv1.0/shop/DBIRTH/main/CMP-01", _birth("CMP-01"))
    words = _cmp_words(MINUTE)
    first_half = _metrics(slot, words)[:3] + [_metrics(slot, words)[7]]
    landing.handle("spBv1.0/shop/DDATA/main/CMP-01", sp.encode(first_half, 2))
    landing.handle("spBv1.0/shop/DDATA/main/CMP-01",
                   sp.encode(_metrics(slot, words)[3:7], 3))
    assert not landing.cmp_rows
    landing.handle("spBv1.0/shop/DDATA/main/CMP-01",
                   sp.encode(_metrics((slot + 1) % rm.RING_SLOTS, [0] * 8), 4))
    assert landing.cmp_rows[("CMP-01", MINUTE)]["discharge_pressure_bar"] == 6.85


def test_inconsistent_snapshot_is_rejected(tmp_path):
    landing = _landing(tmp_path)
    slot = rm.ring_slot(MINUTE)
    words = _cmp_words(MINUTE)
    words[7] = (words[7] + 1) & 0xFFFF
    landing.handle("spBv1.0/shop/DBIRTH/main/CMP-01", _birth("CMP-01"))
    for seq in (2, 3):
        landing.handle("spBv1.0/shop/DDATA/main/CMP-01", sp.encode(_metrics(slot, words), seq))
    assert not landing.cmp_rows


def test_data_before_birth_requests_rebirth(tmp_path):
    landing = _landing(tmp_path)
    slot = rm.ring_slot(MINUTE)
    node = landing.handle("spBv1.0/shop/DDATA/main/CMP-02",
                          sp.encode(_metrics(slot, _cmp_words(MINUTE)), 1))
    assert node == "main"
    assert landing.stats.data_before_birth == 1


def test_room_and_landing_files_have_export_dtypes(tmp_path, export_root):
    landing = _landing(tmp_path)
    slot = rm.ring_slot(MINUTE)
    landing.handle("spBv1.0/shop/DBIRTH/main/room", _birth("room"))
    landing.handle("spBv1.0/shop/DDATA/main/room",
                   sp.encode(_metrics(slot, _room_words(MINUTE)), 2))
    landing.handle("spBv1.0/shop/DDATA/main/room", sp.encode([], 3))
    landing.handle("spBv1.0/shop/DBIRTH/main/CMP-01", _birth("CMP-01"))
    landing.handle("spBv1.0/shop/DDATA/main/CMP-01",
                   sp.encode(_metrics(slot, _cmp_words(MINUTE)), 4))
    landing.handle("spBv1.0/shop/DDATA/main/CMP-01", sp.encode([], 5))
    written = landing.write_all()
    assert len(written) == 2
    room = pd.read_parquet(tmp_path / "compressor_ctrl" / "room" / "month=2026-06" /
                           "room.parquet")
    assert list(room.columns) == ["minute_utc", "ambient_c", "dryer_dew_point_c", "run_id",
                                  "config_hash"]
    export_room = pd.read_parquet(export_root / "compressor_ctrl" / "room" / "month=2026-06" /
                                  "room.parquet")
    assert room.dtypes.astype(str).to_dict() == export_room.dtypes.astype(str).to_dict()
    assert room.loc[0, "dryer_dew_point_c"] == -3.4
    cmp = pd.read_parquet(tmp_path / "compressor_ctrl" / "minute" / "month=2026-06" /
                          "compressor_minute.parquet")
    assert list(cmp.columns)[:3] == ["compressor_id", "minute_utc", "state"]
    export_cmp = pd.read_parquet(export_root / "compressor_ctrl" / "minute" / "month=2026-06" /
                                 "compressor_minute.parquet")
    assert cmp.dtypes.astype(str).to_dict() == export_cmp.dtypes.astype(str).to_dict()
    landing.write_all()
    again = pd.read_parquet(tmp_path / "compressor_ctrl" / "minute" / "month=2026-06" /
                            "compressor_minute.parquet")
    pd.testing.assert_frame_equal(cmp, again)


def test_expected_gaps_lists_missing_minutes():
    start = pd.Timestamp("2026-06-01T00:00Z")
    landed = {"CMP-01": {start + pd.Timedelta(minutes=i) for i in (0, 1, 3)}}
    gaps = expected_gaps(landed, ["CMP-01", "room"], start, 4)
    assert gaps["CMP-01"] == [(start + pd.Timedelta(minutes=2)).isoformat()]
    assert len(gaps["room"]) == 4
