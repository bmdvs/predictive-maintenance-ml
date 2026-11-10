"""Ring-buffer layout, register images and the ring-buffer server's slot writes."""

from __future__ import annotations

import pandas as pd

from idc.replay import build_native as native
from idc.replay import register_map as rm
from idc.replay.modbus_server import RingBuffers

UNIT_IDS = {"CMP-01": 1, "CMP-02": 2, "CMP-03": 3, "CMP-04": 4, "room": 5}


def _frames(minutes: int = 20):
    times = pd.date_range("2026-06-01T00:00Z", periods=minutes, freq="1min").as_unit("ns")
    cmp_rows = []
    for device in ("CMP-01", "CMP-02", "CMP-03", "CMP-04"):
        for i, t in enumerate(times):
            cmp_rows.append({"compressor_id": device, "minute_utc": t,
                             "state": ["off", "loaded", "unloaded"][i % 3],
                             "discharge_pressure_bar": 6.85 if i % 3 == 1 else 0.0,
                             "discharge_temp_c": round(-0.2 + i / 10, 1),
                             "run_hours": round(21101.0 + i / 100, 2)})
    cmp = pd.DataFrame(cmp_rows)
    room = pd.DataFrame({"minute_utc": times, "ambient_c": [round(30.1 - i / 10, 1)
                                                       for i in range(minutes)],
                         "dryer_dew_point_c": [-3.4] * minutes})
    return cmp, room


def test_minute_index_and_slot_are_stable():
    assert rm.minute_index(rm.MINUTE_EPOCH) == 0
    t = pd.Timestamp("2026-06-01T00:07Z")
    assert rm.ring_slot(t) == rm.minute_index(t) % rm.RING_SLOTS
    assert rm.ring_slot(t + pd.Timedelta(minutes=rm.RING_SLOTS)) == rm.ring_slot(t)


def test_register_image_round_trips_exactly():
    cmp, room = _frames()
    image = native.encode_register_image(cmp, room, UNIT_IDS)
    assert list(image.columns) == ["minute_utc", "unit_id", *native.REGISTER_COLUMNS]
    assert len(image) == len(cmp) + len(room)
    assert image.loc[image["unit_id"] == 5, ["r5", "r6", "r7"]].isna().all().all()
    native.verify_register_image(image, cmp, room, UNIT_IDS)


def test_register_image_verification_detects_a_changed_value():
    cmp, room = _frames()
    image = native.encode_register_image(cmp, room, UNIT_IDS)
    image.loc[0, "r3"] = int(image.loc[0, "r3"]) + 1
    try:
        native.verify_register_image(image, cmp, room, UNIT_IDS)
    except ValueError:
        return
    raise AssertionError("a changed register was not detected")


def test_ring_write_places_minutes_in_their_slots_atomically():
    cmp, room = _frames()
    blocks = native.image_blocks(native.encode_register_image(cmp, room, UNIT_IDS))
    buffers = RingBuffers(UNIT_IDS)
    before = buffers.images
    minutes = sorted(blocks)[:5]
    buffers.write_minutes([(m, blocks[m]) for m in minutes])
    assert buffers.images is not before
    for m in minutes:
        base = rm.ring_slot(m) * rm.SLOT_WIDTH
        for uid, words in blocks[m].items():
            assert buffers.images[uid][base:base + rm.SLOT_WIDTH] == words
        room_words = buffers.images[5][base:base + rm.SLOT_WIDTH]
        decoded = rm.decode_block(room_words[:5], rm.ROOM_MAP)
        assert decoded["record_ts"] == m
        assert room_words[5:] == [0, 0, 0]


def test_ring_slot_is_overwritten_after_a_full_cycle():
    cmp, room = _frames(minutes=rm.RING_SLOTS + 1)
    blocks = native.image_blocks(native.encode_register_image(cmp, room, UNIT_IDS))
    buffers = RingBuffers(UNIT_IDS)
    ordered = sorted(blocks)
    buffers.write_minutes([(m, blocks[m]) for m in ordered])
    first, last = ordered[0], ordered[-1]
    assert rm.ring_slot(first) == rm.ring_slot(last)
    base = rm.ring_slot(last) * rm.SLOT_WIDTH
    assert rm.decode_block(buffers.images[1][base:base + 8], rm.COMPRESSOR_MAP)["record_ts"] \
        == last


def test_first_tick_is_held_for_the_start_hold():
    import asyncio
    import time

    from idc.replay.modbus_server import ring_replay

    class _Buffers:
        def __init__(self):
            self.writes = []

        def write_minutes(self, batch):
            self.writes.append((time.monotonic(), batch))

    minute = pd.Timestamp("2026-04-01T00:00:00Z")
    minutes = [(minute + pd.Timedelta(minutes=i), None) for i in range(4)]
    buffers = _Buffers()
    served, _ = asyncio.run(ring_replay(buffers, minutes, 2, 0.0, None, None,
                                        start_hold_seconds=0.4))
    assert served == 4
    assert buffers.writes[1][0] - buffers.writes[0][0] >= 0.35
