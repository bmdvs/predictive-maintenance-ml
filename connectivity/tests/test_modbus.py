"""Register map round trips, common Modbus decoding errors, and a live server exchange."""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pandas as pd
import pytest
from pymodbus.client import AsyncModbusTcpClient, ModbusTcpClient

from idc.replay import modbus_server as ms
from idc.replay import register_map as rm

UNIT_IDS = {"CMP-01": 1, "CMP-02": 2, "CMP-03": 3, "CMP-04": 4, "room": 5}
MINUTE = pd.Timestamp("2026-06-01T11:00:00Z").as_unit("ns")


def test_scaled_values_round_trip_exactly():
    pressure = rm.COMPRESSOR_MAP[2]
    temp = rm.COMPRESSOR_MAP[3]
    hours = rm.COMPRESSOR_MAP[4]
    for v in [0.0, 0.01, 6.85, 7.78, 0.29]:
        assert rm.decode_scalar(rm.encode_scalar(v, pressure), pressure) == v
    for v in [30.1, 111.4, 0.0, -0.2, -12.3]:
        assert rm.decode_scalar(rm.encode_scalar(v, temp), temp) == v
    for v in [21101.0, 28298.0, 50123.45, 0.01]:
        assert rm.decode_scalar(rm.encode_scalar(v, hours), hours) == v


def test_decode_divides_rather_than_multiplies():
    # 685 * 0.01 is not 6.85 in binary floating point; 685 / 100 is.
    assert 685 * 0.01 != 6.85
    assert rm.decode_scalar([685], rm.COMPRESSOR_MAP[2]) == 6.85


def test_signed_register_trap():
    dew = rm.ROOM_MAP[2]
    words = rm.encode_scalar(-0.2, dew)
    assert words == [65534]                       # two's complement on the wire
    assert rm.decode_scalar(words, dew) == -0.2   # read as signed: correct
    unsigned = rm.Register("x", 0, "uint16", 10)
    assert rm.decode_scalar(words, unsigned) == 6553.4   # read as unsigned: wrong


def test_uint32_word_order_is_high_word_first():
    hours = rm.COMPRESSOR_MAP[4]
    words = rm.encode_scalar(28298.0, hours)       # 2,829,800 = 0x002B_2DE8
    assert words == [0x002B, 0x2DE8]
    assert rm.decode_scalar(words, hours) == 28298.0
    swapped = rm.decode_scalar(words[::-1], hours)  # a little-endian-word reader
    assert swapped != 28298.0 and swapped > 7_000_000  # word-swapped read is out of range


def test_pymodbus_agrees_with_our_word_order():
    words = rm.encode_scalar(28298.0, rm.COMPRESSOR_MAP[4])
    raw = ModbusTcpClient.convert_from_registers(words, ModbusTcpClient.DATATYPE.UINT32,
                                                 word_order="big")
    assert raw == 2_829_800
    assert ModbusTcpClient.convert_from_registers([65534], ModbusTcpClient.DATATYPE.INT16) == -2


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        rm.encode_scalar(700.0, rm.COMPRESSOR_MAP[2])    # 70,000 > uint16
    with pytest.raises(ValueError):
        rm.encode_scalar(-1.0, rm.COMPRESSOR_MAP[2])     # negative into uint16


def test_block_round_trip_and_timestamp():
    values = {"record_ts": MINUTE, "state": "unloaded", "discharge_pressure_bar": 6.85,
              "discharge_temp_c": 78.3, "run_hours": 21101.0, "snapshot": 7}
    block = rm.encode_block(values, rm.COMPRESSOR_MAP)
    assert len(block) == rm.COMPRESSOR_BLOCK
    assert (block[0] << 16 | block[1]) == int(MINUTE.timestamp())
    back = rm.decode_block(block, rm.COMPRESSOR_MAP)
    assert back == values
    assert back["record_ts"].tz is not None and back["record_ts"].unit == "ns"


def test_room_block_round_trip_with_negative_dew_point():
    values = {"record_ts": MINUTE, "ambient_c": 24.9, "dryer_dew_point_c": -3.5, "snapshot": 1}
    assert rm.decode_block(rm.encode_block(values, rm.ROOM_MAP), rm.ROOM_MAP) == values


def test_register_map_document_renders():
    text = rm.render_markdown(UNIT_IDS)
    assert "| 5-6 | 40006-40007 | run_hours | uint32 | 100 |" in text
    assert "| room | 5 |" in text


def test_device_blocks_swap_whole_and_count_snapshots():
    blocks = ms.DeviceBlocks(UNIT_IDS)
    before = blocks.blocks[1]
    blocks.update("CMP-01", {"record_ts": MINUTE, "state": "off", "discharge_pressure_bar": 0.0,
                             "discharge_temp_c": 30.1, "run_hours": 21101.0})
    assert blocks.blocks[1] is not before            # replaced, not edited in place
    assert blocks.snapshot[1] == 1
    assert rm.decode_block(blocks.blocks[1], rm.COMPRESSOR_MAP)["snapshot"] == 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_live_server_serves_one_consistent_block():
    """Start the real server on a free port, read every unit with a real client, decode."""
    blocks = ms.DeviceBlocks(UNIT_IDS)
    rows = [(MINUTE, {
        "CMP-01": {"record_ts": MINUTE, "state": "loaded", "discharge_pressure_bar": 7.01,
                   "discharge_temp_c": 85.6, "run_hours": 21101.5},
        "CMP-02": {"record_ts": MINUTE, "state": "off", "discharge_pressure_bar": 0.0,
                   "discharge_temp_c": 30.0, "run_hours": 50123.45},
        "CMP-03": {"record_ts": MINUTE, "state": "unloaded", "discharge_pressure_bar": 6.85,
                   "discharge_temp_c": 60.1, "run_hours": 1.0},
        "CMP-04": {"record_ts": MINUTE, "state": "off", "discharge_pressure_bar": 0.0,
                   "discharge_temp_c": 29.9, "run_hours": 28298.0},
        "room": {"record_ts": MINUTE, "ambient_c": 24.9, "dryer_dew_point_c": -0.2},
    })]
    port = _free_port()

    async def scenario():
        server = ms.ModbusTcpServer(ms.build_devices(blocks), address=("127.0.0.1", port))
        task = asyncio.create_task(server.serve_forever())
        try:
            await asyncio.sleep(0.2)
            await ms.replay(blocks, rows, hold_seconds=0.0)
            client = AsyncModbusTcpClient("127.0.0.1", port=port)
            assert await client.connect()
            out = {}
            for device, uid in UNIT_IDS.items():
                regmap = rm.ROOM_MAP if device == "room" else rm.COMPRESSOR_MAP
                rr = await client.read_holding_registers(0, count=rm.block_size(regmap),
                                                         device_id=uid)
                assert not rr.isError(), rr
                out[device] = rm.decode_block(rr.registers, regmap)
            # Writes are refused: the device is read-only to clients.
            wr = await client.write_register(2, 1, device_id=1)
            assert wr.isError()
            client.close()
            return out
        finally:
            await server.shutdown()
            task.cancel()

    out = asyncio.run(scenario())
    for device, values in rows[0][1].items():
        got = out[device]
        for k, v in values.items():
            assert got[k] == v, (device, k, got[k], v)
        assert got["snapshot"] == 1


def _export_root() -> Path:
    """Export root from IDC_EXPORT_ROOT, else from the repository's chain configuration."""
    if "IDC_EXPORT_ROOT" in os.environ:
        return Path(os.environ["IDC_EXPORT_ROOT"])
    from idc.checks.compare_exports import load_config
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "chain.yaml")
    return Path(config["paths"]["export_root"])


EXPORT = _export_root()


@pytest.mark.skipif(not (EXPORT / "compressor_ctrl").exists(), reason="exports not present")
def test_every_june_value_round_trips():
    cmp, room = ms.load_minutes(EXPORT, "2026-06")
    for reg, col in [(rm.COMPRESSOR_MAP[2], "discharge_pressure_bar"),
                     (rm.COMPRESSOR_MAP[3], "discharge_temp_c"),
                     (rm.COMPRESSOR_MAP[4], "run_hours")]:
        vals = cmp[col].to_numpy()
        for v in set(vals.tolist()):
            assert rm.decode_scalar(rm.encode_scalar(v, reg), reg) == v, (col, v)
    for reg, col in [(rm.ROOM_MAP[1], "ambient_c"), (rm.ROOM_MAP[2], "dryer_dew_point_c")]:
        for v in set(room[col].tolist()):
            assert rm.decode_scalar(rm.encode_scalar(v, reg), reg) == v, (col, v)
