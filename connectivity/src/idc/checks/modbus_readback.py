"""Read the replay server's registers with a Modbus client and check them against the export.

For each device the client reads the whole block in one request, decodes it with the
register map, looks up the export row for the record minute the block names, and compares
every field exactly. It repeats until a number of distinct record minutes have been seen,
which verifies the encode/decode path and the atomic block update against live traffic.
Exits 0 when every field matches, 1 if the server is unreachable and 2 on any mismatch.

Usage:
    python -m idc.checks.modbus_readback [--config config/chain.yaml] [--snapshots 5]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from pymodbus.client import ModbusTcpClient

from idc.replay.modbus_server import load_minutes
from idc.replay.register_map import COMPRESSOR_MAP, ROOM_MAP, block_size, decode_block

log = logging.getLogger("idc.modbus_readback")


def read_device(client: ModbusTcpClient, uid: int, device: str) -> dict:
    """Read and decode one unit's full register block; the raw words are kept under "_raw"."""
    regmap = ROOM_MAP if device == "room" else COMPRESSOR_MAP
    rr = client.read_holding_registers(0, count=block_size(regmap), device_id=uid)
    if rr.isError():
        raise RuntimeError(f"unit {uid}: {rr}")
    decoded = decode_block(rr.registers, regmap)
    decoded["_raw"] = list(rr.registers)
    return decoded


def expected_row(cmp: pd.DataFrame, room: pd.DataFrame, device: str,
                 minute: pd.Timestamp) -> dict:
    """Export values for one device at one record minute, keyed by register field name."""
    if device == "room":
        r = room[room["minute_utc"] == minute].iloc[0]
        return {"ambient_c": r["ambient_c"], "dryer_dew_point_c": r["dryer_dew_point_c"]}
    r = cmp[(cmp["minute_utc"] == minute) & (cmp["compressor_id"] == device)].iloc[0]
    return {"state": r["state"], "discharge_pressure_bar": r["discharge_pressure_bar"],
            "discharge_temp_c": r["discharge_temp_c"], "run_hours": r["run_hours"]}


def check_once(client, cmp, room, unit_ids: dict[str, int]) -> tuple[int, pd.Timestamp]:
    """Read every device once; return the field failure count and the latest record minute."""
    failures = 0
    minutes = set()
    for device, uid in unit_ids.items():
        got = read_device(client, uid, device)
        minute = got["record_ts"]
        minutes.add(minute)
        want = expected_row(cmp, room, device, minute)
        for field, value in want.items():
            if got[field] != value:
                failures += 1
                log.error("%s @ %s: %s expected %r got %r (raw %s)", device, minute, field,
                          value, got[field], got["_raw"])
    if len(minutes) != 1:
        log.warning("devices report different record minutes in one pass: %s",
                    sorted(minutes))
    return failures, max(minutes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modbus_readback")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--month", default="2026-06",
                        help="export month the replay server is serving, YYYY-MM")
    parser.add_argument("--snapshots", type=int, default=5,
                        help="distinct record minutes to verify before exiting")
    parser.add_argument("--host", default="localhost", help="Modbus replay server address")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    unit_ids = config["modbus"]["unit_ids"]
    cmp, room = load_minutes(Path(config["paths"]["export_root"]), args.month)

    client = ModbusTcpClient(args.host, port=int(config["modbus"]["port"]))
    if not client.connect():
        log.error("cannot connect to %s:%s", args.host, config["modbus"]["port"])
        return 1
    seen: list[pd.Timestamp] = []
    failures = 0
    try:
        while len(seen) < args.snapshots:
            f, minute = check_once(client, cmp, room, unit_ids)
            failures += f
            if not seen or minute != seen[-1]:
                seen.append(minute)
                log.info("record minute %s: %s", minute, "OK" if f == 0 else f"{f} FAILURES")
            time.sleep(0.25)
    finally:
        client.close()
    log.info("%d record minutes verified, %d field failures", len(seen), failures)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
