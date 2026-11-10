"""Modbus TCP server that replays logged compressor and room controller minutes.

One server, five unit IDs: four compressor controllers and the room/dryer controller.
Requests are answered from the unit's current register image, which is replaced whole, so
one read request always sees one consistent state. Writes are refused.

Modes:

- Single block (default): each record minute's block (8 or 5 registers) is loaded from the
  export and held for ``--hold`` seconds before the next minute.
- Ring buffer (``--ring``): each unit exposes 15 slots of 8 registers (120 registers, one
  read) loaded from the register images under ``paths.native_root/modbus``. Each tick writes
  the next ``--per-tick`` record minutes into slot ``minute_index mod 15`` for every unit at
  once, then holds ``--hold`` seconds. While ``--pause-file`` exists the server keeps
  serving but stops advancing. Progress (next ``--start``) is logged and, with
  ``--progress-file``, written as JSON after every tick.

Usage:
    python -m idc.replay.modbus_server --start 2026-06-01T11:00Z --minutes 240 --hold 2
    python -m idc.replay.modbus_server --ring --start 2026-04-01T00:00Z --minutes 1440 \
        --per-tick 5 --hold 0.5 --pause-file runs/modbus.pause --progress-file runs/modbus.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd
from pymodbus.server import ModbusTcpServer
from pymodbus.simulator.simdata import SimData
from pymodbus.simulator.simdevice import SimDevice
from pymodbus.simulator.simutils import DataType

from idc.replay.register_map import (
    COMPRESSOR_MAP,
    RING_REGISTERS,
    ROOM_MAP,
    SLOT_WIDTH,
    block_size,
    encode_block,
    ring_slot,
)

log = logging.getLogger("idc.modbus_server")


class DeviceBlocks:
    """Current register block per unit ID. Replaced whole; never edited in place."""

    def __init__(self, unit_ids: dict[str, int]):
        self.unit_ids = unit_ids
        self.blocks: dict[int, list[int]] = {}
        self.snapshot: dict[int, int] = {uid: 0 for uid in unit_ids.values()}
        for device, uid in unit_ids.items():
            size = block_size(ROOM_MAP if device == "room" else COMPRESSOR_MAP)
            self.blocks[uid] = [0] * size

    def update(self, device: str, values: dict) -> None:
        uid = self.unit_ids[device]
        regmap = ROOM_MAP if device == "room" else COMPRESSOR_MAP
        self.snapshot[uid] = (self.snapshot[uid] + 1) & 0xFFFF
        self.blocks[uid] = encode_block({**values, "snapshot": self.snapshot[uid]}, regmap)

    def action_for(self, uid: int):
        """Build the per-device callback pymodbus runs before answering each request.

        The callback copies the current block into the server's register memory, so every
        response reflects the block as it stood when the request arrived. Writes are refused.
        """
        async def action(function_code, start_address, address, count, registers, values):
            if values is not None:
                from pymodbus.constants import ExcCodes
                return ExcCodes.ILLEGAL_FUNCTION
            block = self.blocks[uid]
            registers[0:len(block)] = block
            return None
        return action


class RingBuffers:
    """Per-unit ring-buffer register images (RING_REGISTERS words each), replaced whole."""

    def __init__(self, unit_ids: dict[str, int]):
        self.unit_ids = unit_ids
        self.images: dict[int, list[int]] = {uid: [0] * RING_REGISTERS
                                             for uid in unit_ids.values()}

    def write_minutes(self, minutes: list[tuple[pd.Timestamp, dict[int, list[int]]]]) -> None:
        """Write record minutes into their slots for every unit, then swap all images."""
        staged = {uid: list(image) for uid, image in self.images.items()}
        for minute, blocks in minutes:
            base = ring_slot(minute) * SLOT_WIDTH
            for uid, words in blocks.items():
                padded = (list(words) + [0] * SLOT_WIDTH)[:SLOT_WIDTH]
                staged[uid][base:base + SLOT_WIDTH] = padded
        self.images = staged

    def action_for(self, uid: int):
        """Per-unit request callback: copy the current image into the response registers."""
        async def action(function_code, start_address, address, count, registers, values):
            if values is not None:
                from pymodbus.constants import ExcCodes
                return ExcCodes.ILLEGAL_FUNCTION
            registers[0:RING_REGISTERS] = self.images[uid]
            return None
        return action

    def build_devices(self) -> list[SimDevice]:
        return [SimDevice(id=uid,
                          simdata=[SimData(address=0, count=RING_REGISTERS, values=0,
                                           datatype=DataType.REGISTERS)],
                          action=self.action_for(uid))
                for uid in self.unit_ids.values()]


def ring_minutes(native_root: Path, start: pd.Timestamp, count: int | None):
    """Record minutes from `start` as (minute, {unit_id: words}), across month images."""
    from idc.replay.build_native import image_blocks, read_register_image

    month = start.strftime("%Y-%m")
    served = 0
    while True:
        path = native_root / "modbus" / f"{month}.parquet"
        if not path.exists():
            return
        blocks = image_blocks(read_register_image(native_root, month))
        for minute in sorted(m for m in blocks if m >= start):
            if count is not None and served >= count:
                return
            yield minute, blocks[minute]
            served += 1
        month = (pd.Timestamp(f"{month}-01") + pd.offsets.MonthBegin(1)).strftime("%Y-%m")


def write_progress(path: Path, body: dict) -> None:
    """Replace the progress file atomically; skipped for a tick if a reader holds it open."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body), encoding="utf-8")
    try:
        tmp.replace(path)
    except PermissionError:
        pass


async def ring_replay(buffers: RingBuffers, minutes, per_tick: int, hold_seconds: float,
                      pause_file: Path | None, progress_file: Path | None,
                      start_hold_seconds: float = 0.0) -> tuple[int, object]:
    """Advance the ring buffers `per_tick` minutes per tick; honors the pause file.

    The first tick is held for at least `start_hold_seconds`, so a client that connects when
    the server starts has read the first window before the replay advances.
    """
    served = 0
    last = None
    ticks = 0
    batch: list = []
    iterator = iter(minutes)
    exhausted = False
    while not exhausted:
        while pause_file is not None and pause_file.exists():
            await asyncio.sleep(0.2)
        batch = []
        for _ in range(per_tick):
            try:
                batch.append(next(iterator))
            except StopIteration:
                exhausted = True
                break
        if not batch:
            break
        tick_start = time.monotonic()
        buffers.write_minutes(batch)
        served += len(batch)
        last = batch[-1][0]
        ticks += 1
        nxt = last + pd.Timedelta(minutes=1)
        if progress_file is not None:
            write_progress(progress_file, {
                "last_minute": last.isoformat(), "next_start": nxt.strftime("%Y-%m-%dT%H:%MZ"),
                "served": served, "ticks": ticks, "updated": time.time()})
        if ticks == 1 or ticks % 120 == 0:
            log.info("tick %d: through %s (%d minutes); next --start %s", ticks, last, served,
                     nxt.strftime("%Y-%m-%dT%H:%MZ"))
        hold = max(hold_seconds, start_hold_seconds) if ticks == 1 else hold_seconds
        delay = hold - (time.monotonic() - tick_start)
        if delay > 0:
            await asyncio.sleep(delay)
    return served, last


async def serve_ring(buffers: RingBuffers, minutes, port: int, per_tick: int,
                     hold_seconds: float, pause_file: Path | None, progress_file: Path | None,
                     host: str = "0.0.0.0", linger_seconds: float = 0.0,
                     start_hold_seconds: float = 0.0) -> None:
    server = ModbusTcpServer(buffers.build_devices(), address=(host, port))
    server_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.2)
    log.info("Modbus TCP ring-buffer server on %s:%d, unit IDs %s, %d minutes per tick, "
             "hold %.2fs", host, port, dict(buffers.unit_ids), per_tick, hold_seconds)
    try:
        served, last = await ring_replay(buffers, minutes, per_tick, hold_seconds, pause_file,
                                         progress_file, start_hold_seconds)
        log.info("ring replay complete: %d record minutes, last %s", served, last)
        if last is not None:
            log.info("resume the next chunk with --start %s",
                     (last + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%MZ"))
        await asyncio.sleep(max(linger_seconds, hold_seconds * 4))
    finally:
        await server.shutdown()
        server_task.cancel()


def load_minutes(export_root: Path, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmp_path = export_root / "compressor_ctrl" / "minute" / f"month={month}" / \
        "compressor_minute.parquet"
    room_path = export_root / "compressor_ctrl" / "room" / f"month={month}" / "room.parquet"
    cmp = pd.read_parquet(cmp_path).sort_values(["minute_utc", "compressor_id"],
                                                kind="mergesort")
    room = pd.read_parquet(room_path).sort_values("minute_utc", kind="mergesort")
    return cmp, room


def minute_rows(cmp: pd.DataFrame, room: pd.DataFrame, start: pd.Timestamp,
                count: int | None):
    """Yield (minute, {device: values}) for each record minute from `start`."""
    minutes = room["minute_utc"][room["minute_utc"] >= start]
    if count is not None:
        minutes = minutes.iloc[:count]
    cmp_by_minute = {k: g for k, g in cmp.groupby("minute_utc")}
    room_by_minute = room.set_index("minute_utc")
    for minute in minutes:
        devices: dict[str, dict] = {}
        for row in cmp_by_minute[minute].itertuples(index=False):
            devices[row.compressor_id] = {
                "record_ts": minute, "state": row.state,
                "discharge_pressure_bar": row.discharge_pressure_bar,
                "discharge_temp_c": row.discharge_temp_c, "run_hours": row.run_hours,
            }
        r = room_by_minute.loc[minute]
        devices["room"] = {"record_ts": minute, "ambient_c": r["ambient_c"],
                           "dryer_dew_point_c": r["dryer_dew_point_c"]}
        yield minute, devices


def build_devices(blocks: DeviceBlocks) -> list[SimDevice]:
    devices = []
    for device, uid in blocks.unit_ids.items():
        size = block_size(ROOM_MAP if device == "room" else COMPRESSOR_MAP)
        devices.append(SimDevice(
            id=uid,
            simdata=[SimData(address=0, count=size, values=0, datatype=DataType.REGISTERS)],
            action=blocks.action_for(uid),
        ))
    return devices


async def replay(blocks: DeviceBlocks, rows,
                 hold_seconds: float) -> tuple[int, pd.Timestamp | None]:
    served = 0
    last: pd.Timestamp | None = None
    next_due = time.monotonic()
    for minute, devices in rows:
        for device, values in devices.items():
            blocks.update(device, values)          # no await between devices: one tick
        served += 1
        last = minute
        if served == 1 or served % 60 == 0:
            log.info("serving record minute %s (%d so far)", minute, served)
        next_due += hold_seconds
        delay = next_due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    return served, last


async def serve(blocks: DeviceBlocks, rows, port: int, hold_seconds: float,
                host: str = "0.0.0.0") -> None:
    server = ModbusTcpServer(build_devices(blocks), address=(host, port))
    server_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.2)
    log.info("Modbus TCP server listening on %s:%d, unit IDs %s", host, port,
             dict(blocks.unit_ids))
    try:
        served, last = await replay(blocks, rows, hold_seconds)
        log.info("replay complete: %d record minutes, last %s; holding the final block",
                 served, last)
        if last is not None:
            nxt = last + pd.Timedelta(minutes=1)
            log.info("resume the next chunk with --start %s", nxt.strftime("%Y-%m-%dT%H:%MZ"))
        await asyncio.sleep(hold_seconds * 2)
    finally:
        await server.shutdown()
        server_task.cancel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modbus_server")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--start", default=None,
                        help="first record minute to serve, e.g. 2026-06-01T11:00Z "
                             "(default: the dev slice start)")
    parser.add_argument("--minutes", type=int, default=None,
                        help="how many record minutes to serve (default: the dev slice)")
    parser.add_argument("--hold", type=float, default=None,
                        help="seconds each record minute is held (default from config)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--ring", action="store_true",
                        help="ring-buffer mode served from the native register images")
    parser.add_argument("--per-tick", type=int, default=None,
                        help="ring mode: record minutes written per tick")
    parser.add_argument("--pause-file", type=Path, default=None,
                        help="ring mode: hold without advancing while this file exists")
    parser.add_argument("--progress-file", type=Path, default=None,
                        help="ring mode: JSON progress written after every tick")
    parser.add_argument("--linger", type=float, default=0.0,
                        help="ring mode: seconds to keep serving after the last tick")
    parser.add_argument("--start-hold", type=float, default=0.0,
                        help="ring mode: minimum seconds the first tick is held")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    modbus = config["modbus"]
    if args.ring:
        if args.start is None:
            parser.error("--ring requires --start")
        start = pd.Timestamp(args.start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        per_tick = args.per_tick or int(modbus["ring_minutes_per_tick"])
        hold = args.hold if args.hold is not None else float(modbus["ring_hold_seconds"])
        buffers = RingBuffers(modbus["unit_ids"])
        minutes = ring_minutes(Path(config["paths"]["native_root"]), start, args.minutes)
        try:
            asyncio.run(serve_ring(buffers, minutes, args.port or int(modbus["port"]), per_tick,
                                   hold, args.pause_file, args.progress_file,
                                   linger_seconds=args.linger,
                                   start_hold_seconds=args.start_hold))
        except KeyboardInterrupt:
            log.info("stopped")
        return 0
    slice_cfg = config["dev_slices"]["compressor"]
    start = pd.Timestamp(args.start or slice_cfg["start"])
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    minutes = args.minutes if args.minutes is not None else int(slice_cfg["hours"]) * 60
    hold = args.hold if args.hold is not None else float(modbus["hold_seconds"])
    port = args.port or int(modbus["port"])

    cmp, room = load_minutes(Path(config["paths"]["export_root"]), args.month)
    blocks = DeviceBlocks(modbus["unit_ids"])
    rows = minute_rows(cmp, room, start, minutes)
    log.info("replaying %d record minutes from %s, hold %.1fs each (about %.0f minutes of "
             "wall time)", minutes, start, hold, minutes * hold / 60)
    try:
        asyncio.run(serve(blocks, rows, port, hold))
    except KeyboardInterrupt:
        log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
