"""Build the native device files the replay components serve.

Two outputs, both derived from the bronze exports and verified on write:

- HMC SHDR logs: ``native/cnc/<machine>/<YYYY-MM>.shdr.gz``, the exact line stream the SHDR
  adapter emits for that machine-month (record timestamps included), one line per SHDR line.
- Compressor register images: ``native/modbus/<YYYY-MM>.parquet`` with columns
  ``minute_utc, unit_id, r0 .. r7``: one row per device per record minute holding the
  register block that device presents for that minute. Room rows (unit 5) use r0..r4;
  r5..r7 are null. The snapshot register holds the record-minute sequence
  (``minute_index mod 65536``), so a reader can confirm a block's registers belong together.

Usage:
    python -m idc.replay.build_native [shdr|modbus] [--machines HMC-01,...] [--months 2026-04,...]

With no kind, both outputs are built.
"""

from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from idc.replay.register_map import (
    COMPRESSOR_MAP,
    ROOM_MAP,
    decode_block,
    encode_block,
    minute_index,
)
from idc.replay.shdr_adapter import load_slice, shdr_lines

log = logging.getLogger("idc.build_native")

REGISTER_COLUMNS = [f"r{i}" for i in range(8)]
ROOM_UNIT_KEY = "room"


def shdr_log_path(native_root: Path, machine_id: str, month: str) -> Path:
    return native_root / "cnc" / machine_id / f"{month}.shdr.gz"


def register_image_path(native_root: Path, month: str) -> Path:
    return native_root / "modbus" / f"{month}.parquet"


def write_shdr_log(export_root: Path, native_root: Path, machine_id: str, month: str,
                   day: str | None = None) -> int:
    """Write one machine-month SHDR log (or one UTC day of it) and return its line count."""
    frame = load_slice(export_root, machine_id, month, day)
    path = shdr_log_path(native_root, machine_id, day or month)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    lines = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="\n", compresslevel=6) as out:
        for _, line in shdr_lines(frame):
            out.write(line)
            out.write("\n")
            lines += 1
    tmp.replace(path)
    return lines


def read_shdr_log(path: Path):
    """Yield the lines of an SHDR log without their line terminators."""
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as src:
        for line in src:
            yield line.rstrip("\n")


def verify_shdr_log(export_root: Path, native_root: Path, machine_id: str, month: str,
                    day: str | None = None) -> int:
    """Confirm a log equals the adapter's line stream for the export. Returns the line count."""
    frame = load_slice(export_root, machine_id, month, day)
    expected = (line for _, line in shdr_lines(frame))
    logged = read_shdr_log(shdr_log_path(native_root, machine_id, day or month))
    count = 0
    for count, (want, got) in enumerate(zip(expected, logged, strict=True), start=1):
        if want != got:
            raise ValueError(f"{machine_id} {month} line {count}: expected {want!r}, got {got!r}")
    return count


def encode_register_image(cmp: pd.DataFrame, room: pd.DataFrame,
                          unit_ids: dict[str, int]) -> pd.DataFrame:
    """Encode one month of compressor and room minutes as per-device register blocks."""
    rows: list[list] = []
    for rec in cmp.itertuples(index=False):
        block = encode_block({
            "record_ts": rec.minute_utc, "state": rec.state,
            "discharge_pressure_bar": rec.discharge_pressure_bar,
            "discharge_temp_c": rec.discharge_temp_c, "run_hours": rec.run_hours,
            "snapshot": minute_index(rec.minute_utc) & 0xFFFF,
        }, COMPRESSOR_MAP)
        rows.append([rec.minute_utc, unit_ids[rec.compressor_id], *block])
    for rec in room.itertuples(index=False):
        block = encode_block({
            "record_ts": rec.minute_utc, "ambient_c": rec.ambient_c,
            "dryer_dew_point_c": rec.dryer_dew_point_c,
            "snapshot": minute_index(rec.minute_utc) & 0xFFFF,
        }, ROOM_MAP)
        rows.append([rec.minute_utc, unit_ids[ROOM_UNIT_KEY], *block, None, None, None])
    image = pd.DataFrame(rows, columns=["minute_utc", "unit_id", *REGISTER_COLUMNS])
    image["minute_utc"] = pd.DatetimeIndex(image["minute_utc"]).tz_convert("UTC").as_unit("ns")
    image["unit_id"] = image["unit_id"].astype("int64")
    for column in REGISTER_COLUMNS:
        image[column] = image[column].astype("UInt16")
    return image.sort_values(["minute_utc", "unit_id"], kind="mergesort").reset_index(drop=True)


def verify_register_image(image: pd.DataFrame, cmp: pd.DataFrame, room: pd.DataFrame,
                          unit_ids: dict[str, int]) -> None:
    """Decode every block and require exact equality with the export rows."""
    devices = {uid: name for name, uid in unit_ids.items()}
    expected_cmp = cmp.set_index(["compressor_id", "minute_utc"])
    expected_room = room.set_index("minute_utc")
    seen_cmp = seen_room = 0
    for rec in image.itertuples(index=False):
        device = devices[rec.unit_id]
        regmap = ROOM_MAP if device == ROOM_UNIT_KEY else COMPRESSOR_MAP
        width = 5 if device == ROOM_UNIT_KEY else 8
        block = [int(getattr(rec, f"r{i}")) for i in range(width)]
        values = decode_block(block, regmap)
        if values["record_ts"] != rec.minute_utc:
            raise ValueError(f"unit {rec.unit_id} {rec.minute_utc}: record_ts decodes to "
                             f"{values['record_ts']}")
        if values["snapshot"] != minute_index(rec.minute_utc) & 0xFFFF:
            raise ValueError(f"unit {rec.unit_id} {rec.minute_utc}: snapshot mismatch")
        if device == ROOM_UNIT_KEY:
            want = expected_room.loc[rec.minute_utc]
            fields = ("ambient_c", "dryer_dew_point_c")
            seen_room += 1
        else:
            want = expected_cmp.loc[(device, rec.minute_utc)]
            fields = ("state", "discharge_pressure_bar", "discharge_temp_c", "run_hours")
            seen_cmp += 1
        for name in fields:
            if values[name] != want[name]:
                raise ValueError(f"{device} {rec.minute_utc} {name}: export {want[name]!r}, "
                                 f"register decode {values[name]!r}")
    if seen_cmp != len(cmp) or seen_room != len(room):
        raise ValueError(f"row counts differ: compressor {seen_cmp}/{len(cmp)}, "
                         f"room {seen_room}/{len(room)}")


def load_compressor_month(export_root: Path, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = export_root / "compressor_ctrl"
    cmp = pd.read_parquet(base / "minute" / f"month={month}" / "compressor_minute.parquet")
    room = pd.read_parquet(base / "room" / f"month={month}" / "room.parquet")
    return cmp, room


def read_register_image(native_root: Path, month: str) -> pd.DataFrame:
    return pd.read_parquet(register_image_path(native_root, month))


def image_blocks(image: pd.DataFrame) -> dict[pd.Timestamp, dict[int, list[int]]]:
    """Index an image as {minute: {unit_id: register words}}, room blocks padded to 8 words."""
    words = image[REGISTER_COLUMNS].fillna(0).to_numpy(dtype=np.int64)
    out: dict[pd.Timestamp, dict[int, list[int]]] = {}
    for minute, unit_id, block in zip(image["minute_utc"], image["unit_id"], words, strict=True):
        out.setdefault(minute, {})[int(unit_id)] = [int(w) for w in block]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_native")
    parser.add_argument("kind", nargs="?", choices=["shdr", "modbus"], default=None,
                        help="build only this output (default: both)")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--machines", default=None, help="comma-separated machine IDs")
    parser.add_argument("--months", default=None, help="comma-separated YYYY-MM months")
    parser.add_argument("--day", default=None,
                        help="shdr only: write a single UTC day log, YYYY-MM-DD")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    export_root = Path(config["paths"]["export_root"])
    native_root = Path(config["paths"]["native_root"])
    months = args.months.split(",") if args.months else list(config["replay"]["months"])

    if args.kind in (None, "shdr"):
        machines = args.machines.split(",") if args.machines else list(config["replay"]["hmcs"])
        if args.day:
            month = args.day[:7]
            for machine_id in machines:
                lines = write_shdr_log(export_root, native_root, machine_id, month, args.day)
                verified = verify_shdr_log(export_root, native_root, machine_id, month,
                                           args.day)
                log.info("%s %s: %d lines written and verified", machine_id, args.day, verified)
            return 0
        for machine_id in machines:
            for month in months:
                lines = write_shdr_log(export_root, native_root, machine_id, month)
                verified = verify_shdr_log(export_root, native_root, machine_id, month)
                log.info("%s %s: %d lines written and verified", machine_id, month, verified)
                if verified != lines:
                    raise SystemExit(f"{machine_id} {month}: wrote {lines}, read {verified}")
        if args.kind == "shdr":
            return 0

    unit_ids = config["modbus"]["unit_ids"]
    for month in months:
        cmp, room = load_compressor_month(export_root, month)
        image = encode_register_image(cmp, room, unit_ids)
        verify_register_image(image, cmp, room, unit_ids)
        path = register_image_path(native_root, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.to_parquet(path, index=False)
        log.info("%s: %d register blocks (%d compressor, %d room) written and verified", month,
                 len(image), len(cmp), len(room))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
