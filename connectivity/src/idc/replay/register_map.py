"""Modbus register map for the compressor controllers and the room/dryer controller.

The map is the contract between the Modbus replay server, the Ignition tag definitions and
the landing writers. Conventions:

- Holding registers (function code 3), zero-based wire addresses. The one-based 4xxxx form
  (address 0 is 40001) is listed alongside for reference.
- Registers are big-endian; 32-bit values span two registers, high word first.
- Scaled integers store ``round(value * scale)`` and decode as ``raw / scale``. Division, not
  multiplication by the reciprocal, returns the exact float64 held in the export.
- Signed fields are two's-complement int16.
- The record timestamp is a uint32 of UTC epoch seconds.

Ring-buffer layout (buffered replay): each unit exposes ``RING_SLOTS`` slots of
``SLOT_WIDTH`` registers, 120 holding registers read in one request. A record minute is
written to slot ``minute_index(minute) % RING_SLOTS``, where ``minute_index`` counts minutes
from ``MINUTE_EPOCH``. Room blocks (5 registers) are padded to the slot width with zeros. In
this layout the snapshot register holds ``minute_index % 65536``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STATE_CODES: dict[str, int] = {"off": 0, "loaded": 1, "unloaded": 2}
STATE_NAMES: dict[int, str] = {v: k for k, v in STATE_CODES.items()}

MINUTE_EPOCH = pd.Timestamp("2026-01-01T00:00:00Z")
RING_SLOTS = 15
SLOT_WIDTH = 8
RING_REGISTERS = RING_SLOTS * SLOT_WIDTH


@dataclass(frozen=True)
class Register:
    name: str          # export column name or device field
    address: int       # zero-based address of the first word within the block
    kind: str          # uint16 | int16 | uint32
    scale: int = 1     # stored value = round(value * scale)
    unit: str = ""
    note: str = ""

    @property
    def words(self) -> int:
        return 2 if self.kind.endswith("32") else 1


COMPRESSOR_MAP: list[Register] = [
    Register("record_ts", 0, "uint32", 1, "s", "record minute, UTC epoch seconds"),
    Register("state", 2, "uint16", 1, "", "0 off, 1 loaded, 2 unloaded"),
    Register("discharge_pressure_bar", 3, "uint16", 100, "bar", "0.01 bar resolution"),
    Register("discharge_temp_c", 4, "int16", 10, "C", "0.1 C resolution, signed"),
    Register("run_hours", 5, "uint32", 100, "h", "cumulative, 0.01 h resolution"),
    Register("snapshot", 7, "uint16", 1, "", "block sequence counter; wraps at 65536"),
]

ROOM_MAP: list[Register] = [
    Register("record_ts", 0, "uint32", 1, "s", "record minute, UTC epoch seconds"),
    Register("ambient_c", 2, "int16", 10, "C", "0.1 C resolution, signed"),
    Register("dryer_dew_point_c", 3, "int16", 10, "C", "0.1 C resolution, signed"),
    Register("snapshot", 4, "uint16", 1, "", "block sequence counter; wraps at 65536"),
]

COMPRESSOR_BLOCK = 8
ROOM_BLOCK = 5


def block_size(regmap: list[Register]) -> int:
    return max(r.address + r.words for r in regmap)


def minute_index(minute: pd.Timestamp) -> int:
    """Whole minutes from MINUTE_EPOCH to a UTC record minute."""
    delta = pd.Timestamp(minute).tz_convert("UTC") - MINUTE_EPOCH
    return int(delta // pd.Timedelta(minutes=1))


def ring_slot(minute: pd.Timestamp) -> int:
    return minute_index(minute) % RING_SLOTS


def encode_scalar(value: float | int, reg: Register) -> list[int]:
    """Scale and pack one value into one or two 16-bit words, with range checks."""
    raw = int(round(float(value) * reg.scale))
    if reg.kind == "uint16":
        if not 0 <= raw <= 0xFFFF:
            raise ValueError(f"{reg.name}={value} out of uint16 range after scaling ({raw})")
        return [raw]
    if reg.kind == "int16":
        if not -0x8000 <= raw <= 0x7FFF:
            raise ValueError(f"{reg.name}={value} out of int16 range after scaling ({raw})")
        return [raw & 0xFFFF]
    if reg.kind == "uint32":
        if not 0 <= raw <= 0xFFFFFFFF:
            raise ValueError(f"{reg.name}={value} out of uint32 range after scaling ({raw})")
        return [(raw >> 16) & 0xFFFF, raw & 0xFFFF]
    raise ValueError(reg.kind)


def decode_scalar(words: list[int], reg: Register) -> float | int:
    """Unpack one or two words and unscale by division."""
    if reg.kind == "uint16":
        raw = words[0]
    elif reg.kind == "int16":
        raw = words[0] - 0x10000 if words[0] >= 0x8000 else words[0]
    elif reg.kind == "uint32":
        raw = (words[0] << 16) | words[1]
    else:
        raise ValueError(reg.kind)
    return raw if reg.scale == 1 else raw / reg.scale


def encode_block(values: dict, regmap: list[Register]) -> list[int]:
    """Build a device's register block from field values (state as name, record_ts as time)."""
    block = [0] * block_size(regmap)
    for reg in regmap:
        v = values[reg.name]
        if reg.name == "state":
            v = STATE_CODES[v]
        elif reg.name == "record_ts":
            v = int(pd.Timestamp(v).tz_convert("UTC").timestamp())
        block[reg.address:reg.address + reg.words] = encode_scalar(v, reg)
    return block


def decode_block(block: list[int], regmap: list[Register]) -> dict:
    """Decode a device's register block into field values."""
    out: dict = {}
    for reg in regmap:
        v = decode_scalar(list(block[reg.address:reg.address + reg.words]), reg)
        if reg.name == "state":
            v = STATE_NAMES[int(v)]
        elif reg.name == "record_ts":
            v = pd.Timestamp(int(v), unit="s", tz="UTC").as_unit("ns")
        out[reg.name] = v
    return out


def render_markdown(unit_ids: dict[str, int]) -> str:
    """Render the register map document."""
    lines: list[str] = [
        "# Modbus register map", "",
        "Holding registers (function code 3), zero-based wire addresses; the 4xxxx column is "
        "the one-based form used in vendor manuals. Registers are big-endian; 32-bit values "
        "are two registers, high word first. Scaled values decode as raw / scale.", "",
        "## Unit IDs", "", "| Device | Unit ID | Block registers |", "|---|---|---|",
    ]
    for device, uid in unit_ids.items():
        regmap = ROOM_MAP if device == "room" else COMPRESSOR_MAP
        lines.append(f"| {device} | {uid} | {block_size(regmap)} |")
    for title, regmap in (("Compressor block (CMP-01 to CMP-04)", COMPRESSOR_MAP),
                          ("Room and dryer block", ROOM_MAP)):
        lines += ["", f"## {title}", "",
                  "| Offset | 4xxxx (single block) | Field | Type | Scale | Unit | Note |",
                  "|---|---|---|---|---|---|---|"]
        for r in regmap:
            last = r.address + r.words - 1
            span = f"{r.address}" if r.words == 1 else f"{r.address}-{last}"
            four = f"{40001 + r.address}" if r.words == 1 else \
                f"{40001 + r.address}-{40001 + last}"
            lines.append(f"| {span} | {four} | {r.name} | {r.kind} | {r.scale} | {r.unit} | "
                         f"{r.note} |")
    lines += [
        "", "## Ring-buffer layout", "",
        f"Each unit exposes {RING_SLOTS} slots of {SLOT_WIDTH} registers: holding registers "
        f"0 to {RING_REGISTERS - 1}, read in one request. Slot s occupies registers "
        f"{SLOT_WIDTH}*s to {SLOT_WIDTH}*s+{SLOT_WIDTH - 1}; offsets within a slot follow the "
        "block tables above. Room slots use offsets 0 to 4; offsets 5 to 7 read as zero.", "",
        f"A record minute m is written to slot `minute_index(m) mod {RING_SLOTS}`, where "
        f"`minute_index` counts whole minutes from {MINUTE_EPOCH.isoformat()}. The snapshot "
        "register holds `minute_index(m) mod 65536`, so record_ts and snapshot together "
        "identify the minute a slot holds.", "",
        "Each server tick writes the next N record minutes into their slots for all units at "
        "once, then holds. With N minutes per tick a slot is overwritten every "
        f"{RING_SLOTS}/N ticks, so each minute stays readable for that many ticks.", "",
        "## Reading rules", "",
        "- Read a unit's registers in one request; the server swaps all blocks atomically.",
        "- A slot belongs to the minute in its record_ts registers, not to the time it was "
        "read. Accept a slot only when record_ts and snapshot agree.",
        "- int16 fields: 0x8000 to 0xFFFF are negative.",
        "- Unscale by dividing by the scale.", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="register_map")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--out", type=Path, default=Path("docs/register_map.md"))
    args = parser.parse_args(argv)
    from idc.checks.compare_exports import load_config
    unit_ids = load_config(args.config)["modbus"]["unit_ids"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(unit_ids), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
