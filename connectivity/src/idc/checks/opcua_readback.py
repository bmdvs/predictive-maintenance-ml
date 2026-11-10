"""Read the compressor ring-buffer tags from Ignition's OPC UA server and check them.

Connects to the gateway's OPC UA endpoint, reads every slot register tag of each unit
(``[default]compressors/<unit>/sNN_rK``) in one request per unit, decodes the slots whose
record_ts and snapshot agree, and compares each decoded minute with the bronze export.
Reports, per unit, the slots checked and matched and the spread of source and server
timestamps. Repeats for ``--samples`` reads spaced ``--interval`` seconds apart. Exit code 1
on any mismatch.

Usage:
    python -m idc.checks.opcua_readback [--samples 5] [--interval 2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from asyncua import Client, ua

from idc.checks.compare_exports import load_config
from idc.replay.register_map import (
    COMPRESSOR_MAP,
    RING_SLOTS,
    ROOM_MAP,
    SLOT_WIDTH,
    decode_block,
    minute_index,
)

CMP_FIELDS = ("state", "discharge_pressure_bar", "discharge_temp_c", "run_hours")
ROOM_FIELDS = ("ambient_c", "dryer_dew_point_c")


def load_exports(export_root: Path, months: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = export_root / "compressor_ctrl"
    cmp = pd.concat([pd.read_parquet(base / "minute" / f"month={m}" /
                                     "compressor_minute.parquet") for m in months])
    room = pd.concat([pd.read_parquet(base / "room" / f"month={m}" / "room.parquet")
                      for m in months])
    return cmp.set_index(["compressor_id", "minute_utc"]), room.set_index("minute_utc")


async def read_unit(client: Client, folder: str, unit: str):
    names = [f"s{s:02d}_r{k}" for s in range(RING_SLOTS) for k in range(SLOT_WIDTH)]
    nodes = [client.get_node(f"ns=2;s=[default]{folder}/{unit}/{n}") for n in names]
    values = await client.read_attributes(nodes, ua.AttributeIds.Value)
    return values


async def sample(config: dict, samples: int, interval: float) -> dict:
    ign = config["ignition"]
    cmp, room = load_exports(Path(config["paths"]["export_root"]),
                             list(config["replay"]["months"]))
    results: dict = {"endpoint": ign["opcua_url"], "samples": []}
    async with Client(ign["opcua_url"]) as client:
        for _ in range(samples):
            snap = {"read_utc": datetime.now(UTC).isoformat(), "units": {}}
            for unit in config["modbus"]["unit_ids"]:
                dvs = await read_unit(client, ign["tag_folder"], unit)
                words = [dv.Value.Value if dv.StatusCode.is_good() else None for dv in dvs]
                src = [dv.SourceTimestamp for dv in dvs if dv.SourceTimestamp]
                srv = [dv.ServerTimestamp for dv in dvs if dv.ServerTimestamp]
                checked = matched = 0
                minutes = []
                regmap = ROOM_MAP if unit == "room" else COMPRESSOR_MAP
                width = 5 if unit == "room" else 8
                snap_offset = 4 if unit == "room" else 7
                for slot in range(RING_SLOTS):
                    block = words[slot * SLOT_WIDTH:(slot + 1) * SLOT_WIDTH]
                    if any(w is None for w in block) or (block[0] == 0 and block[1] == 0):
                        continue
                    values = decode_block([int(w) for w in block[:width]], regmap)
                    minute = values["record_ts"]
                    if block[snap_offset] != minute_index(minute) & 0xFFFF:
                        continue
                    checked += 1
                    if unit == "room":
                        want = room.loc[minute]
                        ok = all(values[f] == want[f] for f in ROOM_FIELDS)
                    else:
                        want = cmp.loc[(unit, minute)]
                        ok = all(values[f] == want[f] for f in CMP_FIELDS)
                    matched += ok
                    minutes.append(minute.isoformat())
                snap["units"][unit] = {
                    "slots_checked": checked, "slots_matched": matched,
                    "minutes": [min(minutes), max(minutes)] if minutes else None,
                    "source_ts": [min(src).isoformat(), max(src).isoformat()] if src else None,
                    "server_ts": [min(srv).isoformat(), max(srv).isoformat()] if srv else None,
                }
            results["samples"].append(snap)
            await asyncio.sleep(interval)
    results["slots_checked"] = sum(u["slots_checked"] for s in results["samples"]
                                   for u in s["units"].values())
    results["slots_matched"] = sum(u["slots_matched"] for s in results["samples"]
                                   for u in s["units"].values())
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opcua_readback")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    results = asyncio.run(sample(config, args.samples, args.interval))
    run_dir = Path(config["paths"]["run_reports"])
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = run_dir / f"opcua_readback_{stamp}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"slots checked {results['slots_checked']}, matched {results['slots_matched']}")
    for unit, u in results["samples"][-1]["units"].items():
        print(f"  {unit}: minutes {u['minutes']} source_ts {u['source_ts']} "
              f"server_ts {u['server_ts']}")
    print(f"report {out}")
    return 0 if results["slots_checked"] and \
        results["slots_checked"] == results["slots_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
