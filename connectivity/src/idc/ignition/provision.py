"""Provision the Ignition gateway's file-based configuration for the compressor chain.

Writes, under the gateway data folder (``ignition/data`` by default):

- MQTT Transmission: the server URL (``tcp://mosquitto:1883``) and the transmitter's Sparkplug
  group ID, edge node ID, tag path and tag pacing period.
- Tag group ``Fast``: direct mode at the configured rate.
- UDT ``Compressor`` and instances ``CMP-01`` .. ``CMP-04`` and ``room`` in the
  ``compressors`` folder of the default provider. Each instance has one unsigned 16-bit OPC tag
  per ring-buffer register (``s00_r0`` .. ``s14_r7``), read from the Modbus device with the
  unit-ID prefix ``<unit>.HRUS<n>``. The device uses one-based addressing (zero-based
  addressing off), so register address a is ``HRUS<a+1>``. No scaling, deadband or history.

Also writes a tag import file in Ignition's tag export format to
``ignition/tags/compressors.json`` for import through the Designer.

The gateway reads these files at startup; restart it after provisioning.

Usage:
    python -m idc.ignition.provision [--data ignition/data]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idc.replay.register_map import RING_SLOTS, SLOT_WIDTH

TRANSMISSION = "com.cirruslink.mqtt.transmission.gateway"
OPC_SERVER = "Ignition OPC UA Server"
UDT_NAME = "Compressor"
OPEN_PERMISSIONS = {"securityLevels": [], "type": "AllOf"}


def register_tag_names() -> list[tuple[str, int]]:
    """(tag name, one-based HRUS address) for every ring-buffer register."""
    return [(f"s{slot:02d}_r{word}", slot * SLOT_WIDTH + word + 1)
            for slot in range(RING_SLOTS) for word in range(SLOT_WIDTH)]


def udt_member(name: str, address: int, device: str, tag_group: str) -> dict:
    return {
        "name": name,
        "tagType": "AtomicTag",
        "valueSource": "opc",
        "dataType": "Int4",
        "opcServer": OPC_SERVER,
        "opcItemPath": {"bindType": "parameter",
                        "binding": f"ns=1;s=[{device}]{{UnitId}}.HRUS{address}"},
        "tagGroup": tag_group,
        "historyEnabled": False,
        "deadband": 0.0,
        "scaleMode": "Off",
    }


def udt_definition(device: str, tag_group: str) -> dict:
    return {
        "name": UDT_NAME,
        "tagType": "UdtType",
        "parameters": {"UnitId": {"dataType": "Integer", "value": 1}},
        "tags": [udt_member(n, a, device, tag_group) for n, a in register_tag_names()],
    }


def udt_instance(name: str, unit_id: int) -> dict:
    return {"name": name, "tagType": "UdtInstance", "typeId": UDT_NAME,
            "parameters": {"UnitId": {"dataType": "Integer", "value": unit_id}}}


def tag_export(unit_ids: dict[str, int], folder: str, device: str, tag_group: str) -> dict:
    """The full tag tree in Ignition's export format (UDT definition plus instances)."""
    return {"name": "", "tagType": "Provider", "tags": [
        {"name": "_types_", "tagType": "Folder", "tags": [udt_definition(device, tag_group)]},
        {"name": folder, "tagType": "Folder",
         "tags": [udt_instance(name, uid) for name, uid in unit_ids.items()]},
    ]}


def _resource(files: list[str], config: dict | None = None, unary: bool = False) -> dict:
    body: dict = {"scope": "G" if unary else "A", "version": 1, "restricted": False,
                  "overridable": True, "files": files}
    body["attributes"] = {"config": config or {}} if unary else {}
    return body


def _write_json(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")


def _update_json(path: Path, changes: dict) -> None:
    body = json.loads(path.read_text(encoding="utf-8"))
    body.update(changes)
    _write_json(path, body)


def provision(data_root: Path, config: dict) -> list[Path]:
    """Write the gateway configuration resources; returns the files written."""
    ign = config["ignition"]
    unit_ids = config["modbus"]["unit_ids"]
    core = data_root / "config" / "resources" / "core"
    written: list[Path] = []

    servers = sorted((core / TRANSMISSION / "server").iterdir())
    transmitters = sorted((core / TRANSMISSION / "transmitter").iterdir())
    if len(servers) != 1 or len(transmitters) != 1:
        raise RuntimeError("expected exactly one MQTT Transmission server and transmitter")
    _update_json(servers[0] / "config.json", {"url": "tcp://mosquitto:1883"})
    _update_json(transmitters[0] / "config.json", {
        "groupId": ign["sparkplug_group"], "edgeNodeId": ign["sparkplug_edge_node"],
        "tagPath": ign["tag_folder"], "tagProvider": "default",
        "tagPacingPeriod": int(ign.get("tag_pacing_ms", 100)),
        "deviceLevelUdtsAsDevices": True,
    })
    written += [servers[0] / "config.json", transmitters[0] / "config.json"]

    group_dir = core / "ignition" / "tag-group" / "default" / ign["tag_group"]
    _write_json(group_dir / "config.json", {"config": {
        "mode": "Direct", "oneShot": False, "opcDataMode": "Subscribed",
        "optWriteTimeout": 60000, "optWrites": False, "rate": int(ign["tag_group_rate_ms"]),
        "readAfterWrite": False}})
    _write_json(group_dir / "resource.json", _resource(["config.json"]))
    written.append(group_dir / "config.json")

    types_root = core / "ignition" / "tag-type-definition" / "default"
    stale = types_root / UDT_NAME
    if stale.exists():
        for child in stale.iterdir():
            child.unlink()
        stale.rmdir()
    _write_json(types_root / "tags.json", [udt_definition(ign["modbus_device"],
                                                          ign["tag_group"])])
    _write_json(types_root / "unary-resource.json", _resource(["tags.json"], {}, True))
    written.append(types_root / "tags.json")

    folder_dir = core / "ignition" / "tag-definition" / "default" / ign["tag_folder"]
    _write_json(folder_dir / "tags.json",
                [udt_instance(name, uid) for name, uid in unit_ids.items()])
    _write_json(folder_dir / "unary-resource.json", _resource(["tags.json"], {}, True))
    written.append(folder_dir / "tags.json")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="provision")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--data", type=Path, default=Path("ignition/data"))
    parser.add_argument("--export", type=Path, default=Path("ignition/tags/compressors.json"))
    args = parser.parse_args(argv)
    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    ign = config["ignition"]
    _write_json(args.export, tag_export(config["modbus"]["unit_ids"], ign["tag_folder"],
                                        ign["modbus_device"], ign["tag_group"]))
    print(f"wrote {args.export}")
    for path in provision(args.data, config):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
