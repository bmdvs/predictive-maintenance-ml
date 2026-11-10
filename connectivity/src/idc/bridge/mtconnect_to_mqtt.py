"""Bridge the MTConnect agent's sample stream to MQTT on an ISA-95 topic tree.

The bridge reads the agent's observation buffer in sequence order, groups observations by
machine and timestamp, and publishes each group as one JSON message on the raw edge layer of
the Unified Namespace. Groups that carry a controller record (a record_ts value) go to the
machine's controller topic; groups that carry only availability or link state (the adapter's
segment-end UNAVAILABLE marker, or the agent marking every item UNAVAILABLE when an adapter
connection drops) go to the controller/status topic. Every message carries the agent sequence
range it was built from and the agent's instance id, so a consumer can prove continuity.

Sequence handling:
- The first poll reads /current to learn nextSequence, then every poll asks /sample?from=N.
- The response's nextSequence becomes the bookmark for the next poll.
- If firstSequence > bookmark, observations were overwritten before they were read (the bridge
  fell behind the buffer); the bridge raises and stops rather than publishing a stream with a
  hole in it.
- If instanceId changes, the agent restarted and sequence numbers reset; the bridge raises.
- A poll can end in the middle of a record second, so each machine's newest group is held
  back until a later poll shows a newer timestamp for that machine (the second is complete) or
  returns nothing new (no partial line is in flight), so each second is published as one
  message.

Usage:
    python -m idc.bridge.mtconnect_to_mqtt [--config config/chain.yaml] [--once]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import paho.mqtt.client as mqtt
import requests
import yaml

log = logging.getLogger("idc.bridge")

NS = {"s": "urn:mtconnect.org:MTConnectStreams:2.7"}

# Data item name (SHDR key) -> field name in the JSON payload. These are the export's column
# names so the landing writer maps fields to columns without a second table.
FIELDS: dict[str, str] = {
    "exec": "execution",
    "program": "program",
    "Srpm": "spindle_speed_rpm",
    "Sload": "spindle_load_pct",
    "Xload": "x_load_pct",
    "Yload": "y_load_pct",
    "Zload": "z_load_pct",
    "Bload": "b_load_pct",
    "Stemp": "spindle_temp_c",
    "Fovr": "feed_override_pct",
    "Hpres": "hydraulic_pressure_bar",
    "Cpres": "coolant_pressure_bar",
}

EXECUTION_VALUES = {"ACTIVE": "active", "READY": "ready", "STOPPED": "stopped",
                    "INTERRUPTED": "interrupted"}


class SequenceGap(RuntimeError):
    """Raised when the agent's buffer no longer holds the next observations to read."""


class AgentRestarted(RuntimeError):
    """Raised when the agent's instanceId changes between polls."""


@dataclass
class Observation:
    """One data item observation from an MTConnectStreams response."""

    device: str
    name: str
    sequence: int
    timestamp: str
    value: str
    kind: str  # "sample", "event", "condition"
    level: str | None = None
    native_code: str | None = None


@dataclass
class BridgeStats:
    """Counters for one bridge run, logged at exit."""

    polls: int = 0
    observations: int = 0
    messages: int = 0  # records + status
    records: int = 0
    status: int = 0
    merged: int = 0  # held-back groups completed by observations from a later poll
    first_sequence: int | None = None
    last_sequence: int | None = None
    instance_id: int | None = None
    by_device: dict[str, int] = field(default_factory=dict)


def parse_sample(xml_text: str) -> tuple[dict, list[Observation]]:
    """Parse a /sample or /current response into its header and observations.

    Observations are returned in sequence order, which is also the order the agent accepted
    them from the adapters.
    """
    root = ET.fromstring(xml_text)
    header = root.find("s:Header", NS)
    if header is None:
        raise ValueError("no MTConnectStreams Header in response")
    head = {
        "instanceId": int(header.get("instanceId")),
        "firstSequence": int(header.get("firstSequence", 0)),
        "lastSequence": int(header.get("lastSequence", 0)),
        "nextSequence": int(header.get("nextSequence")),
        "bufferSize": int(header.get("bufferSize")),
    }
    obs: list[Observation] = []
    for dev in root.iterfind(".//s:DeviceStream", NS):
        device = dev.get("name")
        for comp in dev.iterfind("s:ComponentStream", NS):
            for group in comp:
                kind = group.tag.split("}")[1].lower()  # Samples / Events / Condition
                kind = {"samples": "sample", "events": "event", "condition": "condition"}[kind]
                for item in group:
                    tag = item.tag.split("}")[1]
                    name = item.get("name") or item.get("dataItemId")
                    obs.append(Observation(
                        device=device,
                        name=name,
                        sequence=int(item.get("sequence")),
                        timestamp=item.get("timestamp"),
                        value=(item.text or "").strip(),
                        kind=kind,
                        level=tag.upper() if kind == "condition" else None,
                        native_code=item.get("nativeCode") if kind == "condition" else None,
                    ))
    obs.sort(key=lambda o: o.sequence)
    return head, obs


@dataclass
class SecondState:
    """Observations accumulated for one machine at one timestamp."""
    device: str
    ts: str
    fields: dict = field(default_factory=dict)
    seq_lo: int | None = None
    seq_hi: int | None = None
    unavailable: int = 0  # items the agent marked UNAVAILABLE (adapter link lost)

    @property
    def is_record(self) -> bool:
        """True when the group carries a controller record, not just availability or status."""
        return "ts_utc" in self.fields

    def add(self, o: Observation) -> None:
        """Fold one observation into the group; UNAVAILABLE values are counted, not stored."""
        self.seq_lo = o.sequence if self.seq_lo is None else min(self.seq_lo, o.sequence)
        self.seq_hi = o.sequence if self.seq_hi is None else max(self.seq_hi, o.sequence)
        if o.name == "avail":
            self.fields["avail"] = o.value
        elif o.name == "record_ts":
            if o.value == "UNAVAILABLE":
                self.unavailable += 1
            else:
                self.fields["ts_utc"] = o.value
        elif o.name == "alarm":
            if o.level == "UNAVAILABLE":
                self.unavailable += 1
            else:
                self.fields["alarm_code"] = (o.native_code or "") if o.level == "FAULT" else ""
        elif o.name in FIELDS:
            key = FIELDS[o.name]
            if o.value == "UNAVAILABLE":
                self.unavailable += 1
                return
            if key == "execution":
                self.fields[key] = EXECUTION_VALUES.get(o.value, o.value.lower())
            elif key == "program":
                self.fields[key] = o.value
            else:
                self.fields[key] = float(o.value)

    def payload(self, instance_id: int) -> dict:
        """Controller record message: the changed fields plus agent sequence range."""
        body = {"machine_id": self.device, "ts_utc": self.ts}
        body.update(self.fields)
        body["agent_seq"] = [self.seq_lo, self.seq_hi]
        body["instance_id"] = instance_id
        return body

    def status_payload(self, instance_id: int) -> dict:
        """Availability or link-state message for a group that carries no record.

        source is "agent" when the agent marked items UNAVAILABLE (adapter link lost) and
        "adapter" when the adapter sent the availability change itself.
        """
        avail = self.fields.get("avail", "UNAVAILABLE" if self.unavailable else None)
        return {
            "machine_id": self.device,
            "ts_utc": self.ts,
            "avail": avail,
            "source": "agent" if self.unavailable else "adapter",
            "agent_seq": [self.seq_lo, self.seq_hi],
            "instance_id": instance_id,
        }

    def absorb(self, other: SecondState) -> None:
        """Merge another group of the same device into this one (used to collapse status)."""
        self.fields.update(other.fields)
        self.unavailable += other.unavailable
        self.seq_lo = min(self.seq_lo, other.seq_lo)
        self.seq_hi = max(self.seq_hi, other.seq_hi)


def group_by_second(observations: list[Observation],
                    pending: dict[tuple[str, str], SecondState] | None = None,
                    stats: BridgeStats | None = None) -> list[SecondState]:
    """Group observations by (device, timestamp) preserving first-seen order.

    The adapter stamps every observation for a record second with that second's timestamp,
    so timestamp equality is the grouping key. An UNAVAILABLE marker carries the first
    unpowered second's timestamp and forms its own group. Groups held back from the previous
    poll are passed in as `pending` so observations for the same second merge into them.
    """
    groups: dict[tuple[str, str], SecondState] = dict(pending or {})
    merged: set[tuple[str, str]] = set()
    for o in observations:
        key = (o.device, o.timestamp)
        if key in groups and pending and key in pending:
            merged.add(key)
        if key not in groups:
            groups[key] = SecondState(device=o.device, ts=o.timestamp)
        groups[key].add(o)
    if stats is not None:
        stats.merged += len(merged)
    return list(groups.values())


def split_complete(groups: list[SecondState]) -> tuple[list[SecondState], dict]:
    """Separate groups that are known complete from each device's newest (possibly partial).

    Groups arrive in sequence order, so the last group seen for a device is its newest. Every
    earlier group for that device is complete because a newer second has already started.
    """
    newest: dict[str, SecondState] = {}
    for g in groups:
        newest[g.device] = g
    held: set[int] = {id(g) for g in newest.values()}
    # A status-only newest group may be the tail of an agent link-loss burst whose other items
    # are earlier in this batch; hold those back too so they collapse into one message later.
    for g in newest.values():
        if g.is_record:
            continue
        for other in reversed(groups):
            if other.device != g.device or other is g or id(other) in held:
                continue
            if other.is_record or other.ts[:19] != g.ts[:19]:
                break
            held.add(id(other))
    complete = [g for g in groups if id(g) not in held]
    pending = {(g.device, g.ts): g for g in groups if id(g) in held}
    return complete, pending


def collapse_status(groups: list[SecondState]) -> list[SecondState]:
    """Merge consecutive status-only groups of one device that fall in the same second.

    When an adapter connection drops, the agent marks every data item UNAVAILABLE with its
    own clock, microseconds apart. Those are one event, so they become one status message.
    """
    out: list[SecondState] = []
    for g in groups:
        prev = out[-1] if out else None
        if (prev is not None and not g.is_record and not prev.is_record
                and prev.device == g.device and prev.ts[:19] == g.ts[:19]):
            prev.absorb(g)
        else:
            out.append(g)
    return out


def topic_for(machine_id: str, config: dict, cells: dict[str, str],
              kind: str = "controller") -> str:
    """ISA-95 topic for a machine's controller records, or its status when kind='status'."""
    key = "controller_topic" if kind == "controller" else "status_topic"
    return config["mqtt"][key].format(
        root=config["mqtt"]["topic_root"], cell=cells[machine_id], machine_id=machine_id)


class Bridge:
    """Polls the agent's sample stream and publishes grouped observations to MQTT.

    Connects to the broker on construction. `machine_cells` maps machine id to its ISA-95 cell;
    observations for devices not in the map are dropped.
    """

    def __init__(self, config: dict, machine_cells: dict[str, str],
                 devices: list[str] | None = None, client_id: str = "idc-bridge"):
        self.config = config
        self.cells = machine_cells
        self.device_filter = None
        if devices:
            names = " or ".join(f'@name="{d}"' for d in devices)
            self.device_filter = f"//Device[{names}]"
        self.agent = config["mtconnect"]["agent_url"].rstrip("/")
        self.count = int(config["mtconnect"].get("sample_count", 5000))
        self.idle = float(config["mtconnect"].get("idle_poll_seconds", 0.5))
        self.qos = int(config["mqtt"].get("qos", 1))
        self.stats = BridgeStats()
        self.bookmark: int | None = None
        self.instance_id: int | None = None
        self.pending: dict[tuple[str, str], SecondState] = {}
        self.http = requests.Session()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                                  protocol=mqtt.MQTTv5)
        self.client.max_inflight_messages_set(1000)
        self.client.max_queued_messages_set(0)
        self.client.connect(config["mqtt"]["host"], int(config["mqtt"]["port"]))
        self.client.loop_start()

    def _get(self, endpoint: str, **params) -> str:
        r = self.http.get(f"{self.agent}/{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.text

    def prime(self) -> None:
        """Read /current to learn where the buffer is now; publishing starts from there."""
        head, _ = parse_sample(self._get("current"))
        self.instance_id = head["instanceId"]
        self.bookmark = head["nextSequence"]
        self.stats.instance_id = head["instanceId"]
        log.info("agent instanceId=%d nextSequence=%d bufferSize=%d",
                 head["instanceId"], head["nextSequence"], head["bufferSize"])

    def poll_once(self) -> int:
        """Read one batch from the buffer, publish it, advance the bookmark. Returns count."""
        params = {"from": self.bookmark, "count": self.count}
        if self.device_filter:
            params["path"] = self.device_filter
        head, obs = parse_sample(self._get("sample", **params))
        self.stats.polls += 1
        if head["instanceId"] != self.instance_id:
            raise AgentRestarted(
                f"agent instanceId changed {self.instance_id} -> {head['instanceId']}; "
                f"sequence bookmark {self.bookmark} is no longer valid")
        if head["firstSequence"] > self.bookmark:
            raise SequenceGap(
                f"buffer overran: wanted from {self.bookmark}, oldest available is "
                f"{head['firstSequence']} "
                f"({head['firstSequence'] - self.bookmark} observations lost)")
        if not obs:
            # No observations for this bridge's devices in the scanned window. Advance past
            # the window so a filtered bridge keeps pace with the buffer while its devices
            # are idle; once caught up with the agent, the held-back seconds are complete.
            self.bookmark = max(self.bookmark, head["nextSequence"])
            if head["nextSequence"] > head["lastSequence"]:
                self.flush()
            return 0
        complete, self.pending = split_complete(group_by_second(obs, self.pending, self.stats))
        self._publish(complete)
        self.stats.observations += len(obs)
        self.stats.first_sequence = self.stats.first_sequence or obs[0].sequence
        self.stats.last_sequence = obs[-1].sequence
        self.bookmark = head["nextSequence"]
        return len(obs)

    def _publish(self, groups: list[SecondState]) -> None:
        """Publish groups at QoS 1; returns once the broker has acknowledged every message."""
        pending = []
        for group in collapse_status(groups):
            if group.device not in self.cells:
                continue
            if group.is_record:
                topic = topic_for(group.device, self.config, self.cells)
                body = group.payload(self.instance_id)
                self.stats.records += 1
                self.stats.by_device[group.device] = self.stats.by_device.get(group.device, 0) + 1
            else:
                topic = topic_for(group.device, self.config, self.cells, kind="status")
                body = group.status_payload(self.instance_id)
                self.stats.status += 1
            pending.append(self.client.publish(topic, json.dumps(body, separators=(",", ":")),
                                               qos=self.qos))
            self.stats.messages += 1
        for info in pending:
            info.wait_for_publish(timeout=60)
            if not info.is_published():
                raise RuntimeError(f"broker did not acknowledge message {info.mid}")

    def flush(self) -> None:
        """Publish every held-back group; called when the agent has nothing new and at exit."""
        groups = list(self.pending.values())
        self.pending = {}
        self._publish(groups)

    def run(self, once: bool = False) -> BridgeStats:
        """Poll until interrupted, or until the buffer is drained when `once` is set.

        Held-back groups are flushed and the broker connection closed on every exit path.
        Raises SequenceGap or AgentRestarted when continuity cannot be guaranteed.
        """
        self.prime()
        try:
            while True:
                n = self.poll_once()
                if once and n == 0:
                    break
                if n == 0:
                    time.sleep(self.idle)
                elif self.stats.polls % 20 == 0:
                    log.info("polls=%d observations=%d messages=%d bookmark=%d",
                             self.stats.polls, self.stats.observations, self.stats.messages,
                             self.bookmark)
        finally:
            try:
                self.flush()
            finally:
                self.client.loop_stop()
                self.client.disconnect()
        return self.stats


def load_machine_cells(register_path: Path, cells_by_class: dict[str, str]) -> dict[str, str]:
    """Map each machine id in the fleet register to the cell configured for its class."""
    register = yaml.safe_load(register_path.read_text(encoding="utf-8"))
    return {m["machine_id"]: cells_by_class[m["class"]] for m in register["machines"]}


def main(argv: list[str] | None = None) -> int:
    """Run the bridge. Exits 2 on a sequence gap or agent restart, 0 otherwise."""
    parser = argparse.ArgumentParser(prog="mtconnect_to_mqtt")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--once", action="store_true",
                        help="drain the buffer once and exit instead of polling forever")
    parser.add_argument("--devices", default=None,
                        help="comma-separated device names to bridge (default: all)")
    parser.add_argument("--client-id", default="idc-bridge")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    cells = load_machine_cells(Path(config["paths"]["register"]), config["cells"])
    devices = args.devices.split(",") if args.devices else None
    bridge = Bridge(config, cells, devices, args.client_id)
    try:
        stats = bridge.run(once=args.once)
    except (SequenceGap, AgentRestarted) as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        stats = bridge.stats
    log.info("done: polls=%d observations=%d records=%d status=%d merged=%d "
             "sequences %s..%s instanceId=%s", stats.polls, stats.observations, stats.records,
             stats.status, stats.merged, stats.first_sequence, stats.last_sequence,
             stats.instance_id)
    for device, n in sorted(stats.by_device.items()):
        log.info("  %s: %d records", device, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
