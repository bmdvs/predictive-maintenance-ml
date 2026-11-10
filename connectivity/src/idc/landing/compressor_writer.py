"""Land compressor and room controller minutes from the Sparkplug B namespace.

Subscribes to ``spBv1.0/<group>/+/<edge node>/#`` with a persistent MQTT v5 session and
rebuilds record minutes from the ring-buffer register metrics each device publishes
(``sNN_rK``: slot NN, register K; see docs/register_map.md):

- Births (NBIRTH/DBIRTH) define each device's metric names and aliases and seed its register
  values. Data for a device without a birth triggers one rebirth request (NCMD) per edge node
  session and is otherwise ignored until the birth arrives. Deaths clear the device state.
- After each DDATA message, every slot whose record_ts registers and snapshot register agree
  (snapshot equals the record minute's index mod 65536) is a candidate. A candidate is
  accepted when the device's next message shows the slot unchanged, which proves all
  register changes from the same read have arrived. When a device goes quiet (end of a
  replay chunk, a paused replay, a death, or writer shutdown) candidates unchanged for at
  least ``settle_seconds`` are accepted, since no further changes from their read can follow.
- Accepted slots decode by division per the register map and are keyed on
  (compressor_id, minute_utc) or (minute_utc) for the room; a later observation of the same
  minute with different values is counted as a conflict and the later value kept.

Output mirrors the bronze export layout under ``landing_root``:
``compressor_ctrl/minute/month=YYYY-MM/compressor_minute.parquet`` and
``compressor_ctrl/room/month=YYYY-MM/room.parquet``, with the export's columns, order and
dtypes plus the source run constants. Files are merged on their keys, so reruns are
idempotent. A run report with minutes landed, gaps against an expected window, births,
deaths and conflicts is written under ``run_reports``.

Usage:
    python -m idc.landing.compressor_writer [--idle-exit 60] [--stop-file runs/cmp.stop]
        [--expect-start 2026-06-01T00:00Z --expect-minutes 1440]
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import paho.mqtt.client as mqtt
import pandas as pd
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

from idc.landing import sparkplug
from idc.replay.register_map import (
    COMPRESSOR_MAP,
    RING_SLOTS,
    ROOM_MAP,
    SLOT_WIDTH,
    decode_block,
    minute_index,
)

log = logging.getLogger("idc.compressor_writer")

REGISTER_METRIC = re.compile(r"^s(\d{2})_r(\d)$")
ROOM_DEVICE = "room"
CMP_COLUMNS = ["compressor_id", "minute_utc", "state", "discharge_pressure_bar",
               "discharge_temp_c", "run_hours", "run_id", "config_hash"]
ROOM_COLUMNS = ["minute_utc", "ambient_c", "dryer_dew_point_c", "run_id", "config_hash"]
CMP_FIELDS = ["state", "discharge_pressure_bar", "discharge_temp_c", "run_hours"]
ROOM_FIELDS = ["ambient_c", "dryer_dew_point_c"]


@dataclass
class WriterStats:
    messages: int = 0
    data_messages: int = 0
    births: list[dict] = field(default_factory=list)
    deaths: list[dict] = field(default_factory=list)
    rebirth_requests: int = 0
    data_before_birth: int = 0
    accepted: int = 0
    duplicates: int = 0
    conflicts: list[dict] = field(default_factory=list)
    rows_written: dict[str, int] = field(default_factory=dict)


class DeviceState:
    """Register values and slot candidates for one Sparkplug device."""

    def __init__(self, name: str):
        self.name = name
        self.aliases: dict[int, str] = {}
        self.registers: dict[tuple[int, int], int | None] = {}
        self.candidates: dict[int, tuple[pd.Timestamp, tuple[int, ...]]] = {}
        self.candidate_since: dict[int, float] = {}
        self.last_update = 0.0
        self.born = False

    @property
    def regmap(self):
        return ROOM_MAP if self.name == ROOM_DEVICE else COMPRESSOR_MAP

    def birth(self, metrics: list[sparkplug.Metric]) -> None:
        self.aliases = {}
        self.registers = {}
        self.candidates = {}
        for metric in metrics:
            if metric.name is None:
                continue
            if metric.alias is not None:
                self.aliases[metric.alias] = metric.name
            self._set(metric.name, metric)
        self.born = True

    def death(self) -> None:
        self.registers = {}
        self.candidates = {}
        self.candidate_since = {}
        self.born = False

    def stable_candidates(self, now: float, settle_seconds: float) -> list[dict]:
        """Decode candidates unchanged for settle_seconds, removing them from the set."""
        out = []
        for slot, since in list(self.candidate_since.items()):
            if now - since >= settle_seconds and now - self.last_update >= settle_seconds:
                _, words = self.candidates.pop(slot)
                del self.candidate_since[slot]
                width = 5 if self.name == ROOM_DEVICE else 8
                out.append(decode_block(list(words[:width]), self.regmap))
        return out

    def _set(self, name: str, metric: sparkplug.Metric) -> None:
        match = REGISTER_METRIC.match(name)
        if match:
            value = None if metric.is_null or metric.value is None else int(metric.value)
            self.registers[(int(match.group(1)), int(match.group(2)))] = value

    def apply(self, metrics: list[sparkplug.Metric]) -> None:
        for metric in metrics:
            name = metric.name if metric.name is not None else self.aliases.get(metric.alias)
            if name is not None:
                self._set(name, metric)

    def slot_words(self, slot: int) -> tuple[int, ...] | None:
        words = tuple(self.registers.get((slot, k)) for k in range(SLOT_WIDTH))
        return None if any(w is None for w in words) else words

    def consistent_minute(self, words: tuple[int, ...]) -> pd.Timestamp | None:
        """The record minute a slot holds, if its timestamp, snapshot and slot agree."""
        epoch = (words[0] << 16) | words[1]
        if epoch == 0 or epoch % 60:
            return None
        minute = pd.Timestamp(epoch, unit="s", tz="UTC").as_unit("ns")
        snapshot_offset = 4 if self.name == ROOM_DEVICE else 7
        index = minute_index(minute)
        if words[snapshot_offset] != index & 0xFFFF:
            return None
        return minute

    def settle(self, slot: int, words: tuple[int, ...] | None) -> dict | None:
        """Advance one slot after a message; return a decoded record once it is confirmed."""
        now = time.monotonic()
        if words is None:
            self.candidates.pop(slot, None)
            self.candidate_since.pop(slot, None)
            return None
        minute = self.consistent_minute(words)
        if minute is None or minute_index(minute) % RING_SLOTS != slot:
            self.candidates.pop(slot, None)
            self.candidate_since.pop(slot, None)
            return None
        previous = self.candidates.get(slot)
        if previous != (minute, words):
            self.candidates[slot] = (minute, words)
            self.candidate_since[slot] = now
            return None
        width = 5 if self.name == ROOM_DEVICE else 8
        return decode_block(list(words[:width]), self.regmap)


class CompressorLanding:
    """Accepted records keyed per table; merge-written per month."""

    def __init__(self, landing_root: Path, run_id: str, config_hash: str,
                 settle_seconds: float = 1.0):
        self.settle_seconds = settle_seconds
        self.root = landing_root
        self.run_id = run_id
        self.config_hash = config_hash
        self.devices: dict[str, DeviceState] = {}
        self.cmp_rows: dict[tuple[str, pd.Timestamp], dict] = {}
        self.room_rows: dict[pd.Timestamp, dict] = {}
        self.landed_minutes: dict[str, set] = {}
        self.stats = WriterStats()

    def device(self, name: str) -> DeviceState:
        if name not in self.devices:
            self.devices[name] = DeviceState(name)
        return self.devices[name]

    def handle(self, topic: str, payload: bytes) -> str | None:
        """Process one Sparkplug message. Returns an edge node needing a rebirth, if any."""
        self.stats.messages += 1
        t = sparkplug.Topic.parse(topic)
        if t.message_type in ("NCMD", "DCMD", "STATE"):
            return None
        ts, seq, metrics = sparkplug.decode(payload)
        event = {"type": t.message_type, "device": t.device, "timestamp_ms": ts,
                 "received_utc": datetime.now(UTC).isoformat()}
        if t.message_type == "NBIRTH":
            self.stats.births.append(event)
            return None
        if t.message_type == "NDEATH":
            self.stats.deaths.append(event)
            for state in self.devices.values():
                self.accept_stable(state, force=True)
                state.death()
            return None
        if t.device is None:
            return None
        state = self.device(t.device)
        if t.message_type == "DBIRTH":
            self.stats.births.append(event)
            state.birth(metrics)
            self._settle(state)
            return None
        if t.message_type == "DDEATH":
            self.stats.deaths.append(event)
            self.accept_stable(state, force=True)
            state.death()
            return None
        if t.message_type == "DDATA":
            self.stats.data_messages += 1
            if not state.born:
                self.stats.data_before_birth += 1
                return t.edge_node
            state.apply(metrics)
            state.last_update = time.monotonic()
            self._settle(state)
        return None

    def accept_stable(self, state: DeviceState, force: bool = False) -> None:
        """Accept a device's settled candidates (all of them when `force`)."""
        settle = 0.0 if force else self.settle_seconds
        for values in state.stable_candidates(time.monotonic() + (1e9 if force else 0),
                                              settle):
            self._accept(state.name, values)

    def accept_all_stable(self) -> None:
        for state in self.devices.values():
            self.accept_stable(state)

    def _settle(self, state: DeviceState) -> None:
        for slot in range(RING_SLOTS):
            values = state.settle(slot, state.slot_words(slot))
            if values is not None:
                self._accept(state.name, values)

    def _accept(self, device: str, values: dict) -> None:
        minute = values["record_ts"]
        if device == ROOM_DEVICE:
            key, store, fields = minute, self.room_rows, ROOM_FIELDS
        else:
            key, store, fields = (device, minute), self.cmp_rows, CMP_FIELDS
        row = {f: values[f] for f in fields}
        existing = store.get(key)
        if existing is not None:
            if existing == row:
                self.stats.duplicates += 1
                return
            self.stats.conflicts.append({"device": device, "minute_utc": minute.isoformat(),
                                         "first": existing, "later": row})
        store[key] = row
        self.stats.accepted += 1
        self.landed_minutes.setdefault(device, set()).add(minute)

    def frames(self) -> dict[str, tuple[Path, pd.DataFrame, list[str]]]:
        """Frames per output file: {path: (path, frame, key columns)}."""
        out: dict[str, tuple[Path, pd.DataFrame, list[str]]] = {}
        by_month: dict[str, list] = {}
        for (device, minute), row in self.cmp_rows.items():
            by_month.setdefault(minute.strftime("%Y-%m"), []).append(
                {"compressor_id": device, "minute_utc": minute, **row})
        for month, rows in by_month.items():
            path = self.root / "compressor_ctrl" / "minute" / f"month={month}" / \
                "compressor_minute.parquet"
            out[str(path)] = (path, self._frame(rows, CMP_COLUMNS),
                              ["compressor_id", "minute_utc"])
        by_month = {}
        for minute, row in self.room_rows.items():
            by_month.setdefault(minute.strftime("%Y-%m"), []).append({"minute_utc": minute,
                                                                      **row})
        for month, rows in by_month.items():
            path = self.root / "compressor_ctrl" / "room" / f"month={month}" / "room.parquet"
            out[str(path)] = (path, self._frame(rows, ROOM_COLUMNS), ["minute_utc"])
        return out

    def _frame(self, rows: list[dict], columns: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        frame["minute_utc"] = pd.DatetimeIndex(frame["minute_utc"]).tz_convert("UTC") \
            .as_unit("ns")
        for column in columns:
            if column in ("compressor_id", "state"):
                frame[column] = frame[column].astype("str")
            elif column not in ("minute_utc", "run_id", "config_hash"):
                frame[column] = frame[column].astype("float64")
        frame["run_id"] = pd.array([self.run_id] * len(frame), dtype="str")
        frame["config_hash"] = pd.array([self.config_hash] * len(frame), dtype="str")
        return frame[columns]

    def write_all(self) -> list[Path]:
        written = []
        for path, frame, keys in self.frames().values():
            if path.exists():
                merged = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
                frame = merged.drop_duplicates(subset=keys, keep="last")
            frame = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            self.stats.rows_written[str(path)] = len(frame)
            written.append(path)
        return written


def expected_gaps(landed: dict[str, set], devices: list[str], start: pd.Timestamp,
                  minutes: int) -> dict[str, list[str]]:
    """Missing record minutes per device within [start, start + minutes)."""
    window = pd.date_range(start, periods=minutes, freq="1min", tz="UTC").as_unit("ns")
    gaps: dict[str, list[str]] = {}
    for device in devices:
        have = landed.get(device, set())
        missing = [m.isoformat() for m in window if m not in have]
        if missing:
            gaps[device] = missing
    return gaps


def run(config: dict, idle_exit: float, session_expiry: int, clean: bool,
        stop_file: Path | None, expect_start: pd.Timestamp | None,
        expect_minutes: int | None, flush_seconds: float = 300.0) -> int:
    """Consume Sparkplug messages until idle or stopped; write landing files and a report.

    Returns 2 if an expected window was given and minutes are missing, else 0.
    """
    ign = config["ignition"]
    landing = CompressorLanding(Path(config["paths"]["landing_root"]),
                                config["source_run"]["run_id"],
                                config["source_run"]["config_hash"])
    inbox: queue.Queue = queue.Queue()
    group, node = ign["sparkplug_group"], ign["sparkplug_edge_node"]
    subscription = f"{sparkplug.NAMESPACE}/{group}/+/{node}/#"

    def on_connect(client, userdata, flags, reason_code, properties):
        log.info("connected (session present=%s); subscribing %s",
                 getattr(flags, "session_present", False), subscription)
        client.subscribe(subscription, qos=1)

    def on_message(client, userdata, msg):
        inbox.put((msg.topic, msg.payload))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="idc-compressor-writer",
                         protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    props = Properties(PacketTypes.CONNECT)
    props.SessionExpiryInterval = session_expiry
    client.connect(config["mqtt"]["host"], int(config["mqtt"]["port"]), clean_start=clean,
                   properties=props)
    client.loop_start()

    rebirth_sent: set[str] = set()
    last_message = None
    last_flush = last_log = time.monotonic()
    written: dict[str, Path] = {}
    try:
        while True:
            try:
                topic, payload = inbox.get(timeout=0.5)
            except queue.Empty:
                landing.accept_all_stable()
                if stop_file is not None and stop_file.exists():
                    log.info("stop file %s present; finishing", stop_file)
                    break
                if idle_exit > 0 and last_message is not None and \
                        time.monotonic() - last_message > idle_exit:
                    log.info("no messages for %.0fs; finishing", idle_exit)
                    break
                continue
            last_message = time.monotonic()
            needs_rebirth = landing.handle(topic, payload)
            if needs_rebirth and needs_rebirth not in rebirth_sent:
                client.publish(f"{sparkplug.NAMESPACE}/{group}/NCMD/{needs_rebirth}",
                               sparkplug.rebirth_command(int(time.time() * 1000)), qos=0)
                rebirth_sent.add(needs_rebirth)
                landing.stats.rebirth_requests += 1
                log.info("data before birth from %s; rebirth requested", needs_rebirth)
            if topic.split("/")[2] == "NBIRTH":
                rebirth_sent.discard(topic.split("/")[3])
            if time.monotonic() - last_log > 15:
                last_log = time.monotonic()
                log.info("messages=%d accepted=%d compressor_rows=%d room_rows=%d",
                         landing.stats.messages, landing.stats.accepted,
                         len(landing.cmp_rows), len(landing.room_rows))
            if time.monotonic() - last_flush > flush_seconds:
                for path in landing.write_all():
                    written[str(path)] = path
                last_flush = time.monotonic()
    except KeyboardInterrupt:
        log.info("interrupted; writing what was received")
    finally:
        client.loop_stop()
        client.disconnect()

    for state in landing.devices.values():
        landing.accept_stable(state, force=True)
    for path in landing.write_all():
        written[str(path)] = path
    gaps: dict[str, list[str]] = {}
    if expect_start is not None and expect_minutes:
        devices = list(config["modbus"]["unit_ids"])
        gaps = expected_gaps(landing.landed_minutes, devices, expect_start, expect_minutes)
    report = write_report(Path(config["paths"]["run_reports"]), landing, list(written.values()),
                          gaps, expect_start, expect_minutes)
    s = landing.stats
    log.info("done: messages=%d data=%d accepted=%d duplicates=%d conflicts=%d births=%d "
             "deaths=%d rebirth_requests=%d gap_minutes=%d", s.messages, s.data_messages,
             s.accepted, s.duplicates, len(s.conflicts), len(s.births), len(s.deaths),
             s.rebirth_requests, sum(len(v) for v in gaps.values()))
    log.info("report %s", report)
    return 2 if gaps else 0


def write_report(run_dir: Path, landing: CompressorLanding, written: list[Path],
                 gaps: dict[str, list[str]], expect_start, expect_minutes) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = run_dir / f"compressor_landing_{stamp}.json"
    s = landing.stats
    body = {
        "finished_utc": stamp,
        "messages": s.messages, "data_messages": s.data_messages, "accepted": s.accepted,
        "duplicates": s.duplicates, "conflicts": s.conflicts[:100],
        "conflict_count": len(s.conflicts), "rebirth_requests": s.rebirth_requests,
        "data_before_birth": s.data_before_birth,
        "births": s.births[-200:], "deaths": s.deaths[-200:],
        "minutes_landed": {d: len(v) for d, v in landing.landed_minutes.items()},
        "expected_window": {"start": expect_start.isoformat() if expect_start is not None
                            else None, "minutes": expect_minutes},
        "minutes_expected_per_device": expect_minutes,
        "gaps": {d: {"count": len(v), "first": v[:20]} for d, v in gaps.items()},
        "files": {str(p): s.rows_written.get(str(p)) for p in written},
    }
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compressor_writer")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--idle-exit", type=float, default=60.0,
                        help="seconds without a message before finishing (0 disables)")
    parser.add_argument("--session-expiry", type=int, default=3600)
    parser.add_argument("--clean", action="store_true",
                        help="discard any queued session at the broker and start fresh")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="finish, write and exit once this file exists")
    parser.add_argument("--expect-start", default=None,
                        help="first record minute of the expected window, for the gap report")
    parser.add_argument("--expect-minutes", type=int, default=None,
                        help="length of the expected window in minutes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    start = None
    if args.expect_start:
        start = pd.Timestamp(args.expect_start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    return run(config, args.idle_exit, args.session_expiry, args.clean, args.stop_file, start,
               args.expect_minutes)


if __name__ == "__main__":
    raise SystemExit(main())
