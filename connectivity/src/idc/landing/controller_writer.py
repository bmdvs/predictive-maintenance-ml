"""Land controller records from the raw UNS layer as bronze hz1 Parquet files.

The writer subscribes to every machine's controller and controller/status topics, rebuilds the
full 1 Hz row from report-by-exception messages (a message carries only the values that
changed, so unchanged values are carried forward from the previous second), and writes one
Parquet file per machine per month in the bronze export layout:

    <landing_root>/controller/hz1/month=YYYY-MM/<machine_id>.parquet

The month is the plant-local month of the row (``landing.controller_partition_timezone``),
matching the export, which files each local shift day under its local date's month.

Behavior:
- Rows are keyed on (machine_id, ts_utc). A message delivered twice (QoS 1) or a day replayed
  twice produces one row; the write merges with any existing file on the key.
- Carry-forward follows the agent's own state model. The agent holds the current value of
  every data item and emits an observation only when a value changes, so a value that is the
  same after a shutdown as before it is never re-sent. Values therefore carry across the
  adapter's segment markers. State is cleared only when the agent reports the adapter link
  lost (status source "agent"), because the agent itself marked every item UNAVAILABLE.
- Within a segment the record timestamp must advance by exactly one second. A larger step is
  a missing message and is reported as a gap; the run exits non-zero if any gap was seen.
- alarm_code is not carried forward: the export holds it only on the alarm second.
- A message that cannot be decoded (invalid JSON, a missing field or an unparseable record
  time) is counted and logged as rejected and does not change machine state; the run exits
  non-zero if any message was rejected.

Usage:
    python -m idc.landing.controller_writer [--config config/chain.yaml] [--idle-exit 15]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import paho.mqtt.client as mqtt
import pandas as pd
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

log = logging.getLogger("idc.controller_writer")

# Bronze hz1 contract: machine_id, ts_utc, then 13 signals. Mirrors
# pdm.ingest.mtconnect_stream.HZ1_COLUMNS; export files add run_id and config_hash.
SIGNALS: list[str] = [
    "execution", "program", "spindle_speed_rpm", "spindle_load_pct", "x_load_pct",
    "y_load_pct", "z_load_pct", "b_load_pct", "spindle_temp_c", "feed_override_pct",
    "hydraulic_pressure_bar", "coolant_pressure_bar", "alarm_code",
]
HZ1_COLUMNS: list[str] = ["machine_id", "ts_utc", *SIGNALS]
FILE_COLUMNS: list[str] = [*HZ1_COLUMNS, "run_id", "config_hash"]
STRING_COLUMNS = {"machine_id", "execution", "program", "alarm_code", "run_id", "config_hash"}
CARRIED: list[str] = [s for s in SIGNALS if s != "alarm_code"]
KEY = ["machine_id", "ts_utc"]
ONE_SECOND = pd.Timedelta(seconds=1)
DEFAULT_PARTITION_TZ = "UTC"
REJECT_SAMPLES = 20


def partition_month(ts: pd.Timestamp, tz: str) -> str:
    """YYYY-MM month of `ts` in time zone `tz`, the landing file's partition."""
    return ts.tz_convert(tz).strftime("%Y-%m")


@dataclass
class Gap:
    """Seconds missing between two consecutive records of one segment."""

    machine_id: str
    after: str
    before: str
    missing_seconds: int


@dataclass
class WriterStats:
    """Counters for one writer run, written to the run report."""

    messages: int = 0
    records: int = 0
    status: int = 0
    duplicates: int = 0
    segments: int = 0
    link_losses: int = 0
    gaps: list[Gap] = field(default_factory=list)
    rejected: int = 0
    rejected_samples: list[dict] = field(default_factory=list)
    instance_ids: set = field(default_factory=set)
    rows_written: dict[str, int] = field(default_factory=dict)


class MachineState:
    """Carry-forward state for one machine between record messages."""

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.values: dict = {}
        self.prev_ts: pd.Timestamp | None = None

    def reset(self) -> None:
        """Forget all carried values and the continuity reference."""
        self.values = {}
        self.prev_ts = None

    def apply(self, record: dict, stats: WriterStats) -> dict:
        """Fold one record message into the state and return the full row."""
        ts = pd.Timestamp(record["ts_utc"]).tz_convert("UTC").as_unit("ns")
        if record.get("avail") == "AVAILABLE":
            # Segment start. Values persist in the agent across the break, so the state is
            # kept; only the one-second continuity check restarts here.
            self.prev_ts = None
            stats.segments += 1
        if self.prev_ts is not None and ts - self.prev_ts != ONE_SECOND:
            missing = int((ts - self.prev_ts) / ONE_SECOND) - 1
            stats.gaps.append(Gap(self.machine_id, self.prev_ts.isoformat(), ts.isoformat(),
                                  missing))
        for column in CARRIED:
            if column in record:
                self.values[column] = record[column]
        row = {"machine_id": self.machine_id, "ts_utc": ts}
        for column in CARRIED:
            row[column] = self.values.get(column, math.nan if column not in STRING_COLUMNS
                                          else None)
        row["alarm_code"] = record.get("alarm_code", "")
        self.prev_ts = ts
        return row

    def close(self, status: dict, stats: WriterStats) -> None:
        """Handle a status message.

        A link loss (source "agent") clears the state: the agent marked every item
        UNAVAILABLE, so nothing can be carried into the next record. A segment end from the
        adapter only ends the one-second continuity check; the agent still holds the values.
        """
        if status.get("source") == "agent":
            stats.link_losses += 1
            self.reset()
            return
        if self.prev_ts is not None and status.get("avail") == "UNAVAILABLE":
            ts = pd.Timestamp(status["ts_utc"]).tz_convert("UTC").as_unit("ns")
            if ts - self.prev_ts != ONE_SECOND:
                missing = int((ts - self.prev_ts) / ONE_SECOND) - 1
                stats.gaps.append(Gap(self.machine_id, self.prev_ts.isoformat(),
                                      ts.isoformat(), missing))
        self.prev_ts = None


class Landing:
    """Accumulates full rows keyed on (machine_id, ts_utc) and writes them per machine-month."""

    def __init__(self, landing_root: Path, run_id: str, config_hash: str,
                 partition_tz: str = DEFAULT_PARTITION_TZ):
        self.root = landing_root
        self.run_id = run_id
        self.config_hash = config_hash
        self.partition_tz = partition_tz
        self.rows: dict[tuple[str, pd.Timestamp], dict] = {}
        self.states: dict[str, MachineState] = {}
        self.stats = WriterStats()

    def state(self, machine_id: str) -> MachineState:
        """Return the carry-forward state for a machine, creating it on first use."""
        if machine_id not in self.states:
            self.states[machine_id] = MachineState(machine_id)
        return self.states[machine_id]

    def reject(self, topic: str, raw, reason: str) -> None:
        """Count a message that cannot be decoded, keeping a few samples for the report."""
        self.stats.rejected += 1
        text = raw if isinstance(raw, str) else repr(raw)
        if len(self.stats.rejected_samples) < REJECT_SAMPLES:
            self.stats.rejected_samples.append({"topic": topic, "reason": reason,
                                                "payload": text[:500]})
        log.warning("rejected message on %s: %s", topic, reason)

    def handle_raw(self, topic: str, payload: bytes | str) -> None:
        """Decode and handle one message, rejecting it if it cannot be decoded."""
        try:
            decoded = json.loads(payload)
        except ValueError as exc:
            self.stats.messages += 1
            self.reject(topic, payload, f"invalid JSON: {exc}")
            return
        self.handle(topic, decoded)

    def handle(self, topic: str, payload: dict) -> None:
        """Route one decoded message: status topics update state, records add a row.

        A record whose key is already held replaces the earlier row and counts as a duplicate.
        A record that cannot be read is rejected without changing the machine's state.
        """
        try:
            self._handle(topic, payload)
        except (KeyError, TypeError, ValueError) as exc:
            self.reject(topic, json.dumps(payload, default=str), f"{type(exc).__name__}: {exc}")

    def _handle(self, topic: str, payload: dict) -> None:
        self.stats.messages += 1
        self.stats.instance_ids.add(payload.get("instance_id"))
        machine_id = payload["machine_id"]
        if topic.endswith("/status"):
            self.stats.status += 1
            self.state(machine_id).close(payload, self.stats)
            return
        self.stats.records += 1
        row = self.state(machine_id).apply(payload, self.stats)
        key = (machine_id, row["ts_utc"])
        if key in self.rows:
            self.stats.duplicates += 1
        self.rows[key] = row

    def frame(self, machine_id: str, month: str) -> pd.DataFrame:
        """Build the bronze-shaped frame for one machine-month from accumulated rows."""
        rows = [r for (m, ts), r in self.rows.items()
                if m == machine_id and partition_month(ts, self.partition_tz) == month]
        return build_frame(rows, self.run_id, self.config_hash)

    def partitions(self) -> list[tuple[str, str]]:
        """Sorted (machine_id, YYYY-MM) pairs that hold at least one row."""
        seen = {(m, partition_month(ts, self.partition_tz)) for (m, ts) in self.rows}
        return sorted(seen)

    def write_all(self) -> list[Path]:
        """Write every machine-month partition and return the paths written.

        An existing file is merged on the key, with the newly received row winning, so
        rewriting a partition is idempotent.
        """
        written = []
        for machine_id, month in self.partitions():
            path = self.root / "controller" / "hz1" / f"month={month}" / f"{machine_id}.parquet"
            new = self.frame(machine_id, month)
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, new], ignore_index=True)
                merged = merged.drop_duplicates(subset=KEY, keep="last")
                new = merged.sort_values(KEY, kind="mergesort").reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            new.to_parquet(path, index=False)
            self.stats.rows_written[str(path)] = len(new)
            written.append(path)
        return written


def build_frame(rows: list[dict], run_id: str, config_hash: str) -> pd.DataFrame:
    """Assemble rows into a frame with the export's exact column order and dtypes."""
    data: dict[str, object] = {}
    for column in HZ1_COLUMNS:
        values = [r.get(column) for r in rows]
        if column == "ts_utc":
            data[column] = pd.DatetimeIndex(values, tz="UTC").as_unit("ns")
        elif column in STRING_COLUMNS:
            data[column] = pd.array(values, dtype="str")
        else:
            data[column] = pd.array(values, dtype="float64")
    frame = pd.DataFrame(data)
    frame["run_id"] = pd.array([run_id] * len(frame), dtype="str")
    frame["config_hash"] = pd.array([config_hash] * len(frame), dtype="str")
    return frame.sort_values(KEY, kind="mergesort").reset_index(drop=True)[FILE_COLUMNS]


def subscribe_topics(config: dict, machines: list[str] | None = None) -> list[str]:
    """Subscriptions for every machine's controller and status topics, or for `machines`."""
    root = config["mqtt"]["topic_root"]
    template = config["mqtt"]["controller_topic"]
    if not machines:
        pattern = template.format(root=root, cell="+", machine_id="+")
        return [pattern, pattern + "/status"]
    topics = []
    for machine_id in machines:
        topic = template.format(root=root, cell="+", machine_id=machine_id)
        topics += [topic, topic + "/status"]
    return topics


def run(config: dict, idle_exit: float, session_expiry: int, clean: bool,
        flush_rows: int = 0, stop_file: Path | None = None,
        machines: list[str] | None = None, client_id: str = "idc-controller-writer") -> int:
    """Consume messages until idle (or until `stop_file` exists), then write and report.

    Uses a persistent MQTT v5 session (unless `clean`) so messages published while the
    writer is offline are queued at the broker. With `flush_rows` > 0, accumulated rows are
    merged into the landing files whenever that many are held, which bounds memory; carried
    machine state is kept across flushes. `idle_exit` <= 0 disables the idle exit. Returns 2
    if any gap was seen, else 0.
    """
    partition_tz = config.get("landing", {}).get("controller_partition_timezone",
                                                 DEFAULT_PARTITION_TZ)
    landing = Landing(Path(config["paths"]["landing_root"]), config["source_run"]["run_id"],
                      config["source_run"]["config_hash"], partition_tz)
    inbox: queue.Queue = queue.Queue()
    qos = int(config["mqtt"].get("qos", 1))

    def on_connect(client, userdata, flags, reason_code, properties):
        present = getattr(flags, "session_present", False)
        log.info("connected to broker (session present=%s); subscribing", present)
        for topic in subscribe_topics(config, machines):
            client.subscribe(topic, qos=qos)

    def on_message(client, userdata, msg):
        inbox.put((msg.topic, msg.payload))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                         protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    props = Properties(PacketTypes.CONNECT)
    props.SessionExpiryInterval = session_expiry
    client.connect(config["mqtt"]["host"], int(config["mqtt"]["port"]),
                   clean_start=clean, properties=props)
    client.loop_start()

    last_message = None
    exit_code = 0
    written_paths: dict[str, Path] = {}
    try:
        while True:
            try:
                topic, payload = inbox.get(timeout=1.0)
            except queue.Empty:
                if stop_file is not None and stop_file.exists():
                    log.info("stop file %s present; finishing", stop_file)
                    break
                if idle_exit > 0 and last_message is not None and \
                        time.monotonic() - last_message > idle_exit:
                    log.info("no messages for %.0fs; finishing", idle_exit)
                    break
                continue
            landing.handle_raw(topic, payload)
            last_message = time.monotonic()
            if landing.stats.messages % 10000 == 0:
                log.info("messages=%d records=%d rows=%d", landing.stats.messages,
                         landing.stats.records, len(landing.rows))
            if flush_rows > 0 and len(landing.rows) >= flush_rows:
                for path in landing.write_all():
                    written_paths[str(path)] = path
                landing.rows.clear()
    except KeyboardInterrupt:
        log.info("interrupted; writing what was received")
    finally:
        client.loop_stop()
        client.disconnect()

    for path in landing.write_all():
        written_paths[str(path)] = path
    written = list(written_paths.values())
    report = write_report(Path(config["paths"]["run_reports"]), landing.stats, written)
    s = landing.stats
    log.info("done: messages=%d records=%d status=%d duplicates=%d segments=%d link_losses=%d "
             "gaps=%d rejected=%d rows=%d", s.messages, s.records, s.status, s.duplicates,
             s.segments, s.link_losses, len(s.gaps), s.rejected, len(landing.rows))
    for path in written:
        log.info("  wrote %s (%d rows)", path, s.rows_written[str(path)])
    log.info("  report %s", report)
    if s.gaps:
        for g in s.gaps[:10]:
            log.error("gap on %s: %d seconds missing between %s and %s", g.machine_id,
                      g.missing_seconds, g.after, g.before)
        exit_code = 2
    if s.rejected:
        log.error("%d messages rejected", s.rejected)
        exit_code = 2
    return exit_code


def write_report(run_dir: Path, stats: WriterStats, written: list[Path]) -> Path:
    """Write the run's counters, gaps and file row counts as a timestamped JSON report."""
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = run_dir / f"controller_landing_{stamp}.json"
    body = {
        "finished_utc": stamp,
        "messages": stats.messages,
        "records": stats.records,
        "status": stats.status,
        "duplicates": stats.duplicates,
        "segments": stats.segments,
        "link_losses": stats.link_losses,
        "instance_ids": sorted(i for i in stats.instance_ids if i is not None),
        "gaps": [g.__dict__ for g in stats.gaps],
        "rejected": stats.rejected,
        "rejected_samples": stats.rejected_samples,
        "files": {str(p): stats.rows_written[str(p)] for p in written},
    }
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="controller_writer")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--idle-exit", type=float, default=15.0,
                        help="seconds without a message (after the first) before writing and "
                             "exiting")
    parser.add_argument("--session-expiry", type=int, default=3600,
                        help="seconds the broker keeps this client's subscription and queued "
                             "messages while it is offline")
    parser.add_argument("--clean", action="store_true",
                        help="discard any queued session at the broker and start fresh")
    parser.add_argument("--flush-rows", type=int, default=0,
                        help="merge rows into the landing files whenever this many are held "
                             "(0 writes once at exit)")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="finish, write and exit once this file exists")
    parser.add_argument("--machines", default=None,
                        help="comma-separated machine IDs to subscribe to (default: all)")
    parser.add_argument("--client-id", default="idc-controller-writer",
                        help="MQTT client ID, which names the persistent session")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    machines = args.machines.split(",") if args.machines else None
    return run(config, args.idle_exit, args.session_expiry, args.clean, args.flush_rows,
               args.stop_file, machines, args.client_id)


if __name__ == "__main__":
    raise SystemExit(main())
