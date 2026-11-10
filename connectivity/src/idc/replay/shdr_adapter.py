"""MTConnect SHDR adapter that replays a machine's logged 1 Hz controller record.

The adapter listens on the machine's adapter port, waits for the agent to connect, and
writes SHDR lines carrying the record timestamp and only the values that changed since the
previous second (report by exception). Powered segments are bracketed by AVAILABLE and
UNAVAILABLE. Alarms are a SYSTEM condition: FAULT with the native code on the alarm second,
NORMAL on the next second.

Sources: an hz1 export (encoded on the fly) or an SHDR log written by ``idc.replay.build_native``
(``--from-log``), which is served line for line without reading the export.

Values are exact across the text hop: integers print without a decimal point, other floats
with Python's shortest round-trip repr, so float(text) returns the original float64.

Usage:
    python -m idc.replay.shdr_adapter --machine HMC-01 --day 2026-06-09 [--speed 60]
    python -m idc.replay.shdr_adapter --machine HMC-01 --month 2026-06 --from-log --speed 600
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

from idc.replay.devices import adapter_ports

log = logging.getLogger("idc.shdr_adapter")

# Column in the hz1 export -> SHDR key declared in Devices.xml. Order sets the order of pairs
# on a line; it has no effect on meaning.
SHDR_KEYS: dict[str, str] = {
    "execution": "exec",
    "program": "program",
    "spindle_speed_rpm": "Srpm",
    "spindle_load_pct": "Sload",
    "x_load_pct": "Xload",
    "y_load_pct": "Yload",
    "z_load_pct": "Zload",
    "b_load_pct": "Bload",
    "spindle_temp_c": "Stemp",
    "feed_override_pct": "Fovr",
    "hydraulic_pressure_bar": "Hpres",
    "coolant_pressure_bar": "Cpres",
}

# MTConnect Execution event values are upper case; the export stores them lower case.
EXECUTION_VALUES = {"active": "ACTIVE", "ready": "READY", "stopped": "STOPPED",
                    "interrupted": "INTERRUPTED"}


def format_value(column: str, value) -> str | None:
    """Render one export value as SHDR text, or None if the machine lacks the signal."""
    if column == "execution":
        return EXECUTION_VALUES[value]
    if column == "program":
        return str(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(float(value))


def format_timestamp(ts: pd.Timestamp) -> str:
    """ISO-8601 UTC with millisecond precision, the form the agent parses."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


@dataclass
class ReplayStats:
    records: int = 0
    lines: int = 0
    pairs: int = 0
    segments: int = 0
    alarms: int = 0
    first_ts: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    changed_by_key: dict[str, int] = field(default_factory=dict)
    confirmed: bool | None = None


def shdr_lines(frame: pd.DataFrame, stats: ReplayStats | None = None):
    """Yield (timestamp, line) pairs for one machine's rows in time order.

    A new powered segment starts whenever the gap to the previous row exceeds one second.
    At a segment start every value is sent (the agent may hold stale state from the previous
    segment); within a segment only changed values are sent.
    """
    stats = stats if stats is not None else ReplayStats()
    columns = [c for c in SHDR_KEYS if c in frame.columns]
    last: dict[str, str | None] = {}
    prev_ts: pd.Timestamp | None = None
    alarm_open = False

    for row in frame.itertuples(index=False):
        ts: pd.Timestamp = row.ts_utc
        tstr = format_timestamp(ts)
        new_segment = prev_ts is None or (ts - prev_ts) != pd.Timedelta(seconds=1)
        if new_segment:
            if prev_ts is not None:
                # Close the previous segment at the first unpowered second.
                off = format_timestamp(prev_ts + pd.Timedelta(seconds=1))
                yield prev_ts, f"{off}|avail|UNAVAILABLE"
                stats.lines += 1
            last = {}
            stats.segments += 1
            yield ts, f"{tstr}|avail|AVAILABLE"
            stats.lines += 1

        pairs = [f"record_ts|{tstr}"]
        for column in columns:
            text = format_value(column, getattr(row, column))
            if text is None:
                continue
            if last.get(column) != text:
                pairs.append(f"{SHDR_KEYS[column]}|{text}")
                last[column] = text
                stats.changed_by_key[column] = stats.changed_by_key.get(column, 0) + 1
        yield ts, f"{tstr}|" + "|".join(pairs)
        stats.lines += 1
        stats.pairs += len(pairs)

        # Alarm condition: FAULT with the native code on the alarm second, NORMAL afterward.
        code = row.alarm_code
        if code:
            yield ts, f"{tstr}|alarm|FAULT|{code}|||{code}"
            stats.lines += 1
            stats.alarms += 1
            alarm_open = True
        elif alarm_open:
            yield ts, f"{tstr}|alarm|NORMAL||||"
            stats.lines += 1
            alarm_open = False

        stats.records += 1
        stats.first_ts = stats.first_ts or ts
        stats.last_ts = ts
        prev_ts = ts

    if prev_ts is not None:
        yield prev_ts, f"{format_timestamp(prev_ts + pd.Timedelta(seconds=1))}|avail|UNAVAILABLE"
        stats.lines += 1


def load_slice(export_root: Path, machine_id: str, month: str,
               day: str | None = None) -> pd.DataFrame:
    path = export_root / "controller" / "hz1" / f"month={month}" / f"{machine_id}.parquet"
    frame = pd.read_parquet(path)
    if day:
        start = pd.Timestamp(day, tz="UTC")
        frame = frame[(frame["ts_utc"] >= start) & (frame["ts_utc"] < start + pd.Timedelta(days=1))]
    return frame.sort_values("ts_utc", kind="mergesort").reset_index(drop=True)


def log_lines(path: Path) -> Iterator[tuple[datetime, str]]:
    """Yield (record time, line) from an SHDR log; the time is the line's leading timestamp."""
    last_text = ""
    last_ts: datetime | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as src:
        for raw in src:
            line = raw.rstrip("\n")
            text = line.split("|", 1)[0]
            if text != last_text:
                last_ts = datetime.fromisoformat(text)
                last_text = text
            yield last_ts, line


def _elapsed_seconds(ts, start) -> float:
    delta = ts - start
    return delta.total_seconds()


def line_timestamp(line: str) -> datetime:
    """The leading timestamp of an SHDR line as an aware UTC datetime."""
    return datetime.fromisoformat(line.split("|", 1)[0].replace("Z", "+00:00"))


AVAILABILITY = re.compile(r'<Availability [^>]*>')
ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')
NEXT_SEQUENCE = re.compile(r'nextSequence="(\d+)"')


def agent_next_sequence(agent_url: str, machine_id: str) -> int:
    """The agent's next sequence number; observations from this replay are at or above it."""
    response = requests.get(f"{agent_url.rstrip('/')}/current",
                            params={"path": f'//Device[@name="{machine_id}"]'}, timeout=10)
    response.raise_for_status()
    return int(NEXT_SEQUENCE.search(response.text).group(1))


def agent_has(agent_url: str, machine_id: str, final_line: str, min_sequence: int = 0) -> bool:
    """True when the agent's current Availability for the device is the replay's final line.

    Every replay ends with an ``avail|UNAVAILABLE`` line, so its timestamp marks the end of the
    stream; the agent reports it only after reading every line before it. `min_sequence`
    excludes an observation with the same timestamp left by an earlier replay.
    """
    response = requests.get(f"{agent_url.rstrip('/')}/current",
                            params={"path": f'//Device[@name="{machine_id}"]'}, timeout=10)
    response.raise_for_status()
    match = AVAILABILITY.search(response.text)
    if match is None:
        return False
    attrs = dict(ATTRIBUTE.findall(match.group(0)))
    if int(attrs.get("sequence", -1)) < min_sequence:
        return False
    return datetime.fromisoformat(attrs["timestamp"].replace("Z", "+00:00")) == \
        line_timestamp(final_line)


async def serve(lines: Iterable[tuple], port: int, speed: float, host: str = "0.0.0.0",
                stats: ReplayStats | None = None,
                confirm: Callable[[str], bool] | None = None,
                confirm_timeout: float = 900.0) -> ReplayStats:
    """Listen for the agent, then stream (timestamp, line) pairs at `speed` times real time.

    The agent connects to the adapter, sends '* PING' heartbeats and expects '* PONG'
    replies, which are answered on the connection's reader task. Speed 0 streams unpaced.

    With `confirm`, the connection stays open after the last line until `confirm(last_line)`
    returns True (the agent has read the whole stream) or `confirm_timeout` seconds pass;
    closing earlier would discard lines still in the socket buffers. ``stats.confirmed``
    records the outcome.
    """
    stats = stats if stats is not None else ReplayStats()
    connected = asyncio.Event()
    writer_holder: dict[str, asyncio.StreamWriter] = {}

    async def on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        log.info("agent connected from %s", peer)
        writer_holder["w"] = writer
        connected.set()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                if line.strip().upper() == b"* PING":
                    writer.write(b"* PONG 10000\n")
                    await writer.drain()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            log.info("agent disconnected")

    server = await asyncio.start_server(on_connect, host, port)
    log.info("adapter listening on %s:%d, waiting for the agent", host, port)
    async with server:
        await connected.wait()
        writer = writer_holder["w"]
        writer.write(b"* adapterVersion: 2.7\n* realTime: no\n")
        await writer.drain()

        wall_start = time.monotonic()
        rec_start = None
        sent = 0
        last_line = None
        for ts, line in lines:
            if speed > 0:
                rec_start = rec_start or ts
                ahead = _elapsed_seconds(ts, rec_start) / speed - (time.monotonic() - wall_start)
                if ahead > 0:
                    await asyncio.sleep(ahead)
            writer.write(line.encode() + b"\n")
            last_line = line
            sent += 1
            if sent % 500 == 0:
                await writer.drain()
        await writer.drain()
        stats.lines = max(stats.lines, sent)
        log.info("replay complete: %d lines sent, %.0f s wall time", sent,
                 time.monotonic() - wall_start)
        if confirm is not None and last_line is not None:
            deadline = time.monotonic() + confirm_timeout
            while True:
                try:
                    done = await asyncio.to_thread(confirm, last_line)
                except requests.RequestException as exc:
                    log.warning("agent check failed: %s", exc)
                    done = False
                if done:
                    stats.confirmed = True
                    log.info("agent confirmed the final line after %.0f s",
                             confirm_timeout - (deadline - time.monotonic()))
                    break
                if time.monotonic() > deadline:
                    stats.confirmed = False
                    log.error("agent did not confirm the final line within %.0f s",
                              confirm_timeout)
                    break
                await asyncio.sleep(1)
        else:
            await asyncio.sleep(2)
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shdr_adapter")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--machine", required=True)
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--day", default=None, help="restrict to one UTC day, YYYY-MM-DD")
    parser.add_argument("--speed", type=float, default=60.0,
                        help="replay speed multiple; 0 streams as fast as the socket allows")
    parser.add_argument("--port", type=int, default=None,
                        help="override the adapter port from the register order")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the SHDR lines instead of serving them")
    parser.add_argument("--from-log", action="store_true",
                        help="serve the machine-month SHDR log under paths.native_root")
    parser.add_argument("--no-confirm", action="store_true",
                        help="close 2 s after the last line instead of waiting for the agent")
    parser.add_argument("--confirm-timeout", type=float, default=900.0,
                        help="seconds to wait for the agent to report the final line")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    if args.port is None:
        register = yaml.safe_load(Path(config["paths"]["register"]).read_text(encoding="utf-8"))
        args.port = adapter_ports(register["machines"])[args.machine]

    confirm = None
    if not args.no_confirm:
        agent_url = config["mtconnect"]["agent_url"]
        start_sequence = agent_next_sequence(agent_url, args.machine)

        def confirm(final_line: str) -> bool:
            return agent_has(agent_url, args.machine, final_line, start_sequence)

    if args.from_log:
        stem = args.day or args.month
        path = Path(config["paths"]["native_root"]) / "cnc" / args.machine / f"{stem}.shdr.gz"
        if not path.exists():
            log.error("no SHDR log at %s", path)
            return 1
        stats = asyncio.run(serve(log_lines(path), args.port, args.speed,
                                  confirm=confirm, confirm_timeout=args.confirm_timeout))
        return 0 if stats.confirmed is not False else 3

    export_root = Path(config["paths"]["export_root"])
    frame = load_slice(export_root, args.machine, args.month, args.day)
    if frame.empty:
        when = f" on {args.day}" if args.day else ""
        log.error("no rows for %s in %s%s", args.machine, args.month, when)
        return 1

    if args.dry_run:
        stats = ReplayStats()
        for _, line in shdr_lines(frame, stats):
            print(line)
        log.info("%d records, %d lines, %d pairs, %d segments, %d alarms",
                 stats.records, stats.lines, stats.pairs, stats.segments, stats.alarms)
        return 0

    stats = ReplayStats()
    asyncio.run(serve(shdr_lines(frame, stats), args.port, args.speed, stats=stats,
                      confirm=confirm, confirm_timeout=args.confirm_timeout))
    log.info("%d records, %d segments, %d alarms", stats.records, stats.segments, stats.alarms)
    return 0 if stats.confirmed is not False else 3


if __name__ == "__main__":
    raise SystemExit(main())
