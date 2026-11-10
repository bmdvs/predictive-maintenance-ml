"""Run the connectivity chain end to end: replay, publish, land and acceptance.

Replays the native device files through both paths concurrently, publishes them to the
broker, lands them in the bronze layout and checks the landed files against the exports:

- HMC path: one MTConnect bridge and one controller writer per machine group
  (``--groups``) run from start to finish; each machine replays its months in order from its
  SHDR logs (``shdr_adapter --from-log``) with up to ``--hmc-parallel`` machines at once. If a
  bridge stops on a sequence gap or agent restart, it is restarted and every machine-month of
  its group in flight is rerun once at half speed.
  After the replay each landed hz1 file is compared with its export; a failing machine-month
  is rerun once at half speed.
- Compressor path: the compressor writer runs from start to finish; the buffered Modbus
  server replays the quarter in resumable chunks. A watchdog restarts a stalled server from
  its recorded next minute. The Ignition trial is renewed on a timer by restarting the
  gateway with the replay paused, then confirming data flows again. Minutes missing from the
  landing at the end are replayed once.
- Finally ``minute_writer`` and ``acceptance`` run.

Progress is logged to ``<run_reports>/chain_<stamp>.log`` and a summary written to
``<run_reports>/chain_<stamp>.json``. Creating ``<run_reports>/chain_<stamp>/stop``
stops the run cleanly. The process never waits for input; an unrecoverable failure is written
to the summary and the run stops.

Usage:
    python -m idc.run_chain [--hmc-speed 400] [--hmc-parallel 8] [--limit-seconds 600]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from idc.checks.compare_exports import compare_files, load_config
from idc.replay.build_native import read_register_image
from idc.replay.register_map import RING_SLOTS

log = logging.getLogger("idc.run_chain")
PY = sys.executable
CREATE_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class RunFailed(RuntimeError):
    """Unrecoverable failure; recorded in the summary."""


class ChainRun:
    """Shared state, child processes and reporting for one chain run."""

    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.reports = Path(config["paths"]["run_reports"])
        self.dir = self.reports / f"chain_{self.stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = self.dir / "stop"
        self.deadline = time.monotonic() + args.limit_seconds if args.limit_seconds else None
        self.lock = threading.Lock()
        self.summary: dict = {
            "stamp": self.stamp, "started_utc": now_iso(), "args": vars(args),
            "status": "running", "hmc": {"machine_months": {}, "bridge_restarts": [],
                                         "writer_restarts": [], "sequence_gap_reruns": [],
                                         "unconfirmed_reruns": [], "verify_reruns": []},
            "compressor": {"chunks": [], "trial_resets": [], "server_restarts": [],
                           "gap_reruns": [], "path": "ignition-sparkplug"},
            "errors": [],
        }
        self.procs: dict[str, subprocess.Popen] = {}

    # Process helpers -----------------------------------------------------------------------

    def spawn(self, name: str, argv: list[str]) -> subprocess.Popen:
        out = open(self.dir / f"{name}.log", "ab")
        proc = subprocess.Popen([PY, *argv], stdout=out, stderr=subprocess.STDOUT,
                                creationflags=CREATE_FLAGS)
        with self.lock:
            self.procs[name] = proc
        return proc

    def stopping(self) -> bool:
        if self.stop_file.exists():
            return True
        return self.deadline is not None and time.monotonic() > self.deadline

    def event(self, section: str, key: str, value) -> None:
        with self.lock:
            self.summary[section][key].append(value)
        self.save()

    def save(self) -> None:
        with self.lock:
            body = json.dumps(self.summary, indent=2, default=str)
        (self.reports / f"chain_{self.stamp}.json").write_text(body, encoding="utf-8")

    def kill_all(self) -> None:
        for proc in list(self.procs.values()):
            if proc.poll() is None:
                proc.kill()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def cutoff_iso(seconds: float) -> str:
    return datetime.fromtimestamp(time.time() - seconds, UTC).isoformat(timespec="seconds")


# Preflight -----------------------------------------------------------------------------------


def preflight(config: dict) -> dict:
    """Confirm the agent, broker and gateway are up; returns their status."""
    status = {}
    agent = requests.get(config["mtconnect"]["agent_url"] + "/probe", timeout=10)
    status["agent"] = agent.status_code
    with socket.create_connection((config["mqtt"]["host"], int(config["mqtt"]["port"])), 5):
        status["broker"] = "reachable"
    ping = requests.get(config["ignition"]["gateway_url"] + "/StatusPing", timeout=10).json()
    status["ignition"] = ping.get("state")
    if agent.status_code != 200 or status["ignition"] != "RUNNING":
        raise RunFailed(f"preflight failed: {status}")
    return status


def agent_sequence(config: dict) -> dict:
    """Agent buffer header: bufferSize, firstSequence, nextSequence."""
    text = requests.get(config["mtconnect"]["agent_url"] + "/current", timeout=10).text
    header = next(el for el in ET.fromstring(text).iter() if el.tag.endswith("Header"))
    return {k: int(header.attrib[k]) for k in ("bufferSize", "firstSequence", "nextSequence")}


def last_match(path: Path, pattern: str) -> re.Match | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 20000))
        tail = f.read().decode("utf-8", "replace")
    matches = list(re.finditer(pattern, tail))
    return matches[-1] if matches else None


# HMC path ------------------------------------------------------------------------------------


class HmcPath:
    def __init__(self, chain: ChainRun):
        self.s = chain
        cfg = chain.config
        self.machines = list(chain.args.machines or cfg["replay"]["hmcs"])
        self.months = list(chain.args.months or cfg["replay"]["months"])
        self.speed = float(chain.args.hmc_speed)
        n = max(1, min(int(chain.args.groups), len(self.machines)))
        self.groups = [self.machines[i::n] for i in range(n)]
        self.bridge_epoch = 0
        self.in_flight: dict[str, tuple[str, int]] = {}
        self.rerun: set[tuple[str, str]] = set()
        self.unconfirmed: set[tuple[str, str]] = set()
        # Machine-months replayed since each group's writer last started; a writer that exits
        # loses its unflushed rows, so these are rerun.
        self.since_writer_start: list[set[tuple[str, str]]] = [set() for _ in self.groups]
        self.writers_restarting = False
        self.lock = threading.Lock()
        self.failed: str | None = None

    def group_of(self, machine: str) -> int:
        return next(i for i, group in enumerate(self.groups) if machine in group)

    def start_writer(self, i: int, clean: bool) -> None:
        """Start group `i`'s writer and wait until it has subscribed at the broker."""
        stop = self.s.dir / "controller.stop"
        log_path = self.s.dir / f"controller_writer_{i}.log"
        offset = log_path.stat().st_size if log_path.exists() else 0
        argv = ["-m", "idc.landing.controller_writer", "--idle-exit", "0", "--stop-file",
                str(stop), "--flush-rows", "300000", "--machines", ",".join(self.groups[i]),
                "--client-id", f"idc-controller-writer-{i}", "--session-expiry", "86400"]
        proc = self.s.spawn(f"controller_writer_{i}", argv + (["--clean"] if clean else []))
        with self.lock:
            self.since_writer_start[i] = set()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RunFailed(f"controller writer {i} exited at start ({proc.returncode})")
            with open(log_path, "rb") as f:
                f.seek(offset)
                if b"subscribing" in f.read():
                    time.sleep(2)  # let the SUBACKs return
                    return
            time.sleep(0.5)
        raise RunFailed(f"controller writer {i} did not subscribe within 60 s")

    def start_writers(self, clean: bool) -> None:
        for i in range(len(self.groups)):
            self.start_writer(i, clean)

    def start_services(self) -> None:
        self.start_writers(clean=True)
        for i in range(len(self.groups)):
            self.start_bridge(i)

    def start_bridge(self, i: int) -> None:
        self.bridge_epoch += 1
        self.s.spawn(f"bridge_{i}", ["-m", "idc.bridge.mtconnect_to_mqtt", "--devices",
                                     ",".join(self.groups[i]), "--client-id", f"idc-bridge-{i}"])

    def writer_check(self) -> None:
        """Restart a writer that exited and queue the machine-months it may have lost."""
        if self.writers_restarting:
            return
        for i in range(len(self.groups)):
            proc = self.s.procs.get(f"controller_writer_{i}")
            if proc is None or proc.poll() is None or self.writers_restarting:
                continue
            with self.lock:
                affected = set(self.since_writer_start[i])
                affected |= {(m, self.in_flight[m][0]) for m in self.in_flight
                             if m in self.groups[i]}
                self.rerun |= affected
            tail = last_match(self.s.dir / f"controller_writer_{i}.log",
                              r"\w+(Error|Exception)[^\n]*")
            reason = tail.group(0) if tail else f"exit code {proc.returncode}"
            self.s.event("hmc", "writer_restarts", {
                "utc": now_iso(), "writer": i, "reason": reason,
                "rerun": sorted(f"{m} {mo}" for m, mo in affected)})
            log.warning("controller writer %d exited (%s); restarting; rerun queued for %s", i,
                        reason, sorted(affected))
            recent = [r for r in self.s.summary["hmc"]["writer_restarts"]
                      if r.get("writer") == i and r["utc"] >= cutoff_iso(600)]
            if len(recent) > 3:
                self.failed = f"controller writer {i} keeps exiting: {reason}"
                return
            self.start_writer(i, clean=False)

    def bridge_watch(self) -> None:
        """Restart a stopped bridge or writer and queue the affected machine-months for rerun."""
        while not self.s.stopping() and not self.done.is_set():
            self.writer_check()
            if self.failed:
                return
            for i, group in enumerate(self.groups):
                proc = self.s.procs.get(f"bridge_{i}")
                if proc is None or proc.poll() is None:
                    continue
                tail = last_match(self.s.dir / f"bridge_{i}.log",
                                  r"(SequenceGap|AgentRestarted|HTTPError)[^\n]*")
                reason = tail.group(0) if tail else f"exit code {proc.returncode}"
                with self.lock:
                    affected = sorted(m for m in self.in_flight if m in group)
                    for machine in affected:
                        self.rerun.add((machine, self.in_flight[machine][0]))
                self.s.event("hmc", "bridge_restarts", {"utc": now_iso(), "bridge": i,
                                                        "reason": reason, "in_flight": affected})
                log.warning("bridge %d stopped (%s); restarting; rerun queued for %s", i,
                            reason, affected)
                recent = [r for r in self.s.summary["hmc"]["bridge_restarts"]
                          if r.get("bridge") == i and r["utc"] >= cutoff_iso(300)]
                if len(recent) > 5:
                    self.failed = f"bridge {i} crash loop: {reason}"
                    return
                time.sleep(min(30, 2 * len(recent)))
                self.start_bridge(i)
            time.sleep(2)

    def replay(self, machine: str, month: str, speed: float, tag: str) -> int:
        with self.lock:
            self.in_flight[machine] = (month, self.bridge_epoch)
            self.since_writer_start[self.group_of(machine)].add((machine, month))
        name = f"adapter_{machine}_{month}_{tag}"
        proc = self.s.spawn(name, ["-m", "idc.replay.shdr_adapter", "--machine", machine,
                                   "--month", month, "--from-log", "--speed", str(speed)])
        started = now_iso()
        while proc.poll() is None:
            if self.s.stopping() or self.failed:
                proc.kill()
                break
            time.sleep(1)
        with self.lock:
            self.in_flight.pop(machine, None)
            if proc.returncode == 3:
                # The agent never reported the final line: part of the stream may be lost.
                self.unconfirmed.add((machine, month))
        record = {"started_utc": started, "finished_utc": now_iso(), "speed": speed,
                  "exit": proc.returncode}
        with self.s.lock:
            self.s.summary["hmc"]["machine_months"].setdefault(f"{machine} {month}", []) \
                .append({"run": tag, **record})
        self.s.save()
        return proc.returncode

    def machine_worker(self, machine: str) -> None:
        for month in self.months:
            if self.s.stopping() or self.failed:
                return
            self.replay(machine, month, self.speed, "primary")

    def run(self) -> None:
        self.done = threading.Event()
        self.start_services()
        watcher = threading.Thread(target=self.bridge_watch, daemon=True)
        watcher.start()
        sem = threading.Semaphore(int(self.s.args.hmc_parallel))

        def guarded(machine: str) -> None:
            with sem:
                self.machine_worker(machine)

        threads = [threading.Thread(target=guarded, args=(m,)) for m in self.machines]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if self.failed:
            raise RunFailed(self.failed)
        if self.s.stopping():
            return
        while self.rerun or self.unconfirmed:
            if self.s.stopping() or self.failed:
                return
            with self.lock:
                batch = sorted(self.rerun | self.unconfirmed)
                unconfirmed = set(self.unconfirmed)
                self.rerun.clear()
                self.unconfirmed.clear()
            for machine, month in batch:
                if self.s.stopping() or self.failed:
                    return
                key = "unconfirmed_reruns" if (machine, month) in unconfirmed and \
                    (machine, month) not in self.rerun else "sequence_gap_reruns"
                self.s.event("hmc", key, {"machine_month": f"{machine} {month}",
                                          "utc": now_iso()})
                self.replay(machine, month, self.speed / 2, "rerun")
        self.drain()
        failed = self.verify()
        for machine, month in failed:
            if self.s.stopping():
                return
            self.s.event("hmc", "verify_reruns", {"machine_month": f"{machine} {month}",
                                                  "utc": now_iso()})
            self.restart_writers()
            self.replay(machine, month, self.speed / 2, "rerun_verify")
            self.drain()
        self.done.set()

    def drain(self, quiet_seconds: float = 30.0) -> None:
        """Wait until the bridge has published everything and the writers are idle."""
        last = None
        stable_since = time.monotonic()
        def mark(name: str, pattern: str) -> str:
            m = last_match(self.s.dir / f"{name}.log", pattern)
            return m.group(0) if m else ""

        while not self.s.stopping():
            current = tuple(mark(f"controller_writer_{i}", r"messages=\d+") +
                            mark(f"bridge_{i}", r"bookmark=\d+")
                            for i in range(len(self.groups)))
            if current != last:
                last, stable_since = current, time.monotonic()
            elif time.monotonic() - stable_since > quiet_seconds:
                return
            time.sleep(5)

    def restart_writers(self) -> None:
        """Stop the controller writers (they flush) and start them again."""
        stop = self.s.dir / "controller.stop"
        self.writers_restarting = True
        try:
            stop.touch()
            for i in range(len(self.groups)):
                proc = self.s.procs.get(f"controller_writer_{i}")
                if proc is not None:
                    proc.wait(timeout=1800)
            stop.unlink()
            self.start_writers(clean=False)
        finally:
            self.writers_restarting = False

    def verify(self) -> list[tuple[str, str]]:
        """Flush the writers, then compare every landed hz1 file with its export."""
        self.restart_writers()
        export = Path(self.s.config["paths"]["export_root"])
        landing = Path(self.s.config["paths"]["landing_root"])
        failed = []
        for machine in self.machines:
            for month in self.months:
                rel = Path("controller") / "hz1" / f"month={month}" / f"{machine}.parquet"
                ok = (landing / rel).exists() and \
                    compare_files("ctrl_hz1", export / rel, landing / rel).ok
                if not ok:
                    failed.append((machine, month))
        log.info("hz1 verification: %d of %d machine-months match",
                 len(self.machines) * len(self.months) - len(failed),
                 len(self.machines) * len(self.months))
        return failed

    def stop_services(self) -> None:
        (self.s.dir / "controller.stop").touch()
        for i in range(len(self.groups)):
            proc = self.s.procs.get(f"controller_writer_{i}")
            if proc is not None:
                proc.wait(timeout=1800)
        self.done.set()
        for i in range(len(self.groups)):
            bridge = self.s.procs.get(f"bridge_{i}")
            if bridge is not None and bridge.poll() is None:
                bridge.kill()


# Compressor path -----------------------------------------------------------------------------


class CompressorPath:
    def __init__(self, chain: ChainRun):
        self.s = chain
        cfg = chain.config
        months = list(chain.args.months or cfg["replay"]["months"])
        self.start = pd.Timestamp(f"{months[0]}-01T00:00:00Z")
        end = pd.Timestamp(f"{months[-1]}-01T00:00:00Z") + pd.offsets.MonthBegin(1)
        self.total = int((end - self.start) / pd.Timedelta(minutes=1))
        self.pause = self.s.dir / "modbus.pause"
        self.progress = self.s.dir / "modbus_progress.json"
        self.writer_stop = self.s.dir / "compressor.stop"
        self.last_reset = time.monotonic()
        self.per_tick = int(chain.args.cmp_per_tick)
        self.hold = float(chain.args.cmp_hold)
        self.start_hold = float(chain.args.cmp_start_hold)

    def start_writer(self, clean: bool = False) -> None:
        argv = ["-m", "idc.landing.compressor_writer", "--idle-exit", "0", "--stop-file",
                str(self.writer_stop), "--expect-start", self.start.strftime("%Y-%m-%dT%H:%MZ"),
                "--expect-minutes", str(self.total)]
        self.s.spawn("compressor_writer", argv + (["--clean"] if clean else []))
        time.sleep(3)

    def trial_due(self) -> bool:
        """True when the renewal timer has elapsed or under 30 minutes of trial remain."""
        if time.monotonic() - self.last_reset > self.s.args.trial_minutes * 60:
            return True
        if time.monotonic() - getattr(self, "last_trial_check", 0.0) < 60:
            return False
        self.last_trial_check = time.monotonic()
        try:
            gw = self.s.config["ignition"]["gateway_url"]
            left = requests.get(gw + "/data/api/v1/trial", timeout=10).json()["trialSecondsLeft"]
        except (requests.RequestException, ValueError, KeyError):
            return False
        return int(left) < 1800

    def accepted(self) -> int:
        m = last_match(self.s.dir / "compressor_writer.log", r"accepted=(\d+)")
        return int(m.group(1)) if m else 0

    def trial_reset(self) -> None:
        """Renew the gateway trial; one retry if data does not resume."""
        try:
            self._trial_reset()
        except (RunFailed, requests.RequestException, subprocess.CalledProcessError) as exc:
            log.warning("trial reset failed (%s); retrying once", exc)
            self._trial_reset()

    def _trial_reset(self) -> None:
        """Pause the replay, restart the gateway, wait for data, resume."""
        gw = self.s.config["ignition"]["gateway_url"]
        record = {"utc": now_iso()}
        self.pause.touch()
        time.sleep(4)
        before = requests.get(gw + "/data/api/v1/trial", timeout=10).json()
        record["seconds_left_before"] = before.get("trialSecondsLeft")
        subprocess.run(["docker", "restart", self.s.config["ignition"]["container"]],
                       check=True, capture_output=True, creationflags=CREATE_FLAGS)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                if requests.get(gw + "/StatusPing", timeout=5).json().get("state") == "RUNNING":
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(3)
        else:
            raise RunFailed("gateway did not return to RUNNING within 10 minutes")
        after = requests.get(gw + "/data/api/v1/trial", timeout=10).json()
        record["seconds_left_after"] = after.get("trialSecondsLeft")
        time.sleep(20)
        accepted_before = self.accepted()
        self.pause.unlink()
        resume = time.monotonic()
        while time.monotonic() - resume < 180:
            time.sleep(5)
            if self.accepted() > accepted_before:
                break
        record["data_flowing"] = self.accepted() > accepted_before
        record["resume_seconds"] = round(time.monotonic() - resume, 1)
        self.last_reset = time.monotonic()
        self.s.event("compressor", "trial_resets", record)
        log.info("trial reset: %s", record)
        if not record["data_flowing"]:
            raise RunFailed(f"no compressor data after trial reset: {record}")

    def serve(self, start: pd.Timestamp, minutes: int, tag: str) -> None:
        """Replay `minutes` from `start`, restarting a stalled server from its progress."""
        remaining_start, remaining = start, minutes
        attempts = 0
        while remaining > 0 and not self.s.stopping():
            self.progress.unlink(missing_ok=True)
            proc = self.s.spawn("modbus_server", [
                "-m", "idc.replay.modbus_server", "--ring", "--start",
                remaining_start.strftime("%Y-%m-%dT%H:%MZ"), "--minutes", str(remaining),
                "--per-tick", str(self.per_tick), "--hold", str(self.hold), "--pause-file",
                str(self.pause), "--progress-file", str(self.progress), "--linger", "5",
                "--start-hold", str(self.start_hold)])
            launched = time.monotonic()
            self.launched_wall = time.time()
            self.last_progress = None
            stalled = False
            while proc.poll() is None:
                if self.s.stopping():
                    proc.kill()
                    return
                if self.trial_due():
                    self.trial_reset()
                time.sleep(2)
                age = self._progress_age(launched)
                if age is not None and age > 20 and not self.pause.exists():
                    stalled = True
                    proc.kill()
                    break
            done = self._progress()
            served = int(done["served"]) if done else 0
            if not stalled:
                return
            attempts += 1
            # Restart one ring's worth of minutes early so slots in flight are re-served.
            back = min(served, RING_SLOTS)
            remaining_start = remaining_start + pd.Timedelta(minutes=served - back)
            remaining = remaining - (served - back)
            self.s.event("compressor", "server_restarts", {
                "utc": now_iso(), "chunk": tag, "resume_from": remaining_start.isoformat(),
                "attempt": attempts})
            log.warning("modbus server stalled; restarting from %s", remaining_start)
            if attempts > 20:
                raise RunFailed("modbus server stalled more than 20 times in one chunk")

    def _progress(self) -> dict | None:
        """Latest progress record; the last good reading if the file is mid-replace."""
        try:
            self.last_progress = json.loads(self.progress.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return getattr(self, "last_progress", None)

    def _progress_age(self, launched: float) -> float | None:
        """Seconds since the server last advanced, or None while it is starting."""
        data = self._progress()
        if data is None or float(data["updated"]) < self.launched_wall:
            elapsed = time.monotonic() - launched
            return elapsed - 70 if elapsed > 90 else None
        return time.time() - float(data["updated"])

    def landed_gaps(self) -> list[tuple[pd.Timestamp, int]]:
        """Contiguous ranges of record minutes missing for any unit."""
        landing = Path(self.s.config["paths"]["landing_root"]) / "compressor_ctrl"
        native = Path(self.s.config["paths"]["native_root"])
        missing: set[pd.Timestamp] = set()
        month = self.start
        while month < self.start + pd.Timedelta(minutes=self.total):
            m = month.strftime("%Y-%m")
            image = read_register_image(native, m)
            expected = set(zip(image["unit_id"], image["minute_utc"], strict=True))
            unit_ids = self.s.config["modbus"]["unit_ids"]
            have: set = set()
            cmp_path = landing / "minute" / f"month={m}" / "compressor_minute.parquet"
            room_path = landing / "room" / f"month={m}" / "room.parquet"
            if cmp_path.exists():
                c = pd.read_parquet(cmp_path)
                have |= {(unit_ids[d], t) for d, t in zip(c["compressor_id"], c["minute_utc"],
                                                          strict=True)}
            if room_path.exists():
                r = pd.read_parquet(room_path)
                have |= {(unit_ids["room"], t) for t in r["minute_utc"]}
            missing |= {t for (_, t) in expected - have}
            month = month + pd.offsets.MonthBegin(1)
        ranges: list[tuple[pd.Timestamp, int]] = []
        for t in sorted(missing):
            if ranges and ranges[-1][0] + pd.Timedelta(minutes=ranges[-1][1]) == t:
                ranges[-1] = (ranges[-1][0], ranges[-1][1] + 1)
            else:
                ranges.append((t, 1))
        return ranges

    def stop_writer(self) -> None:
        self.writer_stop.touch()
        proc = self.s.procs.get("compressor_writer")
        if proc is not None:
            proc.wait(timeout=1800)
        self.writer_stop.unlink(missing_ok=True)

    def run(self) -> None:
        gaps_only = bool(self.s.args.cmp_gaps_only)
        if not gaps_only:
            self.start_writer(clean=True)
        chunk = int(self.s.args.cmp_chunk_minutes)
        offset = self.total if gaps_only else 0
        while offset < self.total and not self.s.stopping():
            count = min(chunk, self.total - offset)
            start = self.start + pd.Timedelta(minutes=offset)
            record = {"start": start.isoformat(), "minutes": count, "started_utc": now_iso()}
            self.serve(start, count, start.strftime("%Y-%m-%dT%H:%MZ"))
            record["finished_utc"] = now_iso()
            self.s.event("compressor", "chunks", record)
            offset += count
        if not gaps_only:
            time.sleep(10)
            self.stop_writer()
        if self.s.stopping():
            return
        gaps = self.landed_gaps()
        if gaps:
            self.start_writer()
            for start, count in gaps:
                lead = min(RING_SLOTS, int((start - self.start) / pd.Timedelta(minutes=1)))
                replay_start = start - pd.Timedelta(minutes=lead)
                self.s.event("compressor", "gap_reruns", {"start": start.isoformat(),
                                                          "minutes": count,
                                                          "utc": now_iso()})
                self.serve(replay_start, count + lead + RING_SLOTS, "gap")
            time.sleep(10)
            self.stop_writer()
        with self.s.lock:
            self.s.summary["compressor"]["gaps_after_rerun"] = [
                {"start": s.isoformat(), "minutes": n} for s, n in self.landed_gaps()]
        self.s.save()


# Orchestration -------------------------------------------------------------------------------


def progress_loop(chain: ChainRun, hmc: HmcPath | None, cmp: CompressorPath | None,
                  finished: threading.Event) -> None:
    while not finished.wait(timeout=float(chain.args.report_seconds)):
        try:
            parts = [f"elapsed={int(time.monotonic() - chain.t0)}s"]
            if hmc is not None:
                mm = chain.summary["hmc"]["machine_months"]
                done = sum(1 for runs in mm.values() if runs[-1].get("exit") == 0)
                records = 0
                lags = []
                seq = agent_sequence(chain.config)
                for i in range(len(hmc.groups)):
                    m = last_match(chain.dir / f"controller_writer_{i}.log", r"records=(\d+)")
                    records += int(m.group(1)) if m else 0
                    b = last_match(chain.dir / f"bridge_{i}.log", r"bookmark=(\d+)")
                    if b:
                        lags.append(seq["nextSequence"] - int(b.group(1)))
                lag = max(lags) if lags else None
                parts += [f"hmc_done={done}", f"hmc_in_flight={sorted(hmc.in_flight)}",
                          f"writer_records={records}", f"agent_buffer={seq['bufferSize']}",
                          f"bridge_lag_obs={lag}"]
            if cmp is not None:
                p = cmp._progress()
                parts += [f"cmp_last_minute={p['last_minute'] if p else None}",
                          f"cmp_accepted={cmp.accepted()}",
                          f"trial_resets={len(chain.summary['compressor']['trial_resets'])}"]
            log.info("progress %s", " ".join(parts))
        except Exception as exc:  # progress reporting must never stop the run
            log.warning("progress report failed: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_chain")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--hmc-speed", type=float, default=400.0)
    parser.add_argument("--hmc-parallel", type=int, default=8)
    parser.add_argument("--groups", type=int, default=2,
                        help="machine groups, each with its own bridge and controller writer")
    parser.add_argument("--machines", type=lambda s: s.split(","), default=None)
    parser.add_argument("--months", type=lambda s: s.split(","), default=None)
    parser.add_argument("--skip-hmc", action="store_true")
    parser.add_argument("--skip-compressors", action="store_true")
    parser.add_argument("--cmp-per-tick", type=int, default=5)
    parser.add_argument("--cmp-hold", type=float, default=0.5)
    parser.add_argument("--cmp-chunk-minutes", type=int, default=10080)
    parser.add_argument("--cmp-start-hold", type=float, default=15.0,
                        help="seconds the first window of each compressor replay is held")
    parser.add_argument("--cmp-gaps-only", action="store_true",
                        help="replay only the compressor minutes missing from the landing")
    parser.add_argument("--trial-minutes", type=float, default=90.0)
    parser.add_argument("--report-seconds", type=float, default=180.0)
    parser.add_argument("--limit-seconds", type=float, default=0.0,
                        help="stop after this many seconds (throughput measurement)")
    parser.add_argument("--skip-finish", action="store_true",
                        help="do not run minute_writer and acceptance at the end")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    chain = ChainRun(config, args)
    chain.t0 = time.monotonic()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(chain.reports /
                                                      f"chain_{chain.stamp}.log"),
                                  logging.StreamHandler()])
    (chain.reports / f"chain_{chain.stamp}.pid").write_text(str(os.getpid()))
    log.info("chain run %s; stop with: New-Item %s", chain.stamp, chain.stop_file)
    hmc = None if args.skip_hmc else HmcPath(chain)
    cmp = None if args.skip_compressors else CompressorPath(chain)
    finished = threading.Event()
    failures: list[str] = []

    def guarded(name: str, target) -> None:
        try:
            target()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            chain.summary["errors"].append({"path": name, "utc": now_iso(),
                                              "error": traceback.format_exc()})
            chain.save()
            log.error("%s path failed: %s", name, exc)

    try:
        chain.summary["preflight"] = preflight(config)
        threads = []
        if hmc is not None:
            threads.append(threading.Thread(target=guarded, args=("hmc", hmc.run)))
        if cmp is not None:
            threads.append(threading.Thread(target=guarded, args=("compressor", cmp.run)))
        reporter = threading.Thread(target=progress_loop, args=(chain, hmc, cmp, finished),
                                    daemon=True)
        reporter.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if hmc is not None:
            hmc.stop_services()
            chain.summary["hmc"]["finished_utc"] = now_iso()
        if cmp is not None:
            chain.summary["compressor"]["finished_utc"] = now_iso()
        if not failures and not args.skip_finish and not chain.stopping():
            minutes = subprocess.run([PY, "-m", "idc.landing.minute_writer"],
                                     capture_output=True, text=True, creationflags=CREATE_FLAGS)
            (chain.dir / "minute_writer.log").write_text(minutes.stdout + minutes.stderr)
            accept = subprocess.run([PY, "-m", "idc.checks.acceptance"], capture_output=True,
                                    text=True, creationflags=CREATE_FLAGS)
            (chain.dir / "acceptance.log").write_text(accept.stdout + accept.stderr)
            chain.summary["acceptance_exit"] = accept.returncode
            chain.summary["acceptance_tail"] = accept.stdout.splitlines()[-3:]

            log.info("acceptance exit %d: %s", accept.returncode,
                     " | ".join(accept.stdout.splitlines()[-2:]))
        chain.summary["status"] = "failed" if failures else (
            "stopped" if chain.stopping() else "finished")
    except Exception as exc:
        chain.summary["status"] = "failed"
        chain.summary["errors"].append({"path": "orchestrator", "utc": now_iso(),
                                          "error": traceback.format_exc()})
        log.error("run failed: %s", exc)
    finally:
        finished.set()
        chain.kill_all()
        chain.summary["finished_utc"] = now_iso()
        chain.save()
        if not args.limit_seconds:
            report = subprocess.run([PY, "-m", "idc.checks.run_report", "--summary",
                                     str(chain.reports / f"chain_{chain.stamp}.json")],
                                    capture_output=True, text=True, creationflags=CREATE_FLAGS)
            (chain.dir / "run_report.log").write_text(report.stdout + report.stderr)
        log.info("summary %s status=%s", chain.reports / f"chain_{chain.stamp}.json",
                 chain.summary["status"])
    return 0 if chain.summary["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
