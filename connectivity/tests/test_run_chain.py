"""Orchestrator start-up order: no adapter starts until every bridge has read its position."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from idc import run_chain as rc


class _Proc:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


class _Chain:
    """The parts of ChainRun that HmcPath uses, with bridges that confirm after a delay."""

    def __init__(self, tmp_path, ready_after: dict[int, float | None], timeout: float = 5.0):
        self.dir = tmp_path
        self.config = {"replay": {"hmcs": ["HMC-01", "HMC-04"], "months": ["2026-04"]},
                       "mtconnect": {"bridge_ready_timeout_seconds": timeout},
                       "paths": {"export_root": str(tmp_path), "landing_root": str(tmp_path)}}
        self.args = SimpleNamespace(machines=None, months=None, hmc_speed=450.0, groups=2,
                                    hmc_parallel=2, verify="off")
        self.lock = threading.Lock()
        self.procs: dict[str, _Proc] = {}
        self.summary = {"hmc": {"machine_months": {}, "bridge_restarts": [],
                                "writer_restarts": []}}
        self.ready_after = ready_after
        self.ready_at: dict[int, float] = {}

    def spawn(self, name, argv):
        proc = _Proc()
        self.procs[name] = proc
        if name.startswith("bridge_"):
            i = int(name.split("_")[1])
            delay = self.ready_after[i]
            if delay is not None:
                def confirm():
                    time.sleep(delay)
                    with open(self.dir / f"{name}.log", "ab") as f:
                        f.write(b"INFO starting position: agent instanceId=7 "
                                b"nextSequence=260421917 bufferSize=1048576\n")
                    self.ready_at[i] = time.monotonic()
                threading.Thread(target=confirm, daemon=True).start()
        return proc

    def stopping(self):
        return False

    def event(self, section, key, value):
        self.summary[section].setdefault(key, []).append(value)

    def save(self):
        pass


def _path(chain, monkeypatch) -> tuple[rc.HmcPath, list[tuple[str, float]]]:
    hmc = rc.HmcPath(chain)
    adapters: list[tuple[str, float]] = []
    monkeypatch.setattr(hmc, "start_writers", lambda clean: None)
    monkeypatch.setattr(hmc, "drain", lambda: None)
    monkeypatch.setattr(hmc, "replay",
                        lambda machine, month, speed, tag: adapters.append(
                            (machine, time.monotonic())) or 0)
    return hmc, adapters


def test_adapters_start_only_after_every_bridge_confirms(tmp_path, monkeypatch):
    chain = _Chain(tmp_path, ready_after={0: 0.3, 1: 1.0})
    hmc, adapters = _path(chain, monkeypatch)
    hmc.run()
    assert sorted(m for m, _ in adapters) == ["HMC-01", "HMC-04"]
    first_adapter = min(t for _, t in adapters)
    assert sorted(chain.ready_at) == [0, 1], "an adapter started before every bridge confirmed"
    assert first_adapter >= max(chain.ready_at.values())


def test_run_fails_when_a_bridge_never_confirms_and_starts_no_adapter(tmp_path, monkeypatch):
    chain = _Chain(tmp_path, ready_after={0: 0.1, 1: None}, timeout=1.0)
    hmc, adapters = _path(chain, monkeypatch)
    with pytest.raises(rc.RunFailed, match=r"bridge\(s\) \[1\] did not read a starting position"):
        hmc.run()
    assert adapters == []


def test_run_fails_when_a_bridge_exits_before_confirming(tmp_path, monkeypatch):
    chain = _Chain(tmp_path, ready_after={0: 0.1, 1: None})
    hmc, adapters = _path(chain, monkeypatch)
    original = chain.spawn

    def spawn(name, argv):
        proc = original(name, argv)
        if name == "bridge_1":
            proc.returncode = 2
        return proc

    chain.spawn = spawn
    with pytest.raises(rc.RunFailed, match="bridge 1 exited before reading its starting"):
        hmc.run()
    assert adapters == []


def test_drain_returns_when_published_counts_hold_while_idle_bookmarks_advance(tmp_path):
    chain = _Chain(tmp_path, ready_after={0: None, 1: None})
    hmc = rc.HmcPath(chain)
    for i in range(2):
        (tmp_path / f"controller_writer_{i}.log").write_text("INFO messages=3602 records=3600\n")
    stop = threading.Event()

    def idle_bridges():
        bookmark = 58629
        while not stop.is_set():
            bookmark += 2
            for i in range(2):
                with open(tmp_path / f"bridge_{i}.log", "a") as f:
                    f.write(f"INFO polls=1 observations=1 messages=3602 bookmark={bookmark}\n")
            time.sleep(0.05)

    threading.Thread(target=idle_bridges, daemon=True).start()
    started = time.monotonic()
    hmc.drain(quiet_seconds=0.5, poll_seconds=0.1)
    stop.set()
    assert time.monotonic() - started < 5
