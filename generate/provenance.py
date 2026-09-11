"""Run identity and seeding for the export replay: run id, config hash, generator version, seeds.

Input: the resolved configuration. Output: a RunContext stamped into every exported table, and
deterministic random generators derived from the run seed and a key path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pdm.paths import REPO_ROOT

RUNS_DIR = REPO_ROOT / "runs"


def config_hash(resolved: dict[str, Any]) -> str:
    blob = json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def derive_seed(run_seed: int, *parts: object) -> int:
    key = "|".join([str(run_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def rng_for(run_seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(run_seed, *parts))


@dataclass(frozen=True)
class RunContext:
    run_id: str
    config_hash: str
    generator_version: str
    run_seed: int
    label: str

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_id

    def stamp(self) -> dict[str, str]:
        return {"run_id": self.run_id, "config_hash": self.config_hash}


def new_run(resolved: dict[str, Any], label: str) -> RunContext:
    h = config_hash(resolved)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return RunContext(
        run_id=f"{label}_{ts}_{h[:8]}",
        config_hash=h,
        generator_version=str(resolved["config"]["meta"]["generator_version"]),
        run_seed=int(resolved["config"]["meta"]["run_seed"]),
        label=label,
    )
