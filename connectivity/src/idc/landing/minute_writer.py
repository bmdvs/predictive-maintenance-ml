"""Build the controller minute_agg landing files from the landed hz1 files.

For each landed ``controller/hz1/month=YYYY-MM/<machine>.parquet`` this writes
``controller/minute_agg/month=YYYY-MM/<machine>.parquet`` with the pipeline's own aggregation
(``pdm.ingest.mtconnect_stream.aggregate_minutes``) plus the source ``run_id`` and
``config_hash`` columns, the same construction the bronze export uses. Rewrites are
idempotent: the output is a pure function of the hz1 file.

Usage:
    python -m idc.landing.minute_writer [--machines HMC-01,...] [--months 2026-04,...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("idc.minute_writer")


def build_minutes(hz1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an hz1 frame to minute_agg rows with the source run columns appended."""
    from pdm.ingest.mtconnect_stream import HZ1_COLUMNS, aggregate_minutes

    minutes = aggregate_minutes(hz1[HZ1_COLUMNS])
    minutes["run_id"] = hz1["run_id"].iloc[0]
    minutes["config_hash"] = hz1["config_hash"].iloc[0]
    return minutes


def write_minutes(landing_root: Path, machine_id: str, month: str) -> tuple[Path, int]:
    """Write one machine-month minute_agg file from its landed hz1 file."""
    source = landing_root / "controller" / "hz1" / f"month={month}" / f"{machine_id}.parquet"
    target = landing_root / "controller" / "minute_agg" / f"month={month}" / \
        f"{machine_id}.parquet"
    minutes = build_minutes(pd.read_parquet(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    minutes.to_parquet(target, index=False)
    return target, len(minutes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minute_writer")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    parser.add_argument("--machines", default=None, help="comma-separated machine IDs")
    parser.add_argument("--months", default=None, help="comma-separated YYYY-MM months")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from idc.checks.compare_exports import load_config
    config = load_config(args.config)
    landing_root = Path(config["paths"]["landing_root"])
    machines = args.machines.split(",") if args.machines else list(config["replay"]["hmcs"])
    months = args.months.split(",") if args.months else list(config["replay"]["months"])
    missing = 0
    for machine_id in machines:
        for month in months:
            source = landing_root / "controller" / "hz1" / f"month={month}" / \
                f"{machine_id}.parquet"
            if not source.exists():
                log.error("no landed hz1 file for %s %s", machine_id, month)
                missing += 1
                continue
            path, rows = write_minutes(landing_root, machine_id, month)
            log.info("wrote %s (%d rows)", path, rows)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
