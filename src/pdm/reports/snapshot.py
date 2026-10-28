"""Read access to the gold snapshot (data/gold_snapshot) with the Store's read interface, so the page
build reuses the pipeline's own functions without a workspace or Spark.

Inputs: the snapshot's Parquet files by schema and table. Output: pandas DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pdm import paths

ROOT = paths.GOLD_SNAPSHOT_DIR
RENAMED = {("silver", "controller_hourly"): "controller_hourly_hmc"}


class SnapshotStore:
    """The snapshot behind Store.read_pd; there is no catalog and nothing is written."""

    catalog = None

    def __init__(self, root: Path = ROOT) -> None:
        self.root = Path(root)

    def path(self, schema: str, table: str) -> Path:
        return self.root / schema / f"{RENAMED.get((schema, table), table)}.parquet"

    def read_pd(self, schema: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        p = self.path(schema, table)
        if not p.exists():
            raise FileNotFoundError(f"{schema}.{table} is not in the snapshot ({p})")
        return pd.read_parquet(p, columns=columns)

    def exists(self, schema: str, table: str) -> bool:
        return self.path(schema, table).exists()

    def file(self, relative: str) -> Path:
        return self.root / relative
