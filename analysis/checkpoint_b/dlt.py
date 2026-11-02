"""Scratch: read the current version of a local Delta table by replaying its JSON log (no checkpoints)."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2] / "data" / "local_gold"


def read(table: str, columns: list[str] | None = None, version: int | None = None) -> pd.DataFrame:
    d = ROOT / table
    files: set[str] = set()
    logs = sorted((d / "_delta_log").glob("*.json"))
    if version is not None:
        logs = [lg for lg in logs if int(lg.stem) <= version]
    for log in logs:
        for line in log.read_text(encoding="utf-8").splitlines():
            a = json.loads(line)
            if "add" in a:
                files.add(a["add"]["path"])
            elif "remove" in a:
                files.discard(a["remove"]["path"])
    from urllib.parse import unquote

    parts = [pq.read_table(d / unquote(f), columns=columns).to_pandas() for f in sorted(files)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
