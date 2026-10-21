"""Truth archive for one run: degradation state per point-hour, event timeline, work orders,
PM rows, the resolved configuration and seeds. Written under runs/<run_id>/truth/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from generate.asset_group import GroupOutput
from generate.provenance import RunContext


def _frame(rows: list[dict[str, Any]], ctx: RunContext) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(
                lambda v: v if v is None or isinstance(v, (str, bool, int, float, pd.Timestamp)) else str(v)
            )
    return df.assign(**ctx.stamp())


def write_truth(outputs: list[GroupOutput], ctx: RunContext, resolved: dict[str, Any]) -> Path:
    d = ctx.run_dir / "truth"
    d.mkdir(parents=True, exist_ok=True)
    _frame([r for o in outputs for r in o.truth_hours], ctx).to_parquet(
        d / "point_hours.parquet", index=False
    )
    _frame([e for o in outputs for e in o.events], ctx).to_parquet(d / "events.parquet", index=False)
    _frame([w for o in outputs for w in o.work_orders], ctx).to_parquet(
        d / "work_orders.parquet", index=False
    )
    _frame([p for o in outputs for p in o.pm_rows], ctx).to_parquet(d / "pm.parquet", index=False)
    with open(ctx.run_dir / "resolved_config.json", "w", encoding="utf-8") as fh:
        json.dump(resolved, fh, indent=1, default=str)
    with open(d / "seeds.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_seed": ctx.run_seed,
                "run_id": ctx.run_id,
                "config_hash": ctx.config_hash,
                "generator_version": ctx.generator_version,
            },
            fh,
            indent=1,
        )
    return d
