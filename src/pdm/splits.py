"""Chronological, entity and random-window splits with the configured gap at each boundary.

Input: configuration and timestamps. Output: split labels (training, validation, shadow, gap) and
leave-one-asset-out folds. The random-window split exists only for the leakage audit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Chicago"


def boundaries(cfg: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Split start (inclusive) and end (exclusive) in UTC, from the local split dates."""
    s = cfg["splits"]
    out = {}
    for name in ("train", "validation", "shadow"):
        a = pd.Timestamp(str(s[name][0]), tz=LOCAL_TZ).tz_convert("UTC")
        b = pd.Timestamp(str(s[name][1]), tz=LOCAL_TZ).tz_convert("UTC") + pd.Timedelta(days=1)
        out[name] = (a, b)
    return out


def chronological(ts: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """training, validation, shadow, or gap for each timestamp (UTC); the gap is removed at each boundary."""
    b = boundaries(cfg)
    gap = pd.Timedelta(days=cfg["splits"]["gap_days"])
    t = pd.to_datetime(ts, utc=True)
    label = pd.Series("outside", index=ts.index, dtype=object)
    label[(t >= b["train"][0]) & (t < b["train"][1] - gap)] = "training"
    label[(t >= b["validation"][0]) & (t < b["validation"][1] - gap)] = "validation"
    label[(t >= b["shadow"][0]) & (t < b["shadow"][1])] = "shadow"
    label[
        ((t >= b["train"][1] - gap) & (t < b["train"][1]))
        | ((t >= b["validation"][1] - gap) & (t < b["validation"][1]))
    ] = "gap"
    return label


def training_cutoff(cfg: dict[str, Any]) -> pd.Timestamp:
    """Last instant usable for training: the training end less the gap."""
    b = boundaries(cfg)
    return b["train"][1] - pd.Timedelta(days=cfg["splits"]["gap_days"])


def leave_one_asset_out(assets: list[str]) -> list[tuple[list[str], str]]:
    return [([a for a in assets if a != held], held) for held in assets]


def random_windows(ts: pd.Series, window_days: int, frac_test: float, seed: int) -> pd.Series:
    """Leakage audit only: windows of window_days assigned to test at random."""
    t = pd.to_datetime(ts, utc=True)
    w = ((t - t.min()) / pd.Timedelta(days=window_days)).astype(int)
    rng = np.random.default_rng(seed)
    ids = np.unique(w)
    test = set(rng.choice(ids, size=max(1, int(round(frac_test * len(ids)))), replace=False))
    return pd.Series(np.where(w.isin(test), "test", "train"), index=ts.index)
