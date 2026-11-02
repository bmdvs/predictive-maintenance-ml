"""Chronological, entity and random-window splits with the configured gap at each boundary.

Input: configuration and timestamps. Output: split labels (training, shadow, gap), the expanding-window
folds inside training, and leave-one-asset-out folds. The random-window split exists only for the
leakage audit. There is no validation period (revision 2; originally training, validation, shadow).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Chicago"


def _span(dates: list[Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(str(dates[0]), tz=LOCAL_TZ).tz_convert("UTC")
    b = pd.Timestamp(str(dates[1]), tz=LOCAL_TZ).tz_convert("UTC") + pd.Timedelta(days=1)
    return a, b


def boundaries(cfg: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Split start (inclusive) and end (exclusive) in UTC, from the local split dates: train, shadow,
    and fold<k>_fit and fold<k>_score for each expanding-window fold (k from 1)."""
    s = cfg["splits"]
    out = {name: _span(s[name]) for name in ("train", "shadow")}
    for k, f in enumerate(s["folds"], start=1):
        out[f"fold{k}_fit"] = _span(f["fit"])
        out[f"fold{k}_score"] = _span(f["score"])
    return out


def folds(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expanding-window folds inside training: fit start, fit cutoff (the fit end less the gap), and the
    scored span, which starts at the fit end."""
    b = boundaries(cfg)
    gap = pd.Timedelta(days=cfg["splits"]["gap_days"])
    out = []
    for k in range(1, len(cfg["splits"]["folds"]) + 1):
        fit, score = b[f"fold{k}_fit"], b[f"fold{k}_score"]
        out.append({"fold": k, "fit_start": fit[0], "cutoff": fit[1] - gap, "score": score})
    return out


def chronological(ts: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """training, shadow, or gap for each timestamp (UTC); the gap is removed at the end of training."""
    b = boundaries(cfg)
    gap = pd.Timedelta(days=cfg["splits"]["gap_days"])
    t = pd.to_datetime(ts, utc=True)
    label = pd.Series("outside", index=ts.index, dtype=object)
    label[(t >= b["train"][0]) & (t < b["train"][1] - gap)] = "training"
    label[(t >= b["train"][1] - gap) & (t < b["train"][1])] = "gap"
    label[(t >= b["shadow"][0]) & (t < b["shadow"][1])] = "shadow"
    return label


def reference_window(cfg: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The last `splits.reference_months` of training before the cutoff: the reference for everything
    previously set from the validation period (revision 2)."""
    end = training_cutoff(cfg)
    return end - pd.DateOffset(months=int(cfg["splits"]["reference_months"])), end


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
