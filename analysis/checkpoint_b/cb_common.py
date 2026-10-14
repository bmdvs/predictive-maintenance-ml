"""Checkpoint (b) rev2 (run tag diag25): shared loading and markdown helpers for the section scripts."""

import copy
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import diag_folds as F
import diag_lib as L
import r19_common as R

D = R.S / "diag25"
REPO = Path(__file__).resolve().parents[2]
CB = REPO / "build" / "cb_cache"
DAY = pd.Timedelta(days=1)
TZ = "America/Chicago"
METH = ("ML Model", "Vendor Rules")
DEFN = {"E7": "E2 to E7 (headline)", "E4": "E2 to E4", "orig": "original + E2, E3"}


def core() -> dict:
    return pickle.loads((D / "core.pkl").read_bytes())


def conds51() -> pd.DataFrame:
    return pickle.loads((D / "conds51.pkl").read_bytes())


def cb(name: str) -> pd.DataFrame:
    d = pd.read_parquet(CB / f"{name}.parquet")
    for c in d.columns:
        if c.endswith("_ts") or c in ("opened", "closed", "started", "last", "hour_utc", "created", "alert_opened"):
            try:
                d[c] = pd.to_datetime(d[c], utc=True)
            except (TypeError, ValueError):
                pass
    return d


@lru_cache(maxsize=1)
def inputs():
    from pdm import ml_tasks as M
    from pdm import splits

    cfg = copy.deepcopy(L.CFG)
    return M.load_inputs(F._Store(), cfg, splits.boundaries(cfg)["shadow"][1])


def loc(ts, fmt="%Y-%m-%d") -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert(TZ).strftime(fmt)


def fmtv(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:,.1f}"
    if isinstance(v, pd.Timestamp):
        return loc(v, "%Y-%m-%d %H:%M")
    return "" if v is None else str(v)


def md(df: pd.DataFrame, index: bool = False) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in d.itertuples(index=False):
        out.append("| " + " | ".join(fmtv(v).replace("|", "/") for v in r) + " |")
    return "\n".join(out) + "\n"


def sec(title: str, level: int = 2) -> str:
    return "\n" + "#" * level + " " + title + "\n\n"


def note(s: str) -> str:
    """A one-line source or rule statement under a table (rules as coded; no prose)."""
    return f"Rule as coded: {s}\n\n"


LONG = "inside an episode opened more than 120 days earlier"


def miss_reason(lev: pd.DataFrame, asset: str, t: pd.Timestamp, coded: str | None) -> str:
    """rev2.miss_reason with the long-episode category (2026-11-01): an episode on the asset open at the
    confirmation that opened more than 120 days before it; otherwise the reason as coded in prod."""
    e = lev[(lev["asset_id"] == asset) & (lev["opened"] < t - pd.Timedelta(days=120)) & (lev["closed"] > t)]
    return LONG if len(e) else (coded or "not caught")
