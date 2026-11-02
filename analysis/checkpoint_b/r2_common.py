"""Scratch: shared inputs for the round 2 fold diagnosis (all read before the training end)."""

from __future__ import annotations

import os
import copy
from pathlib import Path

import numpy as np
import pandas as pd

import diag_folds as F
import diag_lib as L
from dlt import read
from pdm import ml_tasks as M
from pdm import splits
from pdm.evaluation import rev2 as r2

D = Path(os.environ.get("PDM_ANALYSIS_DIR", Path(__file__).parent)) / "diag3"
cfg = copy.deepcopy(L.CFG)
cfg["evaluation"]["holdout_shadow"] = True
store = F._Store()
B = splits.boundaries(cfg)
END = B["train"][1]
inp = M.load_inputs(store, cfg, END)
conf = r2.confirmations(inp["conds"], inp["wo"], END)
ctx = {"inspection_hours_each": 1.2, "margin": {}}
pts = inp["pts"]
P2A = pts.set_index("point_id")["asset_id"].to_dict()
ASSETS = sorted(pts["asset_id"].unique())
NA = len(ASSETS)
W = float(cfg["evaluation"]["catch"]["opened_within_days"])
GAP = float(cfg["evaluation"]["catch"]["episode_gap_days"])
FOLDS = splits.folds(cfg)
POOL = (FOLDS[0]["score"][0], FOLDS[-1]["score"][1])
FLOOR = float(read("gold/ml_settings")["curve_floor_percentile"].iat[0])
from pdm.models import nbm  # noqa: E402

PCTS = [p for p in nbm.sweep_percentiles(cfg) if p >= FLOOR]
OP = 99.5
DRAWS = 2000
SEED = int(cfg["models"]["seed"])


def cp_of(k: int) -> pd.DataFrame:
    f = FOLDS[k - 1]
    return M.asset_conditions(inp["conds"], *f["score"])


def vendor_levels_fold(k: int) -> pd.DataFrame:
    a, z = FOLDS[k - 1]["score"]
    return M.vendor_levels(inp["vendor"][(inp["vendor"]["ts"] >= a) & (inp["vendor"]["ts"] < z)], inp["ind"], cfg)


def truth_onsets(conds: pd.DataFrame) -> pd.DataFrame:
    """Per condition: the start of the truth-active run on the asset that reaches the confirmation (active
    hours merged across gaps up to the episode gap), within the catch window; NaT when none."""
    th = L.truth_hours()
    th = th.assign(asset_id=th["point_id"].map(P2A))
    act = th[~th["truth_healthy"].fillna(False).astype(bool)]
    out = []
    for c in conds.itertuples():
        t = c.credit_ts
        a = act[(act["asset_id"] == c.asset_key) & (act["hour_utc"] <= t) & (act["hour_utc"] > t - pd.Timedelta(days=W))]
        if a.empty:
            out.append({"wo_id": c.wo_id, "truth_onset": pd.NaT})
            continue
        h = np.sort(a["hour_utc"].unique())
        j = len(h) - 1
        gap = np.timedelta64(int(GAP * 24), "h")
        while j > 0 and h[j] - h[j - 1] <= gap:
            j -= 1
        out.append({"wo_id": c.wo_id, "truth_onset": pd.Timestamp(h[j]).tz_localize("UTC") if pd.Timestamp(h[j]).tzinfo is None else pd.Timestamp(h[j])})
    return pd.DataFrame(out)
