"""Checkpoint (b) rev2 core (run tag diag25): evaluation-year episodes of both methods from prod
gold.level_episodes (version 13, D2), checked against the helpers' rebuild from the prod scores; per-condition
results under E2 to E7, E2 to E4 and original + E2, E3 for both methods. Writes diag25/core.pkl."""

import copy
import pickle

import pandas as pd

import diag_folds as F
import diag_lib as L
import r19_common as R
import r21_common as Q
import r2_common as C
from pdm import ml_tasks as M
from pdm import splits, tasks
from pdm.evaluation import rev2 as r2

D = R.S / "diag25"
D.mkdir(exist_ok=True)
REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
store = F._Store()
cfg = copy.deepcopy(L.CFG)
B = splits.boundaries(cfg)
SPAN = B["shadow"]
inp = M.load_inputs(store, cfg, SPAN[1])
P2A = inp["pts"].set_index("point_id")["asset_id"].to_dict()
conf = r2.confirmations(inp["conds"], inp["wo"], SPAN[1])
cp = R.scope(M.asset_conditions(inp["conds"], *SPAN))
cp_all = M.asset_conditions(inp["conds"], *SPAN)
AM = M.asset_months(SPAN, C.NA)
conds_all = tasks._conditions(store, cfg)
lev = pd.read_parquet(REPO / "build" / "cb_cache" / "level_episodes.parquet")
for c in [c for c in lev.columns if c.endswith("_ts") or c in ("opened", "last", "closed")]:
    lev[c] = pd.to_datetime(lev[c], utc=True)
lev = lev[lev["period"] == "shadow"].reset_index(drop=True)
ml = lev[lev["method"] == "ml_model"].reset_index(drop=True)
vd = lev[lev["method"] == "vendor_rules"].reset_index(drop=True)
# check: the helpers' rebuild of the ML Model's episodes from the prod scores equals prod
import pickle as pk

sc = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
sc = sc[sc["hour_utc"] >= SPAN[0]].reset_index(drop=True)
sg = pd.read_parquet(R.S / "diag22" / "prod_A_segs.parquet")
for c in ("segment_start", "level_ready"):
    sg[c] = pd.to_datetime(sg[c], utc=True)
ctrl = pk.loads((R.S / "diag18" / "ctrl_eval.pkl").read_bytes())
rb = R.ml_lev(sc, sg, copy.deepcopy(cfg), inp["ind"], ctrl)
key = ["asset_id", "opened", "closed", "alarm_ts"]
same = rb[key].sort_values(key).reset_index(drop=True).equals(ml[key].sort_values(key).reset_index(drop=True))
print("ML episodes: prod", len(ml), "rebuild", len(rb), "identical (asset, opened, closed, alarm)", same, flush=True)
res = {}
for d in Q.DEFS3:
    for name, lv in (("ML Model", ml), ("Vendor Rules", vd)):
        m = Q.evaluate3(lv, cp, conf, inp["wo"], SPAN, SPAN[1], d, P2A)
        res[(d, name)] = m
        print(d, name, {k: m[k] for k in ("conditions", "caught", "strict", "false_conf", "false_truth")}, flush=True)
(D / "core.pkl").write_bytes(
    pickle.dumps({"res": res, "ml": ml, "vd": vd, "cp": cp, "cp_all": cp_all, "conf": conf, "AM": AM, "P2A": P2A, "SPAN": SPAN, "conds_all": conds_all, "episodes_same": same, "rebuild": rb})
)
