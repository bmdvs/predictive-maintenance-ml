"""Checkpoint (b) rev2 (diag25), item 8: the ML Model's percentile sweep (84th to 99.99th) from the prod
scores' sweep thresholds, and the Vendor Rules' sweep (10% to 100% of the zone B width), both under E2 to E7:
strict catches against truth-based false alerts per asset-month (and the confirmation-based rate)."""

import copy
import json
import pickle

import numpy as np
import pandas as pd

import diag_lib as L
import r19_common as R
import r21_common as Q
from cb_common import D, core, inputs
from pdm import ml_tasks as M
from pdm.models import nbm

K = core()
cp, conf, SPAN, AM, P2A = K["cp"], K["conf"], K["SPAN"], K["AM"], K["P2A"]
inp = inputs()
cfg = copy.deepcopy(L.CFG)
ctrl = pickle.loads((R.S / "diag18" / "ctrl_eval.pkl").read_bytes())
sc = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
sc = sc[sc["hour_utc"] >= SPAN[0]].reset_index(drop=True)
sg = pd.read_parquet(R.S / "diag22" / "prod_A_segs.parquet")
for c in ("segment_start", "level_ready"):
    sg[c] = pd.to_datetime(sg[c], utc=True)
rows = []
op = float(cfg["models"]["nbm"]["threshold_percentile"])
for p in [x for x in nbm.sweep_percentiles(cfg) if x >= 84.0]:
    col = f"thr_{p:g}"
    lp = R.ml_lev(sc, sg, copy.deepcopy(cfg), inp["ind"], ctrl, thr=col, alarm=False)
    m = Q.evaluate3(lp, cp, conf, inp["wo"], SPAN, SPAN[1], "E7", P2A, alarm=False)
    rows.append({"method": "ML Model", "setting": f"{p:g}th percentile", "operating point": p == op, "caught": m["caught"], "strict": m["strict"],
                 "false per asset-month (truth-based)": m["false_truth"] / AM, "false per asset-month (confirmation-based)": m["false_conf"] / AM})
    print(rows[-1], flush=True)
# operating point from the threshold column (the prod episodes)
m = K["res"][("E7", "ML Model")]
rows.append({"method": "ML Model", "setting": "operating threshold (prod)", "operating point": True, "caught": m["caught"], "strict": m["strict"],
             "false per asset-month (truth-based)": m["false_truth"] / AM, "false per asset-month (confirmation-based)": m["false_conf"] / AM})
import diag_folds as F

store = F._Store()
rf = store.read_pd("gold", "rule_floor_exceedances")
rf["ts"] = pd.to_datetime(rf["ts"], utc=True)
if "rules" not in rf:
    rf["rules"] = rf["level"]
pts = inp["pts"][["point_id", "asset_id"]]
vop = float(cfg["rules"]["vendor"]["alert_frac_zone_b_width"])
for s_ in cfg["evaluation"]["vendor_sweep_alert_frac"]:
    v = rf[(rf["method"] == "iso") | ((rf["method"] == "vendor") & np.isclose(rf["sweep"], s_))].merge(pts, on="point_id", how="left")
    v = v[(v["ts"] >= SPAN[0]) & (v["ts"] < SPAN[1])]
    vl = M.vendor_levels(v, inp["ind"], R.ECFG)
    m = Q.evaluate3(vl, cp, conf, inp["wo"], SPAN, SPAN[1], "E7", P2A, alarm=False)
    rows.append({"method": "Vendor Rules", "setting": f"{s_:.0%} of zone B width", "operating point": bool(np.isclose(s_, vop)), "caught": m["caught"], "strict": m["strict"],
                 "false per asset-month (truth-based)": m["false_truth"] / AM, "false per asset-month (confirmation-based)": m["false_conf"] / AM})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_parquet(D / "curves.parquet", index=False)
