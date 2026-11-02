"""Checkpoint (b) rev2 (diag25), items 18 and 19: version 13's fold models (diag23 fold scores) and the
Vendor Rules on each fold under E2 to E7 and E2 to E4; persistence 3 and 12 hours on the folds; the evaluation
year by quarter; catch windows of 60 and 90 days; 90% intervals from resampling whole conditions."""

import copy
import pickle

import numpy as np
import pandas as pd

import diag_folds as F
import diag_lib as L
import r19_common as R
import r21_common as Q
import r2_common as C
from cb_common import D, core, inputs
from pdm import ml_tasks as M
from pdm import splits

K = core()
cfg = copy.deepcopy(L.CFG)
inp = inputs()
P2A = K["P2A"]
ctrl9 = pickle.loads((R.S / "diag9" / "ctrl.pkl").read_bytes())
fs = pd.read_parquet(R.S / "diag23" / "fold_scores.parquet")
fs["hour_utc"] = pd.to_datetime(fs["hour_utc"], utc=True)
fseg = pd.read_parquet(R.S / "diag23" / "ml_fold_segments.parquet")
for c in ("segment_start", "level_ready"):
    fseg[c] = pd.to_datetime(fseg[c], utc=True)
store = F._Store()
v = inp["vendor"]
rows, cts = [], {}


def run(name, lev_ml, cp, conf, span, end, am):
    vl = M.vendor_levels(v[(v["ts"] >= span[0]) & (v["ts"] < span[1])], inp["ind"], R.ECFG)
    for d in ("E7", "E4"):
        for meth, lv in (("ML Model", lev_ml), ("Vendor Rules", vl)):
            m = Q.evaluate3(lv, cp, conf, inp["wo"], span, end, d, P2A)
            cts[(name, d, meth)] = m["ct"]
            rows.append({"period": name, "definition": d, "method": meth, "conditions": m["conditions"], "caught": m["caught"], "strict": m["strict"] if d == "E7" else np.nan,
                         "false (truth-based) per asset-month": m["false_truth"] / am, "false (confirmation-based) per asset-month": m["false_conf"] / am,
                         "false ALARMs per asset-month (sensor)": m["alarm_false"] / am})
            print(rows[-1], flush=True)


pers = []
for k, f in enumerate(C.FOLDS, start=1):
    span, cut = f["score"], f["cutoff"]
    sc = M.low_pressure_flags(store, fs[fs["fold"] == k].drop(columns=["fold"]).reset_index(drop=True))
    sg = fseg[fseg["fold"] == k].drop(columns=["fold"]).reset_index(drop=True)
    cp = R.scope(C.cp_of(k))
    am = M.asset_months(span, C.NA)
    lev = R.ml_lev(sc, sg, copy.deepcopy(cfg), inp["ind"], ctrl9)
    run(f"fold {k}", lev, cp, C.conf, span, C.END, am)
    ss = sc[sc["state"] == "scored"].sort_values(["point_id", "hour_utc"], kind="stable").reset_index(drop=True)
    for hrs in (3, 6, 12):
        eps = M.persisted_episodes(ss, "threshold", hrs, cfg)
        m = Q.evaluate3(eps, cp, C.conf, inp["wo"], span, C.END, "E7", P2A, alarm=False)
        pers.append({"fold": k, "persistence (h)": hrs, "caught": m["caught"], "strict": m["strict"], "false (truth-based)": m["false_truth"], "asset_months": am})
# evaluation year by quarter (the year's episodes, conditions credited in the quarter)
SPAN = K["SPAN"]
qs = pd.date_range(SPAN[0].tz_convert("America/Chicago").tz_localize(None), periods=5, freq="3MS")
for i in range(4):
    a = pd.Timestamp(qs[i], tz="America/Chicago").tz_convert("UTC")
    b = pd.Timestamp(qs[i + 1], tz="America/Chicago").tz_convert("UTC")
    cp = K["cp"][(K["cp"]["credit_ts"] >= a) & (K["cp"]["credit_ts"] < b)].reset_index(drop=True)
    run(f"evaluation year Q{i + 1}", K["ml"], cp, K["conf"], (a, b), SPAN[1], M.asset_months((a, b), C.NA))
# catch windows
win = []
for wd in (60.0, 90.0, 120.0):
    R.WD = wd
    for meth, lv in (("ML Model", K["ml"]), ("Vendor Rules", K["vd"])):
        m = Q.evaluate3(lv, K["cp"], K["conf"], inp["wo"], SPAN, SPAN[1], "E7", P2A, alarm=False)
        win.append({"catch window (days)": int(wd), "method": meth, "caught": m["caught"], "strict": m["strict"], "false (truth-based) per asset-month": m["false_truth"] / K["AM"]})
R.WD = 120.0
pd.DataFrame(rows).to_parquet(D / "folds_periods.parquet", index=False)
pd.DataFrame(pers).to_parquet(D / "persistence.parquet", index=False)
pd.DataFrame(win).to_parquet(D / "windows.parquet", index=False)
(D / "fold_cts.pkl").write_bytes(pickle.dumps(cts))
print("done")
