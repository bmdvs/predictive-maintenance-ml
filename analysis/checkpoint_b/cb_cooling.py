"""Checkpoint (b) rev2 (diag25), correction item 4: spindle cooling conditions alone in their episode, measured over
the condition's own window (onset to confirmation): housing temperature residual change (daily median less the
segment level-window median), vibration in dB above the segment level, and the controller thermistor residual."""

import json

import numpy as np
import pandas as pd

import r19_common as R
import r19_eval_helpers as H
import r2_common as C
from cb_common import REPO, D, core, inputs, loc
from pdm.advisory import requests as adv

K = core()
P2A = K["P2A"]
inp = inputs()
ind = inp["ind"]
cases = [("HMC-01-WO00010", "fold1"), ("HMC-06-WO00034", "eval")]
ev = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
fs = pd.read_parquet(R.S / "diag23" / "fold_scores.parquet")
tr = pd.read_parquet(R.S / "diag24" / "train_scores.parquet")
segs_e = pd.read_parquet(R.S / "diag22" / "prod_A_segs.parquet")
segs_f = pd.read_parquet(R.S / "diag23" / "ml_fold_segments.parquet")
segs_t = pd.read_parquet(R.S / "diag23" / "ml_train_segments.parquet")
for d in (ev, fs, tr):
    d["hour_utc"] = pd.to_datetime(d["hour_utc"], utc=True)
for d in (segs_e, segs_f, segs_t):
    for c in ("segment_start", "level_ready"):
        d[c] = pd.to_datetime(d[c], utc=True)
ch = pd.read_parquet(REPO / "build" / "queue_cache" / "controller_daily.parquet")
ch["day"] = pd.to_datetime(ch["local_date"]).dt.tz_localize(None) if pd.to_datetime(ch["local_date"]).dt.tz is None else pd.to_datetime(ch["local_date"]).dt.tz_convert("UTC").dt.tz_localize(None)
conds = inp["conds"]
out = []
for wo, period in cases:
    c = conds[conds["wo_id"] == wo].iloc[0]
    own = H.own_start(c["condition_type"], c["asset_key"], c["credit_ts"], P2A)
    on = own if pd.notna(own) else C.truth_onsets(conds[conds["wo_id"] == wo]).set_index("wo_id")["truth_onset"].get(wo)
    t0, t1 = on, c["credit_ts"]
    if period == "eval":
        sc, sg = ev, segs_e
    else:
        sc = pd.concat([tr[tr["hour_utc"] < pd.Timestamp("2024-07-01", tz="UTC")], fs[fs["fold"] == 1].drop(columns=["fold"])], ignore_index=True)
        sg = pd.concat([segs_t, segs_f[segs_f["fold"] == 1].drop(columns=["fold"])], ignore_index=True).drop_duplicates(["point_id", "segment_start"])
    pts = [p for p, a in P2A.items() if a == c["asset_key"]]
    temp = adv.temp_residual_daily(sc[sc["point_id"].isin(pts)], sg[sg["point_id"].isin(pts)])
    temp["day"] = pd.to_datetime(temp["day"])
    td = temp["day"].dt.tz_localize("UTC") if temp["day"].dt.tz is None else temp["day"]
    w = temp[(td >= t0.floor("D")) & (td <= t1)]
    r = {"wo_id": wo, "period": period, "asset": c["asset_key"], "onset": loc(t0), "confirmed": loc(t1), "days": round((t1 - t0) / pd.Timedelta(days=1), 1),
         "temperature residual change, max daily median (C)": float(w["temp_dc"].max()) if len(w) else np.nan,
         "last 7 days median (C)": float(w[pd.to_datetime(w["day"]).dt.tz_localize(None) >= (t1 - pd.Timedelta(days=7)).tz_localize(None).floor("D")]["temp_dc"].median()) if len(w) else np.nan}
    for col, name, f in (("vel_rms_mm_s", "velocity", 20), ("hf_rms_g", "HF", 20), ("env_bpfo_1_g2", "BPFO", 10), ("env_bpfi_1_g2", "BPFI", 10), ("env_bsf_1_g2", "BSF", 10)):
        vals = []
        for p in pts:
            s = sg[(sg["point_id"] == p) & (sg["segment_start"] <= t0)].sort_values("segment_start").tail(1)
            if s.empty:
                continue
            lvl = float(s[f"level_log10_{col}"].iloc[0])
            x = ind[(ind["point_id"] == p) & (ind["hour_utc"] >= t0) & (ind["hour_utc"] < t1)][col]
            if len(x):
                vals.append(f * (np.log10(x.clip(lower=1e-12)).median() - lvl))
        r[f"{name} (dB, median over the window)"] = max(vals) if vals else np.nan
    mch = c["asset_key"][:6]
    th = ch[(ch["machine_id"] == mch) & (ch["day"] >= t0.tz_localize(None).floor("D")) & (ch["day"] <= t1.tz_localize(None))]["thermistor_residual_c"]
    r["controller thermistor residual, max daily (C)"] = float(th.max()) if len(th) else np.nan
    out.append(r)
    print(r, flush=True)
(D / "cooling.json").write_text(json.dumps(out, indent=1, default=str))
