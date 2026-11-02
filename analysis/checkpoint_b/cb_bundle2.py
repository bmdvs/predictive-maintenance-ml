"""Checkpoint (b) rev2 (diag25): the model report bundle's selection tables and re-centering figures."""

import json

import numpy as np
import pandas as pd

import r19_common as R
from cb_common import REPO, D, cb, conds51, core

OUT = REPO / "build" / "report_data"
K = core()
SPAN = K["SPAN"]
DAY = pd.Timedelta(days=1)
B = json.loads((OUT / "bundle.json").read_text())
cs = pd.read_csv(R.S / "diag11" / "candidates_summary.csv")
cs = cs[cs["fold"] == "pooled"].set_index("candidate")
sel = pd.read_csv(R.S / "diag11" / "selection_70.csv").set_index("settings")
names = {"A": ("Autoencoder [16-6-16], 200 epochs", "vendor (13)", "One per point", "autoencoder_vendor [16-6-16] 200 ep, single"),
         "B": ("Autoencoder [16-6-16], 400 epochs", "extended (37)", "By speed band", "autoencoder_extended [16-6-16] 400 ep, speed_band"),
         "C": ("Mahalanobis distance", "extended (37)", "One per point", "mahalanobis_extended, single"),
         "D": ("Autoencoder [16-6-16], 200 epochs", "full spectrum and envelope (52)", "One per point", "autoencoder_full_env [16-6-16] 200 ep, single")}
rows = []
for k, (mt, iset, mode, key) in names.items():
    r = cs.loc[k]
    s = sel.loc[key]
    rows.append({"Model": mt, "Inputs": iset, "Threshold": mode, "Catch at budget": s["caught_share_at_budget"], "Chance": s["chance_share_at_budget"],
                 "Skill above chance": r["skill mean"], "Caught at operating point": r["caught at ALERT mean"], "False episodes per asset-month": r["false ALERT per asset-month mean"],
                 "Selected": "✔" if k == "A" else ""})
pd.DataFrame(rows).sort_values("Skill above chance", ascending=False).to_parquet(OUT / "candidates.parquet", index=False)
pers = pd.read_parquet(D / "persistence.parquet")
d18 = pd.read_csv(R.S / "diag18" / "summary.csv")
d18 = d18[(d18["model"] == "A") & (d18["evaluation"] == "current")].set_index("row")
d20 = pd.read_csv(R.S / "diag20" / "fold_pooled.csv")
d20 = d20[d20["definition"] == "E4"].set_index("row")
d21 = pd.read_csv(R.S / "diag21" / "fold_pooled.csv")
d21 = d21[d21["definition"] == "E7"].set_index("row")
d14 = pd.read_csv(R.S / "diag14" / "confirmed_summary.csv")
d14 = d14[(d14["finalist"] == "A") & (d14["fold"] == "pooled")]
dr = []
for iset, n in (("extended", 37), ("full", 45), ("full_env", 52), ("vendor", 13)):
    s = sel.loc[f"autoencoder_{iset} [16-6-16] 200 ep, single"]
    dr.append((f"{n} inputs" + (" (the 13 measurements, kept)" if n == 13 else " instead of the 13 measurements"), "Selection round", s["caught_operating"], s["false_per_asset_month_operating"], np.nan, "kept" if n == 13 else "not adopted"))
for mode in ("single", "speed_band"):
    r = d14[(d14["index"] == 9) & (d14["mode"] == mode)].iloc[0]
    dr.append(("Thresholds by speed band" if mode == "speed_band" else "One threshold per point (same model)", "Tuning round", r["caught at ALERT mean"], r["false ALERT per asset-month mean"], np.nan, "not adopted" if mode == "speed_band" else "kept"))
for hrs in (3, 6, 12):
    p = pers[pers["persistence (h)"] == hrs]
    dr.append((f"{hrs} hours of persistence" + (" (kept)" if hrs == 6 else " instead of 6"), "Version 13 folds", float(p["strict"].sum()), p["false (truth-based)"].sum() / p["asset_months"].sum(), np.nan, "kept" if hrs == 6 else "not adopted"))
b18 = d18.loc["baseline"]
dr.append(("Fix round 1 baseline", "Fix round 1", b18["caught"], b18["false"], b18["stuck points"], ""))
for fx, res in (("+F1", "adopted"), ("+F2", "set aside: stuck points 8.0 against 6.0"), ("+F3", "adopted"), ("+F4", "adopted (tightened)")):
    r = d18.loc[fx]
    dr.append((f"{fx[1:]}: " + {"+F1": "hold the last accepted level", "+F2": "a shorter post-repair window", "+F3": "spindle temperature residual with regime terms", "+F4": "fast ALERT at 5 times the threshold"}[fx],
               "Fix round 1", r["caught"], r["false"], r["stuck points"], res))
dr.append(("Final round baseline", "Final round", d20.loc["baseline", "caught"], d20.loc["baseline", "false"], d20.loc["baseline", "stuck"], ""))
for rw, lab, res in (("T1", "training without Vendor Rules alert hours", "set aside: false alerts 0.210, stuck points 14.6"), ("T2", "seasonal temperature term only with 9 months", "set aside: stuck points 8.6"),
                     ("T3", "WATCH only under 1,000 training hours", "set aside: caught 14.8"), ("S1", "post-repair level window of up to 60 days", "adopted")):
    r = d20.loc[rw]
    dr.append((f"{rw}: {lab}", "Final round", r["caught"], r["false"], r["stuck"], res))
dr.append(("Improvement round baseline", "Improvement round", d21.loc["baseline", "strict"], d21.loc["baseline", "false_truth"], d21.loc["baseline", "stuck_new"], ""))
for rw, lab, res in (("C1", "the wear-stretch exclusion", "set aside: strict 19.2, stuck points 5.4"), ("C2", "the seasonal clamp on the temperature residual", "set aside: stuck points 5.0"),
                     ("C3", "re-arm of a point flat above its threshold", "adopted"), ("C4", "post-repair window after repairs to other components", "adopted")):
    r = d21.loc[rw]
    dr.append((f"{rw}: {lab}", "Improvement round", r["strict"], r["false_truth"], r["stuck_new"], res))
pd.DataFrame(dr, columns=["Design choice", "Round", "Caught", "False alerts per asset-month", "Stuck points", "Result"]).to_parquet(OUT / "design.parquet", index=False)
# re-centering
segs = cb("ml_segments")
segs["segment_start"] = pd.to_datetime(segs["segment_start"], utc=True)
yr = segs[(segs["segment_start"] >= SPAN[0]) & (segs["segment_start"] < SPAN[1]) & (segs["reason"] != "install")]
s4 = (D / "s4.md").read_text(encoding="utf-8")
import re  # noqa: E402

m = re.search(r"ML Model false ALERT episodes opened within 14 days after one: (\d+) of (\d+)\. Post-repair inspection \(plausibility\) flags raised: (\d+); found real \(a confirmed condition on the asset within 120 days\): (\d+)", s4)
B["recenter"] = {"false_after": int(m.group(1)), "flags": int(m.group(3)), "found": int(m.group(4)),
                 "what": "lubrication degradation on the HMC-01 and HMC-08 spindles and a bearing failure on HMC-03's spindle"}
(OUT / "bundle.json").write_text(json.dumps(B, indent=1, default=str))
print(B["recenter"], len(dr))
