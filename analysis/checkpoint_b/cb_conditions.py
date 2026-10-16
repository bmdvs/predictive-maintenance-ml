"""Checkpoint (b) rev2: the in-scope conditions with onset and confirmation for the timeline shading
(build/report_data/conditions.parquet), and the checks behind the model report's supplied figures, printed.
Writes only conditions.parquet; bundle.json and the other tables are untouched."""

import json

import pandas as pd
import r2_common as C
import r19_common as R
import r19_eval_helpers as H
import r21_common as Q
from cb_common import METH, REPO, cb, conds51, core

from pdm import splits
from pdm.config import load_config
from pdm.evaluation import rev2 as r2

OUT = REPO / "build" / "report_data"
K = core()
cfg = load_config()
SPAN, P2A, cp = K["SPAN"], K["P2A"], K["cp"]
LEV = {"ML Model": K["ml"], "Vendor Rules": K["vd"]}
res = K["res"]
DAY = pd.Timedelta(days=1)
ctm = {m: res[("E7", m)]["ct"].set_index("wo_id") for m in METH}
_, deg_a = Q.degraded_hours(tuple(sorted(P2A.items())))

c = cp.reset_index(drop=True).copy()
own = [H.own_start(x.condition_type, x.asset_key, x.credit_ts, P2A) for x in c.itertuples()]
ons = C.truth_onsets(c).set_index("wo_id")["truth_onset"]
c["onset"] = [o if pd.notna(o) else ons.get(w, pd.NaT) for o, w in zip(own, c["wo_id"], strict=True)]
conds = pd.DataFrame(
    {
        "wo_id": c["wo_id"],
        "asset": c["asset_key"],
        "condition_type": c["condition_type"],
        "onset": c["onset"],
        "confirmed": c["credit_ts"],
        "unplanned": c["unplanned"].astype(bool),
    }
)
for m, k in (("ML Model", "ml"), ("Vendor Rules", "vr")):
    conds[f"{k}_strict"] = conds["wo_id"].map(ctm[m]["strict"]).astype(bool)
    conds[f"{k}_warn"] = conds["wo_id"].map(ctm[m]["attributable_warning"]).astype(float)
    # the episode credited with each strict catch: opening, last alert, close and ALARM start
    ep = conds["wo_id"].map(ctm[m]["credited_episode"]).where(conds[f"{k}_strict"])
    lv = LEV[m]
    for col in ("opened", "last", "closed", "alarm_ts"):
        conds[f"{k}_{col}"] = [lv.at[int(i), col] if pd.notna(i) else pd.NaT for i in ep]
# how each condition was found: the PM or inspection its work order follows from (source_wo_id), and when
wo_all = cb("cmms_work_orders").set_index("wo_id")
src = conds["wo_id"].map(wo_all["source_wo_id"])
conds["found_by"] = [
    str(wo_all.at[s, "wo_type"]) if isinstance(s, str) and s in wo_all.index else "" for s in src
]
conds["found_ts"] = [wo_all.at[s, "opened"] if isinstance(s, str) and s in wo_all.index else pd.NaT for s in src]
conds.to_parquet(OUT / "conditions.parquet", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(conds.sort_values(["asset", "confirmed"]).to_string())

# false ALERT episodes (as in the bundle): opened in the year, not credited, no degradation on the asset while open
out = {}
for m in METH:
    av = r2.alert_view(LEV[m])
    ct = res[("E7", m)]["ct"]
    credited = {i for i, s in zip(ct["credited_episode"], ct["strict"], strict=True) if s and pd.notna(i)}
    fl = [
        r
        for i, r in zip(av.index, av.itertuples(), strict=True)
        if SPAN[0] <= r.opened < SPAN[1]
        and i not in credited
        and not Q.present_between(deg_a.get(r.asset_id), r.opened, r.closed)
    ]
    d = pd.DataFrame(
        [{"asset": r.asset_id, "opened": r.opened, "closed": min(r.closed, SPAN[1])} for r in fl]
    )
    d["days"] = (d["closed"] - d["opened"]) / DAY
    out[m] = d.sort_values("days", ascending=False)
    print(m, "false episodes", len(d))
    print(d.sort_values("days", ascending=False).head(5).to_string())
ml2 = LEV["ML Model"][LEV["ML Model"]["asset_id"] == "HMC-02-SP"]
print("HMC-02-SP ML episodes\n", ml2[["opened", "closed", "alarm_ts"]].to_string())

# Vendor Rules inspections in the training years (alert-driven)
wo = cb("cmms_work_orders")
t0, t1 = pd.Timestamp("2023-07-01 05:00", tz="UTC"), splits.training_cutoff(cfg)
ins = wo[(wo["wo_type"] == "inspection") & (wo["opened"] >= t0) & (wo["opened"] < t1)]
ia = ins[ins["platform_alert_id"].notna()]
print(
    "training inspections all",
    len(ins),
    "nff",
    ins["no_fault_found"].fillna(False).astype(bool).mean(),
    "| alert-driven",
    len(ia),
    "nff",
    ia["no_fault_found"].fillna(False).astype(bool).mean(),
)
ie = ia[ia["no_fault_found"].notna()]
print("alert-driven with a finding recorded", len(ie), "nff", ie["no_fault_found"].astype(bool).mean())
bp = OUT / "bundle.json"
B = json.loads(bp.read_text())
B["training_inspections"] = {
    "all": len(ins),
    "all_nff": float(ins["no_fault_found"].fillna(False).astype(bool).mean()),
    "alert_driven": len(ia),
    "alert_driven_nff": float(ia["no_fault_found"].fillna(False).astype(bool).mean()),
}
c51 = conds51()
on_hc = c51["asset_key"].str.startswith(("HMC", "CMP"))
B["conditions_hmc_cmp"] = {"total": int(on_hc.sum()), "no_sensor": int((on_hc & ~c51["instrumented"]).sum())}
B["false_longest"] = {m: {k: str(v) for k, v in out[m].iloc[0].items()} for m in METH}
bp.write_text(json.dumps(B, indent=1, default=str))

# repairs and remounts per point in the training years
segs = cb("ml_segments")
segs["segment_start"] = pd.to_datetime(segs["segment_start"], utc=True)
print(segs["reason"].value_counts())
s = segs[(segs["segment_start"] >= t0) & (segs["segment_start"] < t1) & (segs["reason"] != "install")]
rr = s.groupby("point_id").size().rename("repairs_remounts")
th = (
    pd.read_parquet(OUT / "train_hours.parquet")
    .set_index("point_id")
    .join(rr)
    .fillna({"repairs_remounts": 0})
)
th["repairs_remounts"] = th["repairs_remounts"].astype(int)
print(th.sort_values("n_train").to_string())
print("spearman n_train vs repairs/remounts", th["n_train"].corr(th["repairs_remounts"], method="spearman"))
th["repairs_remounts"].rename_axis("point_id").reset_index().to_parquet(
    OUT / "train_repairs.parquet", index=False
)
print(s.groupby("reason").size())

# HMC-02-SP score around its December and May conditions
sc = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
sc = sc[(sc["asset_id"] == "HMC-02-SP") & (sc["state"] == "scored")].copy()
sc["ratio"] = sc["score"] / sc["threshold"]
for w in ("PM-000625", "HMC-02-WO00041"):
    r = conds.set_index("wo_id").loc[w]
    before = sc[(sc["hour_utc"] >= r["onset"] - 30 * DAY) & (sc["hour_utc"] < r["onset"])]["ratio"]
    during = sc[(sc["hour_utc"] >= r["onset"]) & (sc["hour_utc"] <= r["confirmed"])]["ratio"]
    print(
        w,
        "onset",
        r["onset"],
        "confirmed",
        r["confirmed"],
        "median ratio before",
        round(before.median(), 2),
        "during",
        round(during.median(), 2),
        "max during",
        round(during.max(), 2),
        "x",
        round(during.median() / before.median(), 1),
    )
print(
    json.dumps(
        {
            "HMC-01-TSC": conds[conds["asset"] == "HMC-01-TSC"][
                ["wo_id", "condition_type", "onset", "confirmed"]
            ]
            .astype(str)
            .to_dict("records")
        },
        indent=1,
    )
)
