"""Checkpoint (b) rev2 (diag25), rotary-union rule (items 6 to 8): the drop-share threshold from healthy training
days on the spindle machines (no rotary-union case used), and D2 names before and after on the training folds and
the evaluation year. Writes diag25/ru.json."""

import copy
import json

import numpy as np
import pandas as pd

import diag_folds as F
import diag_lib as L
import r19_common as R
from cb_common import D, core
from pdm import splits
from pdm.advisory import requests as adv
from pdm.config import load_config

K = core()
cfg = load_config()
store = F._Store()
cut = splits.training_cutoff(cfg)
hmc = set(store.read_pd("bronze", "reg_machines").query("`class` == 'hmc'")["machine_id"])
chh = store.read_pd("silver", "controller_hourly", ["machine_id", "hour_utc", "coolant_pressure_pump_on_bar"])
chh["hour_utc"] = pd.to_datetime(chh["hour_utc"], utc=True)
chh = chh[chh["machine_id"].isin(hmc)]
a = cfg["advisory"]["tsc_pressure_reference_days"]
daily = adv.tsc_pressure_daily(chh, (a[0], a[1]))
daily["local_date"] = pd.to_datetime(daily["local_date"])
conds = K["conds_all"]
c = conds[conds["asset_key"].str.endswith(("-SP", "-TSC"))].copy()
c["machine"] = c["asset_key"].str[:6]
cut_local = cut.tz_convert("America/Chicago").tz_localize(None).normalize()
h = daily[(daily["local_date"] < cut_local) & daily["drop_share"].notna()].copy()
ok = np.ones(len(h), dtype=bool)
for r in c.itertuples():
    t = r.credit_ts.tz_convert("America/Chicago").tz_localize(None).normalize()
    ok &= ~((h["machine_id"] == r.machine) & (h["local_date"] >= t - pd.Timedelta(days=120)) & (h["local_date"] <= t + pd.Timedelta(days=1))).to_numpy()
h = h[ok]
thr = round(float(np.nanpercentile(h["drop_share"], 99)), 4)
basis = (f"99th percentile of the daily pump-on pressure drop share against its reference ({a[0]} to {a[1]} days earlier) on healthy training days of the 8 HMC machines "
         f"({len(h):,} machine-days before the training cutoff, outside 120 days before to 1 day after every confirmed condition on the machine's spindle or TSC pump, rotary-union conditions included, so no rotary-union case sets it)")
print(thr, len(h), flush=True)

# D2 names before and after, per credited episode, folds and the evaluation year
TH0 = {**adv.D2_DEFAULTS, **cfg["advisory"]["d2"]["thresholds"]}
TH1 = {**TH0, "tsc_drop_share": thr}


def share_max(asset, opened, closed, end):
    if not asset.endswith("-SP"):
        return np.nan
    o = pd.Timestamp(opened).tz_convert("America/Chicago").tz_localize(None).normalize()
    e = min(pd.Timestamp(closed) if pd.notna(pd.Timestamp(closed)) else end, end).tz_convert("America/Chicago").tz_localize(None).normalize()
    g = daily[(daily["machine_id"] == asset[:6]) & (daily["local_date"] >= o) & (daily["local_date"] <= e)]["drop_share"]
    return float(g.max()) if g.notna().any() else np.nan


def new_rule(m, kind, th):
    """D2 with the rotary-union step on the drop share (the proposed suspected_d2)."""
    name, det = adv.suspected_d2({k: v for k, v in m.items() if k != "tsc_drop"}, kind, {**th, "tsc_drop_z": np.inf})
    if name == "unclassified" and kind == "spindle" and pd.notna(m.get("tsc_drop_share", np.nan)) and m["tsc_drop_share"] >= th["tsc_drop_share"]:
        return "rotary union", f"TSC pump-on pressure {100 * m['tsc_drop_share']:.1f}% below reference"
    return name, det


rows = []
END = {"fold1": pd.Timestamp("2025-06-17", tz="UTC"), "fold2": pd.Timestamp("2025-06-17", tz="UTC"), "eval": K["SPAN"][1]}
for p in ("fold1", "fold2", "eval"):
    for e in json.loads((R.S / "diag24" / f"measures_{p}.json").read_text()):
        m = dict(e["d2 measures"])
        m["tsc_drop_share"] = share_max(e["asset"], e["opened"], e["closed"], END[p])
        before = adv.suspected_d2(m, e["kind"], TH0)
        after = new_rule(m, e["kind"], TH1)
        for cnd in e["conditions"]:
            rows.append({"period": p, "episode": e["episode"], "asset": e["asset"], "wo_id": cnd["wo_id"], "family": cnd["family"], "strict": cnd["strict"],
                         "before": before[0], "after": after[0], "after detail": after[1], "drop share max": m["tsc_drop_share"]})
d = pd.DataFrame(rows)
d["right before"] = d["before"] == d["family"]
d["right after"] = d["after"] == d["family"]
turned = d[d["right before"] & ~d["right after"]]
adopt = turned.empty
summ = []
for p, g in d.groupby("period", sort=False):
    ep = g.groupby("episode").agg(fams=("family", lambda x: set(x)), b=("before", "first"), a=("after", "first"))
    s = g[g["strict"]]
    summ.append({"period": p, "catches": len(g), "strict": len(s), "episodes": len(ep), "all, before": int(g["right before"].sum()), "all, after": int(g["right after"].sum()),
                 "strict, before": int(s["right before"].sum()), "strict, after": int(s["right after"].sum()),
                 "episodes, before": int(sum(r.b in r.fams for r in ep.itertuples())), "episodes, after": int(sum(r.a in r.fams for r in ep.itertuples()))})
ru = d[d["family"] == "rotary union"]
out = {"threshold": thr, "basis": basis, "healthy machine-days": len(h), "adopted": adopt, "turned wrong": turned.to_dict("records"),
       "summary": summ, "rotary union catches": ru.to_dict("records"), "changed": d[d["before"] != d["after"]].to_dict("records")}
(D / "ru.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ("threshold", "adopted", "summary")}, indent=1, default=str))
print(ru[["period", "wo_id", "asset", "strict", "before", "after", "after detail", "drop share max"]].to_string())
print(pd.DataFrame(out["changed"])[["period", "wo_id", "family", "before", "after"]].to_string() if out["changed"] else "no change")
