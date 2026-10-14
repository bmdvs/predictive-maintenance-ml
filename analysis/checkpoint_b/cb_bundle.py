"""Checkpoint (b) rev2 (diag25): the model report's data bundle (every number and exhibit dataset the page reads),
written to build/report_data (bundle.json and parquet files). Reads the diag25 results, the prod caches and the
frozen model; truth only through the evaluation helpers (E5 to E7)."""

import copy
import json
import pickle

import numpy as np
import pandas as pd

import diag_folds as F
import diag_lib as L
import r19_common as R
import r19_eval_helpers as H
import r21_common as Q
import r2_common as C
from cb_common import METH, REPO, D, cb, conds51, core, inputs, miss_reason
from pdm import splits
from pdm.advisory import requests as adv
from pdm.config import load_config
from pdm.evaluation import rev2 as r2
from pdm.features import burst as fb
from pdm.features import filters, kinematics
from pdm.models import nbm, severity

OUT = REPO / "build" / "report_data"
OUT.mkdir(parents=True, exist_ok=True)
K = core()
cfg = load_config()
SPAN, AM, P2A, cp, conf = K["SPAN"], K["AM"], K["P2A"], K["cp"], K["conf"]
LEV = {"ML Model": K["ml"], "Vendor Rules": K["vd"]}
res = K["res"]
inp = inputs()
ind = inp["ind"]
DAY = pd.Timedelta(days=1)
B = {}
tables = {}
ctm = {m: res[("E7", m)]["ct"].set_index("wo_id") for m in METH}
cpi = cp.set_index("wo_id")
_, deg_a = Q.degraded_hours(tuple(sorted(P2A.items())))

# ------------------------------------------------------------------ section 1
c51 = conds51()
inst = c51[c51["instrumented"]]
B["conditions"] = {"total": len(c51), "in_scope": len(cp), "other": len(c51) - len(cp), "no_sensors": int((~c51["instrumented"]).sum()),
                   "cooler": int((inst["condition_type"] == "cooler (compressor)").sum()),
                   "ru_leak": int(((inst["condition_type"] == "rotary union") & (inst["failure_code"] == "RU-LEAK")).sum()),
                   "no_precursor": int(inst["condition_type"].isin(cfg["evaluation"]["comparison_scope"]["exclude_condition_types"]).sum())}
cls_name = {"spindle": "HMC spindle", "tsc_pump": "TSC pump", "flood_pump": "Flood coolant pump", "compressor": "Compressor"}
tables["conds_by_type"] = cp.assign(asset_class=cp["asset_key"].map(lambda a: cls_name[severity.asset_class(a)]))[["wo_id", "condition_type", "asset_class"]]
biz = json.loads((R.S / "diag22" / "business.json").read_text())
B["business"] = {"hmc_share": biz["by class"]["HMC"]["share of shop cutting hours"], "hmc_util": biz["by class"]["HMC"]["utilization"],
                 "other_util": [v["utilization"] for k, v in biz["by class"].items() if k != "HMC"], "margin": biz["by class"]["HMC"]["margin per hour (USD)"],
                 "ratio": biz["HMC margin per hour / that average"]}

# ------------------------------------------------------------------ section 2: ALERT
sm = {m: ctm[m]["strict"].astype(bool) for m in METH}
both = sm["ML Model"] & sm["Vendor Rules"]
aw = {m: ctm[m].loc[both[both].index, "attributable_warning"].astype(float) for m in METH}
B["alert"] = {"strict": {m: int(sm[m].sum()) for m in METH}, "both": int(both.sum()), "only_ml": int((sm["ML Model"] & ~sm["Vendor Rules"]).sum()),
              "only_vr": int((~sm["ML Model"] & sm["Vendor Rules"]).sum()), "neither": int((~sm["ML Model"] & ~sm["Vendor Rules"]).sum()),
              "warn_both": {m: float(aw[m].median()) for m in METH}, "caught_e4": {m: res[("E4", m)]["caught"] for m in METH},
              "false": {m: res[("E7", m)]["false_truth"] / AM for m in METH}, "false_n": {m: res[("E7", m)]["false_truth"] for m in METH},
              "alert_hours": {m: res[("E7", m)]["alert_hours"] / AM for m in METH}}
tables["warn_both"] = pd.DataFrame({"wo_id": aw["ML Model"].index, "ml": aw["ML Model"].to_numpy(), "vr": aw["Vendor Rules"].to_numpy()})
# assets with no confirmed condition
quiet = sorted(set(P2A.values()) - set(inst["asset_key"]))
wo = cb("cmms_work_orders")
woy = wo[(wo["opened"] >= SPAN[0]) & (wo["opened"] < SPAN[1])]
qq = {}
for m in METH:
    e = LEV[m][LEV[m]["asset_id"].isin(quiet) & (LEV[m]["opened"] >= SPAN[0]) & (LEV[m]["opened"] < SPAN[1])]
    qq[m] = {"episodes": len(e), "assets": int(e["asset_id"].nunique())}
iq = woy[woy["asset_id"].isin(quiet) & (woy["wo_type"] == "inspection")]
qq["inspections"] = len(iq)
qq["inspections_found"] = int((~iq["no_fault_found"].fillna(True).astype(bool)).sum())
B["quiet"] = qq
# inspection hours
s2 = json.loads((D / "cooling.json").read_text())
fl = {}
for m in METH:
    lev = LEV[m]
    av = r2.alert_view(lev)
    ct = res[("E7", m)]["ct"]
    credited = {i for i, s in zip(ct["credited_episode"], ct["strict"], strict=True) if s and pd.notna(i)}
    fl[m] = [r for i, r in zip(av.index, av.itertuples(), strict=True) if SPAN[0] <= r.opened < SPAN[1] and i not in credited and not Q.present_between(deg_a.get(r.asset_id), r.opened, r.closed)]
insp = wo[wo["wo_type"] == "inspection"].copy()
insp["alert_ts"] = pd.to_datetime(insp["platform_alert_id"].str[-10:], format="%Y%m%d%H", utc=True, errors="coerce")
v_act = sum(float(insp[(insp["asset_id"].str.startswith(r.asset_id)) & (insp["alert_ts"] >= r.opened) & (insp["alert_ts"] < r.closed)]["labor_hours"].sum()) for r in fl["Vendor Rules"])
per = v_act / len(fl["Vendor Rules"])
mon = {}
for m in METH:
    s = pd.Series([r.opened.tz_convert("America/Chicago").strftime("%Y-%m") for r in fl[m]]).value_counts()
    mon[m] = s
vr_month = {}
for r in fl["Vendor Rules"]:
    k = r.opened.tz_convert("America/Chicago").strftime("%Y-%m")
    vr_month[k] = vr_month.get(k, 0) + float(insp[(insp["asset_id"].str.startswith(r.asset_id)) & (insp["alert_ts"] >= r.opened) & (insp["alert_ts"] < r.closed)]["labor_hours"].sum())
B["inspection"] = {"vr_hours": v_act, "vr_n": len(fl["Vendor Rules"]), "ml_n": len(fl["ML Model"]), "per": per, "ml_hours": len(fl["ML Model"]) * per,
                   "saved": v_act - len(fl["ML Model"]) * per, "vr_peak_share": max(vr_month.values()) / 520, "vr_peak_month": max(vr_month, key=vr_month.get),
                   "ml_peak_share": mon["ML Model"].max() * per / 520, "ml_peak_month": mon["ML Model"].idxmax()}
tables["curves"] = pd.read_parquet(D / "curves.parquet")

# ------------------------------------------------------------------ ALARM
c43 = inst.reset_index(drop=True).copy()
own = [H.own_start(c.condition_type, c.asset_key, c.credit_ts, P2A) for c in c43.itertuples()]
ons = C.truth_onsets(c43).set_index("wo_id")["truth_onset"]
c43["onset"] = [o if pd.notna(o) else ons.get(w, pd.NaT) for o, w in zip(own, c43["wo_id"], strict=True)]


def alarms(m, sensor=False):
    e = LEV[m]
    col = "alarm_sensor_ts" if sensor else "alarm_ts"
    a = e[e[col].notna() & (e[col] >= SPAN[0]) & (e[col] < SPAN[1])].copy()
    a["start"] = a[col]
    a["conds"] = [",".join(c43[(c43["asset_key"] == r.asset_id) & (c43["credit_ts"] >= r.start) & c43["onset"].notna() & (c43["onset"] < r.closed)]["wo_id"]) for r in a.itertuples()]
    return a


al = {m: alarms(m) for m in METH}
fa = {m: r2.alarm_outcomes(LEV[m], conf, inp["wo"], SPAN, SPAN[1], False, False)["false_alarm"].sum() for m in METH}
fas = r2.alarm_outcomes(LEV["ML Model"], conf, inp["wo"], SPAN, SPAN[1], False, True)["false_alarm"].sum()
wrows = []
for c in c43[c43["wo_id"].isin(cp["wo_id"])].itertuples():
    w = {}
    for m in METH:
        a = al[m][(al[m]["asset_id"] == c.asset_key) & al[m]["conds"].str.contains(c.wo_id)]
        w[m] = max(((c.credit_ts - max(x.start, c.onset)) / DAY for x in a.itertuples()), default=np.nan)
    wrows.append({"wo_id": c.wo_id, "asset": c.asset_key, "type": c.condition_type, "ml": w["ML Model"], "vr": w["Vendor Rules"]})
wd = pd.DataFrame(wrows).dropna(subset=["ml", "vr"])
B["alarm"] = {"n": {m: len(al[m]) for m in METH}, "with": {m: int((al[m]["conds"] != "").sum()) for m in METH},
              "share": {m: float((al[m]["conds"] != "").mean()) for m in METH}, "false": {m: float(fa[m] / AM) for m in METH}, "false_sensor_ml": float(fas / AM),
              "warn": {"ML Model": float(wd["ml"].median()), "Vendor Rules": float(wd["vr"].median()), "n": len(wd)}}
tables["alarm_warn"] = wd
# Vendor Rules ALARMs before unplanned failures
fails = cp[cp["unplanned"].astype(bool)]
fr = []
for c in fails.itertuples():
    v = LEV["Vendor Rules"]
    a = v[(v["asset_id"] == c.asset_key) & v["alarm_ts"].notna() & (v["alarm_ts"] <= c.credit_ts) & (v["closed"] > c.credit_ts - 120 * DAY) & (v["alarm_ts"] >= c.credit_ts - 120 * DAY)]
    ins = insp[insp["asset_id"].str.startswith(c.asset_key) & (insp["opened"] >= c.credit_ts - 120 * DAY) & (insp["opened"] < c.credit_ts)]
    fr.append({"wo_id": c.wo_id, "asset": c.asset_key, "type": c.condition_type, "vr_alarm_before": len(a) > 0, "inspections": len(ins),
               "found": int((~ins["no_fault_found"].fillna(True).astype(bool)).sum()), "nff": int(ins["no_fault_found"].fillna(False).astype(bool).sum())})
tables["failures"] = pd.DataFrame(fr)
B["failures"] = {"n": len(fails), "vr_alarm_before": int(tables["failures"]["vr_alarm_before"].sum())}

# ------------------------------------------------------------------ diagnosis after the rotary-union change
ru = json.loads((D / "ru.json").read_text())
B["ru"] = {"threshold": ru["threshold"], "summary": ru["summary"], "catches": ru["rotary union catches"]}
sc24 = pd.DataFrame(json.loads((R.S / "diag24" / "score_conditions_fold1_fold2_eval.json").read_text()))
ev = sc24[sc24["period"] == "eval"].copy()
chg = {(r["wo_id"]): r["after"] for r in ru["changed"] if r["period"] == "eval"}
ev["d2"] = [chg.get(w, x) for w, x in zip(ev["wo_id"], ev["D2"], strict=True)]
evs = ev[ev["strict"]]
tables["diag_grid"] = evs[["wo_id", "family", "d2"]]
bear = evs[evs["family"] == "bearing"]
loc_ok = sum(1 for w in bear["wo_id"] if str(ev.set_index("wo_id").at[w, "D2 rule"]).startswith(("outer", "inner", "ball", "cage")))
B["diag"] = {"strict_named": int((evs["d2"] == evs["family"]).sum()), "strict": len(evs), "bearing": len(bear), "bearing_named": int((bear["d2"] == "bearing").sum()), "bearing_located": loc_ok,
             "lub": int((evs["family"] == "lubrication").sum()), "lub_named": int(((evs["family"] == "lubrication") & (evs["d2"] == "lubrication")).sum()),
             "pump_imbalance": int(((evs["family"] == "cavitation or wear") & (evs["d2"] == "imbalance")).sum()), "episodes": ru["summary"][2]["episodes"], "episodes_named": ru["summary"][2]["episodes, after"]}

# ------------------------------------------------------------------ section 3
ti = cb("ml_training_info")
ti["n_train"] = pd.to_numeric(ti["n_train"])
ti["segments"] = pd.to_numeric(ti["segments_before_cutoff"])
tables["train_hours"] = ti[["point_id", "point_class", "n_train", "segments"]]
B["train"] = {"total": int(ti["n_train"].sum()), "min": int(ti["n_train"].min()), "max": int(ti["n_train"].max()), "min_point": ti.loc[ti["n_train"].idxmin(), "point_id"]}
# healthy measurements by speed band, spindle front points
ts = pd.read_parquet(R.S / "diag24" / "train_scores.parquet")
ts["hour_utc"] = pd.to_datetime(ts["hour_utc"], utc=True)
cut = splits.training_cutoff(cfg)
h = ts[(ts["state"] == "scored") & (ts["hour_utc"] < cut) & ts["point_id"].str.endswith("-SP-F")][["point_id", "hour_utc", "speed_band"]] if "speed_band" in ts else ts[(ts["state"] == "scored") & (ts["hour_utc"] < cut) & ts["point_id"].str.endswith("-SP-F")][["point_id", "hour_utc"]]
h["asset"] = h["point_id"].map(P2A)
ok = np.ones(len(h), dtype=bool)
for a, g in K["conds_all"].groupby("asset_key"):
    m_ = (h["asset"] == a).to_numpy()
    for t in g["credit_ts"]:
        ok[m_] &= ~((h.loc[m_, "hour_utc"] >= t - 120 * DAY) & (h.loc[m_, "hour_utc"] < t + DAY)).to_numpy()
h = h[ok].merge(ind[["point_id", "hour_utc", "speed_rpm_at_measurement", "vel_rms_mm_s", "hf_rms_g", "env_bpfo_1_g2", "env_bpfi_1_g2", "env_bsf_1_g2", "regime"]], on=["point_id", "hour_utc"])
rated = 12000.0
edges = cfg["regime"]["speed_bands_frac_rated"]
fr_ = h["speed_rpm_at_measurement"] / rated
h["band"] = np.where(fr_ < edges[1] if isinstance(edges[0], (int, float)) else fr_ < 0.5, "low", np.where(fr_ < 0.75, "mid", "high"))
rows = []
for b, g in h.groupby("band"):
    for col in ("vel_rms_mm_s", "hf_rms_g", "env_bpfo_1_g2", "env_bpfi_1_g2", "env_bsf_1_g2"):
        q = g[col].quantile([0.05, 0.5, 0.95])
        rows.append({"band": b, "measure": col, "p05": q.iloc[0], "p50": q.iloc[1], "p95": q.iloc[2], "hours": len(g)})
tables["healthy_band"] = pd.DataFrame(rows)
# scoring summary
sc = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
sc = sc[(sc["hour_utc"] >= SPAN[0]) & (sc["hour_utc"] < SPAN[1])].reset_index(drop=True)
ml = K["ml"]
lvl = np.zeros(len(sc), dtype=int)
for r in ml.itertuples():
    m_ = ((sc["asset_id"] == r.asset_id) & (sc["hour_utc"] >= r.opened) & (sc["hour_utc"] < r.closed)).to_numpy()
    lvl[m_] = np.maximum(lvl[m_], 1)
    if pd.notna(r.alarm_ts):
        lvl[m_ & (sc["hour_utc"] >= r.alarm_ts).to_numpy()] = 2
sc["level"] = lvl
meas = sc[sc["state"] == "scored"].copy()


def grp(c):
    c = str(c).split(",")[0]
    if c == "hf_rms_g" or c.startswith("env_"):
        return "high-frequency or fault-band"
    if c == "temp_resid_c":
        return "temperature residual"
    if c.startswith("ord_"):
        return "shaft-order energies"
    if c == "vel_rms_mm_s":
        return "velocity"
    if c in ("crest_factor", "kurtosis"):
        return "crest factor and kurtosis"
    return "operating indicators"


meas["group"] = meas["top_contributors"].map(grp)
alh = meas[meas["level"] > 0]
eps_y = ml[(ml["opened"] >= SPAN[0]) & (ml["opened"] < SPAN[1])]
B["scoring"] = {"hours": len(meas), "outside": float((meas["level"] == 0).mean()), "alert": float((meas["level"] == 1).mean()), "alarm": float((meas["level"] == 2).mean()),
                "episodes": len(eps_y), "assets": int(eps_y["asset_id"].nunique()), "alarms": int(eps_y["alarm_ts"].notna().sum()),
                "groups": alh["group"].value_counts(normalize=True).to_dict(), "rearms": 5, "rearms_degrading": 3}
meas["month"] = meas["hour_utc"].dt.tz_convert("America/Chicago").dt.strftime("%Y-%m")
tables["level_month"] = meas.groupby("month")["level"].agg(alert=lambda x: (x == 1).mean(), alarm=lambda x: (x == 2).mean()).reset_index()
near = np.zeros(len(meas), dtype=bool)
pre = np.zeros(len(meas), dtype=bool)
for c in c43.itertuples():
    m_ = ((meas["asset_id"] == c.asset_key) & (meas["hour_utc"] >= c.credit_ts - 120 * DAY) & (meas["hour_utc"] < c.credit_ts + DAY)).to_numpy()
    near |= m_
    if c.wo_id in set(cp["wo_id"]):
        pre |= m_ & (meas["hour_utc"] < c.credit_ts).to_numpy()
ratio = (meas["score"] / meas["threshold"]).to_numpy()
tables["score_dist"] = pd.DataFrame({"ratio": np.concatenate([ratio[~near], ratio[pre]]), "group": ["no condition"] * int((~near).sum()) + ["120 days before a confirmed condition"] * int(pre.sum())})
meas["cls"] = meas["asset_id"].map(severity.asset_class)
eps_y = eps_y.assign(cls=eps_y["asset_id"].map(severity.asset_class), days=(eps_y["closed"].clip(upper=SPAN[1]) - eps_y["opened"]) / DAY)
ct_rows = []
for c_, g in meas.groupby("cls"):
    e = eps_y[eps_y["cls"] == c_]
    gg = g[g["level"] > 0]["group"].value_counts()
    ct_rows.append({"Asset class": cls_name[c_], "Measured hours": len(g), "ALERT episodes": len(e), "ALARMs": int(e["alarm_ts"].notna().sum()),
                    "Median episode length, days": float(e["days"].median()) if len(e) else np.nan, "Leading input group": gg.index[0] if len(gg) else ""})
tables["class_summary"] = pd.DataFrame(ct_rows)

# rebuild exhibits: healthy and faulty hour on HMC-01-TSC
A = pickle.loads((R.S / "diag21" / "models_A_final.pkl").read_bytes())
pid = "HMC-01-TSC"
pm, fac, mcfg = A["models"][pid], A["factor"], copy.deepcopy(A["cfg"])
mcfg["models"]["nbm"]["fixes"] = {**nbm.FIX_DEFAULTS, **mcfg["models"]["nbm"]["fixes"], "f1_hold_level": True, "f4_high_ratio": True, "f4_ratio": 5.0, "f4_exclude_pump_low_pressure": True, "s1_on_component_window": True}
win = nbm.component_repairs(inp["win"][inp["win"]["point_id"] == pid], inp["wo"])
rows_p = ind[ind["point_id"] == pid]
s1, _ = nbm.score_point(pm, rows_p, win, mcfg, pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-11-22", tz="UTC"), fac, 3, None, keep_inputs=True)
s1["hour_utc"] = pd.to_datetime(s1["hour_utc"], utc=True)
s1 = s1[s1["state"] == "scored"]


def rebuild(r):
    z13 = np.array([r[f"z:{c}"] for c in pm.columns], dtype=float)
    oh = nbm._one_hot(pd.DataFrame({"regime": [r["regime"]]}), np.array([r["speed_band"]]))[0]
    zi = (oh - pm.scaler.center[len(pm.columns):]) / pm.scaler.scale[len(pm.columns):]
    z = np.concatenate([z13, zi])
    rec = pm.model.predict(z[None, :])[0]
    return z, rec


fh = s1[s1["hour_utc"] == pd.Timestamp("2025-10-16 01:00", tz="UTC")]
fh = fh.iloc[0] if len(fh) else s1.loc[(s1["score"] / s1["threshold"]).idxmax()]
hh = s1[(s1["hour_utc"] < pd.Timestamp("2025-09-01", tz="UTC")) & (s1["score"] < s1["threshold"])]
hh = hh.iloc[(hh["score"] - hh["score"].median()).abs().argsort().iloc[0]]
reb = []
for lab, r in (("healthy", hh), ("faulty", fh)):
    z, rec = rebuild(r)
    for j, c in enumerate([*pm.columns, *[f"regime_{x}" for x in nbm.REGIMES], *[f"band_{x}" for x in nbm.BANDS]]):
        reb.append({"hour": lab, "ts": r["hour_utc"], "input": c, "z": z[j], "rebuild": rec[j], "score": r["score"], "threshold": r["threshold"]})
tables["rebuild"] = pd.DataFrame(reb)
# score to level: HMC-08-SP-F around HMC-08-WO00046
t_c = cpi.at["HMC-08-WO00046", "credit_ts"]
tr_ = sc[(sc["point_id"] == "HMC-08-SP-F") & (sc["hour_utc"] >= t_c - 75 * DAY) & (sc["hour_utc"] <= t_c + 5 * DAY)][["hour_utc", "state", "score", "threshold"]]
tables["score_trace"] = tr_
e8 = ml[(ml["asset_id"] == "HMC-08-SP") & (ml["closed"] > t_c - 75 * DAY) & (ml["opened"] < t_c)]
B["trace"] = {"point": "HMC-08-SP-F", "wo": "HMC-08-WO00046", "confirmed": str(t_c), "alerts": [str(x) for x in e8["opened"]], "alarms": [str(x) for x in e8["alarm_ts"].dropna()]}

# signatures
m24 = json.loads((R.S / "diag24" / "measures_eval.json").read_text())
lub = [e for e in m24 if [c["family"] for c in e["conditions"]] == ["lubrication"] and all(ctm["ML Model"].at[c["wo_id"], "strict"] for c in e["conditions"])]
hfp = [100 * (10 ** (e["d2 measures"]["db_hf"] / 20) - 1) for e in lub if "db_hf" in e["d2 measures"]]
tdc = [e["d2 measures"].get("temp_dc", np.nan) for e in lub]
B["signatures"] = {"lub_hf_pct": [float(min(hfp)), float(max(hfp))] if hfp else None, "lub_temp_c": [float(np.nanmin(tdc)), float(np.nanmax(tdc))] if tdc else None, "lub_n": len(lub),
                   "counts": cp["condition_type"].value_counts().to_dict(), "ru_drop": [9.0, 10.5]}
sig = []
for wo_, pt, cols in (("PM-002016", "HMC-07-SP-F", ["hf_rms_g"]), ("HMC-03-WO00042", "HMC-03-CP", ["ord_01_g2", "env_bpfo_1_g2", "env_bpfi_1_g2", "env_bsf_1_g2"])):
    t = cpi.at[wo_, "credit_ts"]
    g = ind[(ind["point_id"] == pt) & (ind["hour_utc"] >= t - 120 * DAY) & (ind["hour_utc"] <= t)].copy()
    g["day"] = g["hour_utc"].dt.floor("D")
    d = g.groupby("day")[cols].median().reset_index()
    for c in cols:
        sig.append(pd.DataFrame({"wo_id": wo_, "point": pt, "measure": c, "day": d["day"], "value": d[c]}))
cool = sc[(sc["asset_id"] == "HMC-01-SP") & (sc["state"] == "scored") & (sc["hour_utc"] >= cpi.at["HMC-01-WO00036", "credit_ts"] - 60 * DAY) & (sc["hour_utc"] <= cpi.at["HMC-01-WO00036", "credit_ts"])].copy()
cool["day"] = cool["hour_utc"].dt.floor("D")
cd_ = cool.groupby("day")["temp_resid_c"].median().reset_index()
sig.append(pd.DataFrame({"wo_id": "HMC-01-WO00036", "point": "HMC-01-SP", "measure": "temp_resid_c", "day": cd_["day"], "value": cd_["temp_resid_c"]}))
tsc = pd.read_parquet(REPO / "build" / "queue_cache" / "tsc_pressure.parquet")
dd = pd.to_datetime(tsc["local_date"])
tsc["day"] = (dd.dt.tz_convert("UTC").dt.tz_localize(None) if dd.dt.tz is not None else dd).dt.normalize()
t4 = cpi.at["HMC-04-WO00024", "credit_ts"].tz_convert("America/Chicago").tz_localize(None).normalize()
g4 = tsc[(tsc["machine_id"] == "HMC-04") & (tsc["day"] >= t4 - 90 * DAY) & (tsc["day"] <= t4)]
sig.append(pd.DataFrame({"wo_id": "HMC-04-WO00024", "point": "HMC-04", "measure": "pressure_median_bar", "day": g4["day"].dt.tz_localize("UTC"), "value": g4["pressure_median_bar"]}))
sig.append(pd.DataFrame({"wo_id": "HMC-04-WO00024", "point": "HMC-04", "measure": "reference_median_bar", "day": g4["day"].dt.tz_localize("UTC"), "value": g4["reference_median_bar"]}))
tables["signature_series"] = pd.concat(sig, ignore_index=True)
for wo_ in ("PM-002016", "HMC-03-WO00042", "HMC-01-WO00036", "HMC-04-WO00024", "HMC-01-WO00024"):
    B.setdefault("signature_cases", {})[wo_] = {"asset": cpi.at[wo_, "asset_key"], "confirmed": str(cpi.at[wo_, "credit_ts"])}

# bursts: healthy spindle front (five-panel figure) and the bearing panel (HMC-01-TSC nearest confirmation)
bi = pd.read_parquet(REPO / "data" / "bronze" / "cm_platform" / "burst_index.parquet")
bi["hour_utc"] = pd.to_datetime(bi["hour_utc"], utc=True)
xs = fb.ExtractionSettings.from_config(cfg)
se = pd.read_parquet(REPO / "build" / "queue_cache" / "sensor_events.parquet")
se["ts"] = pd.to_datetime(se["ts"], utc=True)
from pdm.reports import queue_bursts as qb  # noqa: E402

bw = []


def burst_rows(pid_, hour, role):
    r = bi[(bi["point_id"] == pid_) & (bi["hour_utc"] == hour)].iloc[0]
    with np.load(REPO / "data" / "bronze" / "cm_platform" / r["path"]) as z:
        x = np.asarray(z["accel_g"], dtype=float)
    sp = float(r["speed_rpm"]) / 60
    band = qb.band_at(se, pid_, hour)
    f_env, a_env = qb.envelope_resultant(x, xs.fs, band, xs)
    xx = filters.remove_dc(x)
    spec, freqs = filters.rfft_with_freqs(xx, xs.fs)
    amp = np.sqrt((np.abs(spec) ** 2).sum(axis=0)) * 2 / x.shape[-1]
    return {"role": role, "point": pid_, "hour": str(hour), "shaft_hz": sp, "band": list(band), "wave": x[1 if x.shape[0] > 1 else 0][: int(0.05 * xs.fs)].tolist(),
            "freqs": freqs[1::4].tolist(), "amp": amp[1::4].tolist(), "env_f": f_env[f_env <= 3000].tolist(), "env_a": a_env[f_env <= 3000].tolist()}


hr_ = bi[(bi["point_id"] == "HMC-08-SP-F") & (bi["reason"] == "weekly_reference") & (bi["hour_utc"] < pd.Timestamp("2024-06-01", tz="UTC"))]["hour_utc"].iloc[-1]
bw.append(burst_rows("HMC-08-SP-F", hr_, "healthy"))
tb = cpi.at["HMC-01-WO00024", "credit_ts"]
nb_ = bi[(bi["point_id"] == "HMC-01-TSC") & (bi["hour_utc"] <= tb)]["hour_utc"].iloc[-1]
ref_ = bi[(bi["point_id"] == "HMC-01-TSC") & (bi["reason"] == "weekly_reference") & (bi["hour_utc"] < pd.Timestamp("2025-06-01", tz="UTC"))]["hour_utc"].iloc[-1]
bw.append(burst_rows("HMC-01-TSC", nb_, "bearing"))
bw.append(burst_rows("HMC-01-TSC", ref_, "bearing reference"))
for b in bw:
    o = kinematics.fault_frequencies_hz(K_ORD := (lambda p: kinematics.fault_orders(*{"HMC-08-SP-F": (19, 10.53, 90.0, 15.0), "HMC-01-TSC": (8, 17.46, 72.5, 0.0)}[p]))(b["point"]), b["shaft_hz"])
    b["fault_hz"] = {k: float(v) for k, v in o.items()}
(OUT / "bursts.json").write_text(json.dumps(bw))
# the 20 inputs at the healthy hour of the spindle front point
s8, _ = nbm.score_point(A["models"]["HMC-08-SP-F"], ind[ind["point_id"] == "HMC-08-SP-F"], nbm.component_repairs(inp["win"][inp["win"]["point_id"] == "HMC-08-SP-F"], inp["wo"]), mcfg, hr_ - DAY, hr_ + pd.Timedelta(hours=1), fac, 3, None, keep_inputs=True)
s8["hour_utc"] = pd.to_datetime(s8["hour_utc"], utc=True)
r8 = s8[s8["hour_utc"] == hr_]
if len(r8):
    r8 = r8.iloc[0]
    pm8 = A["models"]["HMC-08-SP-F"]
    oh = nbm._one_hot(pd.DataFrame({"regime": [r8["regime"]]}), np.array([r8["speed_band"]]))[0]
    tables["inputs20"] = pd.DataFrame({"input": [*pm8.columns, *[f"regime_{x}" for x in nbm.REGIMES], *[f"band_{x}" for x in nbm.BANDS]],
                                       "value": [*[r8[f"z:{c}"] for c in pm8.columns], *oh], "kind": ["sensor"] * len(pm8.columns) + ["controller"] * 7})

# ------------------------------------------------------------------ section 4
fp = pd.read_parquet(D / "folds_periods.parquet")
tables["periods"] = fp
fe7 = fp[(fp["definition"] == "E7") & fp["period"].str.startswith("fold")]
fcts = pickle.loads((D / "fold_cts.pkl").read_bytes())
fw = pd.concat([fcts[(f"fold {k}", "E7", "ML Model")] for k in (1, 2)])
B["folds"] = {"strict": int(fe7[fe7["method"] == "ML Model"]["strict"].sum()), "conditions": int(fe7[fe7["method"] == "ML Model"]["conditions"].sum()),
              "false": float((fe7[fe7["method"] == "ML Model"]["false (truth-based) per asset-month"] * [169.3032, 166.4668]).sum() / (169.3032 + 166.4668)),
              "warn": float(fw.loc[fw["strict"], "attributable_warning"].median()), "eval_warn": float(ctm["ML Model"].loc[ctm["ML Model"]["strict"], "attributable_warning"].median())}
tables["by_type"] = cp.assign(ml=ctm["ML Model"].loc[cp["wo_id"], "strict"].to_numpy(), vr=ctm["Vendor Rules"].loc[cp["wo_id"], "strict"].to_numpy())[["wo_id", "condition_type", "ml", "vr"]]
# intervals (whole conditions; whole assets), false ALARM ratio on all criteria
rng = np.random.default_rng(7)
assets = sorted(set(P2A.values()))
s_ml, s_vr = ctm["ML Model"].loc[cp["wo_id"], "strict"].to_numpy(float), ctm["Vendor Rules"].loc[cp["wo_id"], "strict"].to_numpy(float)
ix = rng.integers(0, len(s_ml), size=(2000, len(s_ml)))
cd = s_ml[ix].sum(1) - s_vr[ix].sum(1)
fa_a = {m: np.array([sum(1 for r in fl[m] if r.asset_id == a) for a in assets], float) for m in METH}
aoA = {m: r2.alarm_outcomes(LEV[m], conf, inp["wo"], SPAN, SPAN[1], False, False) for m in METH}
far_a = {m: np.array([((aoA[m]["asset_id"] == a) & aoA[m]["false_alarm"]).sum() for a in assets], float) for m in METH}
ia = rng.integers(0, len(assets), size=(2000, len(assets)))
fdr = 1 - fa_a["ML Model"][ia].sum(1) / np.maximum(fa_a["Vendor Rules"][ia].sum(1), 1e-9)
frr = far_a["ML Model"][ia].sum(1) / np.maximum(far_a["Vendor Rules"][ia].sum(1), 1e-9)
fpv = fp[fp["definition"] == "E7"].pivot_table(index="period", columns="method", values=["strict", "false (truth-based) per asset-month"])
B["against"] = {"catch_diff": int(s_ml.sum() - s_vr.sum()), "catch_lo": float(np.quantile(cd, 0.05)), "catch_hi": float(np.quantile(cd, 0.95)),
                "false_lower": float(1 - fa_a["ML Model"].sum() / fa_a["Vendor Rules"].sum()), "false_lo": float(np.quantile(fdr, 0.05)), "false_hi": float(np.quantile(fdr, 0.95)),
                "alarm_ratio": float(far_a["ML Model"].sum() / far_a["Vendor Rules"].sum()), "alarm_lo": float(np.quantile(frr, 0.05)), "alarm_hi": float(np.quantile(frr, 0.95)),
                "fold1": int(fpv.loc["fold 1", ("strict", "ML Model")] - fpv.loc["fold 1", ("strict", "Vendor Rules")]),
                "fold2": int(fpv.loc["fold 2", ("strict", "ML Model")] - fpv.loc["fold 2", ("strict", "Vendor Rules")]),
                "fold1_false": float(fpv.loc["fold 1", ("false (truth-based) per asset-month", "ML Model")] - fpv.loc["fold 1", ("false (truth-based) per asset-month", "Vendor Rules")]),
                "fold2_false": float(fpv.loc["fold 2", ("false (truth-based) per asset-month", "ML Model")] - fpv.loc["fold 2", ("false (truth-based) per asset-month", "Vendor Rules")]),
                "false_diff": float((fa_a["ML Model"].sum() - fa_a["Vendor Rules"].sum()) / AM)}
win = pd.read_parquet(D / "windows.parquet")
tables["windows"] = win
# definitions table
drows = []
for m in METH:
    r7, r4, ro = res[("E7", m)], res[("E4", m)], res[("orig", m)]
    c7 = r7["ct"]
    bor = c7["class"].str.startswith("borrowed")
    w7 = c7.loc[c7["strict"], "attributable_warning"].median()
    w4 = (r4["ct"]["credit_ts"] - r4["ct"]["credited_opened"]).dt.total_seconds().div(86400)[r4["ct"]["caught"]].median()
    wo_ = (ro["ct"]["credit_ts"] - ro["ct"]["credited_opened"]).dt.total_seconds().div(86400)[ro["ct"]["caught"]].median()
    wb = c7.loc[c7["strict"] & ~bor, "attributable_warning"].median()
    fa_m = float(fa[m] / AM)
    for dn, cau, fal, wv in (("Headline", r7["strict"], r7["false_truth"] / AM, w7), ("Lookback credit", r4["caught"], r4["false_truth"] / AM, w4),
                             ("Original", ro["caught"], ro["false_conf"] / AM, wo_), ("Borrowed excluded", int((c7["strict"] & ~bor).sum()), r7["false_truth"] / AM, wb)):
        drows.append({"Method": m, "Definition": dn, "Conditions caught": int(cau), "False alerts per asset-month": fal, "False ALARMs per asset-month": fa_m, "Median warning, days": wv})
tables["definitions"] = pd.DataFrame(drows)
# own signal
sig_rows = []
c32 = cp.reset_index(drop=True)
own = [H.own_start(c.condition_type, c.asset_key, c.credit_ts, P2A) for c in c32.itertuples()]
ons = C.truth_onsets(c32).set_index("wo_id")["truth_onset"]
c32["onset"] = [o if pd.notna(o) else ons.get(w, pd.NaT) for o, w in zip(own, c32["wo_id"], strict=True)]


def open_in(lev, a, t0, t1):
    e = lev[lev["asset_id"] == a]
    return bool(((e["opened"] < t1) & (e["closed"] > t0)).any())


for typ, g in c32.groupby("condition_type"):
    if len(g) < 3:
        continue
    for m in METH:
        hit, hw = [], []
        for c in g.itertuples():
            on = max(c.onset if pd.notna(c.onset) else c.credit_ts - 30 * DAY, c.credit_ts - 120 * DAY)
            dur = c.credit_ts - on
            hit.append(open_in(LEV[m], c.asset_key, on, c.credit_ts))
            busy = [(t - 120 * DAY, t + DAY) for t in c43.loc[c43["asset_key"] == c.asset_key, "credit_ts"]]
            dg = deg_a.get(c.asset_key)
            t = SPAN[0]
            while t + dur <= SPAN[1]:
                if not any(a < t + dur and t < b for a, b in busy) and not Q.present_between(dg, t, t + dur):
                    hw.append(open_in(LEV[m], c.asset_key, t, t + dur))
                t += max(dur, DAY)
        sig_rows.append({"type": typ, "method": m, "n": len(g), "alerted": float(np.mean(hit)), "healthy": float(np.mean(hw)) if hw else np.nan, "windows": len(hw)})
tables["own_signal"] = pd.DataFrame(sig_rows)
# false alerts by context (outline definition: programs first run in the evaluation year)
store = F._Store()
jobs = store.read_pd("bronze", "erp_jobs")
jd = pd.to_datetime(jobs["date"])
jobs["day"] = (jd.dt.tz_convert("UTC").dt.tz_localize(None) if jd.dt.tz is not None else jd).dt.normalize()
first = jobs.groupby(["machine_id", "program"])["day"].min().rename("first").reset_index()
jobs = jobs.merge(first, on=["machine_id", "program"]).sort_values(["machine_id", "day"])
y0 = SPAN[0].tz_convert("America/Chicago").tz_localize(None).normalize()
newprog = first[(first["first"] >= y0) & first["machine_id"].str.startswith("HMC")]
B["new_programs"] = {"programs": len(newprog), "cutting_hours": float(jobs[jobs["program"].isin(set(newprog["program"])) & jobs["machine_id"].str.startswith("HMC")]["cutting_hours"].sum())}
JOBS = {m_: g[["day", "first"]].reset_index(drop=True) for m_, g in jobs.groupby("machine_id")}
ben = pd.read_parquet(REPO / "runs" / "run09_20261030T090522Z_3eced8fb" / "truth" / "benign.parquet")
for c in ("ts", "end"):
    ben[c] = pd.to_datetime(ben[c], utc=True)
segs = cb("ml_segments")
segs["segment_start"] = pd.to_datetime(segs["segment_start"], utc=True)


def context(asset, t):
    mach = asset[:6]
    if len(segs[(segs["asset_id"] == asset) & (segs["segment_start"] <= t) & (segs["segment_start"] > t - 14 * DAY) & (segs["reason"] != "install")]):
        return "after re-centering"
    hit = (ben["asset_id"] == mach) | ben["asset_id"].astype(str).str.startswith(asset)
    if asset.startswith("CMP"):
        hit |= ben["event"] == "compressor_room_heat"
    if len(ben[hit & (ben["ts"] <= t) & (ben["end"].fillna(ben["ts"] + DAY) > t)]):
        return "benign event"
    if asset.startswith("CMP"):
        return "other"
    g = JOBS.get(mach)
    day = t.tz_convert("UTC").tz_localize(None).normalize()
    k = g["day"].searchsorted(day, side="right") - 1 if g is not None else -1
    if k < 0:
        return "other"
    return "new program" if g.at[k, "first"] >= y0 else "known program"


healthy = ~near
hs = meas[healthy][["asset_id", "hour_utc"]].drop_duplicates().sample(n=4000, random_state=7)
hctx = pd.Series([context(a, t) for a, t in zip(hs["asset_id"], hs["hour_utc"], strict=True)]).value_counts(normalize=True)
cx = []
for m in METH:
    s = pd.Series([context(r.asset_id, r.opened) for r in fl[m]]).value_counts()
    for k in ("new program", "known program", "after re-centering", "benign event", "other"):
        cx.append({"method": m, "context": k, "false": int(s.get(k, 0)), "share": float(s.get(k, 0) / len(fl[m])), "healthy": float(hctx.get(k, 0.0))})
tables["context"] = pd.DataFrame(cx)
# healthy hours above threshold by month and class
deg_p, _ = Q.degraded_hours(tuple(sorted(P2A.items())))
hh2 = meas[healthy].copy()
dg = np.array([Q.present_at(deg_p.get(p), t) for p, t in zip(hh2["point_id"], hh2["hour_utc"], strict=True)])
hh2 = hh2[~dg]
tables["healthy_above"] = hh2.assign(above=hh2["score"] > hh2["threshold"]).groupby(["month", "cls"])["above"].mean().reset_index()
# misses
ec = cb("eval_conditions").set_index("wo_id")
mi = []
for c in cp.itertuples():
    r = ctm["ML Model"].loc[c.wo_id]
    if r["strict"]:
        continue
    cat = "removed as a coincidence" if bool(r["caught"]) else miss_reason(K["ml"], c.asset_key, c.credit_ts, ec.at[c.wo_id, "ml_miss_reason"])
    mi.append({"Asset": c.asset_key, "Condition": c.condition_type, "Confirmed": c.credit_ts, "Reason": cat, "wo_id": c.wo_id})
tables["misses"] = pd.DataFrame(mi).sort_values("Confirmed")
# timelines (all 28 assets): episodes, inspections, PM findings, failures, confirmations
tl = []
for m in METH:
    for r in LEV[m].itertuples():
        if r.closed <= SPAN[0] or r.opened >= SPAN[1]:
            continue
        tl.append({"asset": r.asset_id, "lane": m, "kind": "ALERT", "start": max(r.opened, SPAN[0]), "end": min(r.closed, SPAN[1])})
        if pd.notna(r.alarm_ts):
            tl.append({"asset": r.asset_id, "lane": m, "kind": "ALARM", "start": max(r.alarm_ts, SPAN[0]), "end": min(r.closed, SPAN[1])})
for r in woy.itertuples():
    a = r.asset_id
    if a not in set(P2A.values()):
        continue
    if r.wo_type == "inspection":
        tl.append({"asset": a, "lane": "Vendor Rules", "kind": "inspection, nothing found" if r.no_fault_found else "inspection, condition confirmed", "start": r.opened, "end": r.opened})
    elif r.wo_type == "unplanned_repair":
        tl.append({"asset": a, "lane": "both", "kind": "failure", "start": r.opened, "end": r.opened})
for r in woy[(woy["wo_type"] == "pm") & woy["failure_code"].notna()].itertuples():
    for a in [x for x in set(P2A.values()) if x.startswith(r.asset_id)][:1] if r.asset_id not in set(P2A.values()) else [r.asset_id]:
        tl.append({"asset": a if r.asset_id not in set(P2A.values()) else r.asset_id, "lane": "both", "kind": "PM finding", "start": r.opened, "end": r.opened})
tables["timeline"] = pd.DataFrame(tl)
B["timeline_assets"] = sorted(set(inst["asset_key"]) | set(woy[(woy["wo_type"] == "unplanned_repair") & woy["asset_id"].isin(set(P2A.values()))]["asset_id"]))
B["stuck"] = json.loads((R.S / "diag21" / "eval_results.json").read_text())["stuck"]
B["cooling_check"] = s2
for k, v in tables.items():
    v.to_parquet(OUT / f"{k}.parquet", index=False)
(OUT / "bundle.json").write_text(json.dumps(B, indent=1, default=str))
print(json.dumps({k: B[k] for k in ("alert", "alarm", "failures", "inspection", "diag", "folds", "against", "new_programs", "scoring")}, indent=1, default=str))
