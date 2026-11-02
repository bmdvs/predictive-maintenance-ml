"""Checkpoint (b) rev2 (diag25): Section 3 (items 13 to 16). Writes diag25/s3.md."""

import copy
import json
import pickle

import numpy as np
import pandas as pd

import diag_lib as L
import r19_common as R
import r21_common as Q
from cb_common import METH, D, cb, conds51, core, inputs, loc, md, note, sec
from pdm import splits
from pdm.config import load_config
from pdm.models import nbm, severity

K = core()
cfg = load_config()
SPAN, AM, P2A, cp = K["SPAN"], K["AM"], K["P2A"], K["cp"]
inp = inputs()
out = [sec("Section 3", 1)]
DAY = pd.Timedelta(days=1)

# ------------------------------------------------------------------ item 13
fz = cfg["models"]["nbm"]["finalists"][0]
cols = [*nbm.VENDOR_LOG, *nbm.VENDOR_LINEAR, nbm.TEMP_RESID]
desc = {"vel_rms_mm_s": "velocity RMS 10 to 1,000 Hz", "hf_rms_g": "HF acceleration RMS 2 to 10 kHz", "ord_01_g2": "1x order energy", "ord_02_g2": "2x order energy",
        "crest_factor": "crest factor", "kurtosis": "kurtosis", "temp_resid_c": "housing temperature residual (C)"}
rows = [{"input": c, "measure": desc.get(c, c.replace("env_", "envelope ").replace("_1_g2", ", fundamental").replace("_2_g2", ", 2nd harmonic").upper().replace("ENVELOPE", "envelope").replace(", FUNDAMENTAL", ", fundamental").replace(", 2ND HARMONIC", ", 2nd harmonic")),
         "scale": "log10, segment-relative" if c in nbm.VENDOR_LOG else ("linear, segment-relative" if c != nbm.TEMP_RESID else "linear (C), segment-relative"), "source": "sensor"} for c in cols]
rows += [{"input": f"regime_{r}", "measure": f"load regime {r} (one-hot)", "scale": "0/1", "source": "controller (load and execution state)"} for r in nbm.REGIMES]
rows += [{"input": f"band_{b}", "measure": f"spindle speed band {b} (one-hot)", "scale": "0/1", "source": "controller (spindle speed)"} for b in nbm.BANDS]
out.append(sec("3.1 Inputs (item 13)") + f"Frozen candidate: {fz['candidate']} {fz['hidden']}, {fz['epochs']} epochs, threshold {fz['threshold_mode']} (config models.nbm.finalists[0]).\n\n" + md(pd.DataFrame(rows)))
out.append("The autoencoder rebuilds all 20 inputs: the network is fitted on the 20-column standardized vector (13 sensor measures less the segment level, then the 7 indicators) as both input and target, and the score is the mean squared rebuild error over the 20 (nbm.PointModel.raw_scores; score_point builds the vector as [x - level, indicators]).\n\n")
out.append(note("nbm.features (input set vendor: VENDOR_LOG, VENDOR_LINEAR, temp_resid_c), nbm._one_hot (REGIMES, BANDS), nbm.score_point"))

# ------------------------------------------------------------------ item 14
out.append(sec("3.2 Healthy hours (item 14)"))
ti = cb("ml_training_info")
ti["n_train"] = pd.to_numeric(ti["n_train"])
ti["segments_before_cutoff"] = pd.to_numeric(ti["segments_before_cutoff"])
out.append(md(pd.DataFrame([{"points": len(ti), "healthy training hours, total": int(ti["n_train"].sum()), "per point, min": int(ti["n_train"].min()),
                             "point with min": ti.loc[ti["n_train"].idxmin(), "point_id"], "per point, max": int(ti["n_train"].max()), "point with max": ti.loc[ti["n_train"].idxmax(), "point_id"],
                             "segments per point, min": int(ti["segments_before_cutoff"].min()), "median": float(ti["segments_before_cutoff"].median()), "max": int(ti["segments_before_cutoff"].max())}])))
out.append("Per point:\n\n" + md(ti[["point_id", "point_class", "n_train", "segments_before_cutoff", "temp_fit"]].sort_values("point_id")))
ex = pd.DataFrame([
    {"event": "sensor remount", "window as coded": f"new segment; level from the first {cfg['models']['nbm']['recenter_days'][0]} to {cfg['models']['nbm']['recenter_days'][1]} days (ends after day {cfg['models']['nbm']['recenter_days'][0]} once {cfg['models']['nbm']['recenter_min_hours']} measured hours exist); no alerts until the level is ready; fewer than {cfg['models']['nbm']['recenter_floor_hours']} hours: no level, not scored", "training": "same", "scoring": "same"},
    {"event": "repair on the monitored component (parts on the measured component)", "window as coded": f"training: as a remount; scoring (S1): level from day {cfg['models']['nbm']['recenter_days'][0]} until {cfg['models']['nbm']['recenter_min_hours']} measured hours, at most 60 days, no level and no alerts meanwhile", "training": "re-centering window", "scoring": "S1 window (scoring only)"},
    {"event": "repair to another component on the asset", "window as coded": "C4: level from days 3 to 14 after the repair (at least 24 measured hours), scored against the previous level until then", "training": "C4", "scoring": "C4"},
    {"event": "confirmed condition (any, incl. failures)", "window as coded": f"training hours excluded from {cfg['models']['nbm']['exclude_before_condition_days']} days before the confirmation to 1 day after (models.nbm.exclude_before_condition_days; was 30)", "training": "excluded", "scoring": "scored"},
    {"event": "open inspection or repair work order", "window as coded": "training hours excluded from opening to close", "training": "excluded", "scoring": "scored"},
    {"event": "baseline flagged by the sanity check", "window as coded": "training hours excluded over the flagged window", "training": "excluded", "scoring": "scored"},
])
out.append("Exclusion and re-centering windows:\n\n" + md(ex))
out.append(note("ml_tasks.excluded_intervals; nbm segment levels (recenter_days, recenter_min_hours, recenter_floor_hours; fixes s1_on_component_window, c4_other_repair_window; F1 holds the previous level where no level forms). The 30-day exclusions asked for are coded as 120 days before every confirmed condition, failures included (a failure is a confirmed condition)"))
ts = pd.read_parquet(R.S / "diag24" / "train_scores.parquet")
ts["hour_utc"] = pd.to_datetime(ts["hour_utc"], utc=True)
cut = splits.training_cutoff(cfg)
conds_all = K["conds_all"]
h = ts[(ts["state"] == "scored") & (ts["hour_utc"] < cut)][["point_id", "hour_utc"]]
h["asset"] = h["point_id"].map(P2A)
ok = np.ones(len(h), dtype=bool)
for a, g in conds_all.groupby("asset_key"):
    m = (h["asset"] == a).to_numpy()
    for t in g["credit_ts"]:
        ok[m] &= ~((h.loc[m, "hour_utc"] >= t - 120 * DAY) & (h.loc[m, "hour_utc"] < t + DAY)).to_numpy()
h = h[ok]
ind = inp["ind"]
sp = h[h["point_id"].str.contains("-SP-")].merge(ind[["point_id", "hour_utc", "vel_rms_mm_s", "kurtosis", "crest_factor", "regime"]], on=["point_id", "hour_utc"])
sp["location"] = np.where(sp["point_id"].str.endswith("-F"), "front", "rear")
rr = []
for (lo, rg), g in sp.groupby(["location", "regime"]):
    r = {"spindle point": lo, "load regime": rg, "hours": len(g)}
    for c in ("vel_rms_mm_s", "kurtosis", "crest_factor"):
        q = g[c].quantile([0.05, 0.5, 0.95])
        r[f"{c} p05 / p50 / p95"] = f"{q.iloc[0]:.2f} / {q.iloc[1]:.2f} / {q.iloc[2]:.2f}"
    rr.append(r)
out.append("Typical healthy ranges on spindles (healthy training hours: scored, before the cutoff, outside 120 days before to 1 day after any confirmed condition):\n\n" + md(pd.DataFrame(rr)))

# ------------------------------------------------------------------ item 15
out.append(sec("3.3 Scoring summary, evaluation year (item 15)"))
sc = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
sc = sc[(sc["hour_utc"] >= SPAN[0]) & (sc["hour_utc"] < SPAN[1])].reset_index(drop=True)
ml = K["ml"]
lvl = np.zeros(len(sc), dtype=int)
for r in ml.itertuples():
    m = ((sc["asset_id"] == r.asset_id) & (sc["hour_utc"] >= r.opened) & (sc["hour_utc"] < r.closed)).to_numpy()
    lvl[m] = np.maximum(lvl[m], 1)
    if pd.notna(r.alarm_ts):
        m2 = m & (sc["hour_utc"] >= r.alarm_ts).to_numpy()
        lvl[m2] = 2
sc["level"] = np.array(["outside any episode", "ALERT", "ALARM"])[lvl]
meas = sc[sc["state"] == "scored"]
st = sc["state"].value_counts()
out.append("Point-hours by state:\n\n" + md(st.rename("point-hours").reset_index()))
sh = meas["level"].value_counts(normalize=True)
eps_y = ml[(ml["opened"] >= SPAN[0]) & (ml["opened"] < SPAN[1])]
out.append(md(pd.DataFrame([{"measured (scored) point-hours": len(meas), "outside any episode": sh.get("outside any episode", 0), "in ALERT": sh.get("ALERT", 0), "in ALARM": sh.get("ALARM", 0),
                             "ALERT episodes opened": len(eps_y), "assets with an episode": eps_y["asset_id"].nunique(), "escalated to ALARM": int(eps_y["alarm_ts"].notna().sum())}])))


def group(c: str) -> str:
    c = str(c).split(",")[0]
    if c.startswith("env_"):
        return "envelope fault bands"
    if c in ("vel_rms_mm_s", "hf_rms_g"):
        return "velocity and HF acceleration"
    if c.startswith("ord_"):
        return "shaft orders (1x, 2x)"
    if c in ("crest_factor", "kurtosis"):
        return "crest factor and kurtosis"
    if c == "temp_resid_c":
        return "temperature residual"
    if c.startswith(("regime_", "band_")):
        return "operating indicators"
    return "other"


al = meas[meas["level"] != "outside any episode"]
g = al["top_contributors"].map(group).value_counts(normalize=True)
out.append("Shares of ALERT hours (ALERT and ALARM) by leading input group (first of the top contributors):\n\n" + md(g.rename("share").rename_axis("leading input group").reset_index()))
meas = meas.assign(month=meas["hour_utc"].dt.tz_convert("America/Chicago").dt.strftime("%Y-%m"))
out.append("Measured point-hours by level by month:\n\n" + md(pd.crosstab(meas["month"], meas["level"]), index=True))
ratio = meas["score"] / meas["threshold"]
c43 = conds51()
c43 = c43[c43["instrumented"]]
near = np.zeros(len(meas), dtype=bool)
pre = np.zeros(len(meas), dtype=bool)
for r in c43.itertuples():
    m = ((meas["asset_id"] == r.asset_key) & (meas["hour_utc"] >= r.credit_ts - 120 * DAY) & (meas["hour_utc"] < r.credit_ts + DAY)).to_numpy()
    near |= m
    if r.wo_id in set(cp["wo_id"]):
        pre |= m & (meas["hour_utc"] < r.credit_ts).to_numpy()
qq = []
for lab, m in (("no confirmed condition within 120 days", ~near), ("120 days before a confirmed condition (in scope)", pre)):
    x = ratio[m]
    qq.append({"hours": lab, "n": int(m.sum()), "median": x.median(), "p90": x.quantile(0.9), "p99": x.quantile(0.99), "share above threshold": (x > 1).mean()})
out.append("Score against threshold (score / threshold):\n\n" + md(pd.DataFrame(qq)))
meas = meas.assign(cls=meas["asset_id"].map(severity.asset_class))
ctm = K["res"][("E7", "ML Model")]["ct"]
cp2 = cp.assign(cls=cp["asset_key"].map(severity.asset_class), strict=ctm["strict"].to_numpy())
br = []
for c_, g_ in meas.groupby("cls"):
    e = eps_y[eps_y["asset_id"].map(severity.asset_class) == c_]
    br.append({"asset class": c_, "assets": g_["asset_id"].nunique(), "measured point-hours": len(g_), "in ALERT": (g_["level"] == "ALERT").mean(), "in ALARM": (g_["level"] == "ALARM").mean(),
               "ALERT episodes": len(e), "ALARMs": int(e["alarm_ts"].notna().sum()), "conditions in scope": int((cp2["cls"] == c_).sum()), "strict catches": int(cp2.loc[cp2["cls"] == c_, "strict"].sum())})
out.append("By asset class:\n\n" + md(pd.DataFrame(br)))
ra = json.loads((R.S / "diag21" / "eval_results.json").read_text())["rearms"]
out.append("Re-arms (C3, flat-only re-arm of a stuck point's threshold), evaluation year:\n\n" + md(pd.DataFrame(ra)))
out.append(note("C3 log from the frozen design's evaluation-year scoring (diag21; prod version 13 reproduces those scores within 7.9e-8, D-006); stuck point = more than half its scored hours outside condition windows above the threshold"))

# ------------------------------------------------------------------ item 16
out.append(sec("3.4 Signatures (item 16)"))
c43["in scope"] = c43["wo_id"].isin(set(cp["wo_id"]))
x = c43.groupby("condition_type").agg(confirmed=("wo_id", "size"), in_scope=("in scope", "sum")).reset_index()
out.append("Condition counts by type, evaluation year (instrumented assets):\n\n" + md(x))
m24 = json.loads((R.S / "diag24" / "measures_eval.json").read_text())
ctv = K["res"][("E7", "Vendor Rules")]["ct"].set_index("wo_id")
ctm2 = ctm.set_index("wo_id")
ex = []
for fam in ("lubrication", "spindle cooling", "bearing", "cavitation or wear", "rotary union"):
    best = None
    for single in (True, False):
      if best is not None:
        break
      for e in m24:
        for c in e["conditions"]:
            if c["family"] != fam or not ctm2.at[c["wo_id"], "strict"]:
                continue
            if single and len(e["conditions"]) > 1 and fam != "rotary union":
                continue
            w = float(ctm2.at[c["wo_id"], "attributable_warning"])
            if best is None or w > best[0]:
                best = (w, e, c)
    if best is None:
        continue
    w, e, c = best
    mm = e["d2 measures"]
    ex.append({"type": fam, "wo_id": c["wo_id"], "condition type": c["condition_type"], "asset": e["asset"], "episode opened": loc(pd.Timestamp(e["opened"]), "%Y-%m-%d %H:%M"),
               "confirmed": loc(cp.set_index("wo_id").at[c["wo_id"], "credit_ts"]), "ML Model attributable warning (days)": w, "Vendor Rules": "caught" if ctv.at[c["wo_id"], "strict"] else "missed",
               "HF (dB)": mm.get("db_hf"), "BPFO / BPFI / BSF (dB)": " / ".join(f"{mm.get('db_env_' + b, np.nan):.1f}" for b in ("bpfo", "bpfi", "bsf")),
               "1x (dB)": mm.get("db_ord_01"), "kurtosis z": mm.get("kurt"), "crest z": mm.get("crest"), "temperature residual change (C)": mm.get("temp_dc"), "TSC drop z": mm.get("tsc_drop")})
out.append("One example of each common type (strict ML Model catch with the longest attributable warning; single-condition episodes where one exists), D2 measures over the episode:\n\n" + md(pd.DataFrame(ex)))
(D / "examples.json").write_text(json.dumps(ex, default=str, indent=1))
# rebuild exhibit
bear = next(x for x in ex if x["type"] == "bearing")
E = next(e for e in m24 if e["asset"] == bear["asset"] and loc(pd.Timestamp(e["opened"]), "%Y-%m-%d %H:%M") == bear["episode opened"])
t0, t1 = pd.Timestamp(E["opened"]), cp.set_index("wo_id").at[bear["wo_id"], "credit_ts"]
cand = meas[(meas["asset_id"] == bear["asset"]) & (meas["hour_utc"] >= t0) & (meas["hour_utc"] < t1)]
rat = cand["score"] / cand["threshold"]
ab = rat[rat > 1].sort_values()
hr = cand.loc[ab.index[len(ab) // 2]]
A = pickle.loads((R.S / "diag21" / "models_A_final.pkl").read_bytes())
pm, fac, mcfg = A["models"][hr["point_id"]], A["factor"], copy.deepcopy(A["cfg"])
mcfg["models"]["nbm"]["fixes"] = {**nbm.FIX_DEFAULTS, **mcfg["models"]["nbm"]["fixes"], "f1_hold_level": True, "f4_high_ratio": True, "f4_ratio": 5.0, "f4_exclude_pump_low_pressure": True, "s1_on_component_window": True}
win = nbm.component_repairs(inp["win"][inp["win"]["point_id"] == hr["point_id"]], inp["wo"])
rows_p = ind[ind["point_id"] == hr["point_id"]]
s1, _ = nbm.score_point(pm, rows_p, win, mcfg, hr["hour_utc"] - 200 * DAY, hr["hour_utc"] + pd.Timedelta(hours=1), fac, 3, None, keep_inputs=True)
s1["hour_utc"] = pd.to_datetime(s1["hour_utc"], utc=True)
r1 = s1[s1["hour_utc"] == hr["hour_utc"]].iloc[0]
z13 = np.array([r1[f"z:{c}"] for c in pm.columns], dtype=float)
oh = nbm._one_hot(pd.DataFrame({"regime": [r1["regime"]]}), np.array([r1["speed_band"]]))[0]
zi = (oh - pm.scaler.center[len(pm.columns):]) / pm.scaler.scale[len(pm.columns):]
z = np.concatenate([z13, zi])
rec = pm.model.predict(z[None, :])[0]
err = (z - rec) ** 2
names = [*pm.columns, *[f"regime_{r}" for r in nbm.REGIMES], *[f"band_{b}" for b in nbm.BANDS]]
rb = pd.DataFrame({"input": names, "standardized input": z, "rebuilt": rec, "squared gap": err, "share of the gap": err / err.sum()}).sort_values("squared gap", ascending=False)
out.append(f"Rebuild exhibit (the median hour above the threshold in the episode): {hr['point_id']} at {loc(hr['hour_utc'], '%Y-%m-%d %H:%M')} (in the episode before {bear['wo_id']}); prod score {hr['score']:.4g}, threshold {hr['threshold']:.4g} ({hr['score'] / hr['threshold']:.1f}x); recomputed mean squared gap {err.mean():.4g}; prod top contributors {hr['top_contributors']}.\n\n" + md(rb))
out.append(note("model A as frozen (diag21 artifact; prod version 13 reproduces it within 7.9e-8, D-006); the 13 sensor inputs as score_point standardizes them (keep_inputs), the 7 indicators standardized with the point's scaler; rebuilt by the point's network"))
(D / "s3.md").write_text("".join(out), encoding="utf-8")
print("done", hr["point_id"], hr["hour_utc"], hr["score"], err.mean())
