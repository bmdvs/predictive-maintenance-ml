"""Checkpoint (b) rev2 (diag25): Section 4 (items 17 to 21). Writes diag25/s4.md."""

import json
import pickle

import numpy as np
import pandas as pd

import diag_folds as F
import r19_common as R
import r19_eval_helpers as H
import r21_common as Q
import r2_common as C
from cb_common import METH, D, REPO, cb, conds51, core, inputs, loc, md, miss_reason, note, sec
from pdm import splits
from pdm.config import load_config
from pdm.evaluation import rev2 as r2
from pdm.models import nbm, severity

K = core()
cfg = load_config()
SPAN, AM, P2A, cp, conf = K["SPAN"], K["AM"], K["P2A"], K["cp"], K["conf"]
LEV = {"ML Model": K["ml"], "Vendor Rules": K["vd"]}
res = K["res"]
inp = inputs()
DAY = pd.Timedelta(days=1)
_, deg_a = Q.degraded_hours(tuple(sorted(P2A.items())))
deg_p, _ = Q.degraded_hours(tuple(sorted(P2A.items())))
out = [sec("Section 4", 1)]
rng = np.random.default_rng(int(cfg["models"]["seed"]))

# ------------------------------------------------------------------ item 17: model card
fz = cfg["models"]["nbm"]["finalists"][0]
nb = cfg["models"]["nbm"]
st = cb("ml_settings").iloc[0]
ti = cb("ml_training_info")
B = splits.boundaries(cfg)
card = [
    ("model", f"{fz['candidate']} (autoencoder on the vendor input set), one model per sensor point (40), per_point_all_segments"),
    ("network", f"hidden layers {fz['hidden']}, {fz['epochs']} epochs, Adam, learning rate 0.001, weight decay 1e-4, seed {cfg['models']['seed']}"),
    ("inputs", "13 sensor measures (segment-relative) and 7 operating indicators (4 load regimes, 3 speed bands); rebuild of all 20"),
    ("threshold", f"{fz['threshold_mode']} per point, {nb['threshold_percentile']}th percentile of the point's healthy training scores"),
    ("ALERT", f"score above the threshold for {nb['persistence_hours']} consecutive scored hours; episode closes {cfg['evaluation']['catch']['episode_gap_days']:g} days after the last alert"),
    ("ALARM", "an open ALERT plus a criterion for the asset class (models.severity; ml_tasks.ALARM_CRITERIA); 14-day escalation a tested variant only"),
    ("re-centering", f"level from days {nb['recenter_days'][0]} to {nb['recenter_days'][1]} after a repair or remount ({nb['recenter_min_hours']} measured hours), no alerts meanwhile; plausibility factor {float(st['plausibility_factor']):.3f}"),
    ("fixes on (frozen)", ", ".join(k for k, v in nbm.fixes(cfg).items() if v is True)),
    ("training span", f"{loc(B['train'][0])} to the training cutoff {loc(splits.training_cutoff(cfg))}; healthy training hours {int(pd.to_numeric(ti['n_train']).sum()):,}"),
    ("training exclusions", f"{nb['exclude_before_condition_days']} days before every confirmed condition; open inspection and repair work; flagged baselines"),
    ("evaluation year", f"{loc(SPAN[0])} to {loc(SPAN[1] - DAY)}"),
    ("registry", "pdm.ml.nbm_point_models version 13, alias champion (prod); dev champion version 5 (copy of 13)"),
    ("code", "commit 8ac6d75, wheel 0.2.9 (model); 8ad4382, wheel 0.2.10 (D2 diagnosis, scores unchanged)"),
    ("diagnosis", "advisory.diagnosis_rule d2 (thresholds in advisory.d2.thresholds)"),
]
out.append(sec("4.1 Model card (item 17)") + md(pd.DataFrame(card, columns=["field", "value (frozen config)"])))

# ------------------------------------------------------------------ item 18: selection
out.append(sec("4.2 Selection (item 18)"))
cs = pd.read_csv(R.S / "diag11" / "candidates_summary.csv")
cs = cs[cs["fold"] == "pooled"]
names = {"A": "autoencoder_vendor [16-6-16] 200 ep, single", "B": "autoencoder_extended [16-6-16] 400 ep, speed_band", "C": "mahalanobis_extended, single", "D": "autoencoder_full_env [16-6-16] 200 ep, single"}
sel = pd.read_csv(R.S / "diag11" / "selection_70.csv").set_index("settings")
rows = []
for r in cs.itertuples():
    s = names[r.candidate]
    one = sel.loc[s] if s in sel.index else None
    rows.append({"candidate": r.candidate, "settings": s, "catch at the Vendor Rules budget (5-seed mean)": r._9 if False else cs.loc[cs["candidate"] == r.candidate, "matched-budget catches mean"].iloc[0],
                 "share at budget (seed 7)": one["caught_share_at_budget"] if one is not None else np.nan, "chance (seed 7)": one["chance_share_at_budget"] if one is not None else np.nan,
                 "skill above chance (5-seed mean)": cs.loc[cs["candidate"] == r.candidate, "skill mean"].iloc[0],
                 "caught at operating point (5-seed mean)": cs.loc[cs["candidate"] == r.candidate, "caught at ALERT mean"].iloc[0],
                 "false ALERT per asset-month at operating point (5-seed mean)": cs.loc[cs["candidate"] == r.candidate, "false ALERT per asset-month mean"].iloc[0]})
out.append("Final candidates, centering-fixed round (diag11; folds pooled, 24 conditions, original definitions of that round):\n\n" + md(pd.DataFrame(rows)))
out.append(note("diag11 candidates_summary (5 seeds) and selection_70 (seed 7): catch share at the Vendor Rules' pooled false-episode budget read off the operating curve, chance share at the same budget, skill = share minus chance; A was later retuned to [16-4-16] 200 epochs (diag14 rule outcome)"))
fp = pd.read_parquet(D / "folds_periods.parquet")
fl = fp[fp["period"].str.startswith("fold") & (fp["method"] == "ML Model")]
am_f = {1: 169.3032, 2: 166.4668}
base_e7 = fl[fl["definition"] == "E7"]
pooled_base = {"caught": int(base_e7["caught"].sum()), "strict": int(base_e7["strict"].sum()), "false": float((base_e7["false (truth-based) per asset-month"] * base_e7["period"].map(lambda p: am_f[int(p[-1])])).sum() / sum(am_f.values()))}
pers = pd.read_parquet(D / "persistence.parquet")
dc = [{"design choice": "frozen design (version 13 fold models, seed 7)", "source": "this run", "definition": "E2 to E7", "caught": pooled_base["caught"], "strict": pooled_base["strict"], "false per asset-month": pooled_base["false"], "rule": "", "result": "adopted"}]
for hrs in (3, 12):
    p = pers[pers["persistence (h)"] == hrs]
    dc.append({"design choice": f"persistence {hrs} hours (6 frozen)", "source": "this run, version 13 fold scores", "definition": "E2 to E7 (ALERT only)", "caught": int(p["caught"].sum()), "strict": int(p["strict"].sum()),
               "false per asset-month": p["false (truth-based)"].sum() / p["asset_months"].sum(), "rule": "", "result": "not adopted (6 hours frozen at step 5)"})
d14 = pd.read_csv(R.S / "diag14" / "confirmed_summary.csv")
for _, r in d14[(d14["finalist"] == "A") & (d14["fold"] == "pooled") & (d14["index"].isin([8, 9]))].iterrows():
    dc.append({"design choice": f"thresholds: {r['mode']} ({r['settings']})", "source": "diag14 (5 seeds)", "definition": "E2, E3 of that round", "caught": r["caught at ALERT mean"], "strict": np.nan,
               "false per asset-month": r["false ALERT per asset-month mean"], "rule": "diag14 tuning rule (skill, then false rate)", "result": "single kept"})
for iset, n in (("vendor", 13), ("extended", 37), ("full", 45), ("full_env", 52)):
    s = f"autoencoder_{iset} [16-6-16] 200 ep, single"
    r = sel.loc[s]
    dc.append({"design choice": f"{n} inputs ({iset}; autoencoder [16-6-16] 200 ep, single)", "source": "diag11 selection_70 (seed 7)", "definition": "original of that round", "caught": r["caught_operating"], "strict": np.nan,
               "false per asset-month": r["false_per_asset_month_operating"], "rule": "selection rule: skill at the budget", "result": f"skill {r['skill']:.3f}" + (" (picked)" if iset == "vendor" else "")})
d18 = pd.read_csv(R.S / "diag18" / "summary.csv")
d18 = d18[(d18["model"] == "A") & (d18["evaluation"] == "current")]
b18 = d18[d18["row"] == "baseline"].iloc[0]
dc.append({"design choice": "fix round 1 baseline (no fixes)", "source": "diag18 (5 seeds)", "definition": "E2, E3", "caught": b18["caught"], "strict": np.nan, "false per asset-month": b18["false"], "rule": "", "result": ""})
for fx, rule, resu in (("+F1", "adopted if not worse on caught, false and stuck points", "adopted"), ("+F2", "same rule: stuck points 8.0 against 6.0", "not adopted"),
                       ("+F3", "same rule", "adopted"), ("+F4", "same rule", "adopted (F4-tight, diag19)")):
    r = d18[d18["row"] == fx].iloc[0]
    dc.append({"design choice": f"fix {fx[1:]} alone on the round 1 baseline", "source": "diag18 (5 seeds)", "definition": "E2, E3", "caught": r["caught"], "strict": np.nan, "false per asset-month": r["false"], "rule": rule, "result": resu})
d20 = pd.read_csv(R.S / "diag20" / "fold_pooled.csv")
d20 = d20[d20["definition"] == "E4"]
for rw, rule, resu in (("baseline", "", ""), ("T1", "caught at least baseline, false at most baseline +10%, stuck no more: false 0.210, stuck 14.6", "not adopted"),
                       ("T2", "same rule: stuck 8.6", "not adopted"), ("T3", "same rule: caught 14.8", "not adopted"), ("S1", "same rule", "adopted")):
    r = d20[d20["row"] == rw].iloc[0]
    dc.append({"design choice": f"final round {rw}", "source": "diag20 (5 seeds)", "definition": "E2 to E4", "caught": r["caught"], "strict": np.nan, "false per asset-month": r["false"], "rule": rule, "result": resu})
d21 = pd.read_csv(R.S / "diag21" / "fold_pooled.csv")
d21 = d21[d21["definition"] == "E7"]
for rw, rule, resu in (("baseline", "", ""), ("C1", "strict at least baseline, truth-based false at most +10%, stuck no more: strict 19.2, stuck 5.4", "not adopted"),
                       ("C2", "same rule: stuck 5.0", "not adopted"), ("C3", "same rule", "adopted"), ("C4", "same rule", "adopted"), ("all adopted: C3, C4", "", "frozen")):
    r = d21[d21["row"] == rw].iloc[0]
    dc.append({"design choice": f"improvement round {rw}", "source": "diag21 (5 seeds)", "definition": "E2 to E7", "caught": r["caught"], "strict": r["strict"], "false per asset-month": r["false_truth"], "rule": rule, "result": resu})
out.append("Design choices, folds pooled:\n\n" + md(pd.DataFrame(dc)))
out.append(note("each fix as tested in its round, against that round's baseline (switching a single adopted fix off the frozen design was not run); rules as fixed before each round (CLAUDE.md R3.10); the 37, 45 and 52-input rows are the autoencoder of the same shape before A's retune to [16-4-16]"))

# ------------------------------------------------------------------ item 19: validation
out.append(sec("4.3 Validation (item 19)"))
yr = []
for d in ("E7", "E4"):
    for m in METH:
        r = res[(d, m)]
        yr.append({"period": "evaluation year", "definition": d, "method": m, "conditions": r["conditions"], "caught": r["caught"], "strict": r["strict"] if d == "E7" else np.nan,
                   "false (truth-based) per asset-month": r["false_truth"] / AM, "false (confirmation-based) per asset-month": r["false_conf"] / AM, "false ALARMs per asset-month (sensor)": r["alarm_false"] / AM})
allp = pd.concat([fp, pd.DataFrame(yr)], ignore_index=True)
allp["definition"] = allp["definition"].map({"E7": "E2 to E7", "E4": "E2 to E4"})
out.append("Results by period:\n\n" + md(allp))
out.append(note("folds: version 13's fold models (diag23 fold scores) and the Vendor Rules on each fold span; quarters: the year's episodes, conditions credited in the quarter, false episodes opened in the quarter over the quarter's asset-months"))
ctm, ctv = (res[("E7", m)]["ct"] for m in METH)
bt = cp.assign(ml=ctm["strict"].to_numpy(), vd=ctv["strict"].to_numpy()).groupby("condition_type").agg(conditions=("wo_id", "size"), ML_Model=("ml", "sum"), Vendor_Rules=("vd", "sum")).reset_index()
bt["ML Model rate"], bt["Vendor Rules rate"] = bt["ML_Model"] / bt["conditions"], bt["Vendor_Rules"] / bt["conditions"]
out.append("Catch rate by condition type (strict, headline):\n\n" + md(bt.rename(columns={"ML_Model": "ML Model strict", "Vendor_Rules": "Vendor Rules strict"})))
# each type against its own signal
c32 = cp.reset_index(drop=True)
own = [H.own_start(c.condition_type, c.asset_key, c.credit_ts, P2A) for c in c32.itertuples()]
ons = C.truth_onsets(c32).set_index("wo_id")["truth_onset"]
c32["onset"] = [o if pd.notna(o) else ons.get(w, pd.NaT) for o, w in zip(own, c32["wo_id"], strict=True)]
c43 = conds51()
c43 = c43[c43["instrumented"]]


def open_in(lev, a, t0, t1):
    e = lev[lev["asset_id"] == a]
    return bool(((e["opened"] < t1) & (e["closed"] > t0)).any())


sig = []
for typ, g in c32.groupby("condition_type"):
    if len(g) < 3:
        continue
    for m in METH:
        hit, hw = [], []
        for c in g.itertuples():
            on = c.onset if pd.notna(c.onset) else c.credit_ts - 30 * DAY
            on = max(on, c.credit_ts - 120 * DAY)
            dur = c.credit_ts - on
            hit.append(open_in(LEV[m], c.asset_key, on, c.credit_ts))
            busy = [(t - 120 * DAY, t + DAY) for t in c43.loc[c43["asset_key"] == c.asset_key, "credit_ts"]]
            dg = deg_a.get(c.asset_key)
            t = SPAN[0]
            while t + dur <= SPAN[1]:
                if not any(a < t + dur and t < b for a, b in busy) and not Q.present_between(dg, t, t + dur):
                    hw.append(open_in(LEV[m], c.asset_key, t, t + dur))
                t += max(dur, DAY)
        sig.append({"condition type": typ, "method": m, "conditions": len(g), "alerted while the signal was present": float(np.mean(hit)),
                    "matched healthy windows": len(hw), "healthy windows alerted": float(np.mean(hw)) if hw else np.nan})
out.append("Each common type against its own signal:\n\n" + md(pd.DataFrame(sig)))
out.append(note("signal present = from the condition's onset (own start, else truth onset; at most 120 days) to its confirmation; alerted = an ALERT episode on the asset open at any time in that span; matched healthy windows = consecutive windows of the same length on the same asset in the evaluation year with no truth degradation on the asset and no confirmed condition from 120 days before to 1 day after"))
# intervals
draws = int(cfg["evaluation"].get("bootstrap_draws", 2000))
sm, sv = ctm["strict"].to_numpy(dtype=float), ctv["strict"].to_numpy(dtype=float)
idx = rng.integers(0, len(sm), size=(draws, len(sm)))
cdiff = (sm[idx].sum(1) - sv[idx].sum(1)) / len(sm)
assets = sorted(set(P2A.values()))


def per_asset_false(m):
    lev = LEV[m]
    av = r2.alert_view(lev)
    ct = res[("E7", m)]["ct"]
    credited = {i for i, s in zip(ct["credited_episode"], ct["strict"], strict=True) if s and pd.notna(i)}
    cnt = dict.fromkeys(assets, 0)
    for i, r in zip(av.index, av.itertuples(), strict=True):
        if SPAN[0] <= r.opened < SPAN[1] and i not in credited and not Q.present_between(deg_a.get(r.asset_id), r.opened, r.closed):
            cnt[r.asset_id] += 1
    return np.array([cnt[a] for a in assets], dtype=float)


def per_asset_false_alarm(m):
    ao = r2.alarm_outcomes(LEV[m], conf, inp["wo"], SPAN, SPAN[1], False, True)
    ao = ao[ao["false_alarm"]]
    return np.array([(ao["asset_id"] == a).sum() for a in assets], dtype=float)


fa = {m: per_asset_false(m) for m in METH}
far = {m: per_asset_false_alarm(m) for m in METH}
ia = rng.integers(0, len(assets), size=(draws, len(assets)))
am_a = AM / len(assets)
fdiff = (fa["ML Model"][ia].sum(1) - fa["Vendor Rules"][ia].sum(1)) / (am_a * len(assets))
fr = far["ML Model"][ia].sum(1) / np.maximum(far["Vendor Rules"][ia].sum(1), 1e-9)
fpr = fp[fp["definition"] == "E7"].pivot_table(index="period", columns="method", values=["strict", "false (truth-based) per asset-month", "false ALARMs per asset-month (sensor)", "conditions"])
iv = []
for lab, x, pt in (("strict catch share difference (ML Model less Vendor Rules)", cdiff, (sm.sum() - sv.sum()) / len(sm)),
                   ("false ALERT per asset-month difference (truth-based)", fdiff, (fa["ML Model"].sum() - fa["Vendor Rules"].sum()) / AM),
                   ("false ALARM ratio (ML Model / Vendor Rules, sensor criteria)", fr, far["ML Model"].sum() / far["Vendor Rules"].sum())):
    r = {"measure": lab, "evaluation year": pt, "90% interval": f"{np.quantile(x, 0.05):.3f} to {np.quantile(x, 0.95):.3f}"}
    for k in (1, 2):
        p = f"fold {k}"
        if lab.startswith("strict"):
            r[p] = (fpr.loc[p, ("strict", "ML Model")] - fpr.loc[p, ("strict", "Vendor Rules")]) / fpr.loc[p, ("conditions", "ML Model")]
        elif lab.startswith("false ALERT"):
            r[p] = fpr.loc[p, ("false (truth-based) per asset-month", "ML Model")] - fpr.loc[p, ("false (truth-based) per asset-month", "Vendor Rules")]
        else:
            r[p] = fpr.loc[p, ("false ALARMs per asset-month (sensor)", "ML Model")] / fpr.loc[p, ("false ALARMs per asset-month (sensor)", "Vendor Rules")]
    iv.append(r)
out.append("90% intervals, headline definitions:\n\n" + md(pd.DataFrame(iv)))
out.append(note(f"{draws} draws, seed {cfg['models']['seed']}; catch difference from resampling whole conditions (32); false-alert-rate difference and false ALARM ratio from resampling whole assets (28; false episodes are not tied to conditions); fold columns are the point values on each fold"))
em = cb("eval_matched").iloc[0]
out.append("As coded in the pipeline (gold.eval_matched, E2 to E4, rev2.bootstrap resampling whole conditions with the episodes that cover them):\n\n" + md(pd.DataFrame([{k: em[k] for k in ("catch_diff", "catch_diff_lo", "catch_diff_hi", "false_diff", "false_diff_lo", "false_diff_hi", "draws")}])))
win = pd.read_parquet(D / "windows.parquet")
out.append("Catch windows (headline):\n\n" + md(win))
out.append("Results under the headline, lookback-credit and original definitions: section 2.1.\n\n")

# ------------------------------------------------------------------ item 20: false alerts and misses
out.append(sec("4.4 False alerts and misses (item 20)"))
store = F._Store()
jobs = store.read_pd("bronze", "erp_jobs")
jd = pd.to_datetime(jobs["date"])
jobs["day"] = (jd.dt.tz_convert("UTC").dt.tz_localize(None) if jd.dt.tz is not None else jd).dt.normalize()
first = jobs.groupby(["machine_id", "program"])["day"].min().rename("first").reset_index()
jobs = jobs.merge(first, on=["machine_id", "program"]).sort_values(["machine_id", "day"])
JOBS = {m: g[["day", "first"]].reset_index(drop=True) for m, g in jobs.groupby("machine_id")}
ben = pd.read_parquet(REPO / "runs" / "run09_20261030T090522Z_3eced8fb" / "truth" / "benign.parquet")
for c in ("ts", "end"):
    ben[c] = pd.to_datetime(ben[c], utc=True)
segs = cb("ml_segments")
segs["segment_start"] = pd.to_datetime(segs["segment_start"], utc=True)


def context(asset: str, t: pd.Timestamp) -> str:
    mach = asset[:6]
    s = segs[(segs["asset_id"] == asset) & (segs["segment_start"] <= t) & (segs["segment_start"] > t - 14 * DAY) & (segs["reason"] != "install")]
    if len(s):
        return "after re-centering (14 days)"
    hit = (ben["asset_id"] == mach) | (ben["asset_id"] == asset) | ben["asset_id"].astype(str).str.startswith(asset)
    if asset.startswith("CMP"):
        hit |= ben["event"] == "compressor_room_heat"
    b = ben[hit & (ben["ts"] <= t) & (ben["end"].fillna(ben["ts"] + DAY) > t)]
    if len(b):
        return "benign event"
    if asset.startswith("CMP"):
        return "other (compressor)"
    day = t.tz_convert("UTC").tz_localize(None).normalize()
    g = JOBS.get(mach)
    if g is None:
        return "other (no job record)"
    k = g["day"].searchsorted(day, side="right") - 1
    if k < 0:
        return "other (no job record)"
    return "new program (first week on the machine)" if (day - g.at[k, "first"]) < pd.Timedelta(days=7) else "known program"


sc = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
sc = sc[(sc["hour_utc"] >= SPAN[0]) & (sc["hour_utc"] < SPAN[1]) & (sc["state"] == "scored")].reset_index(drop=True)
healthy = np.ones(len(sc), dtype=bool)
for r in c43.itertuples():
    healthy &= ~((sc["asset_id"] == r.asset_key) & (sc["hour_utc"] >= r.credit_ts - 120 * DAY) & (sc["hour_utc"] < r.credit_ts + DAY)).to_numpy()
hsamp = sc[healthy][["asset_id", "hour_utc"]].drop_duplicates().sample(n=min(4000, healthy.sum()), random_state=7)
hctx = pd.Series([context(a, t) for a, t in zip(hsamp["asset_id"], hsamp["hour_utc"], strict=True)]).value_counts(normalize=True)
fx_rows = []
for m in METH:
    lev = LEV[m]
    av = r2.alert_view(lev)
    ct = res[("E7", m)]["ct"]
    credited = {i for i, s in zip(ct["credited_episode"], ct["strict"], strict=True) if s and pd.notna(i)}
    fe = [r for i, r in zip(av.index, av.itertuples(), strict=True) if SPAN[0] <= r.opened < SPAN[1] and i not in credited and not Q.present_between(deg_a.get(r.asset_id), r.opened, r.closed)]
    cx = pd.Series([context(r.asset_id, r.opened) for r in fe]).value_counts()
    for k, n in cx.items():
        fx_rows.append({"method": m, "context at opening": k, "false ALERT episodes": int(n), "share of false": n / len(fe), "share of healthy hours": hctx.get(k, 0.0)})
    if m == "ML Model":
        fem = fe
    byasset = pd.Series([r.asset_id for r in fe]).value_counts()
    fx_rows.append({"method": m, "context at opening": f"largest single asset: {byasset.index[0]}", "false ALERT episodes": int(byasset.iloc[0]), "share of false": byasset.iloc[0] / len(fe), "share of healthy hours": np.nan})
out.append("False alerts by context (headline, truth-based false episodes):\n\n" + md(pd.DataFrame(fx_rows)))
out.append(note("context at the episode's opening, first match: a re-centering (repair or remount segment start) on the asset in the 14 days before; a benign event active on the machine or asset (truth benign table: heavy-cut programs, titanium job, coolant drift, sensor remount and others); compressors; otherwise the machine's ERP job that day, new if the program's first job on the machine was under 7 days earlier; healthy-hour shares from 4,000 sampled scored asset-hours outside 120 days before to 1 day after any confirmed condition"))
yr_seg = segs[(segs["segment_start"] >= SPAN[0]) & (segs["segment_start"] < SPAN[1]) & (segs["reason"] != "install")]
rec_false = sum(1 for r in fem if ((yr_seg["asset_id"] == r.asset_id) & (yr_seg["segment_start"] <= r.opened) & (yr_seg["segment_start"] > r.opened - 14 * DAY)).any())
pf = yr_seg[yr_seg["plausibility_flag"].astype(bool)]
real = []
for r in pf.itertuples():
    c = c43[(c43["asset_key"] == r.asset_id) & (c43["credit_ts"] >= r.segment_start) & (c43["credit_ts"] < r.segment_start + 120 * DAY)]
    real.append({"point": r.point_id, "re-centered": loc(r.segment_start), "reason": r.reason, "condition confirmed within 120 days": ", ".join(c["wo_id"]) or "none"})
out.append(f"Post-repair re-centerings in the year: {len(yr_seg)} segments on {yr_seg['asset_id'].nunique()} assets; ML Model false ALERT episodes opened within 14 days after one: {rec_false} of {len(fem)}. Post-repair inspection (plausibility) flags raised: {len(pf)}; found real (a confirmed condition on the asset within 120 days): {sum(1 for x in real if x['condition confirmed within 120 days'] != 'none')}.\n\n" + (md(pd.DataFrame(real)) if real else ""))
hh = sc[healthy].copy()
dg = np.array([Q.present_at(deg_p.get(p), t) for p, t in zip(hh["point_id"], hh["hour_utc"], strict=True)])
hh = hh[~dg]
hh["month"] = hh["hour_utc"].dt.tz_convert("America/Chicago").dt.strftime("%Y-%m")
hh["asset class"] = hh["asset_id"].map(severity.asset_class)
hh["above"] = hh["score"] > hh["threshold"]
out.append("Healthy evaluation hours above the threshold (share), by month and asset class:\n\n" + md(hh.pivot_table(index="month", columns="asset class", values="above", aggfunc="mean").round(4), index=True))
out.append(note("healthy = scored point-hours with no truth degradation on the point and no confirmed condition on the asset from 120 days before to 1 day after"))
ec = cb("eval_conditions").set_index("wo_id")
miss = []
for c, s in zip(cp.itertuples(), ctm["strict"], strict=True):
    if s:
        continue
    r = ctm.set_index("wo_id").loc[c.wo_id]
    cat = f"caught, removed by E5 ({str(r['class']).split(': ', 1)[-1]})" if bool(r["caught"]) else miss_reason(K["ml"], c.asset_key, c.credit_ts, ec.at[c.wo_id, "ml_miss_reason"])
    miss.append({"wo_id": c.wo_id, "asset": c.asset_key, "condition type": c.condition_type, "category": cat})
mi = pd.DataFrame(miss)
out.append(f"The {len(mi)} ML Model misses (headline) by category:\n\n" + md(mi.groupby("category").size().rename("misses").reset_index()) + md(mi))

# ------------------------------------------------------------------ item 21
out.append(sec("4.5 Limitation figures (item 21)"))
c51 = conds51()
inst = c51[c51["instrumented"]]
lim = [("confirmed conditions in the evaluation year", len(c51)), ("on machines without sensors", int((~c51["instrumented"]).sum())),
       ("on instrumented assets but out of the comparison scope (no sensor precursor, oil cooler fouling, rotary-union leaks without ingress)", int(len(inst) - len(cp))),
       ("conditions the sensors cannot see (sum of the two)", int(len(c51) - len(cp))), ("share of all confirmed conditions", (len(c51) - len(cp)) / len(c51))]
out.append(md(pd.DataFrame(lim, columns=["figure", "value"])))
out.append("CMMS failure-code error rate on unplanned repairs (run09 realism check 10, record): wrong code 9 of 60 (15%; configured 12%, binomial p = 0.43); missing code 6 of 60 (10%; configured 15%, p = 0.37).\n\n")
(D / "s4.md").write_text("".join(out), encoding="utf-8")
print("done")
