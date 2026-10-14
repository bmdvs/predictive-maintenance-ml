"""Checkpoint (b) rev2 (diag25): Section 2 items 9 to 12 (false-alert workload, ALARM, value, diagnosis).
Writes diag25/s2c.md."""

import json

import numpy as np
import pandas as pd

import diag_folds as F
import r19_common as R
import r19_eval_helpers as H
import r21_common as Q
import r2_common as C
from cb_common import DEFN, METH, REPO, D, cb, conds51, core, loc, md, note, sec
from pdm import tasks
from pdm.config import load_config
from pdm.evaluation import rev2 as r2
from pdm.models import severity

K = core()
res, cp, SPAN, AM, P2A, conf = K["res"], K["cp"], K["SPAN"], K["AM"], K["P2A"], K["conf"]
cfg = load_config()
LEV = {"ML Model": K["ml"], "Vendor Rules": K["vd"]}
wo = cb("cmms_work_orders")
ctx = tasks._context(F._Store())
MED = ctx["inspection_hours_each"]
_, deg_a = Q.degraded_hours(tuple(sorted(P2A.items())))
out = []


def false_flags(m: str) -> pd.DataFrame:
    """E2 to E7 truth-based false ALERT episodes, per episode, as evaluate3 counts them."""
    lev = LEV[m]
    av = r2.alert_view(lev)
    ct = res[("E7", m)]["ct"]
    credited = {i for i, s in zip(ct["credited_episode"], ct["strict"], strict=True) if s and pd.notna(i)}
    inside = (av["opened"] >= SPAN[0]) & (av["opened"] < SPAN[1])
    f = [bool(ins) and (idx not in credited) and not Q.present_between(deg_a.get(r.asset_id), r.opened, r.closed) for idx, ins, r in zip(av.index, inside, av.itertuples(), strict=True)]
    return av.assign(false=f, inside=inside.to_numpy())


# ------------------------------------------------------------------ item 9
out.append(sec("2.4 False-alert workload (item 9)"))
insp = wo[wo["wo_type"] == "inspection"].copy()
insp["alert_ts"] = pd.to_datetime(insp["platform_alert_id"].str[-10:], format="%Y%m%d%H", utc=True, errors="coerce")
rows, monthly = [], []
for m in METH:
    fl = false_flags(m)
    assert int(fl["false"].sum()) == res[("E7", m)]["false_truth"], (m, int(fl["false"].sum()), res[("E7", m)]["false_truth"])
    fe = fl[fl["false"]].copy()
    if m == "Vendor Rules":
        lab = []
        for r in fe.itertuples():
            w = insp[(insp["asset_id"].str.startswith(r.asset_id)) & insp["alert_ts"].notna() & (insp["alert_ts"] >= r.opened) & (insp["alert_ts"] < r.closed)]
            lab.append(float(w["labor_hours"].sum()))
        fe["actual_h"] = lab
    unrel = r2.unrelated_work(fe, wo)
    rows.append({"method": m, "false ALERT episodes (truth-based)": len(fe), "per asset-month": len(fe) / AM,
                 f"inspection hours at the median labor ({MED:.2f} h each)": len(fe) * MED,
                 "inspection hours at actual CMMS labor": fe["actual_h"].sum() if m == "Vendor Rules" else "",
                 "ended by an unrelated repair or PM first": int(unrel.sum()), "share": unrel.mean() if len(fe) else np.nan})
    fe["month"] = fe["opened"].dt.tz_convert("America/Chicago").dt.strftime("%Y-%m")
    g = fe.groupby("month").agg(episodes=("false", "size"), **({"actual_h": ("actual_h", "sum")} if m == "Vendor Rules" else {}))
    g["median_h"] = g["episodes"] * MED
    for mo, r in g.iterrows():
        monthly.append({"method": m, "month": mo, "false episodes": int(r["episodes"]), "hours (median labor)": r["median_h"], "hours (actual CMMS)": r.get("actual_h", np.nan)})
out.append(md(pd.DataFrame(rows)))
avail = 3 * 2080 / 12
mo = pd.DataFrame(monthly)
pk = []
for m in METH:
    x = mo[mo["method"] == m]
    for col in ("hours (median labor)", "hours (actual CMMS)"):
        if x[col].notna().any():
            r = x.loc[x[col].idxmax()]
            pk.append({"method": m, "basis": "median inspection labor" if "median" in col else "actual CMMS labor", "peak month": r["month"], "hours": r[col], "share of three technicians' available hours": r[col] / avail})
out.append("Peak month:\n\n" + md(pd.DataFrame(pk)))
pv = mo.pivot_table(index="month", columns="method", values=["false episodes", "hours (median labor)"], aggfunc="sum").fillna(0)
pv.columns = [f"{a}, {b}" for a, b in pv.columns]
pv["hours (actual CMMS), Vendor Rules"] = mo[mo["method"] == "Vendor Rules"].set_index("month")["hours (actual CMMS)"]
out.append("By month:\n\n" + md(pv.round(2), index=True))
out.append(note(f"false episodes as in 2.1 (E2 to E7, truth-based); median inspection labor = median labor of all CMMS inspection work orders (tasks._context, {MED:.3f} h); actual CMMS labor = labor of the inspection work orders raised from Vendor Rules alerts inside the episode (platform alert id); available hours = 3 technicians x 2,080 h / 12 = {avail:.0f} h a month; unrelated first = rev2.unrelated_work (a repair, rebuild or PM reached the asset while the episode was open)"))

# ------------------------------------------------------------------ item 10: ALARM
vr = pd.DataFrame(rows).set_index("method")
v_act = float(vr.at["Vendor Rules", "inspection hours at actual CMMS labor"])
n_v, n_m = int(vr.at["Vendor Rules", "false ALERT episodes (truth-based)"]), int(vr.at["ML Model", "false ALERT episodes (truth-based)"])
per = v_act / n_v
mm = mo[mo["method"] == "ML Model"].set_index("month")["false episodes"] * per
vm = mo[mo["method"] == "Vendor Rules"].set_index("month")["hours (actual CMMS)"]
rs = [{"method": "Vendor Rules", "false alerts": n_v, "inspection hours (actual CMMS labor)": v_act, "per false alert (h)": per, "peak month": vm.idxmax(), "peak-month hours": vm.max(), "share of available hours": vm.max() / avail},
      {"method": "ML Model", "false alerts": n_m, "inspection hours (actual CMMS labor)": n_m * per, "per false alert (h)": per, "peak month": mm.idxmax(), "peak-month hours": mm.max(), "share of available hours": mm.max() / avail}]
out.append("Restated at Vendor Rules' actual CMMS labor per false alert (correction item 5):\n\n" + md(pd.DataFrame(rs)))
out.append(note(f"ML Model hours = its false alerts x Vendor Rules' actual CMMS inspection labor per false alert ({v_act:.1f} h / {n_v} = {per:.3f} h); peak month on that basis; available hours 520 a month"))
out.append(sec("2.5 ALARM (item 10)"))
c43 = conds51()
c43 = c43[c43["instrumented"]].reset_index(drop=True)
own = [H.own_start(c.condition_type, c.asset_key, c.credit_ts, P2A) for c in c43.itertuples()]
ons = C.truth_onsets(c43).set_index("wo_id")["truth_onset"]
c43["onset"] = [o if pd.notna(o) else ons.get(w, pd.NaT) for o, w in zip(own, c43["wo_id"], strict=True)]
c43["in_scope"] = c43["wo_id"].isin(set(cp["wo_id"]))


def alarms(m: str, sensor: bool) -> pd.DataFrame:
    e = LEV[m]
    col = "alarm_sensor_ts" if sensor else "alarm_ts"
    a = e[e[col].notna() & (e[col] >= SPAN[0]) & (e[col] < SPAN[1])].copy()
    a["start"] = a[col]
    has, wos = [], []
    for r in a.itertuples():
        c = c43[(c43["asset_key"] == r.asset_id) & (c43["credit_ts"] >= r.start) & c43["onset"].notna()]
        c = c[(c["onset"] < r.closed) & (c["credit_ts"] >= r.start)]
        has.append(len(c) > 0)
        wos.append(",".join(c["wo_id"]))
    a["has_condition"], a["conditions"] = has, wos
    return a


rows = []
AL = {}
for m in METH:
    for sensor in (False, True):
        if m == "Vendor Rules" and sensor:
            continue
        a = alarms(m, sensor)
        AL[(m, sensor)] = a
        fa = r2.alarm_outcomes(LEV[m], conf, wo, SPAN, SPAN[1], False, sensor)
        rows.append({"method": m, "criteria": "sensor criteria only" if sensor else ("all criteria, including controller" if m == "ML Model" else "alarm level (sensor)"),
                     "ALARMs": len(a), "with a confirmed condition": int(a["has_condition"].sum()), "share": a["has_condition"].mean() if len(a) else np.nan,
                     "without (per asset-month)": (~a["has_condition"]).sum() / AM,
                     "false ALARMs as coded (R3.10)": int(fa["false_alarm"].sum()), "per asset-month": fa["false_alarm"].sum() / AM})
out.append(md(pd.DataFrame(rows)))
out.append(note("an ALARM has a confirmed condition when a condition confirmed on the asset at or after the ALARM start (any of the 43 confirmed conditions on instrumented assets: inspection, PM finding or failure) had its onset (own start, else truth onset) before the episode's close; false ALARMs as coded (R3.10, rev2.alarm_outcomes): no confirmation on the asset from the ALARM start to the episode close, pending at the record end not false. ALARMs that started in the evaluation year. The Vendor Rules ALARM is their alarm level (sensor only)"))
# warning on conditions both escalated
w_rows = []
for c in c43[c43["in_scope"]].itertuples():
    r = {"wo_id": c.wo_id, "asset": c.asset_key, "condition type": c.condition_type, "confirmed": loc(c.credit_ts)}
    for m in METH:
        a = AL[(m, False)]
        a = a[(a["asset_id"] == c.asset_key) & a["conditions"].str.contains(c.wo_id)]
        r[m] = max(((c.credit_ts - max(x.start, c.onset)) / pd.Timedelta(days=1) for x in a.itertuples()), default=np.nan)
    w_rows.append(r)
wd = pd.DataFrame(w_rows)
both = wd.dropna(subset=list(METH))
out.append(f"Median ALARM warning (days) on the conditions both methods escalated: n = {len(both)}; ML Model {both['ML Model'].median():.1f}, Vendor Rules {both['Vendor Rules'].median():.1f}.\n\n")
out.append(md(both))
out.append(note("ALARM warning = confirmation minus the later of the ALARM start and the condition's onset; per condition the earliest qualifying ALARM; in-scope conditions (32); ML Model on all criteria"))
ac = cb("alarm_criteria")
ac = ac[(ac["hour_utc"] >= SPAN[0]) & (ac["hour_utc"] < SPAN[1])]
ac["asset_class"] = ac["asset_id"].map(severity.asset_class)
t = ac.groupby(["asset_class", "criterion"]).agg(hours=("hour_utc", "size"), assets=("asset_id", "nunique")).reset_index()
cause = K["ml"][K["ml"]["alarm_ts"].notna()].assign(asset_class=lambda d: d["asset_id"].map(severity.asset_class)).groupby(["asset_class", "alarm_cause"]).size()
t["ALARMs started by it"] = [int(cause.get((a, c), 0)) for a, c in zip(t["asset_class"], t["criterion"], strict=True)]
from pdm import ml_tasks as M

t["signal"] = ["controller" if c in M.CONTROLLER_CRITERIA else "sensor" for c in t["criterion"]]
t["in R3.10 (build plan 7.2 and design freeze round 3)"] = ["yes" if c in M.ALARM_CRITERIA[a] else "no" for a, c in zip(t["asset_class"], t["criterion"], strict=True)]
out.append("ALARM criteria, evaluation year (gold.alarm_criteria), checked against R3.10:\n\n" + md(t))
miss = [(a, c) for a, cs in M.ALARM_CRITERIA.items() for c in cs if not ((t["asset_class"] == a) & (t["criterion"] == c)).any()]
out.append(f"Criteria in the coded set with no hour in the year: {', '.join(f'{a}: {c}' for a, c in miss) or 'none'}.\n\n")
out.append("Temperature limit criteria: \"temperature residual\" (spindle, pumps, compressor) reads the sensor housing temperature residual (sensor); \"thermistor at limit\" (spindle) reads the controller spindle thermistor, per-minute mean 10 consecutive minutes at 44 C or above (controller); \"discharge temperature\" (compressor) reads the compressor controller discharge temperature at or above 100 C (controller). CONTROLLER_CRITERIA in ml_tasks: " + ", ".join(M.CONTROLLER_CRITERIA) + ".\n\n")

# ------------------------------------------------------------------ item 11: value
out.append(sec("2.6 Value of the change (item 11)"))
minw = float(cfg["evaluation"]["plannable_min_warning_days"])
planned = wo[wo["wo_type"] == "condition_based"].assign(cls=lambda d: d["asset_id"].map(severity.asset_class))
med_dt = planned.groupby("cls")["downtime_hours"].median().to_dict()
med_lab = planned.groupby("cls")["labor_hours"].median().to_dict()
margin = ctx["margin"]
ctm = {m: res[("E7", m)]["ct"].set_index("wo_id") for m in METH}
cpi = cp.set_index("wo_id")
vrows = []
for w in cp["wo_id"]:
    c = cpi.loc[w]
    s = {m: bool(ctm[m].at[w, "strict"]) for m in METH}
    aw = {m: float(ctm[m].at[w, "attributable_warning"]) if s[m] else np.nan for m in METH}
    pl = {m: bool(c["unplanned"]) and s[m] and aw[m] >= minw for m in METH}
    if s["ML Model"] == s["Vendor Rules"] and pl["ML Model"] == pl["Vendor Rules"]:
        continue
    cls = severity.asset_class(c["asset_key"])
    rate = margin.get(c["machine_class"], margin.get("hmc"))
    dt_saved = max(float(c["downtime_hours"]) - med_dt.get(cls, np.nan), 0.0)
    lab_saved = max(float(c["labor_hours"]) - med_lab.get(cls, np.nan), 0.0)
    sign = (1 if pl["ML Model"] else 0) - (1 if pl["Vendor Rules"] else 0)
    vrows.append({"wo_id": w, "asset": c["asset_key"], "condition type": c["condition_type"], "unplanned": bool(c["unplanned"]),
                  "ML Model warning (days)": aw["ML Model"], "Vendor Rules warning (days)": aw["Vendor Rules"],
                  "plannable under": ", ".join(m for m in METH if pl[m]) or "neither",
                  "downtime (h)": float(c["downtime_hours"]), "planned downtime (h)": med_dt.get(cls, np.nan),
                  "margin (USD/h)": rate, "signed value, downtime (USD)": sign * dt_saved * rate, "signed repair labor saved (h)": sign * lab_saved})
vt = pd.DataFrame(vrows)
fi = {m: res[("E7", m)]["false_truth"] * MED for m in METH}
tot = {"wo_id": "false-alert inspection hours (median labor)", "signed repair labor saved (h)": fi["Vendor Rules"] - fi["ML Model"]}
vt = pd.concat([vt, pd.DataFrame([tot])], ignore_index=True)
vt.loc[len(vt)] = {"wo_id": "net total", "signed value, downtime (USD)": vt["signed value, downtime (USD)"].sum(), "signed repair labor saved (h)": vt["signed repair labor saved (h)"].sum()}
out.append(md(vt))
out.append(note(f"R3.6 (rev2.value_table) applied both ways: a condition counts when it is a strict catch (E2 to E7) for one method only, or plannable under one only; plannable = unplanned and attributable warning of at least {minw:.0f} days; value = (actual downtime less the median downtime of condition-based repairs of the asset class) x the machine class contribution margin, signed + for the ML Model and - for the Vendor Rules; HMC spindle failures keep the downtime recorded (the spare-spindle case that applied); repair labor and inspection hours are in hours: the record and the config carry no labor rate, so they are not priced"))

# ------------------------------------------------------------------ item 12: diagnosis
out.append(sec("2.7 Diagnosis (item 12)"))
r24 = json.loads((R.S / "diag24" / "rule_fold1_fold2_eval.json").read_text())
out.append("D2 results as reported (diag24):\n\n" + md(pd.DataFrame(r24["summary"])))
sc24 = pd.DataFrame(json.loads((R.S / "diag24" / "score_conditions_fold1_fold2_eval.json").read_text()))
ev24 = sc24[sc24["period"] == "eval"]
for rule in ("current", "D1", "D2"):
    for lab, g in (("all catches", ev24), ("strict catches", ev24[ev24["strict"]])):
        x = pd.crosstab(g["family"], g[rule]).astype(int)
        x["matched"] = [int(x.at[f, f]) if f in x.columns else 0 for f in x.index]
        out.append(f"Evaluation year, {rule}, {lab} ({len(g)}), confirmed against suspected:\n\n" + md(x, index=True))
tsc = pd.read_parquet(REPO / "build" / "queue_cache" / "tsc_pressure.parquet")
dd = pd.to_datetime(tsc["local_date"])
tsc["day"] = (dd.dt.tz_convert("UTC").dt.tz_localize(None) if dd.dt.tz is not None else dd).dt.normalize()
rows = []
for c in cp[cp["condition_type"] == "rotary union"].itertuples():
    mch = c.asset_key[:6]
    day = c.credit_ts.tz_convert("America/Chicago").tz_localize(None).normalize()
    g = tsc[tsc["machine_id"] == mch].set_index("day").sort_index()
    pre = g[(g.index < day) & (g.index >= day - pd.Timedelta(days=7))]
    base = g[(g.index < day - pd.Timedelta(days=30)) & (g.index >= day - pd.Timedelta(days=60))]
    ref = pre["reference_median_bar"].dropna()
    rows.append(
        {
            "wo_id": c.wo_id,
            "asset": c.asset_key,
            "failure code": c.failure_code,
            "confirmed": loc(c.credit_ts),
            "ML Model": "caught" if ctm["ML Model"].at[c.wo_id, "strict"] else "missed",
            "Vendor Rules": "caught" if ctm["Vendor Rules"].at[c.wo_id, "strict"] else "missed",
            "pump-on pressure, 7 days before (bar)": pre["pressure_median_bar"].median(),
            "30 to 60 days before (bar)": base["pressure_median_bar"].median(),
            "reference (bar)": ref.iloc[-1] if len(ref) else np.nan,
            "change vs reference (%)": 100 * (pre["pressure_median_bar"].median() / ref.iloc[-1] - 1) if len(ref) else np.nan,
            "largest daily drop share, 30 days before": g[(g.index < day) & (g.index >= day - pd.Timedelta(days=30))]["drop_share"].max(),
        }
    )
out.append("Rotary-union conditions in scope (coolant ingress): TSC pump-on pressure before confirmation:\n\n" + md(pd.DataFrame(rows)))
out.append(note("gold.tsc_pressure_daily (median pump-on pressure per machine and local day; reference_median_bar and drop_share as the advisory diagnosis reads them); catches are strict catches under E2 to E7"))
(D / "s2c.md").write_text("".join(out), encoding="utf-8")
print("done")
