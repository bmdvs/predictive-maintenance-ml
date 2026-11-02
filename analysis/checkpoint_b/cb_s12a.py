"""Checkpoint (b) rev2 (diag25): Section 1 (item 5) and Section 2 items 6 and 7. Writes diag25/s1.md, s2a.md."""

import json

import numpy as np
import pandas as pd

import r19_common as R
from cb_common import DEFN, METH, D, cb, conds51, core, loc, md, miss_reason, note, sec
from pdm.config import load_config

K = core()
res, cp, SPAN, AM = K["res"], K["cp"], K["SPAN"], K["AM"]
cfg = load_config()
c51 = conds51()

# ------------------------------------------------------------------ Section 1, item 5
out = [sec("Section 1", 1), sec("1.1 Confirmed conditions, July 2025 to June 2026 (item 5)")]
ex_types = cfg["evaluation"]["comparison_scope"]["exclude_condition_types"]
inst = c51[c51["instrumented"]]
groups = [
    ("in scope (E2): the comparison set", len(cp)),
    ("on machines without sensors (controller only)", int((~c51["instrumented"]).sum())),
    ("cooler faults: compressor oil cooler fouling (cooler (compressor), CMP-OILCOOL-FOUL)", int((inst["condition_type"] == "cooler (compressor)").sum())),
    ("rotary-union leaks without coolant ingress (RU-LEAK)", int(((inst["condition_type"] == "rotary union") & (inst["failure_code"] == "RU-LEAK")).sum())),
    (f"no precursor in any data: excluded condition types ({', '.join(ex_types)})", int(inst["condition_type"].isin(ex_types).sum())),
]
g = pd.DataFrame(groups, columns=["group", "conditions"])
g.loc[len(g)] = ["total", len(c51)]
out.append(md(g))
out.append(note("evaluation.comparison_scope (exclude_condition_types, exclude_failure_codes RU-LEAK) and evaluation.fixes.e2_scope (cooler (compressor) out; rotary union in scope only with SP-COOLANT-INGRESS); source silver.confirmed_conditions credited in the evaluation year"))
d = c51[~c51["instrumented"]][["wo_id", "asset_id", "condition_type", "failure_code", "machine_class"]]
out.append("Machines without sensors:\n\n" + md(d))
d = inst[~inst["wo_id"].isin(cp["wo_id"])][["wo_id", "asset_id", "condition_type", "failure_code", "wo_type"]].sort_values(["condition_type", "wo_id"])
out.append("Instrumented assets, out of scope:\n\n" + md(d))
cls = {"SP": "HMC spindle", "TSC": "TSC pump", "CP": "flood pump"}
cp2 = cp.assign(asset_class=[("compressor" if a.startswith("CMP") else cls[a.split("-")[-1]]) for a in cp["asset_key"]])
x = pd.crosstab(cp2["condition_type"], cp2["asset_class"], margins=True, margins_name="total")
out.append("The 32 in scope by condition type and asset class:\n\n" + md(x, index=True))
(D / "s1.md").write_text("".join(out), encoding="utf-8")

# ------------------------------------------------------------------ Section 2, item 6
out = [sec("Section 2", 1), sec("2.1 ALERT: catches and false alerts (item 6)")]
rows = []
for dname in ("E7", "E4", "orig"):
    for m in METH:
        r = res[(dname, m)]
        rows.append(
            {
                "definition": DEFN[dname],
                "method": m,
                "conditions": r["conditions"],
                "caught": r["caught"],
                "strict (E5)": r["strict"] if dname == "E7" else "",
                "removed by E5": r["e5_removed"] if dname == "E7" else "",
                "false episodes (confirmation-based)": r["false_conf"],
                "confirmation-based per asset-month": r["false_conf"] / AM,
                "false episodes (truth-based, E7)": r["false_truth"] if dname != "orig" else "",
                "truth-based per asset-month": r["false_truth"] / AM if dname != "orig" else "",
                "alert-hours per asset-month": r["alert_hours"] / AM,
                "of which no degradation": r["alert_hours_clean"] / AM,
            }
        )
out.append(md(pd.DataFrame(rows)))
out.append(note(f"r21_common.evaluate3 on prod gold.level_episodes (period shadow, version 13); asset-months {AM:.1f} (28 assets x 12 months); a catch is an ALERT episode open at confirmation and opened within 120 days before it (E4 adds lookback credit: an episode open in the 30 days before, opened within 120 days); E5 keeps catches whose episode opened after the condition's onset or with degradation present at opening, or within 7 days before onset; E7 false episodes are episodes with no truth degradation on the asset while open and not credited to a strict catch"))
ct = {m: res[("E7", m)]["ct"].set_index("wo_id") for m in METH}
mlc, vdc = ct["ML Model"]["strict"], ct["Vendor Rules"]["strict"]
only_ml = mlc[mlc & ~vdc.reindex(mlc.index).fillna(False)].index.tolist()
rows = []
for w in only_ml:
    v = ct["Vendor Rules"].loc[w]
    if bool(v["caught"]) and not bool(v["strict"]):
        st = f"coincidental alert: episode opened {loc(v['credited_opened'], '%Y-%m-%d %H:%M')}, {v['class']}"
    else:
        st = "no alert episode credited (none open at confirmation and opened within 120 days)"
    r = ct["ML Model"].loc[w]
    rows.append({"wo_id": w, "asset": cp.set_index("wo_id").at[w, "asset_key"], "condition type": cp.set_index("wo_id").at[w, "condition_type"],
                 "ML Model attributable warning (days)": r["attributable_warning"], "Vendor Rules": st})
out.append(sec(f"Conditions only the ML Model caught (strict, headline): {len(only_ml)}", 3) + md(pd.DataFrame(rows)))
ec = cb("eval_conditions").set_index("wo_id")
neither = [w for w in cp["wo_id"] if not mlc.get(w, False) and not vdc.get(w, False)]
d21 = json.loads((R.S / "diag21" / "eval_results.json").read_text())["detail"]["E7"]
d21_neither = {x["wo_id"] for x in d21 if not str(x["A"]).split(";")[1:2] or "missed" in str(x["A"]) or "removed" in str(x["A"])}
d21_neither = {x["wo_id"] for x in d21 if (str(x["A"]) == "missed" or "removed by E5" in str(x["A"])) and (str(x["Vendor Rules"]) == "missed" or "removed by E5" in str(x["Vendor Rules"]))}
rows = []
for w in neither:
    r = {}
    for m in METH:
        c = ct[m].loc[w]
        if bool(c["caught"]):
            r[m] = f"removed by E5: {c['class'].split(': ', 1)[-1]}"
        else:
            lv = K["ml"] if m == "ML Model" else K["vd"]
            r[m] = miss_reason(lv, cp.set_index("wo_id").at[w, "asset_key"], cp.set_index("wo_id").at[w, "credit_ts"], ec.at[w, "ml_miss_reason" if m == "ML Model" else "vendor_miss_reason"])
    rows.append({"wo_id": w, "asset": cp.set_index("wo_id").at[w, "asset_key"], "condition type": cp.set_index("wo_id").at[w, "condition_type"],
                 "confirmed": loc(cp.set_index("wo_id").at[w, "credit_ts"]), "ML Model reason": r["ML Model"], "Vendor Rules reason": r["Vendor Rules"],
                 "in the diag21 neither list": w in d21_neither})
out.append(sec(f"Conditions neither method caught (strict, headline): {len(neither)}", 3) + md(pd.DataFrame(rows)))
out.append(note("miss reasons from prod gold.eval_conditions (rev2.miss_reason, E2 to E4 basis), with the category added 2026-11-01 (inside an episode opened more than 120 days earlier: an episode on the asset open at the confirmation that opened before the catch window) checked first; others: no precursor in the inputs; precursor under the threshold; point re-centering; dropout; signature outside the inputs); for a catch removed by E5, the E5 class"))
new_vs_21 = sorted(set(neither) - d21_neither)
gone = sorted(d21_neither - set(neither))
out.append(f"Not in the diag21 neither list: {', '.join(new_vs_21) or 'none'}. In diag21's list but caught now: {', '.join(gone) or 'none'}.\n\n")

# assets with no confirmed condition
inst_all = set(K["P2A"].values())
with_cond = set(c51.loc[c51["instrumented"], "asset_key"])
quiet = sorted(inst_all - with_cond)
wo = cb("cmms_work_orders")
wo = wo[(wo["opened"] >= SPAN[0]) & (wo["opened"] < SPAN[1])]
rows = []
for a in quiet:
    r = {"asset": a}
    for m, lv in (("ML Model", K["ml"]), ("Vendor Rules", K["vd"])):
        e = lv[(lv["asset_id"] == a) & (lv["opened"] >= SPAN[0]) & (lv["opened"] < SPAN[1])]
        r[f"{m} ALERT episodes"] = len(e)
        r[f"{m} ALARMs"] = int(e["alarm_ts"].notna().sum())
    i = wo[(wo["asset_id"] == a) & (wo["wo_type"] == "inspection")]
    r["inspections (CMMS, Vendor Rules alerts)"] = len(i)
    r["no fault found"] = int(i["no_fault_found"].fillna(False).astype(bool).sum())
    r["inspection labor (h)"] = float(i["labor_hours"].sum())
    rows.append(r)
q = pd.DataFrame(rows)
q.loc[len(q)] = {"asset": "total", **{c: q[c].sum() for c in q.columns if c != "asset"}}
out.append(sec(f"Alert episodes and inspections on assets with no confirmed condition ({len(quiet)} assets)", 3) + md(q))
out.append(note("assets with no confirmed condition of any type (instrumented, 43 conditions) in the evaluation year; ALERT episodes opened in the year; inspections are CMMS inspection work orders opened in the year (the shop inspected on Vendor Rules alerts only; the ML Model ran in shadow and raised none)"))
(D / "s2a.md").write_text("".join(out), encoding="utf-8")

# ------------------------------------------------------------------ item 7: timeline exhibit data
c43 = c51[c51["instrumented"]]
wo_all = cb("cmms_work_orders")
fail_assets = set(wo_all[(wo_all["wo_type"] == "unplanned_repair") & (wo_all["opened"] >= SPAN[0]) & (wo_all["opened"] < SPAN[1]) & wo_all["asset_id"].isin(inst_all)]["asset_id"])
assets = sorted(set(c43["asset_key"]) | fail_assets)
rows = []
for a in assets:
    for m, lv in (("ML Model", K["ml"]), ("Vendor Rules", K["vd"])):
        e = lv[(lv["asset_id"] == a) & (lv["closed"] > SPAN[0]) & (lv["opened"] < SPAN[1])]
        for r in e.itertuples():
            rows.append({"asset": a, "row": f"{m} ALERT", "start": r.opened, "end": min(r.closed, SPAN[1]), "detail": ""})
            if pd.notna(r.alarm_ts):
                rows.append({"asset": a, "row": f"{m} ALARM", "start": r.alarm_ts, "end": min(r.closed, SPAN[1]), "detail": str(r.alarm_cause)})
    w = wo_all[(wo_all["asset_id"] == a) | ((wo_all["asset_id"] == "-".join(a.split("-")[:2])) & wo_all["component"].notna() & (wo_all["component"] != "AXIS"))]
    w = w[(w["opened"] >= SPAN[0]) & (w["opened"] < SPAN[1])]
    for r in w.itertuples():
        if r.wo_type == "inspection":
            kind, det = "inspection", ("no fault found" if r.no_fault_found else str(r.notes).split(": ", 1)[-1].rstrip("."))
        elif r.wo_type == "pm":
            if pd.isna(r.failure_code):
                continue
            kind, det = "PM finding", f"{r.failure_code}"
        elif r.wo_type in ("unplanned_repair",):
            kind, det = "failure", f"{r.failure_code or 'no code'}, downtime {r.downtime_hours:.0f} h"
        elif r.wo_type in ("condition_based", "corrective"):
            kind, det = "planned repair", f"{r.failure_code or 'no code'}"
        else:
            continue
        rows.append({"asset": a, "row": kind, "start": r.opened, "end": r.closed, "detail": f"{r.wo_id}: {det}"})
for r in c43.itertuples():
    rows.append({"asset": r.asset_key, "row": "confirmed condition", "start": r.credit_ts, "end": r.credit_ts, "detail": f"{r.wo_id}: {r.condition_type}{'' if r.wo_id in set(cp['wo_id']) else ' (out of scope)'}"})
tl = pd.DataFrame(rows).sort_values(["asset", "start", "row"]).reset_index(drop=True)
tl.to_csv(D / "timeline.csv", index=False)
out = [sec("2.2 Timeline exhibit data (item 7)"), f"{len(assets)} assets, {len(tl)} rows (CSV beside this report: predictive-maintenance-ml-rev2-cpb_timeline.csv). Rows per asset and kind:\n\n"]
out.append(md(pd.crosstab(tl["asset"], tl["row"]), index=True))
out.append(note("ALERT from opening to close; ALARM from the ALARM start to the episode close; inspections with their outcome; PM work orders with a finding code; unplanned repairs as failures; condition-based and corrective repairs as planned repairs; work orders on the asset or on its machine with a component"))
(D / "s2b.md").write_text("".join(out), encoding="utf-8")
print("only ML", only_ml, "neither", neither, "new vs diag21", new_vs_21, "gone", gone)
