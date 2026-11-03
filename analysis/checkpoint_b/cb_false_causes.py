"""Checkpoint (b) rev2: the evaluation-year false ALERT episodes of both methods (truth-based, as in the
headline) classified by primary cause, rules applied in order: post-repair re-centering; healthy level outside
the training range (the leading input outside its 5th to 95th percentile training range for most of the
episode, opening to close, or the point above its threshold on more than half of its healthy hours that month,
or led by the temperature residual); operating state outside training (load and speed combination with fewer
than 24 training hours on the point, for most of the alert hours); for Vendor Rules, a load regime under 10%
of the point's latest vendor baseline window for most of the alert hours; transient (alert span under 24 hours);
an unrelated repair on the asset while open; unexplained. Writes build/report_data/false_causes.parquet."""

import pickle

import diag_folds as F_
import numpy as np
import pandas as pd
import r19_common as R
import r21_common as Q
from cb_common import METH, REPO, core

from pdm import ml_tasks as M
from pdm import splits
from pdm.config import load_config
from pdm.evaluation import rev2 as r2
from pdm.models import nbm

OUT = REPO / "build" / "report_data"
K = core()
SPAN, P2A = K["SPAN"], K["P2A"]
LEV = {"ML Model": K["ml"], "Vendor Rules": K["vd"]}
res = K["res"]
_, deg_a = Q.degraded_hours(tuple(sorted(P2A.items())))
deg_p, _ = Q.degraded_hours(tuple(sorted(P2A.items())))

# the false ALERT episodes exactly as the headline counts them
FE = {}
for m in METH:
    av = r2.alert_view(LEV[m])
    ct = res[("E7", m)]["ct"]
    credited = {i for i, s in zip(ct["credited_episode"], ct["strict"], strict=True) if s and pd.notna(i)}
    keep = [
        i
        for i, r in zip(av.index, av.itertuples(), strict=True)
        if SPAN[0] <= r.opened < SPAN[1]
        and i not in credited
        and not Q.present_between(deg_a.get(r.asset_id), r.opened, r.closed)
    ]
    FE[m] = LEV[m].loc[keep].reset_index(drop=True)
    print(m, "false episodes", len(keep), flush=True)

store = F_._Store()
cfg = load_config()
B = splits.boundaries(cfg)
cut = splits.training_cutoff(cfg)
model = pickle.loads((R.S / "diag22" / "pipeline_model.pkl").read_bytes())
it = M.load_inputs(store, cfg, B["train"][1])
args = {a["point_id"]: a for a in M._point_args(it, cfg, cut)}
inp = M.load_inputs(store, cfg, B["shadow"][1])
start = B["shadow"][0] - pd.Timedelta(days=150)

# production scores (state, score, threshold, contributors) and segments
sc = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
sg = pd.read_parquet(R.S / "diag22" / "prod_A_segs.parquet")
for c in ("segment_start", "level_ready"):
    sg[c] = pd.to_datetime(sg[c], utc=True)

# standardized inputs: healthy training hours (the model's own training mask) and the evaluation year
train_rng, ztab, okey_train, ztr_all = {}, [], {}, {}
for pid, pm in sorted(model.models.items()):
    a = args[pid]
    s_tr, _ = nbm.score_point(
        pm,
        a["rows"],
        a["windows"],
        model.cfg,
        pm.info["fit_start"],
        cut,
        1.0,
        0,
        a["ambient"],
        keep_inputs=True,
    )
    healthy = (s_tr["state"] == "scored").to_numpy() & nbm.training_mask(
        s_tr["hour_utc"], cut, a["excluded"], pm.info["fit_start"]
    )
    h = s_tr[healthy]
    train_rng[pid] = {
        c: (np.nanquantile(h[f"z:{c}"], 0.05), np.nanquantile(h[f"z:{c}"], 0.95)) for c in pm.columns
    }
    okey_train[pid] = pd.DataFrame(
        {
            "hour_utc": pd.to_datetime(h["hour_utc"], utc=True),
            "okey": h["regime"].astype(str) + "|" + h["speed_band"].astype(str),
        }
    )
    rows = inp["ind"][inp["ind"]["point_id"] == pid]
    w = inp["win"][inp["win"]["point_id"] == pid]
    amb = inp["ambient"] if str(pm.point_class).startswith("compressor") else None
    s_ev, _ = nbm.score_point(
        pm, rows, w, model.cfg, start, B["shadow"][1], model.plausibility_factor, 3, amb, keep_inputs=True
    )
    s_ev = s_ev[pd.to_datetime(s_ev["hour_utc"], utc=True) >= SPAN[0]]
    ztab.append(
        s_ev[
            ["point_id", "hour_utc", "score", "regime", "speed_band", *[f"z:{c}" for c in pm.columns]]
        ].assign(hour_utc=lambda d: pd.to_datetime(d["hour_utc"], utc=True))
    )
    print(pid, "train healthy", int(healthy.sum()), flush=True)
z = pd.concat(ztab, ignore_index=True)
chk = sc.merge(z[["point_id", "hour_utc", "score"]], on=["point_id", "hour_utc"], suffixes=("", "_z"))
d = (chk["score"] - chk["score_z"]).abs()
print(
    "score check: rows",
    len(chk),
    "max abs diff where both finite",
    float(np.nanmax(d)),
    "rows differing >1e-9",
    int((d > 1e-9).sum()),
    flush=True,
)

# program per machine-hour (most active seconds) for HMC points
ch = pd.read_parquet(
    REPO / "data" / "local_gold" / "silver" / "controller_hourly",
    columns=["machine_id", "hour_utc", "program", "active_s"],
)
ch["hour_utc"] = pd.to_datetime(ch["hour_utc"], utc=True)
ch = (
    ch[ch["machine_id"].str.startswith("HMC")]
    .sort_values("active_s")
    .drop_duplicates(["machine_id", "hour_utc"], keep="last")
)
prog = ch.set_index(["machine_id", "hour_utc"])["program"]

# training-hour counts per point by operating key and by program
okc = {p: v["okey"].value_counts() for p, v in okey_train.items()}
pgc = {}
for p, v in okey_train.items():
    if p.startswith("HMC"):
        mi = pd.MultiIndex.from_arrays([[p[:6]] * len(v), v["hour_utc"]])
        pgc[p] = pd.Series(prog.reindex(mi).to_numpy()).value_counts()

# post-repair windows: ML segments from a repair (S1 on component, C4 other component), start to level ready
segs_ev = sg[
    sg["reason"].astype(str).str.startswith("repair")
    & (sg["segment_start"] >= SPAN[0] - pd.Timedelta(days=60))
]
win = inp["win"].copy()
wo = inp["wo"]
win = nbm.component_repairs(win, wo)
vwin = win[win["reason"].astype(str).str.startswith("repair")]
FACTS = {
    "FE": FE,
    "z": z,
    "train_rng": train_rng,
    "okc": okc,
    "pgc": pgc,
    "prog": prog,
    "segs_ev": segs_ev,
    "vwin": vwin,
    "wo": wo,
    "vendor": inp["vendor"],
    "deg_p": deg_p,
}

F = FACTS
K = core()
SPAN = K["SPAN"]
P2A = K["P2A"]

H = pd.Timedelta(hours=1)
TZ = "America/Chicago"
N = 24
sc = pd.read_parquet(R.S / "diag22" / "prod_A_scores.parquet")
sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
sc = sc.merge(F["z"].drop(columns="score"), on=["point_id", "hour_utc"], how="left")
by_pt = {p: g.set_index("hour_utc").sort_index() for p, g in sc.groupby("point_id")}
zev = {p: g.set_index("hour_utc").sort_index() for p, g in F["z"].groupby("point_id")}
rep = F["wo"][F["wo"]["wo_type"].isin(("unplanned_repair", "condition_based", "corrective"))]


def first(t):
    n = [x for x in str(t).split(",") if x and not x.startswith(("regime_", "band_"))]
    return n[0] if n else None


def lead(rows):
    f = rows["top_contributors"].map(first).dropna()
    return f.mode().iat[0] if len(f) else None


GROUP = {
    "vel_rms_mm_s": "Velocity RMS",
    "hf_rms_g": "HF acceleration",
    "crest_factor": "Crest factor, kurtosis",
    "kurtosis": "Crest factor, kurtosis",
    "temp_resid_c": "Temperature residual",
    "ord_01_g2": "1x, 2x order bands",
    "ord_02_g2": "1x, 2x order bands",
}


def group(c):
    return GROUP.get(c, "Envelope fault bands" if str(c).startswith("env_") else str(c))


def vgroup(rules):
    r = rules.split(",")
    for k, g in (
        ("envelope", "Envelope fault bands"),
        ("zone_", "ISO zone (velocity)"),
        ("temp", "Temperature limit"),
        ("vel_", "Velocity RMS"),
        ("hf_", "HF acceleration"),
    ):
        if any(x.startswith(k) for x in r):
            return g
    return rules


def rare_key(pid, hours):
    if not len(hours):
        return np.nan
    e = zev[pid].reindex(hours)
    key = e["regime"].astype(str) + "|" + e["speed_band"].astype(str)
    return float(np.mean([F["okc"][pid].get(k, 0) < N for k in key]))


def other_repair(asset, points, opened, closed):
    r = rep[rep["asset_id"].str.startswith(asset[:6]) & (rep["opened"] >= opened) & (rep["opened"] <= closed)]
    kind = nbm.point_kind(points[0])
    return [
        f"{x.wo_id} {x.asset_id} {x.failure_code}"
        for x in r.itertuples()
        if not (x.asset_id == asset and any(k in str(x.parts) for k in nbm.COMPONENT_PARTS[kind]))
    ]


def post_repair(points, opened, method):
    if method == "ML Model":
        s = F["segs_ev"]
        s = s[s["point_id"].isin(points) & (s["segment_start"] <= opened) & (s["level_ready"] >= opened)]
        return s["reason"].iat[0] if len(s) else None
    w = F["vwin"]
    w = w[w["point_id"].isin(points) & (w["start"] <= opened) & (w["end"] >= opened)]
    return w["reason"].iat[0] if len(w) else None


# ML Model: every evaluation-year episode (false or not)
ct = K["res"][("E7", "ML Model")]["ct"]
cred = {}
for r in ct.itertuples():
    if r.caught and pd.notna(r.credited_episode):
        cred.setdefault(int(r.credited_episode), []).append(
            f"{r.wo_id} ({'counted' if r.strict else 'removed by E5'})"
        )
ml = K["ml"]
false_idx = set()
for i, r in ml.iterrows():
    if (
        SPAN[0] <= r.opened < SPAN[1]
        and not any(
            i == int(c.credited_episode) and c.strict for c in ct.itertuples() if pd.notna(c.credited_episode)
        )
        and not Q.present_between(deg_a.get(r.asset_id), r.opened, r.closed)
    ):
        false_idx.add(i)
assert len(false_idx) == 47
rows = []
for i, e in ml.iterrows():
    if not (e.closed > SPAN[0] and e.opened < SPAN[1]):
        continue
    pts = str(e.trigger_points).split(",")
    o, c = max(e.opened, SPAN[0]), min(e.closed, SPAN[1])
    ab = {p: by_pt[p].loc[e.opened : e["last"]] for p in pts}
    ab = {p: g[(g["state"] == "scored") & (g["score"] > g["threshold"])] for p, g in ab.items()}
    lp = max(ab, key=lambda p: len(ab[p]))
    li = lead(ab[lp]) if len(ab[lp]) else None
    dg = deg_p.get(lp, pd.DatetimeIndex([]))
    w = by_pt[lp].loc[e.opened : e.closed]
    w = w[(w["state"] == "scored") & ~w.index.isin(dg)]
    if li is not None and len(w):
        lo, hi = F["train_rng"][lp][li]
        v = w[f"z:{li}"]
        out = float(((v < lo) | (v > hi)).mean())
        below = float((v < lo).mean())
    else:
        out = below = np.nan
    m0 = e.opened.tz_convert(TZ).to_period("M")
    mm = by_pt[lp]
    mm = mm[(mm.index.tz_convert(TZ).to_period("M") == m0) & (mm["state"] == "scored") & ~mm.index.isin(dg)]
    month_above = float((mm["score"] > mm["threshold"]).mean()) if len(mm) else np.nan
    rows.append(
        {
            "idx": i,
            "asset": e.asset_id,
            "points": e.trigger_points,
            "lead_point": lp,
            "opened": e.opened,
            "last": e["last"],
            "closed": e.closed,
            "alert_hours": (c - o) / H,
            "span_h": (e["last"] - e.opened) / H + 1,
            "lead_input": li,
            "group": group(li) if li else "",
            "out_healthy": out,
            "below_healthy": below,
            "healthy_h": len(w),
            "month_above": month_above,
            "false": i in false_idx,
            "credited": "; ".join(cred.get(i, [])),
            "post_repair": post_repair(pts, e.opened, "ML Model"),
            "rare_key": rare_key(lp, ab[lp].index),
            "other_repair": "; ".join(other_repair(e.asset_id, pts, e.opened, e.closed)),
            "degraded_while_open": Q.present_between(deg_a.get(e.asset_id), e.opened, e.closed),
        }
    )
A = pd.DataFrame(rows)
A["level_out"] = (A["out_healthy"] > 0.5) | (A["month_above"] > 0.5)


def ml_class(r):
    if r.post_repair:
        return 1
    if r.level_out or r.lead_input == "temp_resid_c":
        return 2
    if r.rare_key > 0.5:
        return 3
    if r.span_h < 24:
        return 5
    if r.other_repair:
        return 6
    return 7


A["cls"] = A.apply(ml_class, axis=1)

# Vendor Rules: the 79, with the regime class after class 3
allw = pd.read_parquet(REPO / "data" / "local_gold" / "silver" / "healthy_windows")
for c in ("start", "end"):
    allw[c] = pd.to_datetime(allw[c], utc=True)
ind = pd.read_parquet(
    REPO / "data" / "local_gold" / "silver" / "hourly_indicators",
    columns=["point_id", "hour_utc", "regime", "vel_rms_mm_s"],
)
ind["hour_utc"] = pd.to_datetime(ind["hour_utc"], utc=True)
byi = {
    p: g.dropna(subset=["vel_rms_mm_s"]).set_index("hour_utc").sort_index()["regime"]
    for p, g in ind.groupby("point_id")
}
v = F["vendor"]
vr = []
for e in F["FE"]["Vendor Rules"].itertuples():
    pts = str(e.trigger_points).split(",")
    s = v[v["point_id"].isin(pts) & (v["ts"] >= e.opened) & (v["ts"] <= e.last)]
    lp = s["point_id"].mode().iat[0] if len(s) else pts[0]
    hrs = pd.DatetimeIndex(s[s["point_id"] == lp]["ts"].dt.floor("h").unique())
    ww = allw[(allw["point_id"] == lp) & (allw["start"] <= e.opened)].sort_values("start").tail(1)
    base = byi[lp].loc[ww["start"].iat[0] : ww["end"].iat[0]].value_counts(normalize=True)
    reg = byi[lp].reindex(hrs)
    thin = float(np.mean([base.get(x, 0.0) < 0.10 for x in reg])) if len(hrs) else np.nan
    vr.append(
        {
            "asset": e.asset_id,
            "lead_point": lp,
            "opened": e.opened,
            "alert_hours": (min(e.closed, SPAN[1]) - e.opened) / H,
            "span_h": (e.last - e.opened) / H + 1,
            "group": s["rules"].map(vgroup).mode().iat[0] if len(s) else "",
            "post_repair": post_repair(pts, e.opened, "Vendor Rules"),
            "rare_key": rare_key(lp, hrs),
            "thin": thin,
            "other_repair": "; ".join(other_repair(e.asset_id, pts, e.opened, e.closed)),
        }
    )
V = pd.DataFrame(vr)


def v_class(r, with_new=True):
    if r.post_repair:
        return 1
    if r.rare_key > 0.5:
        return 3
    if with_new and r.thin > 0.5:
        return 3.5
    if r.span_h < 24:
        return 5
    if r.other_repair:
        return 6
    return 7


V["cls_before"] = V.apply(lambda r: v_class(r, False), axis=1)
V["cls"] = V.apply(v_class, axis=1)

ml_rows = A[A["false"]].assign(method="ML Model")
ml_rows["cause"] = ml_rows["cls"].map(
    {
        1: "Post-repair re-centering",
        2: "Healthy level outside training range",
        3: "Operating state outside training",
        5: "Transient",
        6: "Unrelated repair reached first",
        7: "Unexplained",
    }
)
ml_rows["temperature_led"] = ml_rows["lead_input"] == "temp_resid_c"
vd_rows = V.assign(method="Vendor Rules")
vd_rows["cause"] = vd_rows["cls"].map(
    {
        1: "Post-repair re-centering",
        3: "Operating state outside training",
        3.5: "Load regime missing from baseline",
        5: "Transient",
        6: "Unrelated repair reached first",
        7: "Unexplained",
    }
)
vd_rows["temperature_led"] = False
cols = ["method", "asset", "lead_point", "opened", "alert_hours", "cause", "temperature_led"]
out = pd.concat([ml_rows[cols], vd_rows[cols]], ignore_index=True)
out.to_parquet(OUT / "false_causes.parquet", index=False)
print(
    out.groupby(["method", "cause"])
    .agg(episodes=("asset", "size"), alert_hours=("alert_hours", "sum"), points=("lead_point", "nunique"))
    .to_string()
)
