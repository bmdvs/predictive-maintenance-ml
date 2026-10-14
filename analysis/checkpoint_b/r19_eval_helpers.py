"""Scratch (run tags diag19, diag20): per-condition table with flags (E3 onset) and stuck-point causes."""

from __future__ import annotations

import numpy as np
import pandas as pd

import diag_lib as L
import r19_common as R
from pdm.models import nbm

_th = None
_ev = None
OWN = {"rotary union": [("condition_start", "rotary_union_leak")], "spindle cooling": [("condition_start", "spindle_cooling")],
       "lubrication": [("lube_episode_start", None)], "bearing (condition-based)": [("bearing_onset", None)], "bearing (unplanned)": [("bearing_onset", None)],
       "pump seal or impeller": [("pump_wear_start", None), ("cavitation_start", None)], "imbalance": [("condition_start", "imbalance")],
       "looseness": [("condition_start", "looseness")], "misalignment": [("condition_start", "misalignment")]}


def fmt(t):
    return "" if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def _events(p2a):
    global _ev
    if _ev is None:
        e = L.events().copy()
        e["asset"] = e["asset_id"]
        m = e["point_id"].notna()
        e.loc[m, "asset"] = e.loc[m, "point_id"].map(p2a)
        c = e["component_id"].notna()
        e.loc[c, "asset"] = e.loc[c, "component_id"].map(lambda x: p2a.get(x, p2a.get(str(x) + "-F", x)))
        _ev = e
    return _ev


def truth():
    global _th
    if _th is None:
        _th = L.truth_hours().copy()
    return _th


def own_start(ctype, asset, t, p2a):
    e = _events(p2a)
    hits = []
    for name, k in OWN.get(ctype, []):
        x = e[(e["event"] == name) & (e["ts"] <= t) & (e["ts"] > t - pd.Timedelta(days=R.WD)) & ((e["asset"] == asset) | (e["asset_id"] == asset))]
        if k:
            x = x[x["kind"] == k]
        hits += list(x["ts"])
    return max(hits) if hits else pd.NaT


def per_condition(cp: pd.DataFrame, cts: dict) -> pd.DataFrame:
    import r2_common as C

    p2a = C.P2A
    onset = C.truth_onsets(cp).set_index("wo_id")["truth_onset"]
    rows = []
    for c in cp.itertuples():
        own = own_start(c.condition_type, c.asset_key, c.credit_ts, p2a)
        on_f = own if pd.notna(own) else onset.get(c.wo_id, pd.NaT)
        row = {"wo_id": c.wo_id, "asset": c.asset_key, "condition_type": c.condition_type, "confirmation": fmt(c.credit_ts), "onset for flags": fmt(on_f)}
        for name, ct in cts.items():
            t = ct.set_index("wo_id").loc[c.wo_id]
            row[f"{name} caught"] = bool(t["caught"])
            if t["caught"]:
                st = t["credited_opened"]
                lead = (on_f - st) / R.DAY if pd.notna(on_f) else np.nan
                row[f"{name} start"] = fmt(st)
                row[f"{name} warning"] = round(float(t["warning_days"]), 1)
                row[f"{name} flags"] = ",".join(x for x, ok in (("O", lead > 0), ("7", lead > 7), ("60", t["warning_days"] > 60)) if ok)
        rows.append(row)
    return pd.DataFrame(rows)


def stuck_causes(sc_all: pd.DataFrame, sg: pd.DataFrame, points: list[str], conds_all: pd.DataFrame, inp: dict) -> list[dict]:
    th = truth()
    p2a = inp["pts"].set_index("point_id")["asset_id"].to_dict()
    win_c = nbm.component_repairs(inp["win"], inp["wo"])
    parts = inp["wo"].set_index("wo_id")["parts"]
    s = sc_all[sc_all["state"] == "scored"].copy()
    s["over"] = s["score"] > s["threshold"]
    out = []
    for pid in points:
        g = s[s["point_id"] == pid].sort_values("hour_utc")
        asset = p2a[pid]
        # the stuck period from the hours outside confirmation windows only (as the stuck count)
        okw = np.ones(len(g), dtype=bool)
        for t_ in conds_all.loc[conds_all["asset_key"] == asset, "credit_ts"]:
            okw &= ~((g["hour_utc"] >= t_ - pd.Timedelta(days=R.WD)) & (g["hour_utc"] < t_ + R.DAY)).to_numpy()
        dd = g[okw].set_index("hour_utc")["over"].resample("D").mean().dropna()
        best = cur = 0
        bs = cs = None
        for i, x in enumerate((dd > 0.5).to_numpy()):
            cur = cur + 1 if x else 0
            if cur == 1:
                cs = i
            if cur > best:
                best, bs = cur, cs
        start = dd.index[bs] if bs is not None else pd.NaT
        end = dd.index[bs + best - 1] + R.DAY if bs is not None else pd.NaT
        pre = (start - 30 * R.DAY, start + 2 * R.DAY)
        sp = sg[(sg["point_id"] == pid) & (sg["segment_start"] >= pre[0]) & (sg["segment_start"] <= pre[1])]
        txt = []
        for q in sp.itertuples():
            w = win_c[(win_c["point_id"] == pid) & (pd.to_datetime(win_c["start"], utc=True) == q.segment_start)]
            oc = bool(w["on_component"].any()) if len(w) else False
            wid = str(q.reason).split(" ", 1)[1] if " " in str(q.reason) else ""
            state = "held" if bool(getattr(q, "held", False)) else ("no level" if pd.isna(q.level_ready) else ("flagged" if q.plausibility_flag else "accepted"))
            txt.append(f"{fmt(q.segment_start)} {q.reason} (parts {parts.get(wid, '') if wid else ''}; {'on' if oc else 'not on'} the measured component; level {state})")
        gg = g[(g["hour_utc"] >= start) & (g["hour_utc"] < end)]
        ok = np.ones(len(gg), dtype=bool)
        for t in conds_all.loc[conds_all["asset_key"] == asset, "credit_ts"]:
            ok &= ~((gg["hour_utc"] >= t - pd.Timedelta(days=R.WD)) & (gg["hour_utc"] < t + R.DAY)).to_numpy()
        sh = gg[ok & gg["over"].to_numpy()]
        hp = th[th["point_id"] == pid].set_index("hour_utc")["truth_healthy"]
        dp = float((~hp.reindex(sh["hour_utc"]).fillna(True).astype(bool)).mean()) if len(sh) else np.nan
        reg_b = g[(g["hour_utc"] >= start - 30 * R.DAY) & (g["hour_utc"] < start)]["regime"].value_counts(normalize=True).round(2).to_dict()
        reg_d = gg["regime"].value_counts(normalize=True).round(2).to_dict()
        out.append({"point": pid, "stuck period start": fmt(start), "days": best, "repair or re-centering in the 30 days before": "; ".join(txt) or "none",
                    "regime before to during": f"{reg_b} to {reg_d}", "start month": start.strftime("%b %Y") if pd.notna(start) else "",
                    "stuck hours outside windows": len(sh), "share truth-degraded": round(dp, 3) if pd.notna(dp) else None,
                    "top contributors": sh["top_contributors"].str.split(",").str[0].value_counts().head(3).to_dict()})
    return out
