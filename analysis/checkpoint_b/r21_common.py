"""Scratch (run tag diag21): evaluation under three definitions, identical for every method:
'orig' (original + E2, E3), 'E4' (E2 to E4), 'E7' (E2 to E7: E4 plus E5 strict catches, E6 attributable warning,
E7 truth-based false alerts). Degradation: any truth-degraded field on any point of the asset except cavitation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import diag_lib as L
import r19_common as R
import r19_eval_helpers as H
from pdm.evaluation import events as ev
from pdm.evaluation import rev2 as r2

DEFS3 = ("E7", "E4", "orig")
DN = {"E7": "E2 to E7", "E4": "E2 to E4", "orig": "original + E2, E3"}
H7 = pd.Timedelta(days=7)


@lru_cache(maxsize=1)
def degraded_hours(p2a_items: tuple) -> tuple[dict, dict]:
    p2a = dict(p2a_items)
    th = L.truth_hours()
    d = (th["stage"].fillna(0) > 0) | (th["lube_severity"].fillna(0) > 0) | (th["wear_frac"].fillna(0) > 0) | (th["fault_level"].fillna(0) > 0) | (th["ingress_level"].fillna(0) > 0)
    g = th.loc[d, ["point_id", "hour_utc"]].copy()
    g["asset"] = g["point_id"].map(p2a)
    by_point = {p: pd.DatetimeIndex(x["hour_utc"]).sort_values() for p, x in g.groupby("point_id")}
    by_asset = {a: pd.DatetimeIndex(x["hour_utc"].unique()).sort_values() for a, x in g.groupby("asset")}
    return by_point, by_asset


def present_at(idx: pd.DatetimeIndex | None, t: pd.Timestamp) -> bool:
    if idx is None or not len(idx):
        return False
    h = pd.Timestamp(t).floor("h")
    k = idx.searchsorted(h)
    return bool(k < len(idx) and idx[k] == h)


def present_between(idx: pd.DatetimeIndex | None, a: pd.Timestamp, b: pd.Timestamp) -> bool:
    if idx is None or not len(idx):
        return False
    k = idx.searchsorted(pd.Timestamp(a).floor("h"))
    return bool(k < len(idx) and idx[k] < b)


def evaluate3(lev: pd.DataFrame, cp: pd.DataFrame, conf: pd.DataFrame, wo: pd.DataFrame, span, end, definition: str, p2a: dict, alarm: bool = True) -> dict:
    """Catches, warning, false episodes (confirmed-only and truth-based), alert-hours under one definition."""
    _, deg_a = degraded_hours(tuple(sorted(p2a.items())))
    av = r2.alert_view(lev)
    base = "orig" if definition == "orig" else "E4"
    m = R.evaluate(lev, cp, conf, wo, span, end, base, alarm=alarm)
    ct = m["ct"].copy()
    ct["asset_key"] = cp["asset_key"].to_numpy()
    ct["credit_ts"] = cp["credit_ts"].to_numpy()
    own = [H.own_start(c.condition_type, c.asset_key, c.credit_ts, p2a) for c in cp.itertuples()]
    import r2_common as C

    ons = C.truth_onsets(cp).set_index("wo_id")["truth_onset"]
    ct["onset_ref"] = [o if pd.notna(o) else ons.get(w, pd.NaT) for o, w in zip(own, ct["wo_id"], strict=True)]
    cls, att = [], []
    for r in ct.itertuples():
        if not r.caught:
            cls.append("")
            att.append(np.nan)
            continue
        o = r.credited_opened
        dg = present_at(deg_a.get(r.asset_key), o)
        on = r.onset_ref
        if pd.isna(on):
            c = "counts: degradation present at opening (no onset)" if dg else "removed by E5: no degradation at opening, no onset"
        elif o >= on:
            c = "caught after onset"
        elif dg:
            c = "borrowed: other degradation present at opening"
        elif on - o > H7:
            c = "removed by E5: coincidence (no degradation at opening, onset more than 7 days later)"
        else:
            c = "opened within 7 days before onset, no degradation"
        cls.append(c)
        start = max(o, on) if pd.notna(on) else o
        att.append((r.credit_ts - start) / R.DAY)
    ct["class"] = cls
    ct["attributable_warning"] = att
    ct["strict"] = ct["caught"] & ~ct["class"].str.startswith("removed")
    inside = ((av["opened"] >= span[0]) & (av["opened"] < span[1])).to_numpy()
    if definition == "E7":
        credited = {i for i, s in zip(ct.get("credited_episode", pd.Series([None] * len(ct))), ct["strict"], strict=True) if s and i is not None and not (isinstance(i, float) and np.isnan(i))}
    else:
        credited = {i for i in ct.get("credited_episode", pd.Series(dtype=object)) if i is not None and not (isinstance(i, float) and np.isnan(i))}
    fconf = np.zeros(len(av), dtype=bool)
    if inside.any():
        fconf[inside] = ev.false_episodes(av[inside], conf).to_numpy()
    ftruth = np.zeros(len(av), dtype=bool)
    for j, r in enumerate(av.itertuples()):
        if not inside[j]:
            continue
        idx = av.index[j]
        if idx in credited:
            fconf[j] = False
            continue
        ftruth[j] = not present_between(deg_a.get(r.asset_id), r.opened, r.closed)
    if definition == "orig":
        ftruth[:] = False
    # alert-hours: hours with an episode open on the asset (ALERT view covers ALARM), total and without degradation
    tot = clean = 0
    for a, g in av.groupby("asset_id"):
        iv = sorted((max(x.opened, span[0]), min(x.closed, span[1])) for x in g.itertuples() if x.closed > span[0] and x.opened < span[1])
        hrs = set()
        for s0, s1 in iv:
            hrs.update(pd.date_range(pd.Timestamp(s0).floor("h"), pd.Timestamp(s1), freq="h", inclusive="left"))
        tot += len(hrs)
        dset = set(deg_a.get(a, pd.DatetimeIndex([])))
        clean += len([h for h in hrs if h not in dset])
    return {"ct": ct, "caught": int(ct["caught"].sum()), "strict": int(ct["strict"].sum()), "e5_removed": int((ct["caught"] & ~ct["strict"]).sum()) if definition == "E7" else 0,
            "false_conf": int(fconf.sum()), "false_truth": int(ftruth.sum()), "alert_hours": tot, "alert_hours_clean": clean,
            "alarm_false": m.get("alarm_false", 0), "conditions": len(cp)}


def stuck2(sc: pd.DataFrame, conds_all: pd.DataFrame, p2a: dict) -> tuple[list[str], list[str]]:
    """Stuck points: the previous definition, and the new one (only hours without degradation on the point)."""
    deg_p, _ = degraded_hours(tuple(sorted(p2a.items())))
    s = sc[sc["state"] == "scored"]
    old, new = [], []
    for pid, g in s.groupby("point_id"):
        asset = str(g["asset_id"].iat[0])
        ok = np.ones(len(g), dtype=bool)
        for t in conds_all.loc[conds_all["asset_key"] == asset, "credit_ts"]:
            ok &= ~((g["hour_utc"] >= t - pd.Timedelta(days=R.WD)) & (g["hour_utc"] < t + R.DAY)).to_numpy()
        over = (g["score"] > g["threshold"]).to_numpy()
        if ok.any() and over[ok].mean() > 0.5:
            old.append(pid)
        dp = deg_p.get(pid)
        nd = ~g["hour_utc"].dt.floor("h").isin(dp).to_numpy() if dp is not None else np.ones(len(g), dtype=bool)
        k = ok & nd
        if k.any() and over[k].mean() > 0.5:
            new.append(pid)
    return old, new
