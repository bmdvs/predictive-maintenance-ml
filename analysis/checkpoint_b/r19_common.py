"""Scratch (run tag diag19): shared scoring and evaluation for the E2-to-E4 round. Two definitions, both with
E2 (cooler out of scope) and E3 (own truth start as the onset for flags) and the 7-day close gap: 'orig' (credit
when open at the confirmation) and 'E4' (lookback credit; credited episodes not false)."""

from __future__ import annotations

import os
import copy
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import diag_lib as L
import r2_common as C
from pdm import ml_tasks as M
from pdm.evaluation import events as ev
from pdm.evaluation import rev2 as r2
from pdm.models import nbm, severity

# working folder of the analysis rounds (earlier rounds' saved outputs, diag3 to diag25), outside the repository
S = Path(os.environ.get("PDM_ANALYSIS_DIR", Path(__file__).parent))
D = S / "diag19"
WD = 120.0
LOOKBACK = 30.0
DEFS = ("orig", "E4")
DAY = pd.Timedelta(days=1)


def eval_cfg() -> dict:
    c = copy.deepcopy(L.CFG)
    c["evaluation"]["fixes"] = {**(c["evaluation"].get("fixes") or {}), "e2_scope": True, "e3_own_onset": True}
    assert c["evaluation"]["catch"]["episode_gap_days"] == 7
    return c


ECFG = eval_cfg()


def fix_cfg(cfg: dict, fixes: set[str]) -> dict:
    c = copy.deepcopy(cfg)
    c["models"]["nbm"]["fixes"] = {**nbm.FIX_DEFAULTS, **(c["models"]["nbm"].get("fixes") or {})}
    fx = c["models"]["nbm"]["fixes"]
    fx["f1_hold_level"] = "F1" in fixes or "F1-scope" in fixes
    fx["f1_scope"] = "F1-scope" in fixes
    fx["f4_high_ratio"] = "F4" in fixes or "F4-tight" in fixes
    fx["f4_ratio"] = 5.0 if "F4-tight" in fixes else 3.0
    fx["f4_exclude_pump_low_pressure"] = "F4-tight" in fixes
    return c


@lru_cache(maxsize=1)
def low_pressure_intervals() -> pd.DataFrame:
    import glob

    f = glob.glob(str(L.REPO / "data/bronze/controller/events/*.parquet"))
    e = pd.concat([pd.read_parquet(x, columns=["machine_id", "ts_utc", "event_type", "duration_s"]) for x in f], ignore_index=True)
    e = e[e["event_type"] == "coolant_low_pressure"].copy()
    e["a"] = pd.to_datetime(e["ts_utc"], utc=True).dt.floor("h")
    e["b"] = pd.to_datetime(e["ts_utc"], utc=True) + pd.to_timedelta(e["duration_s"].fillna(0), unit="s")
    return e[["machine_id", "a", "b"]]


def add_low_pressure(sc: pd.DataFrame) -> pd.DataFrame:
    lp = low_pressure_intervals()
    flag = np.zeros(len(sc), dtype=bool)
    mach = sc["point_id"].str[:6].to_numpy()
    h = sc["hour_utc"]
    for r in lp.itertuples():
        flag |= (mach == r.machine_id) & ((h >= r.a) & (h <= r.b)).to_numpy()
    return sc.assign(low_pressure=flag)


@lru_cache(maxsize=1)
def wo_parts() -> pd.DataFrame:
    return L.wo()


def score(models: dict, factor: float, cfg: dict, args_or_inp, span, keep_inputs=False, with_comp=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every point; args (folds, list of point args) or inp (evaluation year, load_inputs dict)."""
    parts, segs = [], []
    if isinstance(args_or_inp, list):
        items = [(a["point_id"], a["asset_id"], a["rows"], a["windows"], a["ambient"]) for a in args_or_inp]
    else:
        inp = args_or_inp
        p2a = inp["pts"].set_index("point_id")["asset_id"].to_dict()
        items = []
        for pid, g in inp["ind"].groupby("point_id", sort=True):
            m = models.get(pid)
            if m is None:
                continue
            amb = inp["ambient"] if str(m.point_class).startswith("compressor") else None
            items.append((pid, p2a[pid], g, inp["win"][inp["win"]["point_id"] == pid], amb))
    for pid, asset, rows, win, amb in items:
        m = models.get(pid)
        if m is None:
            continue
        if with_comp:
            win = nbm.component_repairs(win, wo_parts())
        sc, sg = nbm.score_point(m, rows, win, cfg, span[0], span[1], factor, 3, amb, keep_inputs=keep_inputs)
        if len(sc):
            parts.append(sc.assign(asset_id=asset))
        if len(sg):
            segs.append(sg.assign(asset_id=asset))
    sc = pd.concat(parts, ignore_index=True)
    sc["hour_utc"] = pd.to_datetime(sc["hour_utc"], utc=True)
    sg = pd.concat(segs, ignore_index=True)
    for c in ("segment_start", "level_ready"):
        sg[c] = pd.to_datetime(sg[c], utc=True)
    return add_low_pressure(sc), sg


def with_alarm(eps, ss, crit, cfg):
    hit = (ss["score"] > ss["threshold"]).to_numpy()
    operating = {a: pd.DataFrame({"ts": pd.to_datetime(g["hour_utc"], utc=True).to_numpy(), "point_id": g["point_id"].to_numpy(), "met": hit[g.index]})
                 for a, g in ss.groupby("asset_id")}
    days = int(cfg["models"]["severity"]["escalation_days"])
    full = severity.alarm_starts(eps, crit, operating, days)
    sensor = severity.alarm_starts(eps, crit[~crit["criterion"].isin(M.CONTROLLER_CRITERIA)], operating, days)
    return M._with_sensor(full, sensor)


def ml_lev(sc: pd.DataFrame, sg: pd.DataFrame, cfg: dict, ind: pd.DataFrame, ctrl, thr: str = "threshold", alarm: bool = True) -> pd.DataFrame:
    ss = sc[sc["state"] == "scored"].sort_values(["point_id", "hour_utc"], kind="stable").reset_index(drop=True)
    eps = M.persisted_episodes(ss, thr, 6, cfg)
    if not alarm:
        return eps
    crit = M.ml_alarm_criteria(sc, sg, ind, ctrl, ECFG)
    return with_alarm(eps, ss, crit, cfg)


def evaluate(lev: pd.DataFrame, cp: pd.DataFrame, conf: pd.DataFrame, wo: pd.DataFrame, span, end, definition: str, alarm: bool = True) -> dict:
    av = r2.alert_view(lev) if "alarm_ts" in lev or "last" in lev else lev
    if definition == "orig":
        ct = ev.catch_table(av, cp, WD)
        op = []
        for c in cp.itertuples():
            h = av[(av["asset_id"] == c.asset_key) & (av["opened"] <= c.credit_ts) & (av["closed"] > c.credit_ts) & (av["opened"] >= c.credit_ts - pd.Timedelta(days=WD))]
            op.append(h["opened"].min() if len(h) else pd.NaT)
        ct["credited_opened"] = op
        credited = set()
    else:
        ct, credited = ev.lookback_catch_table(av, cp, WD, LOOKBACK)
        ct["credited_opened"] = [av.at[i, "opened"] if i is not None and not (isinstance(i, float) and np.isnan(i)) else pd.NaT for i in ct["credited_episode"]]
    inside = (av["opened"] >= span[0]) & (av["opened"] < span[1])
    fm = np.zeros(len(av), dtype=bool)
    if inside.any():
        fm[inside.to_numpy()] = ev.false_episodes(av[inside], conf).to_numpy()
    for i in credited:
        fm[av.index.get_loc(i)] = False
    out = {"ct": ct, "caught": int(ct["caught"].sum()), "conditions": len(cp), "false": int(fm.sum()), "false_mask": fm, "av": av}
    if alarm:
        ao = r2.alarm_outcomes(lev, conf, wo, span, end, False, True)
        out["alarm_false"] = int(ao["false_alarm"].sum()) if len(ao) else 0
    return out


def stuck(sc: pd.DataFrame, conds_all: pd.DataFrame) -> list[str]:
    s = sc[sc["state"] == "scored"]
    out = []
    for pid, g in s.groupby("point_id"):
        asset = str(g["asset_id"].iat[0])
        ok = np.ones(len(g), dtype=bool)
        for t in conds_all.loc[conds_all["asset_key"] == asset, "credit_ts"]:
            ok &= ~((g["hour_utc"] >= t - pd.Timedelta(days=WD)) & (g["hour_utc"] < t + DAY)).to_numpy()
        if ok.any() and float((g.loc[ok, "score"] > g.loc[ok, "threshold"]).mean()) > 0.5:
            out.append(pid)
    return out


def scope(cp: pd.DataFrame) -> pd.DataFrame:
    return cp[ev.in_comparison_scope(cp, ECFG).to_numpy()].reset_index(drop=True)


def load_models(tag: str) -> dict:
    return pickle.loads((D / f"models_{tag}.pkl").read_bytes())
