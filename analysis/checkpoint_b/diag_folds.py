"""Scratch: refit the selected design on each fold with optional variants and evaluate on the fold spans
only (catch at the operating threshold and at the fold's Vendor Rules budget, false episodes per asset-month).
Diagnosis only; nothing is written to the store."""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

import diag_lib as L
from dlt import read
from pdm import ml_tasks, splits
from pdm.evaluation import events as ev
from pdm.models import nbm, temperature

BASE = copy.deepcopy(L.CFG)
BASE["models"]["nbm"]["selection_n_jobs"] = 1
SETTINGS = nbm.Settings("pca_t2_spe_extended", (), 0, "speed_band")


@contextmanager
def patched(variant: str):
    saved = (nbm._one_hot, nbm.features, temperature.covariates)
    try:
        if "no_onehot" in variant:
            nbm._one_hot = lambda rows, bands: np.zeros((len(rows), 0))
        if "toct8" in variant:
            orig = saved[1]

            def feats(df, candidate):
                x, cols = orig(df, candidate)
                tc = [i for i, c in enumerate(cols) if c.startswith("toct_")]
                if not tc:
                    return x, cols
                lin = 10 ** x[:, tc]
                agg = np.column_stack([np.log10(lin[:, i : i + 4].sum(axis=1)) for i in range(0, len(tc), 4)])
                rest = [i for i in range(len(cols)) if i not in tc]
                return np.column_stack([agg, x[:, rest]]), [f"toct_grp{k}" for k in range(agg.shape[1])] + [cols[i] for i in rest]

            nbm.features = feats
        if "pump_season" in variant:
            orig_cov = saved[2]

            def cov(point_class, seasonal=True):
                c = orig_cov(point_class, seasonal)
                if not point_class.startswith("spindle"):
                    c = c + [temperature.SEASON_SIN, temperature.SEASON_COS]
                return c

            temperature.covariates = cov
        yield
    finally:
        nbm._one_hot, nbm.features, temperature.covariates = saved


class _Store:
    """The local reference through the Store read interface (pandas only)."""

    catalog = None

    def read_pd(self, schema, table, columns=None):
        d = read(f"{schema}/{table}", columns)
        return d

    def exists(self, schema, table):
        from dlt import ROOT

        return (ROOT / schema / table / "_delta_log").exists()

    def read(self, *a, **k):
        raise RuntimeError("spark not used")


INP = None


def inputs():
    global INP
    if INP is None:
        from pdm import tasks

        store = _Store()
        b = splits.boundaries(BASE)
        ind = L.ind()
        ind = ind[ind["hour_utc"] < b["train"][1]].reset_index(drop=True)
        win = L.windows()
        win = win[win["start"] < b["train"][1]].reset_index(drop=True)
        san = read("gold/baseline_sanity", version=2)
        san = san.merge(L.windows()[["window_id", "start", "end"]].assign(end=lambda d: L.utc(d["end"])), on="window_id", how="left")
        conds = L.conds().copy()
        conds["machine_class"] = "hmc"
        conds["point_scope"] = None
        pts = L.points()
        rated = read("bronze/reg_machines").set_index("machine_id")["rated_rpm"].astype(float).to_dict()
        vend = read("gold/rule_floor_exceedances")
        op = BASE["rules"]["vendor"]["alert_frac_zone_b_width"]
        vend = vend[(vend["method"] == "iso") | ((vend["method"] == "vendor") & np.isclose(vend["sweep"], op))].copy()
        vend["ts"] = L.utc(vend["ts"])
        vend = vend.merge(pts[["point_id", "asset_id"]], on="point_id", how="left")
        INP = {"ind": ind, "win": win, "sanity": san, "conds": conds, "wo": L.wo(), "pts": pts, "rated": rated,
               "ambient": L.ambient(), "vendor": vend}
        INP["ctx"] = {"inspection_hours_each": 1.2, "margin": {}}
    return INP


def run(variant: str = "base", cfg_changes: dict | None = None, trim_quantile: float | None = None, keep_scores: bool = False):
    cfg = copy.deepcopy(BASE)
    for path, v in (cfg_changes or {}).items():
        d = cfg
        ks = path.split(".")
        for k in ks[:-1]:
            d = d[k]
        d[ks[-1]] = v
    inp = inputs()
    nb = cfg["models"]["nbm"]
    pcts = nbm.sweep_percentiles(cfg)
    op_p = float(nb["threshold_percentile"])
    n_assets = inp["pts"]["asset_id"].nunique()
    vf = read("gold/ml_vendor_fold").set_index("fold")
    floor = float(read("gold/ml_settings")["curve_floor_percentile"].iat[0])
    out, scores = [], []
    with patched(variant):
        for f in splits.folds(cfg):
            args = ml_tasks._point_args(inp, cfg, f["cutoff"])
            models, _ = ml_tasks._fit_all(args, [SETTINGS], cfg, f["fit_start"], f["cutoff"])
            if trim_quantile is not None:
                # two-pass: drop training hours whose first-pass score is above the quantile, refit
                for a in args:
                    if a["point_id"] not in models:
                        continue
                    pm = models[a["point_id"]][SETTINGS.key]
                    s, _ = nbm.score_point(pm, a["rows"], a["windows"], cfg, f["fit_start"], f["cutoff"], 1.45, 1, a["ambient"])
                    s = s[s["state"] == "scored"]
                    hi = s[s["score"] > np.quantile(s["score"], trim_quantile)]["hour_utc"]
                    a["excluded"] = list(a["excluded"]) + [(t, t + pd.Timedelta(hours=1)) for t in hi]
                models, _ = ml_tasks._fit_all(args, [SETTINGS], cfg, f["fit_start"], f["cutoff"])
            factor, _ = ml_tasks._factor(models, cfg)
            res = ml_tasks._score_all(args, models, cfg, f["score"], factor, pcts, keep_scores)
            conds_f = ml_tasks.asset_conditions(inp["conds"], *f["score"])
            curve = ml_tasks._curve(res, SETTINGS.key, pcts, 6, conds_f, inp["conds"], f["score"], n_assets, cfg, inp["ctx"])
            c = curve[curve["percentile"] >= floor]
            budget = float(vf.at[f["fold"], "false_per_asset_month"])
            at = ev.at_budget(c, budget)
            o = c[c["percentile"] == op_p].iloc[0]
            out.append({"variant": variant, "fold": f["fold"], "conditions": int(o["conditions"]), "caught_operating": int(o["caught"]),
                        "false_per_asset_month_operating": float(o["false_per_asset_month"]),
                        "caught_share_at_budget": at["caught_share"], "chance_at_budget": at["chance_share"],
                        "within_curve": bool(at["within_curve"]), "vendor_caught": int(vf.at[f["fold"], "caught"]),
                        "vendor_false": float(vf.at[f["fold"], "false_per_asset_month"])})
            if keep_scores:
                scores.append(ml_tasks._frames(res, SETTINGS.key, "scores").assign(fold=f["fold"]))
    return pd.DataFrame(out), (pd.concat(scores, ignore_index=True) if scores else None)


if __name__ == "__main__":
    r, s = run("base", keep_scores=True)
    print(r.round(3).to_string())
    fs = read("gold/ml_fold_scores", ["point_id", "hour_utc", "fold", "score", "state"])
    fs["hour_utc"] = L.utc(fs["hour_utc"])
    s["hour_utc"] = L.utc(s["hour_utc"])
    m = fs.merge(s[["point_id", "hour_utc", "fold", "score"]], on=["point_id", "hour_utc", "fold"], suffixes=("", "_r"))
    print("rows", len(fs), len(s), len(m), "max score diff", float(np.nanmax(np.abs(m["score"] - m["score_r"]))))
