"""Checkpoint (b) rev2 (diag28): the ML Model's monitoring references rebased from in-sample to
out-of-sample.

The version 13 run set the ML Model's alert-rate reference and its score-drift reference from the final
model's scores on its own training hours. Here both come from the fold-2 model's scores (gold.ml_fold_scores,
fold 2, prod) over the reference window, where fold 2 scores it. Reference catch shares for the ML Model and
Vendor Rules are recomputed on E2 to E7 over both training folds (diag25 fold results). Feature drift, the
Vendor Rules' alert rate and the rule itself are unchanged. Every month's verdict is recomputed with the
production rule (pdm.monitoring.performance.retraining_decision). Writes
build/report_data/monitoring_references.parquet, monitoring_score_drift.parquet and
monitoring_decision.parquet."""

import copy
import pickle

import diag_lib as L
import numpy as np
import pandas as pd
from cb_common import REPO, D

from pdm import splits
from pdm.monitoring import performance as mon

C = REPO / "build" / "monitoring_cache"
OUT = REPO / "build" / "report_data"
MONTH = pd.Timedelta(days=365.25 / 12)
N_ASSETS = 28
TZ = "America/Chicago"
cfg = copy.deepcopy(L.CFG)
ref_a, ref_b = splits.reference_window(cfg)
f2 = splits.folds(cfg)[1]
months = [str(p) for p in pd.period_range("2025-07", "2026-06", freq="M")]
pd.set_option("display.width", 250)


def utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def scores(name: str) -> pd.DataFrame:
    d = pd.read_parquet(C / f"{name}.parquet")
    d["hour_utc"] = utc(d["hour_utc"])
    return d


old = pd.read_parquet(C / "references.parquet").set_index("method")
lev = pd.read_parquet(C / "level_episodes.parquet")
for c in ("opened", "closed"):
    lev[c] = utc(lev[c])


def rate(period: str, span: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[float, int, float]:
    e = lev[(lev["period"] == period) & (lev["method"] == "ml_model")]
    n = int(((e["opened"] >= span[0]) & (e["opened"] < span[1])).sum())
    am = N_ASSETS * (span[1] - span[0]) / MONTH
    return n / am, n, am


# 1. the version 13 reference reproduced from the in-sample training episodes
r_in, n_in, am_in = rate("training", (ref_a, ref_b))
assert abs(r_in - float(old.at["ML Model", "alerts_per_asset_month"])) < 1e-6, r_in
print(
    f"in-sample ML rate: {n_in} episodes / {am_in:.2f} asset-months = {r_in:.6f}",
    f"(prod {old.at['ML Model', 'alerts_per_asset_month']:.6f})",
)

# 2. the fold-2 rate over the part of the window fold 2 scores
fs = scores("ml_fold_scores")
f2s = fs[fs["fold"] == 2]
span2 = (max(ref_a, f2["score"][0]), ref_b)
r_out, n_out, am_out = rate("fold 2", span2)
print(f"fold-2 span {span2[0]} to {span2[1]}; first fold-2 score {f2s['hour_utc'].min()}")
print(f"out-of-sample ML rate: {n_out} episodes / {am_out:.2f} asset-months = {r_out:.6f}")

# 3. reference catch shares on E2 to E7 over both training folds
cts = pickle.loads((D / "fold_cts.pkl").read_bytes())
share = {}
for meth in ("ML Model", "Vendor Rules"):
    n = sum(len(cts[(f"fold {k}", "E7", meth)]) for k in (1, 2))
    s = sum(int(cts[(f"fold {k}", "E7", meth)]["strict"].sum()) for k in (1, 2))
    per = {
        k: (int(cts[(f"fold {k}", "E7", meth)]["strict"].sum()), len(cts[(f"fold {k}", "E7", meth)]))
        for k in (1, 2)
    }
    share[meth] = (s / n, s, n, per)
    print(f"{meth} E7 catch share over folds: {s}/{n} = {s / n:.4f} by fold {per}")

# 4. score drift: the version 13 statistic reproduced, then rebased on fold-2 scores
fd = pd.read_parquet(C / "feature_drift.parquet")
trs, shs = scores("ml_train_scores"), scores("nbm_scores")
allsc = pd.concat([trs, shs], ignore_index=True)
allsc = allsc[allsc["state"] == "scored"]
segs = pd.read_parquet(C / "ml_segments.parquet")
for c in ("segment_start", "level_ready"):
    segs[c] = utc(segs[c])
end = splits.boundaries(cfg)["shadow"][1]
ml_sh = lev[(lev["period"] == "shadow") & (lev["method"] == "ml_model")]
p2a = allsc.drop_duplicates("point_id").set_index("point_id")["asset_id"].to_dict()
days = int(cfg["monitoring"]["drift"]["reference_scored_days"])
rebase = {}
for pid, g in segs[(segs["segment_start"] >= ref_a) & segs["level_ready"].notna()].groupby("point_id"):
    e = ml_sh[ml_sh["asset_id"] == p2a.get(pid)]
    iv = list(zip(e["opened"], e["closed"].clip(upper=end), strict=True))
    wins = [(a, r, True) for a, r in zip(g["segment_start"], g["level_ready"], strict=True)]
    rebase[pid] = mon.drift_references(wins, allsc.loc[allsc["point_id"] == pid, "hour_utc"], iv, end, days)
first = (ref_a.tz_convert("America/Chicago"), (ref_b - pd.Timedelta(hours=1)).tz_convert("America/Chicago"))
base_months = [
    str(p) for p in pd.period_range(first[0].strftime("%Y-%m"), first[1].strftime("%Y-%m"), freq="M")
]
F = mon.SCORE_FEATURE
sd_in = mon.feature_drift(allsc.rename(columns={"score": F}), [F], base_months, months, rebase)
prod_sd = fd[fd["feature"] == F].merge(sd_in, on=["point_id", "month"], suffixes=("_prod", "_here"))
assert len(prod_sd) == (fd["feature"] == F).sum() == len(sd_in), (len(prod_sd), len(sd_in))
dev = float((prod_sd["wasserstein_scaled_prod"] - prod_sd["wasserstein_scaled_here"]).abs().max())
print("version 13 score drift reproduced, rows", len(sd_in), "max abs difference", dev)
assert dev < 1e-9
f2w = f2s[(f2s["state"] == "scored") & (f2s["hour_utc"] >= ref_a) & (f2s["hour_utc"] < ref_b)]
out_sc = pd.concat([f2w, shs[shs["state"] == "scored"]], ignore_index=True)
out_sc[F] = out_sc["score"] / out_sc["threshold"]  # each model's score over its own threshold
sd_out = mon.feature_drift(out_sc, [F], base_months, months, rebase)
print(
    "rebased score drift rows",
    len(sd_out),
    "fold-2 reference hours",
    len(f2w),
    "in-sample reference hours",
    int(
        (
            (trs["state"] == "scored")
            & trs["hour_utc"].dt.tz_convert("America/Chicago").dt.strftime("%Y-%m").isin(base_months)
        ).sum()
    ),
)

# scale: the fold-2 model's scores against the final model's on the same hours
j = f2s[f2s["state"] == "scored"].merge(
    trs[trs["state"] == "scored"], on=["point_id", "hour_utc"], suffixes=("_f2", "_fin")
)
sc_ratio = (j["score_f2"] / j["score_fin"]).groupby(j["point_id"]).median()
th = j.groupby("point_id")[["threshold_f2", "threshold_fin"]].median()
th_ratio = th["threshold_f2"] / th["threshold_fin"]
print(
    f"fold-2 / final score, median per point: median {sc_ratio.median():.3f},",
    f"range {sc_ratio.min():.3f} to {sc_ratio.max():.3f}",
)
print(
    f"fold-2 / final threshold per point: median {th_ratio.median():.3f},",
    f"range {th_ratio.min():.3f} to {th_ratio.max():.3f}",
)

# 5. verdicts with the production rule, old and new references
perf = pd.read_parquet(C / "event_performance.parquet")
mm = pd.read_parquet(OUT / "monitoring_month.parquet")
mm["e7"] = np.where(mm["conditions"] > 0, mm["caught"] / mm["conditions"].where(mm["conditions"] > 0), np.nan)
p = (
    perf[perf["method"].isin(["ML Model", "Vendor Rules"])]
    .drop(columns=["recall"])
    .merge(mm[["method", "month", "e7"]].rename(columns={"e7": "recall"}), on=["method", "month"])
)
feat = fd[fd["feature"] != F]
refs = {
    "version 13": {
        m: {
            "recall": float(old.at[m, "recall"]),
            "alerts_per_asset_month": float(old.at[m, "alerts_per_asset_month"]),
        }
        for m in ("ML Model", "Vendor Rules")
    },
    "rates and score drift rebased": {
        "ML Model": {"recall": float(old.at["ML Model", "recall"]), "alerts_per_asset_month": r_out},
        "Vendor Rules": {
            "recall": float(old.at["Vendor Rules", "recall"]),
            "alerts_per_asset_month": float(old.at["Vendor Rules", "alerts_per_asset_month"]),
        },
    },
}
refs["all rebased"] = {
    "ML Model": {"recall": share["ML Model"][0], "alerts_per_asset_month": r_out},
    "Vendor Rules": {
        "recall": share["Vendor Rules"][0],
        "alerts_per_asset_month": float(old.at["Vendor Rules", "alerts_per_asset_month"]),
    },
}
drift_of = {
    "version 13": pd.concat([feat, sd_in]),
    "rates and score drift rebased": pd.concat([feat, sd_out]),
    "all rebased": pd.concat([feat, sd_out]),
}
dec = {}
for k, ref in refs.items():
    dec[k] = pd.concat(
        [mon.retraining_decision(p, drift_of[k], ref, cfg, m) for m in ("ML Model", "Vendor Rules")],
        ignore_index=True,
    )
v13 = pd.read_parquet(C / "retraining_decision.parquet")
chk = dec["version 13"].merge(v13, on=["method", "month"], suffixes=("", "_prod"))
assert (chk["retrain"] == chk["retrain_prod"].astype(bool)).all()
assert np.allclose(chk["score_drift_share"], chk["score_drift_share_prod"])
print("version 13 verdicts reproduced with headline catch shares")
cols = [
    "method",
    "month",
    "recall",
    "alerts_rate_ratio",
    "drift_share",
    "score_drift_share",
    "secondary_trigger",
    "retrain",
]
w = dec["version 13"][cols].merge(
    dec["rates and score drift rebased"][cols], on=["method", "month"], suffixes=("", " B")
)
w = w.merge(
    dec["all rebased"][["method", "month", "retrain"]].rename(columns={"retrain": "retrain C"}),
    on=["method", "month"],
)
print(w.to_string())

new = dec["all rebased"].assign(recall_reference_original=lambda d: d["method"].map(old["recall"]))
new.to_parquet(OUT / "monitoring_decision.parquet", index=False)
sd_out.to_parquet(OUT / "monitoring_score_drift.parquet", index=False)
win = str(old["reference_window"].iat[0])
pd.DataFrame(
    [
        {
            "method": m,
            "recall": share[m][0],
            "recall_caught": share[m][1],
            "recall_conditions": share[m][2],
            "alerts_per_asset_month": refs["all rebased"][m]["alerts_per_asset_month"],
            "reference_window": win,
            "rate_window": f"{span2[0].tz_convert(TZ):%Y-%m-%d} to {span2[1].tz_convert(TZ):%Y-%m-%d}"
            if m == "ML Model"
            else win,
            "recall_original": float(old.at[m, "recall"]),
            "alerts_per_asset_month_original": float(old.at[m, "alerts_per_asset_month"]),
        }
        for m in ("ML Model", "Vendor Rules")
    ]
).to_parquet(OUT / "monitoring_references.parquet", index=False)
print("written")
