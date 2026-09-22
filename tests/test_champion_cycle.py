"""Per-point model version histories, the registered model, promotion gates and scoring from a model."""

import numpy as np
import pandas as pd
import pytest

from pdm import entrypoints, registry, tasks
from pdm.config import load_config
from pdm.models import nbm
from pdm.models.nbm_pyfunc import NBMPointModels
from tests.test_splits_models import _year

CFG = load_config()
NB = CFG["models"]["nbm"]
UTC = "UTC"
END = pd.Timestamp("2024-09-01", tz=UTC)
REPAIR = pd.Timestamp("2024-06-01", tz=UTC)


def _fitted(cand: str = "mahalanobis"):
    rows = _year(4)
    rows = rows[rows["hour_utc"] < END].reset_index(drop=True)
    shift = rows["hour_utc"] >= REPAIR
    rows.loc[shift, "hf_rms_g"] *= 3.0
    rows.loc[shift, [c for c in rows.columns if c.startswith("toct_")]] *= 3.0
    rows["asset_id"] = "A"
    rb = pd.Timedelta(days=NB["rebaseline_days"])
    win = pd.DataFrame(
        {
            "window_id": ["P|install", "P|repair"],
            "point_id": "P",
            "reason": ["install", "repair W1"],
            "start": [rows["hour_utc"].min(), REPAIR],
            "usable": True,
        }
    )
    plan = nbm.segment_plan(win, set(), END, rb)
    cutoff = pd.Timestamp("2024-05-01", tz=UTC)
    s, info, tr, fitted = nbm.fit_score_segments(rows, cand, CFG, cutoff, [], plan, END, 3, [], [])
    hs = pd.DataFrame({"point_id": ["P"], "segment": [0], "indicator": ["hf"], "median": [1.0]})
    model = NBMPointModels(
        cand, NB["threshold_percentile"], {"P": fitted}, 3, NB["sweep_percentiles"], END, hs
    )
    return rows, win, s, info, model


@pytest.mark.parametrize("cand", ["mahalanobis", "autoencoder_third_octave"])
def test_version_history_reproduces_training_scores(cand):
    rows, _, s, info, model = _fitted(cand)
    seg1 = s[s["model_segment"] == "segment 1"]
    assert info[1]["monthly_updates"] == 2 and seg1["model_update"].nunique() == 3
    assert len(model.segments["P"][1]["versions"]) == 3
    pd.testing.assert_frame_equal(model.score_frame(rows), s)
    shuffled = rows.sample(frac=1.0, random_state=1).reset_index(drop=True)
    out = model.predict(None, shuffled)
    assert len(out) == len(shuffled) and (out["point_id"] == shuffled["point_id"]).all()
    m = shuffled.assign(score_out=out["score"]).merge(s[["hour_utc", "score"]], on="hour_utc", how="left")
    assert np.array_equal(m["score_out"].to_numpy(), m["score"].to_numpy(), equal_nan=True)
    assert {f"thr_{p:g}" for p in NB["sweep_percentiles"]} <= set(out.columns)
    window = (shuffled["hour_utc"] >= REPAIR) & (
        shuffled["hour_utc"] < REPAIR + pd.Timedelta(days=NB["rebaseline_days"])
    )
    assert out.loc[window.to_numpy(), "score"].isna().all()
    assert (out["exceeds"] == (out["score"] > out[f"thr_{NB['threshold_percentile']:g}"])).all()


def test_registered_model_round_trip(tmp_path):
    import mlflow.pyfunc

    rows, _, s, _, model = _fitted()
    hs_path = tmp_path / "healthy_stats.parquet"
    model.healthy_stats.to_parquet(hs_path)
    mlflow.pyfunc.save_model(
        str(tmp_path / "m"), python_model=model, artifacts={"healthy_stats": str(hs_path)}
    )
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "m")).unwrap_python_model()
    pd.testing.assert_frame_equal(loaded.healthy_stats, model.healthy_stats)
    pd.testing.assert_frame_equal(loaded.score_frame(rows), s)


def test_score_tables_from_model():
    rows, win, s, _, model = _fitted()
    win = win.assign(end=win["start"] + pd.Timedelta(days=21))
    pts = pd.DataFrame({"point_id": ["P"], "asset_id": ["A"]})
    out = tasks.nbm_score_tables(model, rows, pts, win, CFG, "nbm_point_models/7")
    sc = out["nbm_scores"]
    assert (sc["model_version"] == "nbm_point_models/7").all() and (sc["asset_id"] == "A").all()
    pd.testing.assert_frame_equal(sc.drop(columns=["asset_id", "model_version", "candidate"]), s)
    ex = out["nbm_exceedances"]
    assert (
        set(ex["sweep"]) <= set(NB["sweep_percentiles"])
        and (ex["model_version"] == "nbm_point_models/7").all()
    )
    assert len(out["point_healthy_stats"]) == 1 and len(out["nbm_exceedance_diagnosis"])


def test_gates_selection_parity_segments():
    rows, _, s, info, model = _fitted()
    cands = pd.DataFrame(
        {"candidate": ["mahalanobis", "pca_t2_spe"], "champion": [True, False], "model_version": "nbm-x"}
    )
    tags = {"challenger": "true", "candidate": "mahalanobis", "model_version": "nbm-x"}
    assert registry.gate_selection(tags, cands).passed
    assert not registry.gate_selection({**tags, "candidate": "pca_t2_spe"}, cands).passed
    assert not registry.gate_selection({**tags, "challenger": "false"}, cands).passed
    pred = model.score_frame(rows)
    assert registry.gate_parity(pred, s).passed
    off = s.copy()
    off.loc[off.index[5], "score"] *= 1 + 1e-6
    g = registry.gate_parity(pred, off)
    assert not g.passed and "score" in g.detail
    assert not registry.gate_parity(pred.iloc[1:], s).passed
    near = pred.copy()
    near.loc[near.index[0], "score"] = 1e-15
    zero = s.copy()
    zero.loc[zero.index[0], "score"] = 0.0
    assert registry.gate_parity(near, zero).passed
    mi = pd.DataFrame(info)
    assert registry.gate_segments(model, mi, s).passed
    assert not registry.gate_segments(model, mi, s[s["model_segment"] != "segment 1"]).passed
    model.healthy_stats = None
    assert not registry.gate_segments(model, mi, s).passed


def test_latest_complete_month():
    a, b = registry.latest_complete_month(pd.Timestamp("2026-04-01 05:00", tz=UTC))
    assert a == pd.Timestamp("2026-03-01 06:00", tz=UTC) and b == pd.Timestamp("2026-04-01 05:00", tz=UTC)
    a, b = registry.latest_complete_month(pd.Timestamp("2026-05-17 12:00", tz=UTC))
    assert a == pd.Timestamp("2026-04-01 05:00", tz=UTC) and b == pd.Timestamp("2026-05-01 05:00", tz=UTC)


def _month_scores(bursts: list[str]) -> pd.DataFrame:
    t = pd.date_range("2026-03-01 06:00", "2026-04-01 05:00", freq="h", tz=UTC, inclusive="left")
    s = pd.DataFrame({"point_id": "P", "hour_utc": t, "score": 1.0, "top_contributors": "x"})
    s["thr_99.5"] = 2.0
    for d in bursts:
        h = pd.Timestamp(d, tz=UTC)
        s.loc[(s["hour_utc"] >= h) & (s["hour_utc"] < h + pd.Timedelta(hours=8)), "score"] = 3.0
    return s


def test_false_episode_gate():
    pts = pd.DataFrame({"point_id": ["P", "Q"], "asset_id": ["A", "B"]})
    curves = pd.DataFrame(
        {"method": ["nbm", "nbm"], "level": ["99.0", "99.5"], "false_per_asset_month": [9.0, 0.5]}
    )
    wo = pd.DataFrame(columns=["asset_id", "wo_type", "opened", "closed"])
    conds = pd.DataFrame(columns=["asset_key", "credit_ts"])
    h = pd.Timestamp("2026-04-01 05:00", tz=UTC)
    one = registry.gate_false_episodes(_month_scores(["2026-03-02"]), pts, conds, wo, curves, CFG, h)
    assert one.passed and one.tags["false_episodes.month"] == "2026-03"
    assert one.tags["false_episodes.false_episodes"] == "1" and one.tags["false_episodes.budget"] == "1.0000"
    assert (
        abs(float(one.tags["false_episodes.healthy_asset_months"]) - 2 * 743 / registry.HOURS_PER_MONTH)
        < 1e-3
    )
    three = registry.gate_false_episodes(
        _month_scores(["2026-03-02", "2026-03-12", "2026-03-22"]), pts, conds, wo, curves, CFG, h
    )
    assert not three.passed and three.tags["false_episodes.episodes"] == "3"
    conf = pd.DataFrame({"asset_key": ["A"], "credit_ts": [pd.Timestamp("2026-03-25", tz=UTC)]})
    near = registry.gate_false_episodes(
        _month_scores(["2026-03-02", "2026-03-12", "2026-03-22"]), pts, conf, wo, curves, CFG, h
    )
    assert near.tags["false_episodes.false_episodes"] == "0" and near.passed
    assert float(near.tags["false_episodes.healthy_asset_months"]) < 1.3
    late = pd.DataFrame({"asset_key": ["A"], "credit_ts": [pd.Timestamp("2026-04-20", tz=UTC)]})
    assert (
        registry.gate_false_episodes(_month_scores(["2026-03-02"]), pts, late, wo, curves, CFG, h).tags
        == one.tags
    )
    none = registry.gate_false_episodes(_month_scores([]), pts, conds, wo, curves.iloc[:1], CFG, h)
    assert not none.passed


class FakeRegistry:
    def __init__(self, champion=None):
        self.tags, self.aliases = {}, ({"champion": champion} if champion else {})

    def set_tag(self, name, version, key, value):
        self.tags[(version, key)] = value

    def set_alias(self, name, alias, version):
        self.aliases[alias] = version

    def alias_version(self, name, alias):
        return self.aliases.get(alias)


@pytest.mark.parametrize(
    ("passes", "auto", "status", "champion"),
    [
        ([True, True, True, True], True, "promoted", "2"),
        ([True, False, True, True], True, "rejected", "1"),
        ([True, True, True, True], False, "for_review", "1"),
        ([True, True, True, False], False, "for_review", "1"),
    ],
)
def test_decide(passes, auto, status, champion):
    names = ("selection", "parity", "segments", "false_episodes")
    gates = [
        registry.Gate(n, p, "d", {"false_episodes.month": "2026-03"} if n == "false_episodes" else {})
        for n, p in zip(names, passes, strict=True)
    ]
    reg = FakeRegistry(champion="1")
    assert registry.decide(reg, "pdm.ml.nbm_point_models", "2", gates, auto) == status
    assert reg.aliases["champion"] == champion and reg.tags[("2", "status")] == status
    assert reg.tags[("2", "false_episodes.month")] == "2026-03"
    failed = [n for n, p in zip(names, passes, strict=True) if not p]
    assert reg.tags.get(("2", "failed_gates")) == (",".join(failed) if failed else None)


def test_entrypoint_keyword_arguments():
    a = entrypoints.parse(
        "pdm-promote",
        [
            "--catalog=pdm",
            "--config=c.yaml",
            "--experiment_root=/Shared/pdm",
            "--job_run_id=42",
            "--auto_promote=false",
        ],
    )
    assert (
        a.catalog == "pdm"
        and a.job_run_id == "42"
        and a.auto_promote is False
        and a.experiment_root == "/Shared/pdm"
    )
    b = entrypoints.parse("pdm-promote", ["--catalog", "pdm", "--config", "c.yaml", "--job-run-id", "7"])
    assert b.auto_promote is True and b.run_id is None and b.job_run_id == "7"
    with pytest.raises(SystemExit):
        entrypoints.parse("pdm-promote", ["--catalog=pdm", "--config=c.yaml", "--auto_promote=maybe"])


def test_controller_signal_tie_break():
    t = pd.Timestamp("2025-03-15 04:00", tz=UTC)
    sig = pd.DataFrame(
        {
            "method": "controller",
            "asset_id": "VMC-04",
            "machine_id": "VMC-04",
            "point_id": None,
            "ts": [t, t],
            "level": "alarm",
            "rules": ["thermistor_residual", "spindle_load_drift"],
        }
    )
    a = tasks._daily_first(sig)
    b = tasks._daily_first(sig.iloc[::-1].reset_index(drop=True))
    assert a["rules"].tolist() == b["rules"].tolist() == ["spindle_load_drift"]
