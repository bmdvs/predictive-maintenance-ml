"""The frozen ML Model as a registered model, the promotion gates run once, and the registry decisions."""

import pickle

import numpy as np
import pandas as pd
import pytest

from pdm import entrypoints, registry, tasks
from pdm.config import load_config
from pdm.models import nbm
from pdm.models.nbm_pyfunc import NBMPointModels
from tests.test_splits_models import _settings, _windows, _year

CFG = load_config()
UTC = "UTC"
CUTOFF = pd.Timestamp("2024-07-01", tz=UTC)
REPAIR = pd.Timestamp("2024-07-20", tz=UTC)
END = pd.Timestamp("2024-09-01", tz=UTC)


def _fitted(cand: str = "mahalanobis"):
    rows = _year(11)
    win = _windows(rows["hour_utc"].min(), REPAIR)
    st = _settings(cand)
    models, _ = nbm.fit_point(rows, win, [st], CFG, rows["hour_utc"].min(), CUTOFF, [], None, None)
    hs = pd.DataFrame(
        {
            "point_id": ["P"],
            "regime": ["running"],
            "indicator": ["hf"],
            "median": [0.0],
            "scale": [1.0],
            "n": [99],
            "segment": [0],
            "segment_start": [rows["hour_utc"].min()],
        }
    )
    model = NBMPointModels(st.key, {"P": models[st.key]}, 1.45, CFG, hs)
    scores, segs = model.score_span(rows, win, rows["hour_utc"].min(), CUTOFF)
    return rows, win, st, model, scores, segs


def test_registered_model_round_trip():
    rows, win, st, model, scores, _ = _fitted()
    back = pickle.loads(pickle.dumps(model))
    assert back.healthy_stats is None and back.settings_key == st.key
    x = rows[(rows["hour_utc"] >= REPAIR - pd.Timedelta(days=2)) & (rows["hour_utc"] < END)].copy()
    x["segment_starts"] = ";".join(str(t) for t in win["start"])
    pred = back.predict(None, x)
    direct, _ = model.score_span(x, win, x["hour_utc"].min(), END)
    m = pred.dropna(subset=["score"]).merge(
        direct.assign(hour_utc=direct["hour_utc"].dt.tz_localize(None)),
        on=["point_id", "hour_utc"],
        suffixes=("", "_d"),
    )
    assert len(m) and np.allclose(m["score"], m["score_d"]) and (pred["state"] == "recentering").any()


def test_gates_once():
    rows, win, st, model, scores, segs = _fitted()
    cands = pd.DataFrame({"settings": [st.key, "other"], "champion": [True, False], "model_version": "nbm-x"})
    tags = {"challenger": "true", "settings": st.key, "model_version": "nbm-x"}
    assert registry.gate_selection(tags, cands).passed
    assert not registry.gate_selection({**tags, "settings": "other"}, cands).passed
    assert registry.gate_parity(scores, scores).passed
    bent = scores.copy()
    bent.loc[bent["score"].notna().idxmax(), "score"] *= 1.001
    assert not registry.gate_parity(scores, bent).passed
    pts = pd.DataFrame({"point_id": ["P"], "asset_id": ["A"]})
    assert registry.gate_points(model, pts, segs).passed
    assert not registry.gate_points(
        model, pd.DataFrame({"point_id": ["P", "Q"], "asset_id": ["A", "B"]}), segs
    ).passed
    settings = pd.DataFrame(
        {
            "reference_false_per_asset_month": [0.3],
            "fold_false_per_asset_month": [0.2],
            "reference_window": ["2024-12-17 to 2025-06-17"],
        }
    )
    g = registry.gate_false_episodes(settings)
    assert g.passed and g.tags["false_episodes_limit"] == "0.4000"
    assert not registry.gate_false_episodes(settings.assign(reference_false_per_asset_month=0.5)).passed


class FakeRegistry:
    def __init__(self, champion=None):
        self.tags, self.aliases = {}, ({"champion": champion} if champion else {})
        self.versions = {}

    def set_tag(self, name, version, key, value):
        self.tags[(version, key)] = value

    def delete_tag(self, name, version, key):
        self.tags.pop((version, key), None)

    def set_alias(self, name, alias, version):
        self.aliases[alias] = version

    def find_version(self, name, run_id):
        return self.versions.get(run_id)

    def register(self, run_id, name):
        self.versions[run_id] = str(max([int(x) for x in self.versions.values()] + [1]) + 1)
        return self.versions[run_id]

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
    names = ("selection", "parity", "points", "false_episodes")
    gates = [
        registry.Gate(n, p, "d", {"false_episodes_window": "w"} if n == "false_episodes" else {})
        for n, p in zip(names, passes, strict=True)
    ]
    reg = FakeRegistry(champion="1")
    assert registry.decide(reg, "pdm.ml.nbm_point_models", "2", gates, auto) == status
    assert reg.aliases["champion"] == champion and reg.tags[("2", "status")] == status
    assert reg.tags[("2", "false_episodes_window")] == "w"
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
    assert entrypoints.train_stages("ml_model,survival") == ["ml_model", "survival"]
    with pytest.raises(ValueError):
        entrypoints.train_stages("nbm")


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


def test_tag_keys_avoid_reserved_characters():
    rows, win, st, model, scores, segs = _fitted()
    cands = pd.DataFrame({"settings": [st.key], "champion": [True], "model_version": "nbm-x"})
    tags = {"challenger": "true", "settings": st.key, "model_version": "nbm-x"}
    settings = pd.DataFrame(
        {
            "reference_false_per_asset_month": [0.3],
            "fold_false_per_asset_month": [0.2],
            "reference_window": ["a to b"],
        }
    )
    gates = [
        registry.gate_selection(tags, cands),
        registry.gate_parity(scores, scores),
        registry.gate_points(model, pd.DataFrame({"point_id": ["P"]}), segs),
        registry.gate_false_episodes(settings),
        registry.Gate("parity", False, "d"),
    ]
    reg = FakeRegistry(champion="1")
    registry.decide(reg, "pdm.ml.nbm_point_models", "2", gates, True)
    keys = {k for _, k in reg.tags} | set(registry.IDENTITY_TAGS)
    assert "false_episodes_window" in keys and "gate_false_episodes_detail" in keys and "failed_gates" in keys
    for k in keys:
        assert not set(k) & set(".=><%&?\\"), k
        assert registry.tag_key(k) == k
    for bad in ("gate.parity", "a=b", "x%y", "p\\q", ""):
        with pytest.raises(ValueError):
            registry.tag_key(bad)


def test_promote_reuses_the_version_registered_from_the_same_run():
    reg = FakeRegistry(champion="1")
    reg.versions["run-a"] = "2"
    assert registry.register_or_reuse(reg, "run-a", "pdm.ml.nbm_point_models") == ("2", True)
    v, reused = registry.register_or_reuse(reg, "run-b", "pdm.ml.nbm_point_models")
    assert not reused and v == "3" and reg.versions["run-b"] == "3"
    registry.decide(reg, "m", "2", [registry.Gate("parity", False, "d")], True)
    assert reg.tags[("2", "failed_gates")] == "parity"
    registry.decide(reg, "m", "2", [registry.Gate("parity", True, "d")], False)
    assert ("2", "failed_gates") not in reg.tags and reg.tags[("2", "status")] == "for_review"
