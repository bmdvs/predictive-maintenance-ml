"""Splits, per-point models (shape, determinism, leaked-window refusal), controller health, prognosis,
fusion, advisory and event-based evaluation on small known inputs."""

import numpy as np
import pandas as pd
import pytest

from pdm import splits
from pdm.advisory.requests import DB6, suspected
from pdm.config import load_config
from pdm.evaluation import events as ev
from pdm.fusion.status import cases, status_changes
from pdm.models import controller_health as ch
from pdm.models import nbm, prognosis

CFG = load_config()
UTC = "UTC"


def test_chronological_split_and_gap():
    ts = pd.Series(
        pd.to_datetime(["2023-08-01", "2025-06-25", "2025-08-01", "2026-03-25", "2026-05-01"], utc=True)
    )
    lab = splits.chronological(ts, CFG)
    assert list(lab) == ["training", "gap", "validation", "gap", "shadow"]
    assert splits.training_cutoff(CFG) < pd.Timestamp("2025-07-01", tz=UTC)


def test_leave_one_asset_out():
    folds = splits.leave_one_asset_out(["a", "b", "c"])
    assert [h for _, h in folds] == ["a", "b", "c"] and all(h not in t for t, h in folds)


def _healthy(n: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="h", tz=UTC)
    d = pd.DataFrame(
        {
            "point_id": "P",
            "hour_utc": t,
            "regime": "running",
            "implausible": False,
            "accel_rms_g": rng.lognormal(-1, 0.05, n),
            "vel_rms_mm_s": rng.lognormal(0.5, 0.05, n),
            "hf_rms_g": rng.lognormal(-1.5, 0.05, n),
            "crest_factor": rng.normal(4, 0.1, n),
            "kurtosis": rng.normal(3, 0.05, n),
            "ord_01_g2": rng.lognormal(-7, 0.1, n),
            "ord_02_g2": rng.lognormal(-8, 0.1, n),
            "env_bpfo_1_g2": rng.lognormal(-9, 0.1, n),
            "env_bpfi_1_g2": rng.lognormal(-9, 0.1, n),
            "env_bsf_1_g2": rng.lognormal(-9, 0.1, n),
        }
    )
    for b in range(1, 9):
        d[f"toct_{b:02d}_g2"] = rng.lognormal(-6, 0.1, n)
    return d


@pytest.mark.parametrize("cand", nbm.CANDIDATES)
def test_nbm_shape_determinism_and_anomaly(cand):
    train = _healthy(300, "2023-07-05", 1)
    win = (train["hour_utc"].min(), train["hour_utc"].max() + pd.Timedelta(hours=1))
    m1 = nbm.fit_point_window(train, cand, CFG, win)
    m2 = nbm.fit_point_window(train, cand, CFG, win)
    test = _healthy(50, "2023-08-01", 2)
    s1, s2 = nbm.score_rows(test, m1, 3), nbm.score_rows(test, m2, 3)
    assert len(s1) == len(test) and np.allclose(s1["score"], s2["score"])
    bad = test.copy()
    bad["hf_rms_g"] *= 4
    bad["env_bpfo_1_g2"] *= 50
    bad[[c for c in bad.columns if c.startswith("toct_")]] *= 20
    assert nbm.score_rows(bad, m1, 3)["score_ratio"].median() > s1["score_ratio"].median()


def test_nbm_refuses_leaked_rows():
    train = _healthy(100, "2025-06-20", 3)
    win = (train["hour_utc"].min(), train["hour_utc"].max() + pd.Timedelta(hours=1))
    with pytest.raises(nbm.LeakedWindowError):
        nbm.fit_point_window(train, "mahalanobis", CFG, win, cutoff=pd.Timestamp("2025-06-22", tz=UTC))
    with pytest.raises(nbm.LeakedWindowError):
        nbm.fit_point_window(train, "mahalanobis", CFG, (win[0] + pd.Timedelta(days=1), win[1]))


def test_nbm_alert_persistence():
    s = pd.DataFrame(
        {
            "point_id": "P",
            "hour_utc": pd.date_range("2024-01-01", periods=6, freq="h"),
            "score_ratio": [2, 0.5, 2, 2, 2, 2],
            "top_contributors": "x",
        }
    )
    a = nbm.alerts_from_scores(s, 2, 7)
    assert len(a) == 1 and a["alert_ts"].iat[0] == s["hour_utc"].iat[3]


def test_controller_drift_and_thermistor():
    rng = np.random.default_rng(0)
    hours = pd.date_range("2024-01-01", periods=24 * 200, freq="h", tz=UTC)
    h = pd.DataFrame({"machine_id": "M", "hour_utc": hours, "program": "P1", "active_s": 3600.0})
    h["local_date"] = h["hour_utc"].dt.tz_convert("America/Chicago").dt.date
    drift = np.where(np.arange(len(h)) > 24 * 150, 1.3, 1.0)
    for a in ("x", "y", "z", "b"):
        h[f"{a}_load_p95"] = 20 * (drift if a == "z" else 1.0) * (1 + 0.01 * rng.standard_normal(len(h)))
    h["spindle_load"] = 30.0
    h["spindle_speed_rpm"] = 6000.0
    h["spindle_temp_c"] = 30 + rng.normal(0, 0.2, len(h)) + np.where(np.arange(len(h)) > 24 * 180, 5.0, 0.0)
    scores, _ = ch.daily_scores(h, 12000.0, pd.Timestamp("2024-05-01"))
    al = ch.alerts(scores, CFG)
    assert {"axis_drift", "thermistor_residual"} <= set(al["rule"])


def test_prognosis_exponential_fit_and_alpha_lambda():
    det = pd.Timestamp("2024-01-01", tz=UTC)
    days = pd.date_range(det, periods=40, freq="D")
    energy = pd.Series(np.exp(0.1 * np.arange(40)), index=days)
    inst = prognosis.Instance("A", det, det + pd.Timedelta(days=40), energy)
    thr = float(np.exp(0.1 * 40))
    p = prognosis.alpha_lambda(prognosis.predict(inst, thr, [0.5]), 0.2)
    assert p["within_alpha"].all() and abs(p["pred_rul_days"].iat[0] - 20) < 1.0


def test_fusion_rule_floor_watch_then_model_alarm_and_case():
    t = pd.Timestamp("2024-03-01", tz=UTC)
    sig = pd.DataFrame(
        {
            "method": ["vendor", "nbm"],
            "asset_id": "A",
            "machine_id": "M",
            "point_id": "P",
            "ts": [t, t + pd.Timedelta(hours=2)],
            "level": "alert",
            "rules": "r",
        }
    )
    ch_ = status_changes(sig, CFG)
    assert list(ch_["status"])[:2] == ["WATCH", "ALARM"]
    wo = pd.DataFrame(
        {"asset_id": ["A"], "wo_id": ["W1"], "wo_type": ["inspection"], "closed": [t + pd.Timedelta(days=3)]}
    )
    c = cases(ch_, sig, wo)
    assert len(c) == 1 and c["closing_wo"].iat[0] == "W1"


def test_advisory_rules():
    assert suspected({"env_bpfi_1_ratio_24h": DB6 + 1}, "flood_pump", {}, CFG)[0] == "bearing"
    assert suspected({"order_1x_ratio_24h": DB6 + 1}, "flood_pump", {}, CFG)[0] == "imbalance"
    assert (
        suspected({"hf_rms_g_ratio_24h": 1.5}, "spindle_front", {"coolant_drop": 0.1}, CFG)[0]
        == "contamination"
    )
    assert suspected({"hf_rms_g_ratio_24h": 1.5}, "spindle_front", {}, CFG)[0] == "lubrication"
    assert suspected({}, "spindle_front", {"thermistor_residual_alert": True}, CFG)[0] == "cooler"


def test_evaluation_caught_and_false_alerts():
    t = pd.Timestamp("2025-09-01", tz=UTC)
    conds = pd.DataFrame(
        {
            "wo_id": ["W"],
            "asset_id": ["A"],
            "machine_id": ["M"],
            "condition_type": ["imbalance"],
            "unplanned": [True],
            "period": ["validation"],
            "credit_ts": [t],
            "downtime_hours": [10.0],
            "machine_class": ["hmc"],
        }
    )
    alerts = pd.DataFrame(
        {
            "method": "nbm",
            "asset_id": ["A", "A"],
            "machine_id": "M",
            "ts": [t - pd.Timedelta(days=20), t + pd.Timedelta(days=200)],
        }
    )
    c = ev.caught_table(alerts, conds, ["nbm"], 90)
    assert bool(c["caught"].iat[0]) and abs(c["warning_days"].iat[0] - 20) < 1e-9
    fa = ev.false_alerts(alerts, conds, ["nbm"], 30)
    assert list(fa["confirmed_within_window"]) == [True, False]
    s = ev.summary(c, fa, {"nbm": 1}, 1.0, 1.5, {"hmc": 100.0}, 5)
    assert s["false_alerts"].iat[0] == 1 and s["unplanned_cost_plannable_usd"].iat[0] == 1000.0
