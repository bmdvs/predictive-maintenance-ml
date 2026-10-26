"""Splits, per-point models (shape, determinism, leaked-window refusal, seasonal coverage), controller
health, prognosis, fusion, advisory and event-based evaluation on small known inputs."""

import numpy as np
import pandas as pd
import pytest

from pdm import splits, tasks
from pdm.advisory.requests import asset_kind, family_of, suspected
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


def _year(seed: int) -> pd.DataFrame:
    return _healthy(24 * 400, "2023-07-05", seed)


@pytest.mark.parametrize("cand", nbm.CANDIDATES)
def test_nbm_shape_determinism_and_anomaly(cand):
    rows = _year(1)
    cutoff = pd.Timestamp("2024-07-01", tz=UTC)
    starts = [rows["hour_utc"].min()]
    m1, info = nbm.fit_point(rows, cand, CFG, cutoff, [], starts)
    m2, _ = nbm.fit_point(rows, cand, CFG, cutoff, [], starts)
    assert info["covers_summer"] and info["covers_winter"] and set(m1) == {"running"}
    test = rows[rows["hour_utc"] >= cutoff].head(50)
    s1, s2 = nbm.score_point(test, m1, CFG, starts, 3), nbm.score_point(test, m2, CFG, starts, 3)
    assert len(s1) == len(test) and np.allclose(s1["score"], s2["score"])
    assert {f"thr_{p:g}" for p in CFG["models"]["nbm"]["sweep_percentiles"]} <= set(s1.columns)
    bad = test.copy()
    bad["hf_rms_g"] *= 4
    bad["env_bpfo_1_g2"] *= 50
    bad[[c for c in bad.columns if c.startswith("toct_")]] *= 20
    assert nbm.score_point(bad, m1, CFG, starts, 3)["score"].median() > s1["score"].median()


def test_nbm_seasonal_coverage_required():
    rows = _healthy(24 * 60, "2023-07-05", 2)
    models, info = nbm.fit_point(
        rows, "mahalanobis", CFG, pd.Timestamp("2024-07-01", tz=UTC), [], [rows["hour_utc"].min()]
    )
    assert models == {} and info["covers_summer"] and not info["covers_winter"]


def test_nbm_refuses_leaked_rows():
    train = _healthy(100, "2025-06-20", 3)
    with pytest.raises(nbm.LeakedWindowError):
        nbm.check_training_rows(train["hour_utc"], pd.Timestamp("2025-06-22", tz=UTC), [])
    a = train["hour_utc"].iat[10]
    with pytest.raises(nbm.LeakedWindowError):
        nbm.check_training_rows(
            train["hour_utc"], pd.Timestamp("2026-01-01", tz=UTC), [(a, a + pd.Timedelta(hours=5))]
        )
    keep = nbm.training_mask(
        train["hour_utc"], pd.Timestamp("2026-01-01", tz=UTC), [(a, a + pd.Timedelta(hours=5))]
    )
    assert (~keep).sum() == 5


def test_nbm_exceedance_persistence():
    s = pd.DataFrame(
        {
            "point_id": "P",
            "hour_utc": pd.date_range("2024-01-01", periods=8, freq="h"),
            "score": [2, 0.5, 2, 2, 2, 2, 2, 2],
            "thr_99.5": 1.0,
            "top_contributors": "x",
        }
    )
    ex = nbm.exceedance_hours(s, 99.5, 6)
    assert list(ex["ts"]) == [s["hour_utc"].iat[7]]


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


def _inst(name: str, b: float, life: float, seed: int) -> prognosis.Instance:
    rng = np.random.default_rng(seed)
    det = pd.Timestamp("2024-01-01", tz=UTC)
    days = pd.date_range(det, periods=int(life), freq="D")
    v = 2.0 + np.exp(-1.0 + b * np.arange(len(days)) + rng.normal(0, 0.05, len(days)))
    thr = 2.0 + np.exp(-1.0 + b * life)
    return prognosis.Instance(name, det, det + pd.Timedelta(days=life), pd.Series(v, index=days), 2.0, thr)


def test_prognosis_bayesian_intervals_and_alpha_lambda():
    train = [_inst(f"T{i}", 0.08 + 0.005 * i, 40, i) for i in range(4)]
    prior = prognosis.prior_from_training(train)
    assert prior.mean[1] > 0
    test = _inst("V", 0.09, 40, 9)
    p = prognosis.scores(prognosis.predict(test, prior, [0.5, 0.75], 0.9, 2000, 7), 0.2)
    q = prognosis.predict(test, prior, [0.5, 0.75], 0.9, 2000, 7)
    assert np.allclose(p["pred_rul_days"], q["pred_rul_days"])
    assert p["within_alpha"].all() and p["covered"].all()
    assert (p["pi_low"] <= p["pred_rul_days"]).all() and (p["pred_rul_days"] <= p["pi_high"]).all()


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
        {"asset_id": ["A"], "wo_id": ["W1"], "wo_type": ["corrective"], "closed": [t + pd.Timedelta(days=3)]}
    )
    c = cases(ch_, sig, wo, 7)
    assert len(c) == 1 and c["closing_wo"].iat[0] == "W1" and c["closed_by"].iat[0] == "work order"
    late = wo.assign(closed=[t + pd.Timedelta(days=30)])
    c2 = cases(ch_, sig, late, 7)
    assert c2["closed_by"].iat[0] == "quiet" and c2["closed"].iat[0] == t + pd.Timedelta(hours=2, days=7)
    sig3 = pd.concat(
        [
            sig,
            sig.iloc[[1]].assign(ts=t + pd.Timedelta(days=4)),
            sig.iloc[[1]].assign(ts=t + pd.Timedelta(days=5)),
        ],
        ignore_index=True,
    )
    c3 = cases(status_changes(sig3, CFG), sig3, wo, 7)
    assert len(c3) == 2 and c3["opened"].iat[1] == t + pd.Timedelta(days=4)
    assert c3["chain_opened"].iat[1] == c3["opened"].iat[0]
    insp = wo.assign(wo_type="inspection")
    c4 = cases(status_changes(sig3, CFG), sig3, insp, 7)
    assert len(c4) == 1 and c4["closed_by"].iat[0] == "quiet"


def test_advisory_rules():
    z = 3.0
    assert suspected({"env_bpfi": 5.0, "hf": 40.0}, "flood_pump", z)[0] == "cavitation or wear"
    assert suspected({"env_bpfi": 5.0, "hf": 40.0}, "tsc_pump", z)[0] == "bearing"
    assert suspected({"env_bpfi": 5.0, "hf": 40.0, "tsc_drop": 4.0}, "tsc_pump", z)[0] == "cavitation or wear"
    assert suspected({"ord_01": 5.0}, "flood_pump", z)[0] == "imbalance"
    assert suspected({"ord_01": 5.0}, "spindle", z)[0] == "unclassified"
    assert suspected({"hf": 4.0, "tsc_drop": 6.0, "thermistor": 50.0}, "spindle", z)[0] == "rotary union"
    assert suspected({"tsc_drop": 6.0}, "spindle", z)[0] == "unclassified"
    assert suspected({"hf": 40.0, "env_bpfo": 3.5}, "spindle", z)[0] == "bearing"
    assert suspected({"hf": 4.0}, "spindle", z)[0] == "lubrication"
    assert suspected({"thermistor": 4.0, "env_bpfo": 9.0}, "spindle", z)[0] == "spindle cooling"
    assert suspected({"discharge": 4.0, "env_bsf": 30.0}, "compressor", z)[0] == "cooler"
    assert suspected({"ord_02": 4.0, "ord_01": 3.5}, "compressor", z)[0] == "imbalance"
    assert suspected({"axis_drift": 4.0, "thermistor": 30.0}, "machine", z)[0] == "axis wear"
    assert [asset_kind(a) for a in ("CMP-01", "HMC-01-SP", "HMC-01-TSC", "HMC-01-CP", "VMC-03")] == [
        "compressor",
        "spindle",
        "tsc_pump",
        "flood_pump",
        "machine",
    ]
    assert family_of("other machine stoppage", "LTH-01-AX-Z") == "axis wear"
    assert family_of("other machine stoppage", "VMC-04") is None
    assert family_of("cooler (compressor)", "CMP-01") == "cooler"
    assert family_of("compressor trip", "CMP-02") is None
    assert family_of("compressor trip", "CMP-02", True) == "cooler"


def test_champion_selection_prefers_simpler_within_margin():
    sel = pd.DataFrame(
        {"candidate": ["autoencoder_third_octave", "mahalanobis", "pca_t2_spe"], "skill": [0.40, 0.36, 0.20]}
    )
    assert tasks.select_champion(sel, 34, 2) == "mahalanobis"
    assert tasks.select_champion(sel, 34, 1) == "autoencoder_third_octave"


def test_nbm_segments_rebaseline_and_causal_updates():
    rows = _year(4)
    rows = rows[rows["hour_utc"] < pd.Timestamp("2024-09-01", tz=UTC)].reset_index(drop=True)
    repair = pd.Timestamp("2024-06-01", tz=UTC)
    shift = rows["hour_utc"] >= repair
    rows.loc[shift, "hf_rms_g"] *= 3.0
    rows.loc[shift, [c for c in rows.columns if c.startswith("toct_")]] *= 3.0
    cutoff = pd.Timestamp("2024-05-01", tz=UTC)
    rb = pd.Timedelta(days=CFG["models"]["nbm"]["rebaseline_days"])
    win = pd.DataFrame(
        {
            "window_id": ["P|install", "P|repair"],
            "point_id": "P",
            "reason": ["install", "repair W1"],
            "start": [rows["hour_utc"].min(), repair],
            "usable": True,
        }
    )
    end = pd.Timestamp("2024-09-01", tz=UTC)
    plan = nbm.segment_plan(win, set(), end, rb)
    s, info, tr, _fitted = nbm.fit_score_segments(rows, "mahalanobis", CFG, cutoff, [], plan, end, 3, [], [])
    assert [i["model"] for i in info] == ["segment fit", "window refit"]
    assert info[1]["monthly_updates"] == 2 and info[1]["n_train_final"] > info[1]["n_train"]
    assert set(tr["segment"]) == {0, 1}
    july = [(pd.Timestamp("2024-07-01", tz=UTC), pd.Timestamp("2024-08-01", tz=UTC))]
    _, info_b, _, _ = nbm.fit_score_segments(rows, "mahalanobis", CFG, cutoff, [], plan, end, 3, july, [])
    assert info_b[1]["n_train_final"] < info[1]["n_train_final"]
    assert not ((s["hour_utc"] >= repair) & (s["hour_utc"] < repair + rb)).any()
    after = s[s["hour_utc"] >= repair + rb]
    assert len(after) and (after["score"] > after["thr_99.5"]).mean() < 0.05
    flagged = nbm.segment_plan(win, {"P|repair"}, end, rb)
    s2, info2, _, _ = nbm.fit_score_segments(rows, "mahalanobis", CFG, cutoff, [], flagged, end, 3, [], [])
    assert info2[1]["model"].startswith("rule floor only") and not (s2["hour_utc"] >= repair).any()


def test_evaluation_episodes_catch_chance_and_budget():
    t = pd.Timestamp("2025-09-01", tz=UTC)
    conds = pd.DataFrame(
        {
            "wo_id": ["W"],
            "asset_id": ["A"],
            "asset_key": ["A"],
            "point_scope": [None],
            "condition_type": ["imbalance"],
            "unplanned": [True],
            "credit_ts": [t],
            "downtime_hours": [10.0],
            "machine_class": ["hmc"],
        }
    )
    ex = pd.DataFrame(
        {
            "asset_id": "A",
            "point_id": "P",
            "ts": [t - pd.Timedelta(days=20), t - pd.Timedelta(days=15), t + pd.Timedelta(days=100)],
        }
    )
    ep = ev.episodes(ex, 7)
    assert len(ep) == 2 and ep["closed"].iat[0] == t - pd.Timedelta(days=8)
    assert not ev.catch_table(ep, conds, 60)["caught"].iat[0]
    days = [-20, -14, -8, -2, 100]
    ep2 = ev.episodes(
        pd.DataFrame({"asset_id": "A", "point_id": "P", "ts": [t + pd.Timedelta(days=d) for d in days]}), 7
    )
    c = ev.catch_table(ep2, conds, 60)
    assert bool(c["caught"].iat[0]) and abs(c["warning_days"].iat[0] - 20) < 1e-9
    assert not ev.catch_table(ep2, conds, 10)["caught"].iat[0]
    period = (pd.Timestamp("2025-07-01", tz=UTC), pd.Timestamp("2026-04-01", tz=UTC))
    share = ev.chance_share(ep2, conds, period, 60)
    assert 0 < share < 0.2
    row, _ = ev.summarize("nbm", "op", ep2, conds, conds, period, 1, 60, 1.5, {"hmc": 100.0}, 5)
    assert row["caught"] == 1 and row["false_episodes"] == 1 and row["unplanned_cost_plannable_usd"] == 1000.0
    curve = pd.DataFrame(
        {"false_per_asset_month": [0.1, 0.5], "caught_share": [0.2, 0.6], "chance_share": [0.1, 0.3]}
    )
    at = ev.at_budget(curve, 0.3)
    assert abs(at["caught_share"] - 0.4) < 1e-9 and at["within_curve"]


def test_condition_scope_affected_point():
    conds = pd.DataFrame(
        {
            "asset_id": ["CMP-01", "HMC-01-SP", "HMC-02-SP", "HMC-03-X"],
            "failure_code": ["CMP-BRG-AIREND", "coolant_ingress", "BRG-SPALL", None],
        }
    )
    pts = {
        "CMP-01": ["CMP-01-AE", "CMP-01-MDE"],
        "HMC-01-SP": ["HMC-01-SP-F", "HMC-01-SP-R"],
        "HMC-02-SP": ["HMC-02-SP-F", "HMC-02-SP-R"],
    }
    c = ev.condition_scope(conds, pts)
    assert list(c["point_scope"]) == ["CMP-01-AE", "HMC-01-SP-F", None, None]
    assert list(c["asset_key"]) == ["CMP-01", "HMC-01-SP", "HMC-02-SP", "HMC-03"]


def test_shared_shape_weibull():
    from pdm.models import survival

    rng = np.random.default_rng(3)
    rows = []
    for cls, scale in (("a", 500.0), ("b", 2000.0)):
        life = scale * rng.weibull(1.5, 60)
        cens = rng.uniform(100, 3000, 60)
        rows.append(
            pd.DataFrame(
                {
                    "component_class": cls,
                    "duration_days": np.minimum(life, cens),
                    "event": (life <= cens).astype(int),
                    "entry_days": 0.0,
                }
            )
        )
    lives = pd.concat(rows, ignore_index=True)
    t, shape = survival.shared_shape_weibull(lives, 0.9)
    assert t["shape_rho"].nunique() == 1 and 1.1 < shape["shape_rho"] < 2.0
    a, b = t.set_index("component_class")["scale_lambda_days"]
    assert 300 < a < 800 and 1300 < b < 3000
    assert (t["scale_low"] < t["scale_lambda_days"]).all() and (
        t["scale_lambda_days"] < t["scale_high"]
    ).all()


def _life_inputs(replacements: str, install_year: int | None):
    machines = pd.DataFrame([{"machine_id": "HMC-09", "age_class": "newer", "install_year": install_year}])
    pumps = pd.DataFrame(
        [
            {
                "asset_id": "HMC-09-CP",
                "kind": "flood",
                "machine_id": "HMC-09",
                "replacement_dates": replacements,
            }
        ]
    )
    spindles = pd.DataFrame(columns=["machine_id", "rebuild_dates"])
    comps = pd.DataFrame(columns=["asset_id", "motor_bearing_replaced", "airend_rebuilt", "install_year"])
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    wo = pd.DataFrame(
        [
            {
                "asset_id": "HMC-09-CP",
                "wo_type": "unplanned_repair",
                "failure_code": "PMP-BRG",
                "opened": ts("2024-05-10 12:00"),
                "closed": ts("2024-05-11 09:00"),
                "downtime_hours": 8.0,
            },
        ]
    )
    col = pd.DataFrame(columns=["machine_id", "ts_utc"])
    pm = pd.DataFrame(columns=["pm_task", "asset_id", "due", "outcome"])
    return machines, pumps, spindles, comps, wo, col, pm, (ts("2023-07-01"), ts("2026-07-01"))


def test_lives_register_date_on_failure_day_is_the_same_event():
    from pdm.models import survival

    # register date at midnight of the failure's opening day: it is the failure's replacement
    lives = survival.lifetimes(*_life_inputs("2022-03-01;2024-05-10", 2019), close_lag_days=2)
    assert len(lives) == 2
    first, second = lives.sort_values("start").itertuples()
    assert first.event == 1 and first.start == pd.Timestamp("2022-03-01", tz="UTC")
    assert (
        abs(first.duration_days - (pd.Timestamp("2024-05-10 12:00") - pd.Timestamp("2022-03-01")).days - 0.5)
        < 1e-9
    )
    assert second.start == pd.Timestamp("2024-05-11 09:00", tz="UTC") and second.event == 0
    assert (lives["duration_days"] > 1).all()
    # without the rule the midnight date would cut a 0.5-day life ending in the failure
    lag0 = survival.same_event_removed(
        [pd.Timestamp("2024-05-10", tz="UTC")], _life_inputs("", 2019)[4], pd.Timedelta(0)
    )
    assert lag0 == []


def test_lives_start_at_install_when_no_register_date_precedes_the_record():
    from pdm.models import survival

    lives = survival.lifetimes(*_life_inputs("", 2019), close_lag_days=2).sort_values("start")
    first = lives.iloc[0]
    assert first["start"] == pd.Timestamp("2019-07-01", tz="UTC")
    assert first["entry_days"] == (pd.Timestamp("2023-07-01") - pd.Timestamp("2019-07-01")).days
    assert first["event"] == 1


def test_entrypoint_arguments(monkeypatch):
    from pdm import entrypoints

    seen = {}

    class FakeBuilder:
        def getOrCreate(self):
            class S:
                class conf:
                    @staticmethod
                    def set(k, v):
                        seen[k] = v

            return S()

    import pyspark.sql

    monkeypatch.setattr(pyspark.sql.SparkSession, "builder", FakeBuilder())
    store, cfg, args = entrypoints.setup(
        "pdm-test", ["--catalog", "pdm_dev", "--config", "config/config.yaml"]
    )
    assert store.catalog == "pdm_dev" and args.experiment_root == "/Shared/pdm" and "splits" in cfg
    assert seen["spark.sql.session.timeZone"] == "UTC"


def test_lives_register_date_on_preventive_work_order_day_is_that_replacement():
    from pdm.models import survival

    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    inputs = list(_life_inputs("2022-03-01;2025-03-16", 2019))
    pm = pd.DataFrame(
        [
            {
                "asset_id": "HMC-09-CP",
                "wo_type": "pm",
                "failure_code": None,
                "opened": ts("2025-03-16 05:47"),
                "closed": ts("2025-03-16 15:35"),
                "downtime_hours": 4.0,
            },
            {
                "asset_id": "HMC-09-CP",
                "wo_type": "condition_based",
                "failure_code": "imbalance",
                "opened": ts("2025-09-01 11:00"),
                "closed": ts("2025-09-02 08:00"),
                "downtime_hours": 3.0,
            },
        ]
    )
    inputs[4] = pd.concat([inputs[4], pm], ignore_index=True)
    inputs[1].loc[0, "replacement_dates"] = "2022-03-01;2025-03-16;2025-09-01"
    lives = survival.lifetimes(*inputs, close_lag_days=2).sort_values("start")
    assert (lives["duration_days"] > 1).all()
    starts = list(lives["start"])
    assert ts("2025-03-16") not in starts and ts("2025-09-01") not in starts
    assert ts("2025-03-16 15:35") in starts and ts("2025-09-02 08:00") in starts
