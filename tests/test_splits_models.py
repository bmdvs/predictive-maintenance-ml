"""Splits, per-point models (shape, determinism, leaked-window refusal, seasonal coverage), controller
health, advisory and event-based evaluation on small known inputs."""

import copy

import numpy as np
import pandas as pd
import pytest

from pdm import splits
from pdm.advisory.requests import asset_kind, family_of, suspected
from pdm.config import load_config
from pdm.evaluation import events as ev
from pdm.models import controller_health as ch
from pdm.models import nbm

CFG = load_config()
# the base mechanisms are tested with every model fix off (the frozen switches: test_fix_switches.py)
CFG["models"]["nbm"]["fixes"] = dict(nbm.FIX_DEFAULTS)
UTC = "UTC"


def test_chronological_split_and_gap():
    ts = pd.Series(
        pd.to_datetime(["2023-08-01", "2025-06-25", "2025-08-01", "2026-03-25", "2026-05-01"], utc=True)
    )
    lab = splits.chronological(ts, CFG)
    assert list(lab) == ["training", "gap", "shadow", "shadow", "shadow"]
    assert splits.training_cutoff(CFG) < pd.Timestamp("2025-07-01", tz=UTC)


def test_expanding_folds_inside_training():
    fs = splits.folds(CFG)
    cutoff = splits.training_cutoff(CFG)
    assert [f["fold"] for f in fs] == [1, 2]
    for f in fs:
        assert f["cutoff"] <= f["score"][0] - pd.Timedelta(days=CFG["splits"]["gap_days"])
        assert f["score"][1] <= splits.boundaries(CFG)["train"][1] and f["cutoff"] < cutoff
    assert fs[0]["fit_start"] == fs[1]["fit_start"] and fs[0]["cutoff"] < fs[1]["cutoff"]
    a, b = splits.reference_window(CFG)
    assert b == cutoff and (b - a).days in range(181, 185)


def test_leave_one_asset_out():
    folds = splits.leave_one_asset_out(["a", "b", "c"])
    assert [h for _, h in folds] == ["a", "b", "c"] and all(h not in t for t, h in folds)


def _healthy(n: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="h", tz=UTC)
    d = pd.DataFrame(
        {
            "point_id": "P",
            "asset_id": "A",
            "point_class": "flood_pump",
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
            "env_bpfo_2_g2": rng.lognormal(-9.5, 0.1, n),
            "env_bpfi_2_g2": rng.lognormal(-9.5, 0.1, n),
            "env_bsf_2_g2": rng.lognormal(-9.5, 0.1, n),
            **{c: rng.lognormal(-10, 0.1, n) for c in nbm.FULL_ENV_EXTRA},
            "temp_c": rng.normal(35, 0.3, n),
            "speed_rpm_at_measurement": 3450.0,
            "load_pct_at_measurement": np.nan,
        }
    )
    for b in range(1, 9):
        d[f"toct_{b:02d}_g2"] = rng.lognormal(-6, 0.1, n)
    return d


def _year(seed: int) -> pd.DataFrame:
    return _healthy(24 * 400, "2023-07-05", seed)


def _windows(*starts: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {"point_id": "P", "start": list(starts), "reason": ["install"] + ["repair W"] * (len(starts) - 1)}
    )


def _settings(cand: str, mode: str = "single") -> nbm.Settings:
    if cand in nbm.AUTOENCODERS:
        return nbm.Settings(cand, (8, 4, 8), 50, mode)
    return nbm.Settings(cand, (), 0, mode)


@pytest.mark.parametrize("cand", nbm.CANDIDATES)
def test_nbm_shape_determinism_and_anomaly(cand):
    rows = _year(1)
    cutoff = pd.Timestamp("2024-07-01", tz=UTC)
    win = _windows(rows["hour_utc"].min())
    st = _settings(cand)
    m1, info = nbm.fit_point(rows, win, [st], CFG, rows["hour_utc"].min(), cutoff, [], None, None)
    m2, _ = nbm.fit_point(rows, win, [st], CFG, rows["hour_utc"].min(), cutoff, [], None, None)
    assert set(m1) == {st.key} and info["n_train"] > 1000
    end = cutoff + pd.Timedelta(hours=50)
    s1, _ = nbm.score_point(m1[st.key], rows, win, CFG, cutoff, end, 1.45, 3)
    s2, _ = nbm.score_point(m2[st.key], rows, win, CFG, cutoff, end, 1.45, 3)
    assert len(s1) == 50 and (s1["state"] == "scored").all() and np.allclose(s1["score"], s2["score"])
    assert {f"thr_{p:g}" for p in CFG["models"]["nbm"]["sweep_percentiles"]} <= set(s1.columns)
    bad = rows.copy()
    late = bad["hour_utc"] >= cutoff
    bad.loc[late, "hf_rms_g"] *= 4
    bad.loc[late, "env_bpfo_1_g2"] *= 50
    bad.loc[late, [c for c in bad.columns if c.startswith("toct_")]] *= 20
    s3, _ = nbm.score_point(m1[st.key], bad, win, CFG, cutoff, end, 1.45, 3)
    assert s3["score"].median() > s1["score"].median()


def test_autoencoder_weight_decay_default_is_unchanged_and_early_stopping_keys():
    """2026-11-01: weight decay None trains with the library default 1e-4 (same scores as an explicit 1e-4,
    same key as before); early stopping replaces the epoch count in the key and records the best epoch."""
    rows = _year(1)
    cutoff = pd.Timestamp("2024-07-01", tz=UTC)
    win = _windows(rows["hour_utc"].min())
    cand = "autoencoder_vendor"
    base = nbm.Settings(cand, (8, 4, 8), 50, "single")
    wd = nbm.Settings(cand, (8, 4, 8), 50, "single", weight_decay=1e-4)
    es = nbm.Settings(cand, (8, 4, 8), 0, "single", early_stopping=True)
    assert base.key == "autoencoder_vendor [8-4-8] 50 ep, single"
    assert wd.key == "autoencoder_vendor [8-4-8] 50 ep, wd 0.0001, single"
    assert es.key == "autoencoder_vendor [8-4-8] early stopping, single"
    m, _ = nbm.fit_point(rows, win, [base, wd, es], CFG, rows["hour_utc"].min(), cutoff, [], None, None)
    end = cutoff + pd.Timedelta(hours=50)
    s0, _ = nbm.score_point(m[base.key], rows, win, CFG, cutoff, end, 1.45, 3)
    s1, _ = nbm.score_point(m[wd.key], rows, win, CFG, cutoff, end, 1.45, 3)
    s2, _ = nbm.score_point(m[es.key], rows, win, CFG, cutoff, end, 1.45, 3)
    assert np.array_equal(s0["score"].to_numpy(), s1["score"].to_numpy())
    assert np.isfinite(s2["score"]).all()


def _shifted(factor: float, repair: pd.Timestamp, seed: int) -> pd.DataFrame:
    rows = _year(seed)
    shift = rows["hour_utc"] >= repair
    for c in ("hf_rms_g", "vel_rms_mm_s", "accel_rms_g", *[c for c in rows.columns if c.startswith("toct_")]):
        rows.loc[shift, c] *= factor
    return rows


def test_nbm_recentering_absorbs_a_plausible_level_shift():
    repair = pd.Timestamp("2024-07-15", tz=UTC)
    rows = _shifted(1.2, repair, 5)
    cutoff = pd.Timestamp("2024-07-01", tz=UTC)
    win = _windows(rows["hour_utc"].min(), repair)
    st = _settings("mahalanobis")
    m, _ = nbm.fit_point(rows, win, [st], CFG, rows["hour_utc"].min(), cutoff, [], None, None)
    s, segs = nbm.score_point(m[st.key], rows, win, CFG, cutoff, pd.Timestamp("2024-09-01", tz=UTC), 1.45, 3)
    lo = pd.Timedelta(days=CFG["models"]["nbm"]["recenter_days"][0])
    forming = s[(s["hour_utc"] >= repair) & (s["hour_utc"] < repair + lo)]
    assert len(forming) and (forming["state"] == "recentering").all() and forming["score"].isna().all()
    after = s[(s["hour_utc"] >= repair + pd.Timedelta(days=8)) & (s["state"] == "scored")]
    assert len(after) and (after["score"] > after["threshold"]).mean() < 0.05
    assert not segs["plausibility_flag"].any() and list(segs["trained"]) == [True, False]


def test_nbm_plausibility_flags_a_high_level_after_the_cutoff():
    repair = pd.Timestamp("2024-07-15", tz=UTC)
    rows = _shifted(3.0, repair, 6)
    cutoff = pd.Timestamp("2024-07-01", tz=UTC)
    win = _windows(rows["hour_utc"].min(), repair)
    st = _settings("mahalanobis")
    m, _ = nbm.fit_point(rows, win, [st], CFG, rows["hour_utc"].min(), cutoff, [], None, None)
    s, segs = nbm.score_point(m[st.key], rows, win, CFG, cutoff, pd.Timestamp("2024-09-01", tz=UTC), 1.45, 3)
    assert bool(segs["plausibility_flag"].iat[1]) and "hf_rms_g" in segs["plausibility_over"].iat[1]
    after = s[s["hour_utc"] >= repair + pd.Timedelta(days=8)]
    assert len(after) and (after["state"] == "plausibility").all() and after["score"].isna().all()
    low = _shifted(0.4, repair, 6)
    _, segs_low = nbm.score_point(
        m[st.key], low, win, CFG, cutoff, pd.Timestamp("2024-09-01", tz=UTC), 1.45, 3
    )
    assert not segs_low["plausibility_flag"].any()


def test_nbm_trains_across_segments_before_the_cutoff():
    repair = pd.Timestamp("2024-01-15", tz=UTC)
    rows = _shifted(3.0, repair, 7)
    cutoff = pd.Timestamp("2024-07-01", tz=UTC)
    win = _windows(rows["hour_utc"].min(), repair)
    st = _settings("mahalanobis")
    m, _ = nbm.fit_point(rows, win, [st], CFG, rows["hour_utc"].min(), cutoff, [], None, None)
    pm = m[st.key]
    assert pm.segments_trained == 2 and len(pm.train_levels) == 2
    s, segs = nbm.score_point(pm, rows, win, CFG, rows["hour_utc"].min(), cutoff, 1.45, 3)
    sc = s[s["state"] == "scored"]
    pre, post = sc[sc["hour_utc"] < repair], sc[sc["hour_utc"] >= repair]
    assert len(pre) and len(post) and (post["score"] > post["threshold"]).mean() < 0.05
    assert segs["trained"].all() and not segs["plausibility_flag"].any()


def test_nbm_settings_grid_and_complexity():
    grid = nbm.settings_grid(CFG)
    g = CFG["models"]["nbm"]["selection_grid"]
    n_ae = len(g["autoencoder_hidden"]) * len(g["autoencoder_epochs"])
    # the configured grid: the adopted design only (fold diagnosis round 2 decision, 2026-10-31)
    cands = CFG["models"]["nbm"]["candidates"]
    n_ae = sum(c in nbm.AUTOENCODERS for c in cands) * n_ae
    assert len(grid) == len(g["threshold_mode"]) * (sum(c not in nbm.AUTOENCODERS for c in cands) + n_ae)
    assert [s.key for s in grid] == ["mahalanobis_extended, single"]  # base-model decision 2026-10-31
    # the 42-setting grid the selection ran on (checkpoint (b) diagnosis decisions 2026-10-30)
    full = copy.deepcopy(CFG)
    full["models"]["nbm"]["candidates"] = [c for c in nbm.CANDIDATES if nbm.input_set(c) in nbm.INPUT_SETS]
    full["models"]["nbm"]["selection_grid"]["threshold_mode"] = ["single", "speed_band"]
    big = nbm.settings_grid(full)
    assert len(big) == 2 * len(nbm.INPUT_SETS) * (
        3 + len(g["autoencoder_hidden"]) * len(g["autoencoder_epochs"])
    )
    order = sorted(big, key=nbm.complexity)
    assert nbm.model_type(order[0].candidate) == "pca" and order[0].threshold_mode == "single"
    assert nbm.model_type(order[-1].candidate) == "autoencoder"


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
    ex = nbm.exceedance_hours(s, "thr_99.5", 6)
    assert list(ex["ts"]) == [s["hour_utc"].iat[7]]
    s["state"] = ["scored"] * 7 + ["recentering"]
    assert nbm.exceedance_hours(s, "thr_99.5", 6).empty


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


def test_ml_cases_and_status_from_episodes():
    from pdm import ml_tasks

    t = pd.Timestamp("2025-09-01", tz=UTC)
    eps = pd.DataFrame(
        {
            "asset_id": ["A", "A"],
            "opened": [t, t + pd.Timedelta(days=40)],
            "last": [t + pd.Timedelta(days=2), t + pd.Timedelta(days=41)],
            "closed": [t + pd.Timedelta(days=9), t + pd.Timedelta(days=48)],
            "trigger_points": ["A-F", "A-R"],
            "trigger_rules": ["", ""],
            "alarm_ts": [t + pd.Timedelta(days=1), pd.NaT],
        }
    )
    c = ml_tasks.ml_cases(eps)
    assert list(c["level"]) == ["ALARM", "ALERT"] and c["case_id"].iat[0] == "A|2025090100"
    st = ml_tasks.status_changes(eps)
    assert list(st["status"]) == ["ALERT", "ALARM", "OK", "ALERT", "OK"]


def test_advisory_rules():
    z = 3.0
    assert suspected({"env_bpfi": 5.0, "hf": 40.0}, "flood_pump", z)[0] == "bearing"
    assert suspected({"hf": 40.0}, "flood_pump", z)[0] == "cavitation or wear"
    assert suspected({"env_bpfi": 5.0, "hf": 40.0, "tsc_drop": 4.0}, "tsc_pump", z)[0] == "bearing"
    assert suspected({"hf": 40.0, "tsc_drop": 4.0}, "tsc_pump", z)[0] == "cavitation or wear"
    assert suspected({"ord_01": 5.0, "hf": 9.0}, "flood_pump", z)[0] == "imbalance"
    assert suspected({"ord_01": 5.0}, "spindle", z)[0] == "unclassified"
    assert suspected({"hf": 4.0, "tsc_drop": 6.0, "thermistor": 50.0}, "spindle", z)[0] == "rotary union"
    assert suspected({"tsc_drop": 6.0}, "spindle", z)[0] == "unclassified"
    assert suspected({"hf": 40.0, "env_bpfo": 3.5}, "spindle", z)[0] == "bearing"
    assert suspected({"hf": 4.0, "temp": 4.0}, "spindle", z)[0] == "lubrication"
    assert suspected({"hf": 4.0}, "spindle", z)[0] == "contamination"
    assert suspected({"temp": 4.0}, "spindle", z)[0] == "spindle cooling"
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
    from pdm import ml_tasks

    sel = pd.DataFrame(
        {
            "settings": [
                "autoencoder_third_octave [16-6-16] 200 ep, single",
                "mahalanobis, single",
                "pca_t2_spe, single",
            ],
            "skill": [0.40, 0.36, 0.20],
            "complexity_rank": [6, 2, 0],
        }
    )
    assert ml_tasks.select_champion(sel, 34, 2) == "mahalanobis, single"
    assert ml_tasks.select_champion(sel, 34, 1) == "autoencoder_third_octave [16-6-16] 200 ep, single"


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
            "failure_code": ["CMP-BRG-AIREND", "SP-COOLANT-INGRESS", "BRG-SPALL", None],
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


def test_per_input_limits_alert_alongside_the_score():
    """Fold diagnosis round 2 decision (2026-10-31): any single standardized input persisted above its own
    healthy-training limit raises a persisted exceedance even when the score stays below its threshold."""
    rows = _year(7)
    cutoff = pd.Timestamp("2024-07-01", tz=UTC)
    win = _windows(rows["hour_utc"].min())
    st = _settings("pca_t2_spe_vendor")
    cfg = copy.deepcopy(CFG)
    cfg["models"]["nbm"]["input_limit_percentile"] = 99.9  # design 1 (dropped from the adopted design)
    m, _ = nbm.fit_point(rows, win, [st], cfg, rows["hour_utc"].min(), cutoff, [], None, None)
    pm = m[st.key]
    assert set(pm.input_limits) == set(pm.columns)
    end = cutoff + pd.Timedelta(hours=48)
    calm, _ = nbm.score_point(pm, rows, win, cfg, cutoff, end, 1.45, 3)
    bad = rows.copy()
    late = bad["hour_utc"] >= cutoff + pd.Timedelta(hours=10)
    bad.loc[late, "kurtosis"] += 2.0
    s, _ = nbm.score_point(pm, bad, win, cfg, cutoff, end, 1.45, 3)
    assert s["limit_inputs"].str.contains("kurtosis").sum() >= 30
    ex = nbm.exceedance_hours(s.assign(score=0.0), "threshold", 6)
    assert len(ex) and ex["source"].str.contains("limit:kurtosis").all()
    assert len(nbm.exceedance_hours(calm.assign(score=0.0), "threshold", 6)) <= 2


def test_scoring_identical_under_two_thread_counts():
    """Parity failure decisions (2026-10-31): fitting and scoring give the same scores within 1e-9 under one
    and four BLAS threads."""
    from threadpoolctl import threadpool_limits

    rows = _year(11)
    cutoff = pd.Timestamp("2024-07-01", tz=UTC)
    win = _windows(rows["hour_utc"].min())
    st = _settings("pca_t2_spe_vendor")
    end = cutoff + pd.Timedelta(hours=72)
    out = []
    for n in (1, 4):
        with threadpool_limits(limits=n):
            m, _ = nbm.fit_point(rows, win, [st], CFG, rows["hour_utc"].min(), cutoff, [], None, None)
            s, _ = nbm.score_point(m[st.key], rows, win, CFG, cutoff, end, 1.45, 3)
        out.append(s["score"].to_numpy(float))
    assert np.allclose(out[0], out[1], rtol=1e-9, atol=1e-12)


def test_full_rank_pca_scores_on_t2_alone_and_scale_floor():
    """Where the PCA keeps every component of the training rank there is no residual subspace: the score is
    T2 alone (SPE undefined). The robust scale is floored at the standard deviation."""
    sc = nbm.Scaler.fit(np.column_stack([np.r_[np.zeros(90), np.full(10, 50.0)], np.ones(100)]))
    assert sc.scale[0] >= np.std(np.r_[np.zeros(90), np.full(10, 50.0)]) - 1e-12 and sc.scale[1] == 1.0
    rng = np.random.default_rng(3)
    z = np.column_stack([rng.normal(size=(500, 2)), np.zeros((500, 3))])
    st = _settings("pca_t2_spe_vendor")
    model, extra = nbm._fit_estimator(z, st, CFG)
    assert extra["rank"] == 2 and model.n_components_ == 2 and extra["spe_med"] is None
    pm = nbm.PointModel(
        "P",
        "flood_pump",
        st,
        ["a", "b"],
        None,
        None,
        {},
        nbm.Scaler(np.zeros(5), np.ones(5)),
        model,
        extra,
        {},
        [],
        500,
        1,
    )
    s, _ = pm.raw_scores(z[:10])
    t = model.transform(z[:10])
    assert np.allclose(s, np.sum(t**2 / model.explained_variance_, axis=1) / extra["t2_med"])


def test_numerics_gate_fails_below_the_floor():
    from types import SimpleNamespace

    from pdm import registry

    ok = SimpleNamespace(
        scaler=SimpleNamespace(scale=np.array([0.1, 1.0])), extra={"t2_med": 5.0, "spe_med": None}
    )
    bad = SimpleNamespace(
        scaler=SimpleNamespace(scale=np.array([1e-8, 1.0])), extra={"t2_med": 5.0, "spe_med": 1e-29}
    )
    assert registry.gate_numerics(SimpleNamespace(models={"A": ok})).passed
    g = registry.gate_numerics(SimpleNamespace(models={"A": ok, "B": bad}))
    assert not g.passed and "B input scale" in g.detail and "B spe_med" in g.detail


def test_mahalanobis_scores_from_the_fitted_location():
    """Centering defect fix (2026-10-31): a reading at the fitted location scores 0, and on Gaussian data
    the mean training score is within 10 percent of the column count."""
    rng = np.random.default_rng(5)
    z = rng.normal(loc=0.4, scale=1.0, size=(4000, 8))
    st = _settings("mahalanobis")
    prec, extra = nbm._fit_estimator(z, st, CFG)
    pm = nbm.PointModel(
        "P",
        "flood_pump",
        st,
        [f"c{j}" for j in range(8)],
        None,
        None,
        {},
        nbm.Scaler(np.zeros(8), np.ones(8)),
        prec,
        extra,
        {},
        [],
        len(z),
        1,
    )
    s0, _ = pm.raw_scores(extra["location"][None, :])
    assert abs(float(s0[0])) < 1e-12
    s, _ = pm.raw_scores(z)
    assert abs(float(s.mean()) - 8) <= 0.8
