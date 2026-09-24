"""Per-month horizons, shadow evaluation periods, the scoring log and alert rows for monitoring."""

import copy

import numpy as np
import pandas as pd
import pytest

from pdm import entrypoints, tasks
from pdm.config import load_config

UTC = "UTC"
BASE = load_config()


def _cfg(holdout: bool, as_of: str | None) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg["evaluation"]["holdout_shadow"] = holdout
    cfg["evaluation"]["as_of"] = as_of
    return cfg


def test_horizon_by_month():
    assert tasks.horizon(_cfg(True, None)) == pd.Timestamp("2026-04-01 05:00", tz=UTC)
    assert tasks.horizon(_cfg(False, None)) == pd.Timestamp("2026-07-01 05:00", tz=UTC)
    assert tasks.horizon(_cfg(False, "2026-04")) == pd.Timestamp("2026-05-01 05:00", tz=UTC)
    assert tasks.horizon(_cfg(False, "2026-06")) == pd.Timestamp("2026-07-01 05:00", tz=UTC)
    with pytest.raises(ValueError):
        tasks.horizon(_cfg(False, "2026-03"))
    with pytest.raises(ValueError):
        tasks.horizon(_cfg(False, "2026-07"))
    with pytest.raises(ValueError):
        tasks.horizon(_cfg(True, "2026-04"))


def test_shadow_periods():
    april = tasks.shadow_periods(_cfg(False, "2026-04"))
    assert [p[0] for p in april] == ["2026-04", "2026-Q2"]
    assert april[0][1] == pd.Timestamp("2026-04-01 05:00", tz=UTC) and april[0][2] == pd.Timestamp(
        "2026-05-01 05:00", tz=UTC
    )
    may = tasks.shadow_periods(_cfg(False, "2026-05"))
    assert may[0][1] == pd.Timestamp("2026-05-01 05:00", tz=UTC) and may[1][1] == pd.Timestamp(
        "2026-04-01 05:00", tz=UTC
    )
    june = tasks.shadow_periods(_cfg(False, "2026-06"))
    assert [p[0] for p in june] == ["2026-06", "2026-Q2", "2025-07_to_2026-06"]
    assert june[2][1] == pd.Timestamp("2025-07-01 05:00", tz=UTC) and june[2][2] == pd.Timestamp(
        "2026-07-01 05:00", tz=UTC
    )


def test_scoring_log_tier_and_outcome_window():
    cfg = _cfg(False, "2026-04")
    hours = pd.date_range("2026-03-01", "2026-04-30 23:00", freq="h", tz=UTC)
    sc = pd.DataFrame(
        {
            "point_id": "P",
            "asset_id": "A",
            "hour_utc": hours,
            "score": np.where(hours >= pd.Timestamp("2026-03-20", tz=UTC), 3.0, 1.0),
            "thr_99.5": 2.0,
            "model_version": "nbm_point_models/2",
        }
    )
    status = pd.DataFrame(
        {"asset_id": ["A"], "ts": [pd.Timestamp("2026-03-21", tz=UTC)], "status": ["ALARM"]}
    )
    conds = pd.DataFrame({"asset_key": ["A"], "credit_ts": [pd.Timestamp("2026-03-25", tz=UTC)]})
    end = tasks.horizon(cfg)
    log = tasks.scoring_log(sc, status, conds, "thr_99.5", pd.Timestamp("2026-03-01", tz=UTC), end, 30, cfg)
    assert len(log) == len(hours)
    at = log.set_index("hour_utc")
    assert at.loc[pd.Timestamp("2026-03-10", tz=UTC), "tier"] == "OK"
    assert at.loc[pd.Timestamp("2026-03-22", tz=UTC), "tier"] == "ALARM"
    assert at.loc[pd.Timestamp("2026-03-22", tz=UTC), "prediction"] == 1
    assert at.loc[pd.Timestamp("2026-03-10", tz=UTC), "outcome"] == 1.0
    assert at.loc[pd.Timestamp("2026-03-01", tz=UTC), "outcome"] == 1.0
    assert at.loc[pd.Timestamp("2026-03-26", tz=UTC), "outcome"] == 0.0
    assert at.loc[pd.Timestamp("2026-04-15", tz=UTC), "outcome"] is None
    assert set(log["period"]) <= {"validation", "gap", "shadow"}


def test_alert_rows_window():
    end = pd.Timestamp("2026-05-01 05:00", tz=UTC)
    ep = pd.DataFrame(
        {
            "asset_id": ["A", "A", "B"],
            "opened": pd.to_datetime(["2026-03-01", "2026-03-20", "2026-04-20"], utc=True),
        }
    )
    conds = pd.DataFrame({"asset_key": ["A"], "credit_ts": [pd.Timestamp("2026-04-05", tz=UTC)]})
    r = tasks.alert_rows(ep, conds, end, 30)
    assert r["confirmed_within_window"].iloc[0] == 0.0
    assert r["confirmed_within_window"].iloc[1] == 1.0
    assert np.isnan(r["confirmed_within_window"].iloc[2])
    empty = tasks.alert_rows(ep.iloc[:0], conds, end, 30)
    assert str(empty["ts"].dtype).startswith("datetime64")


def test_entrypoint_as_of():
    a = entrypoints.parse("pdm-score", ["--catalog=pdm", "--config=c.yaml", "--as_of=2026-05"])
    assert a.as_of == "2026-05"
    assert entrypoints.parse("pdm-score", ["--catalog=pdm", "--config=c.yaml", "--as_of="]).as_of == ""
    assert entrypoints.parse("pdm-score", ["--catalog", "pdm", "--config", "c.yaml"]).as_of is None


def test_uniform_types_before_write():
    d = pd.DataFrame({"a": [True, 3, None], "b": ["x", 1, None], "c": [1.0, 2.0, np.nan]})
    u = tasks._uniform_types(d)
    assert u["a"].tolist()[:2] == [1.0, 3.0] and np.isnan(u["a"].iloc[2])
    assert u["b"].tolist() == ["x", "1", None]
    assert u["c"].dtype == float


def test_shadow_months():
    assert tasks.shadow_months(_cfg(False, "2026-04")) == ["2026-04"]
    assert tasks.shadow_months(_cfg(False, "2026-06")) == ["2026-04", "2026-05", "2026-06"]
    assert tasks.shadow_months(_cfg(False, None)) == ["2026-04", "2026-05", "2026-06"]


def test_case_trigger_text_does_not_depend_on_row_order():
    from pdm.fusion.status import cases

    t0 = pd.Timestamp("2026-04-10 10:00", tz=UTC)
    sig = pd.DataFrame(
        {
            "method": ["vendor", "nbm", "iso", "vendor"],
            "asset_id": "A",
            "machine_id": "A",
            "point_id": ["P", "P", "P", "P"],
            "ts": [t0, t0, t0, t0 + pd.Timedelta(hours=1)],
            "level": ["alarm", "alarm", "alarm", "alarm"],
            "rules": ["vendor:alarm", "nbm:x,y", "iso:alarm", "vendor:alarm"],
        }
    )
    changes = pd.DataFrame({"asset_id": ["A"], "ts": [t0], "status": ["ALARM"], "layers": ["vendor"]})
    wo = pd.DataFrame(columns=["asset_id", "wo_id", "wo_type", "failure_code", "opened", "closed"])
    a = cases(changes, sig, wo, 7)
    b = cases(changes, sig.iloc[[2, 0, 1, 3]].reset_index(drop=True), wo, 7)
    assert a["trigger_rules"].tolist() == b["trigger_rules"].tolist()
    assert a["trigger_rules"].iat[0].split(";")[:3] == ["iso:iso:alarm", "nbm:nbm:x,y", "vendor:vendor:alarm"]
