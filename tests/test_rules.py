"""ISO zones, vendor baseline rules, persistence and suppression."""

import pandas as pd
import pytest

from pdm.config import load_config
from pdm.rules.iso_zones import zone, zone_b_width
from pdm.rules.persistence import ConsecutiveCounter, Suppressor, persisted
from pdm.rules.vendor_baseline import VendorRuleSettings, VendorRuleState


@pytest.mark.parametrize(
    "group,foundation,bounds",
    [
        (1, "rigid", (2.3, 4.5, 7.1)),
        (1, "flexible", (3.5, 7.1, 11.0)),
        (2, "rigid", (1.4, 2.8, 4.5)),
        (2, "flexible", (2.3, 4.5, 7.1)),
    ],
)
def test_iso_zone_boundaries(group, foundation, bounds):
    ab, bc, cd = bounds
    assert zone(ab - 1e-9, group, foundation) == "A"
    assert zone(ab, group, foundation) == "B"
    assert zone(bc, group, foundation) == "C"
    assert zone(cd, group, foundation) == "D"
    assert zone_b_width(group, foundation) == pytest.approx(bc - ab)


def _state():
    cfg = load_config()
    st = VendorRuleState("P", VendorRuleSettings.for_point_class(cfg, "spindle", "spindle"))
    st.baseline = {"vel_rms_mm_s": 1.0, "hf_rms_g": 0.1, **{c: 1.0 for c in st.settings.env_columns()}}
    return st


def _row(vel=1.0, hf=0.1, env=1.0, cols=()):
    return {"vel_rms_mm_s": vel, "hf_rms_g": hf, **{c: env for c in cols}}


def test_vendor_thresholds():
    st = _state()
    th = st.thresholds()
    assert th["vel_alert"] == pytest.approx(1.0 + 0.25 * 1.4)
    assert th["vel_alarm"] == pytest.approx(1.0 + 0.50 * 1.4)
    assert th["hf_alert"] == pytest.approx(0.1 + 0.25 * 0.20)


def test_velocity_alert_needs_two_consecutive():
    st = _state()
    cols = st.settings.env_columns()
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    assert st.update(t0, _row(vel=1.4, cols=cols)) is None
    a = st.update(t0 + pd.Timedelta(hours=1), _row(vel=1.4, cols=cols))
    assert a is not None and a["level"] == "alert" and "vel_alert" in a["rules"]


def test_envelope_rule_fires_on_third():
    st = _state()
    cols = st.settings.env_columns()
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    hot = _row(cols=cols)
    hot[cols[0]] = 10**0.6 * 1.01
    assert st.update(t0, hot) is None
    assert st.update(t0 + pd.Timedelta(hours=1), hot) is None
    a = st.update(t0 + pd.Timedelta(hours=2), hot)
    assert a is not None and a["rules"] == "envelope"


def test_no_measurement_neither_counts_nor_breaks():
    assert persisted([True, None, True], 2) == [False, False, True]
    c = ConsecutiveCounter(2)
    c.update(True)
    c.update(None)
    assert c.update(False) is False


def test_suppression_seven_days():
    sup = Suppressor(7)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    assert sup.allow(t0)
    assert not sup.allow(t0 + pd.Timedelta(days=6, hours=23))
    assert sup.allow(t0 + pd.Timedelta(days=7))
