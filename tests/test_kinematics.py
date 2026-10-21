"""Fault frequencies against hand-computed values."""

import pytest

from pdm.config import load_register
from pdm.features.kinematics import bearing_fault_orders, fault_frequencies_hz


def test_hmc_front_set_hand_values():
    g = load_register().bearings["7014C-P4"]
    o = bearing_fault_orders(g)
    assert o.bpfo == pytest.approx(8.426, abs=0.001)
    assert o.bpfi == pytest.approx(10.574, abs=0.001)
    assert o.bsf == pytest.approx(4.219, abs=0.001)
    assert o.ftf == pytest.approx(0.4435, abs=0.0001)
    hz = fault_frequencies_hz(o, 12000 / 60)
    assert hz["bpfo"] == pytest.approx(1685.3, abs=0.1)
    assert hz["ftf"] == pytest.approx(88.7, abs=0.1)


def test_compressor_motor_bearing_hand_values():
    o = bearing_fault_orders(load_register().bearings["6316"])
    assert o.bpfo == pytest.approx(3.035, abs=0.001)
    assert o.bpfi == pytest.approx(4.965, abs=0.001)
    assert o.bsf == pytest.approx(1.952, abs=0.001)
    assert o.ftf == pytest.approx(0.3794, abs=0.0001)
