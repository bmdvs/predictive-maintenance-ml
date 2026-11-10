"""Sparkplug B codec: round trips, signed integers, topics and the rebirth command."""

from __future__ import annotations

import pytest

from idc.landing import sparkplug as sp


def test_scalar_metrics_round_trip():
    metrics = [sp.Metric("s00_r0", 1, 1700000000000, sp.DATATYPE_INT32, 27166),
               sp.Metric("s00_r1", 2, None, sp.DATATYPE_INT32, 65535),
               sp.Metric(None, 3, None, sp.DATATYPE_INT32, -5),
               sp.Metric("flag", None, None, sp.DATATYPE_BOOLEAN, True),
               sp.Metric("text", None, None, sp.DATATYPE_STRING, "loaded"),
               sp.Metric("gone", None, None, sp.DATATYPE_INT32, None, is_null=True)]
    ts, seq, decoded = sp.decode(sp.encode(metrics, 1700000000123, seq=9))
    assert (ts, seq) == (1700000000123, 9)
    assert [m.name for m in decoded] == ["s00_r0", "s00_r1", None, "flag", "text", "gone"]
    assert [m.alias for m in decoded] == [1, 2, 3, None, None, None]
    assert [m.value for m in decoded] == [27166, 65535, -5, True, "loaded", None]
    assert decoded[0].timestamp_ms == 1700000000000
    assert decoded[-1].is_null


def test_topic_parsing():
    t = sp.Topic.parse("spBv1.0/shop/DDATA/main/CMP-01")
    assert (t.group, t.message_type, t.edge_node, t.device) == ("shop", "DDATA", "main",
                                                                "CMP-01")
    assert sp.Topic.parse("spBv1.0/shop/NBIRTH/main").device is None
    with pytest.raises(ValueError):
        sp.Topic.parse("shop/main/machining")


def test_rebirth_command():
    _, _, metrics = sp.decode(sp.rebirth_command(1))
    assert metrics[0].name == "Node Control/Rebirth"
    assert metrics[0].value is True
