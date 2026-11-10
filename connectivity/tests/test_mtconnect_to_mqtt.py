"""Unit tests for the MTConnect-to-MQTT bridge on hand-built agent responses."""

from __future__ import annotations

import json

import pytest

from idc.bridge import mtconnect_to_mqtt as br

NS = "urn:mtconnect.org:MTConnectStreams:2.7"


def _response(instance_id: int, first: int, last: int, nxt: int, body: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<MTConnectStreams xmlns="{NS}">'
        f'<Header creationTime="2026-11-08T20:00:00Z" sender="idc-agent" instanceId="{instance_id}"'
        f' version="2.7.0.0" bufferSize="131072" firstSequence="{first}" lastSequence="{last}"'
        f' nextSequence="{nxt}"/>'
        f'<Streams>{body}</Streams></MTConnectStreams>'
    )


def _device(name: str, inner: str) -> str:
    return f'<DeviceStream name="{name}" uuid="{name}">{inner}</DeviceStream>'


T0 = "2026-06-09T21:31:08.000Z"
T1 = "2026-06-09T21:31:09.000Z"

def _item(tag, name, seq, ts, text="", **attrs):
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f'<{tag} dataItemId="hmc_01_{name.lower()}" name="{name}" sequence="{seq}"' \
           f' timestamp="{ts}"{extra}>{text}</{tag}>'


def _stream(component, name, group, items):
    return (f'<ComponentStream component="{component}" name="{name}" componentId="c_{name}">'
            f'<{group}>{"".join(items)}</{group}></ComponentStream>')


HMC_BODY = _device("HMC-01", (
    _stream("Device", "HMC-01", "Events", [
        _item("Availability", "avail", 101, T0, "AVAILABLE"),
        _item("RecordTimestamp", "record_ts", 102, T0, T0),
        _item("RecordTimestamp", "record_ts", 110, T1, T1),
    ])
    + _stream("Path", "path", "Events", [
        _item("Execution", "exec", 103, T0, "ACTIVE"),
        _item("Program", "program", 104, T0, "HMC-01-P0137"),
        _item("PathFeedrateOverride", "Fovr", 105, T0, "80", subType="PROGRAMMED"),
    ])
    + _stream("Rotary", "S", "Samples", [
        _item("RotaryVelocity", "Srpm", 106, T0, "8723", subType="ACTUAL"),
        _item("Load", "Sload", 107, T0, "4.8"),
        _item("Load", "Sload", 111, T1, "5.1"),
    ])
    + _stream("Linear", "Y", "Samples", [_item("Load", "Yload", 108, T0, "UNAVAILABLE")])
    + _stream("Controller", "controller", "Condition", [
        _item("Fault", "alarm", 109, T0, "atc_fault", type="SYSTEM", nativeCode="atc_fault"),
        _item("Normal", "alarm", 112, T1, "", type="SYSTEM"),
    ])
))

CONFIG = {
    "mtconnect": {"agent_url": "http://agent:5000", "sample_count": 100, "idle_poll_seconds": 0},
    "mqtt": {"host": "x", "port": 1883, "topic_root": "shop/main", "qos": 1,
             "controller_topic": "{root}/machining/{cell}/{machine_id}/controller",
             "status_topic": "{root}/machining/{cell}/{machine_id}/controller/status"},
}
CELLS = {"HMC-01": "cell-hmc"}


def test_parse_sample_reads_header_and_sorts_by_sequence():
    head, obs = br.parse_sample(_response(7, 101, 112, 113, HMC_BODY))
    assert head == {"instanceId": 7, "firstSequence": 101, "lastSequence": 112,
                    "nextSequence": 113, "bufferSize": 131072}
    assert [o.sequence for o in obs] == list(range(101, 113))
    assert obs[8].kind == "condition" and obs[8].level == "FAULT"
    assert obs[8].native_code == "atc_fault"


def test_group_by_second_builds_one_payload_per_record_second():
    _, obs = br.parse_sample(_response(7, 101, 112, 113, HMC_BODY))
    groups = br.group_by_second(obs)
    assert [(g.device, g.ts) for g in groups] == [("HMC-01", T0), ("HMC-01", T1)]
    first = groups[0].payload(7)
    assert first == {
        "machine_id": "HMC-01", "ts_utc": T0, "avail": "AVAILABLE",
        "execution": "active", "program": "HMC-01-P0137", "feed_override_pct": 80.0,
        "spindle_speed_rpm": 8723.0, "spindle_load_pct": 4.8, "alarm_code": "atc_fault",
        "agent_seq": [101, 109], "instance_id": 7,
    }
    assert "y_load_pct" not in first  # UNAVAILABLE signals are omitted, not nulled
    second = groups[1].payload(7)
    assert second == {"machine_id": "HMC-01", "ts_utc": T1, "spindle_load_pct": 5.1,
                      "alarm_code": "", "agent_seq": [110, 112], "instance_id": 7}


def test_float_values_round_trip_exactly():
    _, obs = br.parse_sample(_response(7, 101, 112, 113, HMC_BODY))
    payload = br.group_by_second(obs)[0].payload(7)
    assert payload["spindle_load_pct"] == 4.8
    assert json.loads(json.dumps(payload))["spindle_load_pct"] == 4.8


def test_topic_follows_isa95_template():
    assert br.topic_for("HMC-01", CONFIG, CELLS) == "shop/main/machining/cell-hmc/HMC-01/controller"
    assert br.topic_for("HMC-01", CONFIG, CELLS, kind="status") == (
        "shop/main/machining/cell-hmc/HMC-01/controller/status")


class _PublishInfoStub:
    mid = 1

    def wait_for_publish(self, timeout=None):
        return None

    def is_published(self):
        return True


class _RecordingClient:
    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    def publish(self, topic, payload, qos):
        self.published.append((topic, json.loads(payload)))
        return _PublishInfoStub()

    def loop_stop(self):
        pass

    def disconnect(self):
        pass


def _bridge(responses: list[str]) -> br.Bridge:
    """Build a Bridge without touching the network; _get returns canned responses."""
    b = object.__new__(br.Bridge)
    b.config, b.cells = CONFIG, CELLS
    b.agent, b.count, b.idle, b.qos = "http://agent:5000", 100, 0.0, 1
    b.stats = br.BridgeStats()
    b.bookmark = b.instance_id = None
    b.pending = {}
    b.device_filter = None
    b.client = _RecordingClient()
    queue = list(responses)
    b._get = lambda path, **params: queue.pop(0)
    return b


def test_poll_publishes_and_advances_bookmark():
    b = _bridge([
        _response(7, 1, 100, 101, ""),                # /current
        _response(7, 1, 112, 113, HMC_BODY),          # /sample from 101
        _response(7, 1, 112, 113, ""),                # /sample from 113: nothing new
    ])
    b.prime()
    assert b.bookmark == 101
    assert b.poll_once() == 12
    assert b.bookmark == 113
    # The newest second (T1) is held back until the agent confirms nothing is in flight.
    assert b.stats.messages == 1
    assert b.poll_once() == 0
    assert b.stats.messages == 2 and b.stats.records == 2 and b.stats.status == 0
    assert b.client.published[0][0] == "shop/main/machining/cell-hmc/HMC-01/controller"
    assert b.client.published[1][1]["agent_seq"] == [110, 112]


def test_buffer_overrun_raises_sequence_gap():
    b = _bridge([
        _response(7, 1, 100, 101, ""),
        _response(7, 150, 300, 301, HMC_BODY),  # oldest kept is 150; bookmark is 101
    ])
    b.prime()
    with pytest.raises(br.SequenceGap, match="49 observations lost"):
        b.poll_once()
    assert b.client.published == []  # nothing published past a hole


def test_agent_restart_raises():
    b = _bridge([
        _response(7, 1, 100, 101, ""),
        _response(8, 1, 12, 13, HMC_BODY),
    ])
    b.prime()
    with pytest.raises(br.AgentRestarted):
        b.poll_once()


def test_unknown_device_is_skipped():
    body = _device("XYZ-99", _stream("Device", "XYZ-99", "Events",
                                     [_item("Availability", "avail", 101, T0, "AVAILABLE")]))
    b = _bridge([_response(7, 1, 100, 101, ""), _response(7, 1, 101, 102, body),
                 _response(7, 1, 101, 102, "")])
    b.prime()
    assert b.poll_once() == 1
    assert b.poll_once() == 0
    assert b.stats.messages == 0


def test_second_split_across_polls_is_published_once():
    first = _device("HMC-01", _stream("Rotary", "S", "Samples", [
        _item("Load", "Sload", 101, T0, "4.8"),
    ]) + _stream("Device", "HMC-01", "Events", [
        _item("RecordTimestamp", "record_ts", 102, T0, T0),
    ]))
    rest = _device("HMC-01", _stream("Linear", "X", "Samples", [
        _item("Load", "Xload", 103, T0, "13.8"),
    ]) + _stream("Device", "HMC-01", "Events", [
        _item("RecordTimestamp", "record_ts", 104, T1, T1),
    ]))
    b = _bridge([
        _response(7, 1, 100, 101, ""),
        _response(7, 1, 102, 103, first),   # poll ends mid-second
        _response(7, 1, 104, 105, rest),    # rest of T0, then T1 begins
        _response(7, 1, 104, 105, ""),      # idle: flush T1
    ])
    b.prime()
    b.poll_once()
    assert b.client.published == []         # T0 held back, nothing published yet
    b.poll_once()
    assert len(b.client.published) == 1     # T0 complete once T1 appears
    msg = b.client.published[0][1]
    assert msg["ts_utc"] == T0
    assert msg["spindle_load_pct"] == 4.8 and msg["x_load_pct"] == 13.8
    assert msg["agent_seq"] == [101, 103]
    assert b.stats.merged == 1
    b.poll_once()
    assert len(b.client.published) == 2     # T1 flushed on the empty poll
    assert b.client.published[1][1]["agent_seq"] == [104, 104]


T_OFF = "2026-06-10T00:00:00Z"
T_WALL = "2026-11-08T22:38:30.0223"


def test_adapter_unavailable_marker_goes_to_status_topic():
    body = _device("HMC-01", _stream("Device", "HMC-01", "Events", [
        _item("RecordTimestamp", "record_ts", 101, T0, T0),
        _item("Availability", "avail", 102, T_OFF, "UNAVAILABLE"),
    ]))
    b = _bridge([_response(7, 1, 100, 101, ""), _response(7, 1, 102, 103, body),
                 _response(7, 1, 102, 103, "")])
    b.prime()
    b.poll_once()
    b.poll_once()
    topics = [t for t, _ in b.client.published]
    assert topics == ["shop/main/machining/cell-hmc/HMC-01/controller",
                      "shop/main/machining/cell-hmc/HMC-01/controller/status"]
    status = b.client.published[1][1]
    assert status == {"machine_id": "HMC-01", "ts_utc": T_OFF, "avail": "UNAVAILABLE",
                      "source": "adapter", "agent_seq": [102, 102], "instance_id": 7}
    assert b.stats.records == 1 and b.stats.status == 1


def test_agent_link_loss_burst_collapses_to_one_status_message():
    # The agent marks every item UNAVAILABLE with its own clock, microseconds apart.
    items = [
        _item("RecordTimestamp", "record_ts", 130, T_WALL + "81Z", "UNAVAILABLE"),
        _item("Load", "Xload", 131, T_WALL + "97Z", "UNAVAILABLE"),
        _item("Load", "Sload", 136, T_WALL + "31Z", "UNAVAILABLE"),
        _item("Execution", "exec", 140, T_WALL + "93Z", "UNAVAILABLE"),
        _item("Program", "program", 141, T_WALL + "01Z", "UNAVAILABLE"),
    ]
    body = _device("HMC-01",
                   _stream("Device", "HMC-01", "Events", [_item("RecordTimestamp", "record_ts",
                                                                 129, T0, T0)] + items[:1])
                   + _stream("Linear", "X", "Samples", items[1:3])
                   + _stream("Path", "path", "Events", items[3:])
                   + _stream("Controller", "controller", "Condition", [
                       _item("Unavailable", "alarm", 138, T_WALL + "45Z", "", type="SYSTEM")]))
    b = _bridge([_response(7, 1, 100, 101, ""), _response(7, 1, 141, 142, body),
                 _response(7, 1, 141, 142, "")])
    b.prime()
    b.poll_once()
    b.poll_once()
    assert b.stats.records == 1 and b.stats.status == 1
    record, status = b.client.published[0][1], b.client.published[1][1]
    assert record["ts_utc"] == T0 and "UNAVAILABLE" not in json.dumps(record)
    assert status["source"] == "agent" and status["avail"] == "UNAVAILABLE"
    assert status["agent_seq"] == [130, 141]


def test_filtered_poll_with_no_matches_advances_past_the_window():
    b = _bridge([
        _response(7, 1, 100, 101, ""),           # /current
        _response(7, 1, 5000, 3001, ""),         # nothing for these devices up to 3000
        _response(7, 1, 5000, 5001, ""),         # caught up with the agent
    ])
    b.prime()
    assert b.poll_once() == 0
    assert b.bookmark == 3001
    assert b.poll_once() == 0
    assert b.bookmark == 5001


def test_held_back_second_is_kept_until_the_bridge_is_caught_up():
    b = _bridge([
        _response(7, 1, 100, 101, ""),
        _response(7, 1, 112, 113, HMC_BODY),
        _response(7, 1, 9000, 4001, ""),         # empty window, agent has more to scan
        _response(7, 1, 9000, 9001, ""),         # caught up
    ])
    b.prime()
    b.poll_once()
    assert b.stats.messages == 1
    b.poll_once()
    assert b.stats.messages == 1                 # newest second still held back
    b.poll_once()
    assert b.stats.messages == 2
