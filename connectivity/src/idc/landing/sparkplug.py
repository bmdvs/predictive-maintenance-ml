"""Sparkplug B payload encoding and decoding.

The message types are built at import time from the field definitions of Eclipse Tahu's
``sparkplug_b.proto`` (release v1.0.22, package ``org.eclipse.tahu.protobuf``), restricted to
the fields this chain reads and writes: Payload timestamp, metrics, seq, uuid and body, and
Metric name, alias, timestamp, datatype, flags and scalar values. Other fields (metadata,
property sets, datasets, templates, extensions) are kept by protobuf as unknown fields and
ignored. Field numbers and types match the published schema, so payloads are wire-compatible.

Topic namespace: ``spBv1.0/<group>/<NBIRTH|NDATA|NDEATH|NCMD|DBIRTH|DDATA|DDEATH|DCMD>/
<edge node>[/<device>]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

TAHU_VERSION = "1.0.22"
NAMESPACE = "spBv1.0"

# Sparkplug B data types (subset used for scalar metrics).
DATATYPE_INT8, DATATYPE_INT16, DATATYPE_INT32, DATATYPE_INT64 = 1, 2, 3, 4
DATATYPE_UINT8, DATATYPE_UINT16, DATATYPE_UINT32, DATATYPE_UINT64 = 5, 6, 7, 8
DATATYPE_FLOAT, DATATYPE_DOUBLE, DATATYPE_BOOLEAN, DATATYPE_STRING = 9, 10, 11, 12
DATATYPE_DATETIME, DATATYPE_TEXT = 13, 14

_F = descriptor_pb2.FieldDescriptorProto


def _build_payload_class():
    file = descriptor_pb2.FileDescriptorProto(name="idc_sparkplug_b.proto",
                                              package="org.eclipse.tahu.protobuf",
                                              syntax="proto2")
    payload = file.message_type.add(name="Payload")
    metric = payload.nested_type.add(name="Metric")
    metric.oneof_decl.add(name="value")
    for name, number, kind in [
        ("name", 1, _F.TYPE_STRING), ("alias", 2, _F.TYPE_UINT64),
        ("timestamp", 3, _F.TYPE_UINT64), ("datatype", 4, _F.TYPE_UINT32),
        ("is_historical", 5, _F.TYPE_BOOL), ("is_transient", 6, _F.TYPE_BOOL),
        ("is_null", 7, _F.TYPE_BOOL),
    ]:
        metric.field.add(name=name, number=number, type=kind, label=_F.LABEL_OPTIONAL)
    for name, number, kind in [
        ("int_value", 10, _F.TYPE_UINT32), ("long_value", 11, _F.TYPE_UINT64),
        ("float_value", 12, _F.TYPE_FLOAT), ("double_value", 13, _F.TYPE_DOUBLE),
        ("boolean_value", 14, _F.TYPE_BOOL), ("string_value", 15, _F.TYPE_STRING),
        ("bytes_value", 16, _F.TYPE_BYTES),
    ]:
        metric.field.add(name=name, number=number, type=kind, label=_F.LABEL_OPTIONAL,
                         oneof_index=0)
    payload.field.add(name="timestamp", number=1, type=_F.TYPE_UINT64, label=_F.LABEL_OPTIONAL)
    payload.field.add(name="metrics", number=2, type=_F.TYPE_MESSAGE,
                      label=_F.LABEL_REPEATED,
                      type_name=".org.eclipse.tahu.protobuf.Payload.Metric")
    payload.field.add(name="seq", number=3, type=_F.TYPE_UINT64, label=_F.LABEL_OPTIONAL)
    payload.field.add(name="uuid", number=4, type=_F.TYPE_STRING, label=_F.LABEL_OPTIONAL)
    payload.field.add(name="body", number=5, type=_F.TYPE_BYTES, label=_F.LABEL_OPTIONAL)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("org.eclipse.tahu.protobuf.Payload"))


Payload = _build_payload_class()

_SIGNED_BITS = {DATATYPE_INT8: 8, DATATYPE_INT16: 16, DATATYPE_INT32: 32, DATATYPE_INT64: 64}


@dataclass
class Metric:
    name: str | None
    alias: int | None
    timestamp_ms: int | None
    datatype: int | None
    value: object
    is_null: bool = False
    is_historical: bool = False


@dataclass
class Topic:
    group: str
    message_type: str
    edge_node: str
    device: str | None

    @classmethod
    def parse(cls, topic: str) -> Topic:
        parts = topic.split("/")
        if len(parts) not in (4, 5) or parts[0] != NAMESPACE:
            raise ValueError(f"not a Sparkplug B topic: {topic}")
        return cls(parts[1], parts[2], parts[3], parts[4] if len(parts) == 5 else None)


def _metric_value(m) -> object:
    kind = m.WhichOneof("value")
    if kind is None:
        return None
    value = getattr(m, kind)
    bits = _SIGNED_BITS.get(m.datatype)
    if bits and kind in ("int_value", "long_value") and value >= 1 << (bits - 1):
        value -= 1 << (32 if kind == "int_value" else 64)
    return value


def decode(payload: bytes) -> tuple[int | None, int | None, list[Metric]]:
    """Decode a payload into (timestamp ms, seq, metrics)."""
    msg = Payload()
    msg.ParseFromString(payload)
    metrics = [Metric(name=m.name if m.HasField("name") else None,
                      alias=m.alias if m.HasField("alias") else None,
                      timestamp_ms=m.timestamp if m.HasField("timestamp") else None,
                      datatype=m.datatype if m.HasField("datatype") else None,
                      value=None if m.is_null else _metric_value(m),
                      is_null=m.is_null, is_historical=m.is_historical)
               for m in msg.metrics]
    return (msg.timestamp if msg.HasField("timestamp") else None,
            msg.seq if msg.HasField("seq") else None, metrics)


def encode(metrics: list[Metric], timestamp_ms: int, seq: int | None = None) -> bytes:
    """Encode scalar metrics into a payload (used for commands and test fixtures)."""
    msg = Payload(timestamp=timestamp_ms)
    if seq is not None:
        msg.seq = seq
    for metric in metrics:
        m = msg.metrics.add()
        if metric.name is not None:
            m.name = metric.name
        if metric.alias is not None:
            m.alias = metric.alias
        if metric.timestamp_ms is not None:
            m.timestamp = metric.timestamp_ms
        if metric.datatype is not None:
            m.datatype = metric.datatype
        if metric.is_null:
            m.is_null = True
            continue
        if metric.datatype == DATATYPE_BOOLEAN:
            m.boolean_value = bool(metric.value)
        elif metric.datatype in (DATATYPE_STRING, DATATYPE_TEXT):
            m.string_value = str(metric.value)
        elif metric.datatype == DATATYPE_FLOAT:
            m.float_value = float(metric.value)
        elif metric.datatype == DATATYPE_DOUBLE:
            m.double_value = float(metric.value)
        elif metric.datatype in (DATATYPE_INT64, DATATYPE_UINT64, DATATYPE_DATETIME):
            m.long_value = int(metric.value) & 0xFFFFFFFFFFFFFFFF
        else:
            m.int_value = int(metric.value) & 0xFFFFFFFF
    return msg.SerializeToString()


def rebirth_command(timestamp_ms: int) -> bytes:
    """NCMD payload asking an edge node to republish its births."""
    return encode([Metric("Node Control/Rebirth", None, None, DATATYPE_BOOLEAN, True)],
                  timestamp_ms)
