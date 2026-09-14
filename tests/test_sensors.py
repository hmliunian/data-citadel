import json
from types import SimpleNamespace

import flatbuffers
import pytest

from citadel.flatbuffer import Schema
from citadel.sensors import TOPICS, intervals, prepare_gripper, read_gripper


def vector_fixture(inline):
    b = flatbuffers.Builder(256)
    def field(name, kind, offset, element=0, index=-1):
        label = b.CreateString(name)
        b.StartObject(4)
        b.PrependInt8Slot(0, kind, 0)
        b.PrependInt8Slot(1, element, 0)
        b.PrependInt32Slot(2, index, -1)
        info = b.EndObject()
        b.StartObject(6)
        b.PrependUOffsetTRelativeSlot(0, label, 0)
        b.PrependUOffsetTRelativeSlot(1, info, 0)
        b.PrependUint16Slot(3, offset, 0)
        return b.EndObject()
    def offsets(items):
        b.StartVector(4, len(items), 4)
        for item in reversed(items):
            b.PrependUOffsetTRelative(item)
        return b.EndVector()
    def obj(name, fields, struct=False):
        label = b.CreateString(name)
        fields = offsets(fields)
        b.StartObject(5)
        b.PrependUOffsetTRelativeSlot(0, label, 0)
        b.PrependUOffsetTRelativeSlot(1, fields, 0)
        b.PrependBoolSlot(2, struct, False)
        b.PrependInt32Slot(4, 8 if struct else 0, 0)
        return b.EndObject()
    sample = obj("Sample", [field("value", 12, 0 if inline else 4)], inline)
    root = obj("Vector", [field("points", 14, 4, 15, 0)])
    objects = offsets([sample, root])
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, objects, 0)
    b.PrependUOffsetTRelativeSlot(4, root, 0)
    b.Finish(b.EndObject(), file_identifier=b"BFBS")
    schema = bytes(b.Output())
    b = flatbuffers.Builder(128)
    values = [1.25, -2.5]
    if inline:
        b.StartVector(8, len(values), 8)
        for value in reversed(values):
            b.PrependFloat64(value)
        vector = b.EndVector()
    else:
        tables = []
        for value in values:
            b.StartObject(1)
            b.PrependFloat64Slot(0, value, 0)
            tables.append(b.EndObject())
        vector = offsets(tables)
    b.StartObject(1)
    b.PrependUOffsetTRelativeSlot(0, vector, 0)
    b.Finish(b.EndObject())
    return schema, bytes(b.Output())


@pytest.mark.parametrize("inline", [True, False])
def test_embedded_schema_decodes_struct_and_table_vectors(inline):
    schema, data = vector_fixture(inline)
    assert Schema(schema, "Vector").decode(data) == {"points": [{"value": 1.25}, {"value": -2.5}]}
    with pytest.raises(ValueError, match="Expected"):
        Schema(schema, "WrongRoot")
    if inline:
        with pytest.raises(ValueError, match="vector"):
            Schema(schema, "Vector").decode(data[:-1])


def test_between_frame_extrema_keep_short_cycles_and_missing_channels():
    signals = {"channels": {"left_position": {"samples": [[0, 100], [0.35, 20], [0.39, 100]]}}}
    first, second = intervals(signals, [{"time_s": 0}, {"time_s": 1}])
    assert first == [[0, 0, [100, 100], None, None, None, None, None]]
    assert second == [[0.3, 0.4, [20, 100], None, None, None, None, None]]
    assert intervals(signals, [{"time_s": 0.36}, {"time_s": 0.5}])[1][0][2] == [100, 100]


def test_signal_clock_load_gaps_and_cache_integrity(tmp_path, monkeypatch):
    def message(name, stamp, data):
        topic = TOPICS[name]
        schema_name = "foxglove.JointStates" if name.endswith("_position") else "discover.TactileData"
        schema = SimpleNamespace(id=1 if name.endswith("_position") else 2,
                                 name=schema_name, encoding="flatbuffer", data=b"schema")
        return schema, SimpleNamespace(topic=topic), SimpleNamespace(
            data=json.dumps(data).encode(), log_time=1_000_000_000 + round(stamp * 1e9))
    def joint(stamp, value):
        return {"timestamp": {"sec": 1, "nsec": round(stamp * 1e9)},
                "joints": [{"position": value}]}
    rows = [
        message("left_position", 0, joint(0, 100)),
        message("left_position", 0.05, joint(0.05, 20)),
        message("left_position", 0.21, joint(0.21, 100)),
        message("left_position", 0.22, joint(0.05, 40)),
        message("left_finger0", 0.03, {"points": [{"fx": 3, "fy": 4, "fz": 0}]}),
        message("left_finger0", 0.04, {"points": [{"fx": float("nan"), "fy": 0, "fz": 0}]}),
    ]
    monkeypatch.setattr("citadel.sensors.make_reader",
                        lambda *a, **k: SimpleNamespace(iter_messages=lambda **k: iter(rows)))
    monkeypatch.setattr("citadel.sensors.Schema",
                        lambda *a: SimpleNamespace(decode=lambda data: json.loads(data)))
    source = tmp_path / "sample.mcap"
    source.write_bytes(b"fake container; reader is injected")
    result = prepare_gripper(source, tmp_path, 1_000_000_000, "source-hash")
    assert result["channels"]["left_position"]["samples"] == [[0, 100], [0.05, 20], [0.21, 100]]
    assert result["channels"]["left_finger0"]["samples"] == [[0.03, 5]]
    assert "left_position:sample_gap" in result["warnings"]
    assert "left_position:invalid_sample" in result["warnings"]
    assert "left_finger0:timestamp_fallback" in result["warnings"]
    assert "left_finger0:invalid_sample" in result["warnings"]
    assert "right_position:missing" in result["warnings"]
    assert prepare_gripper(source, tmp_path, 1_000_000_000, "source-hash") == result
    result["channels"]["left_position"]["samples"][0][1] = 0
    (tmp_path / "gripper.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="changed"):
        read_gripper(tmp_path / "gripper.json")
