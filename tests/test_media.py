from fractions import Fraction
from io import BytesIO
import os
from pathlib import Path

import av
import flatbuffers
from mcap.writer import Writer
import numpy as np
from PIL import Image
import pytest

from data_citadel.media import VideoSampler, inspect_mcap
from data_citadel.media.flatbuffer import FlatbufferSchema
from data_citadel.media.sampling import gripper_event_times
from data_citadel.models import Episode, MediaError
from data_citadel.repository import EpisodeRepository


TOPIC = "/camera/coracam_head/left_h264/video"
ORIGIN = 1_000_000_000_000


def _vector(builder, offsets):
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _object(builder, name, specifications):
    fields = []
    for ordinal, (field_name, base, element, index) in enumerate(specifications):
        field_name = builder.CreateString(field_name)
        builder.StartObject(3)
        builder.PrependInt8Slot(0, base, 0)
        builder.PrependInt8Slot(1, element, 0)
        builder.PrependInt32Slot(2, index, -1)
        kind = builder.EndObject()
        builder.StartObject(4)
        builder.PrependUOffsetTRelativeSlot(0, field_name, 0)
        builder.PrependUOffsetTRelativeSlot(1, kind, 0)
        builder.PrependUint16Slot(2, ordinal, 0)
        builder.PrependUint16Slot(3, 4 + ordinal * 2, 0)
        fields.append(builder.EndObject())
    name = builder.CreateString(name)
    field_vector = _vector(builder, fields)
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(0, name, 0)
    builder.PrependUOffsetTRelativeSlot(1, field_vector, 0)
    return builder.EndObject()


def video_schema():
    """Deliberately order fields differently from the production Foxglove schema."""
    builder = flatbuffers.Builder(1024)
    timestamp = _object(builder, "foxglove.Timestamp", [("sec", 9, 0, -1), ("nsec", 8, 0, -1)])
    video = _object(builder, "foxglove.CompressedVideo", [
        ("format", 13, 0, -1), ("timestamp", 15, 0, 0), ("data", 14, 4, -1),
    ])
    objects = _vector(builder, [timestamp, video])
    builder.StartObject(5)
    builder.PrependUOffsetTRelativeSlot(0, objects, 0)
    builder.PrependUOffsetTRelativeSlot(4, video, 0)
    root = builder.EndObject()
    builder.Finish(root, file_identifier=b"BFBS")
    return bytes(builder.Output())


def video_message(data, timestamp_ns):
    builder = flatbuffers.Builder(len(data) + 100)
    encoded = builder.CreateByteVector(data)
    format_name = builder.CreateString("h264")
    builder.StartObject(2)
    builder.PrependInt64Slot(0, timestamp_ns // 1_000_000_000, 0)
    builder.PrependUint32Slot(1, timestamp_ns % 1_000_000_000, 0)
    timestamp = builder.EndObject()
    builder.StartObject(3)
    builder.PrependUOffsetTRelativeSlot(0, format_name, 0)
    builder.PrependUOffsetTRelativeSlot(1, timestamp, 0)
    builder.PrependUOffsetTRelativeSlot(2, encoded, 0)
    root = builder.EndObject()
    builder.Finish(root)
    return bytes(builder.Output())


def synthetic_video(tmp_path, *, frames=61, moving=False, bframes=False):
    path = tmp_path / "episode.mcap"
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 96, 64, "yuv420p"
    encoder.time_base, encoder.framerate = Fraction(1, 10), Fraction(10, 1)
    encoder.options = {"preset": "ultrafast", "bf": "2" if bframes else "0"}
    packets = []
    for index in range(frames):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[:, :, :] = (index * 10) % 255 if moving else 100
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts, frame.time_base = index, Fraction(1, 10)
        packets.extend(encoder.encode(frame))
    packets.extend(encoder.encode(None))
    with path.open("wb") as output:
        writer = Writer(output)
        writer.start()
        schema = writer.register_schema("foxglove.CompressedVideo", "flatbuffer", video_schema())
        channel = writer.register_channel(TOPIC, "flatbuffer", schema)
        first_dts = packets[0].dts
        for packet in packets:
            # MCAP order follows decode order; embedded timestamps follow presentation order.
            timestamp = ORIGIN + int((packet.pts - first_dts) * packet.time_base * 1e9)
            log_time = ORIGIN + int((packet.dts - first_dts) * packet.time_base * 1e9)
            writer.add_message(channel, log_time, video_message(bytes(packet), timestamp), log_time)
        writer.finish()
    return Episode("synthetic", "A_001", "test-task", "test instruction", "collector",
                   path, tmp_path / "unused.json")


def test_reflection_reads_schema_field_order_and_rejects_corruption():
    schema = FlatbufferSchema(video_schema(), "foxglove.CompressedVideo")
    message = schema.decode(video_message(b"encoded", ORIGIN + 123))
    assert message.get("data") == b"encoded"
    assert message.get("format") == "h264"
    assert message.get("timestamp").get("nsec") == 123
    with pytest.raises(MediaError):
        schema.decode(b"\xff" * 8)
    with pytest.raises(MediaError, match="BFBS"):
        FlatbufferSchema(b"not a binary schema")


def test_continuous_decode_detects_static_interval_and_keeps_endpoints(tmp_path):
    episode = synthetic_video(tmp_path)
    sampled = VideoSampler(TOPIC).sample(episode)
    assert [frame.timestamp_s for frame in sampled.frames] == [0, 2, 4, 6]
    assert sampled.motion["analyzed_frames"] == 61
    assert sampled.motion["max_stationary_duration_s"] == pytest.approx(6)
    assert sampled.warnings == []
    assert Image.open(BytesIO(sampled.frames[0].jpeg)).size == (96, 64)
    inspection = inspect_mcap(episode.mcap_path, [TOPIC, "/missing"])
    assert inspection["missing_required_channels"] == ["/missing"]
    assert inspection["message_counts"][TOPIC] == 61


def test_presentation_timestamps_survive_b_frame_reordering(tmp_path):
    episode = synthetic_video(tmp_path, frames=21, moving=True, bframes=True)
    sampled = VideoSampler(TOPIC).sample(episode, interval_s=0.5)
    times = [frame.timestamp_s for frame in sampled.frames]
    assert times == sorted(set(times))
    assert sampled.motion["analyzed_frames"] == 21
    assert times[-1] - times[0] == pytest.approx(2)
    assert sampled.duration_s >= times[-1]


def test_budget_reduction_is_visible_and_cache_preserves_data(tmp_path):
    episode = synthetic_video(tmp_path, moving=True)
    sampler = VideoSampler(TOPIC, max_frames=3, cache_dir=tmp_path / "cache")
    sampled = sampler.sample(episode, interval_s=0.1)
    assert len(sampled.frames) == 3
    assert sampled.frames[0].timestamp_s == 0
    assert sampled.frames[-1].timestamp_s == 6
    assert "frame_budget_reduced_temporal_coverage" in sampled.warnings
    assert sampled.motion["stationary_intervals"] == []
    assert sampler.sample(episode, interval_s=0.1) == sampled


def test_missing_camera_is_operational_error_and_gripper_fallback_is_visible(tmp_path):
    episode = synthetic_video(tmp_path)
    with pytest.raises(MediaError, match="missing_camera_topic"):
        VideoSampler("/missing").sample(episode)
    sampled = VideoSampler(TOPIC).sample(episode, strategy="keyframes")
    assert "keyframe_fallback_to_uniform_no_gripper_events" in sampled.warnings


def test_gripper_events_capture_open_and_close_boundaries():
    times = np.arange(0, 4, 0.01)
    values = np.interp(times, [0, 1, 1.5, 2.5, 3, 4], [0, 0, 1, 1, 0, 0])
    events = gripper_event_times(list(zip(times, values)))
    assert len(events) == 4
    assert events == pytest.approx([1, 1.5, 2.5, 3], abs=0.1)
    assert gripper_event_times(list(zip(times, np.zeros(len(times))))) == []
    with pytest.raises(MediaError, match="sequence"):
        gripper_event_times([(0, 1)] * 10)


@pytest.mark.skipif(not os.environ.get("CITADEL_REAL_MCAP"), reason="Opt-in local real-data smoke test")
@pytest.mark.parametrize("action", ["A_001", "A_004", "A_011"])
def test_real_mcap_both_sampling_strategies(action):
    root = Path(os.environ.get("CITADEL_DATASET_ROOT", str(Path(__file__).resolve().parents[2] / "datasets")))
    episode = next(item for item in EpisodeRepository(root).list_episodes(action) if item.label == "correct")
    sampler = VideoSampler(TOPIC)
    uniform = sampler.sample(episode)
    keyframes = sampler.sample(episode, strategy="keyframes")
    assert len(uniform.frames) >= 2 and len(keyframes.frames) >= 2
    assert uniform.motion["analyzed_frames"] > len(uniform.frames)
    assert uniform.frames[0].timestamp_s == keyframes.frames[0].timestamp_s
    assert uniform.frames[-1].timestamp_s == keyframes.frames[-1].timestamp_s
    for sampled in (uniform, keyframes):
        assert Image.open(BytesIO(sampled.frames[0].jpeg)).format == "JPEG"
