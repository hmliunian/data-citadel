from fractions import Fraction
from io import BytesIO
import json
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
from data_citadel.media.sampling import WRIST_TOPICS, gripper_event_times
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


def synthetic_video(tmp_path, *, frames=61, moving=False, bframes=False, cameras=None,
                    corrupt_topic=None):
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
        cameras = {TOPIC: 0} if cameras is None else cameras
        channels = {topic: writer.register_channel(topic, "flatbuffer", schema) for topic in cameras}
        first_dts = packets[0].dts
        for packet in packets:
            # MCAP order follows decode order; embedded timestamps follow presentation order.
            timestamp = ORIGIN + int((packet.pts - first_dts) * packet.time_base * 1e9)
            log_time = ORIGIN + int((packet.dts - first_dts) * packet.time_base * 1e9)
            for topic, offset in cameras.items():
                if offset is not None:
                    data = b"invalid h264" if topic == corrupt_topic else bytes(packet)
                    writer.add_message(channels[topic], log_time + offset,
                                       video_message(data, timestamp + offset), log_time + offset)
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
    cameras = {TOPIC: 0, WRIST_TOPICS["left_wrist"]: 7_000_000,
               WRIST_TOPICS["right_wrist"]: 17_000_000}
    episode = synthetic_video(tmp_path, frames=21, moving=True, bframes=True, cameras=cameras)
    sampler = VideoSampler(TOPIC)
    sampled = sampler.sample(episode, interval_s=0.5)
    times = [frame.timestamp_s for frame in sampled.frames]
    assert times == sorted(set(times))
    assert sampled.motion["analyzed_frames"] == 21
    assert times[-1] - times[0] == pytest.approx(2)
    assert sampled.duration_s >= times[-1]
    combined = sampler.sample(episode, interval_s=0.5, camera_mode="main_wrist")
    assert combined.duration_s > sampled.duration_s
    views = combined.motion["views"]
    assert views["main"]["duration_s"] == sampled.duration_s
    assert views["left_wrist"]["duration_s"] == pytest.approx(sampled.duration_s + 0.007)
    assert views["right_wrist"]["duration_s"] == pytest.approx(sampled.duration_s + 0.017)
    assert combined.duration_s == views["right_wrist"]["duration_s"]


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


def test_multiview_uses_main_1s_wrist_2s_and_reuses_single_view_cache(tmp_path, monkeypatch):
    cameras = {TOPIC: 0, WRIST_TOPICS["left_wrist"]: 7_000_000,
               WRIST_TOPICS["right_wrist"]: 17_000_000}
    episode = synthetic_video(tmp_path, frames=21, moving=True, cameras=cameras)
    sampler = VideoSampler(TOPIC, cache_dir=tmp_path / "cache")
    main = sampler.sample(episode, interval_s=1)
    combined = sampler.sample(episode, interval_s=1, camera_mode="main_wrist")
    assert [frame for frame in combined.frames if frame.view == "main"] == main.frames
    assert all(frame.view == "main" for frame in main.frames)
    assert {key: value for key, value in combined.motion.items() if key != "views"} == main.motion
    sequence = [(frame.timestamp_s, frame.view) for frame in combined.frames]
    assert sequence == sorted(sequence) and len(sequence) == 7
    assert [frame.timestamp_s for frame in main.frames] == [0, 1, 2]
    assert {view: info["interval_s"] for view, info in combined.motion["views"].items()} == {
        "main": 1, "left_wrist": 2, "right_wrist": 2,
    }
    assert combined.motion["views"]["left_wrist"]["timestamps_s"] == pytest.approx([0.007, 2.007])
    assert combined.motion["views"]["right_wrist"]["timestamps_s"] == pytest.approx([0.017, 2.017])
    assert combined.duration_s >= max(frame.timestamp_s for frame in combined.frames)
    assert combined.warnings == []
    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files) == 3
    assert all("view" not in frame for path in cache_files
               for frame in json.loads(path.read_text())["frames"])

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Existing per-camera cache should avoid decoding again")

    monkeypatch.setattr("data_citadel.media.sampling.decode_video", unexpected_decode)
    cached = VideoSampler(TOPIC, cache_dir=tmp_path / "cache", camera_mode="main_wrist")
    assert cached.sample(episode, interval_s=1) == combined
    assert cached.sample(episode, interval_s=1, camera_mode="main") == main


def test_missing_and_empty_wrist_channels_remain_visible(tmp_path):
    episode = synthetic_video(tmp_path, frames=5, cameras={TOPIC: 0, WRIST_TOPICS["left_wrist"]: None})
    combined = VideoSampler(TOPIC, camera_mode="main_wrist").sample(episode)
    assert {frame.view for frame in combined.frames} == {"main"}
    assert combined.warnings == ["left_wrist: empty_camera_topic", "right_wrist: missing_camera_topic"]
    for view in WRIST_TOPICS:
        assert combined.motion["views"][view]["timestamps_s"] == []
        assert combined.motion["views"][view]["duration_s"] is None
        assert combined.motion["views"][view]["warnings"]


def test_wrist_decoder_failure_is_not_silently_downgraded(tmp_path):
    cameras = {TOPIC: 0, **dict.fromkeys(WRIST_TOPICS.values(), 0)}
    episode = synthetic_video(tmp_path, frames=5, cameras=cameras,
                              corrupt_topic=WRIST_TOPICS["left_wrist"])
    with pytest.raises(MediaError, match="decode"):
        VideoSampler(TOPIC, camera_mode="main_wrist").sample(episode)


def test_frame_budget_is_per_camera_and_every_reduction_is_reported(tmp_path):
    cameras = {TOPIC: 0, **dict.fromkeys(WRIST_TOPICS.values(), 0)}
    episode = synthetic_video(tmp_path, frames=61, cameras=cameras)
    combined = VideoSampler(TOPIC, max_frames=2, camera_mode="main_wrist").sample(
        episode, interval_s=0.1,
    )
    assert len(combined.frames) == 6
    assert all(len(item["timestamps_s"]) == 2 for item in combined.motion["views"].values())
    assert len(combined.warnings) == 3
    assert all("frame_budget_reduced_temporal_coverage" in warning for warning in combined.warnings)


def test_invalid_camera_modes_are_rejected_before_sampling(tmp_path):
    with pytest.raises(ValueError, match="camera mode"):
        VideoSampler(TOPIC, camera_mode="unknown")
    episode = synthetic_video(tmp_path, frames=5)
    with pytest.raises(ValueError, match="camera mode"):
        VideoSampler(TOPIC).sample(episode, camera_mode="unknown")


@pytest.mark.skipif(not os.environ.get("CITADEL_REAL_MCAP"), reason="Opt-in local real-data smoke test")
@pytest.mark.parametrize("episode_id", [
    "005b3292336e7a5ba1c4092695b3ce97", "171f0ddc528c299ac484ef996b563f73",
])
def test_real_main_and_both_wrists_keep_distinct_source_timestamps(episode_id):
    root = Path(os.environ.get("CITADEL_DATASET_ROOT", str(Path(__file__).resolve().parents[2] / "datasets")))
    episode = EpisodeRepository(root).get(episode_id)
    sampler = VideoSampler(TOPIC)
    main = sampler.sample(episode)
    combined = sampler.sample(episode, camera_mode="main_wrist")
    assert {frame.view for frame in combined.frames} == {"main", "left_wrist", "right_wrist"}
    assert [frame for frame in combined.frames if frame.view == "main"] == main.frames
    assert set(main.warnings) <= set(combined.warnings)
    assert len(combined.frames) == sum(len(view["timestamps_s"]) for view in combined.motion["views"].values())
    for view, topic in WRIST_TOPICS.items():
        info = combined.motion["views"][view]
        assert info["topic"] == topic
        assert info["timestamps_s"] != [frame.timestamp_s for frame in main.frames]
        assert info["motion"]["analyzed_frames"] > len(info["timestamps_s"])
