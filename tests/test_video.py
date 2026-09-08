from dataclasses import replace
import hashlib
import os
from pathlib import Path
import struct

import av
from mcap.reader import make_reader
from mcap.writer import Writer
import pytest

from data_citadel.media.flatbuffer import FlatbufferSchema
from data_citadel.media.mcap_reader import decode_video, inspect_mcap
from data_citadel.media.sampling import WRIST_TOPICS
from data_citadel.media.video import export_video
from data_citadel.models import MediaError
from data_citadel.repository import EpisodeRepository
from test_media import ORIGIN, TOPIC, synthetic_video, video_message, video_schema


def preview_times(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.pix_fmt == "yuv420p"
        return [float(frame.pts * frame.time_base) for frame in container.decode(stream)]


def retimed_episode(tmp_path, timestamps_ns):
    episode = synthetic_video(tmp_path, frames=len(timestamps_ns), moving=True)
    with episode.mcap_path.open("rb") as stream:
        packets = [FlatbufferSchema(schema.data, schema.name).decode(message.data).get("data")
                   for schema, _, message in make_reader(stream).iter_messages()]
    path = tmp_path / "retimed.mcap"
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema("foxglove.CompressedVideo", "flatbuffer", video_schema())
        channel = writer.register_channel(TOPIC, "flatbuffer", schema)
        for index, (packet, timestamp) in enumerate(zip(packets, timestamps_ns, strict=True)):
            log_time = ORIGIN + index * 100_000_000
            writer.add_message(channel, log_time, video_message(packet, ORIGIN + timestamp), log_time)
        writer.finish()
    return replace(episode, mcap_path=path)


def test_export_preserves_bframe_timing_resolution_and_faststart(tmp_path):
    topic = WRIST_TOPICS["right_wrist"]
    episode = synthetic_video(tmp_path, frames=21, moving=True, bframes=True,
                              cameras={TOPIC: 0, topic: 17_000_000})
    source_hash = hashlib.sha256(episode.mcap_path.read_bytes()).digest()
    start_ns = inspect_mcap(episode.mcap_path)["start_time_ns"]
    source_times = [timestamp for timestamp, _ in decode_video(episode.mcap_path, topic, start_ns, [])]
    assert source_times[0] > 0
    path = export_video(episode, topic, tmp_path / "cache")
    assert preview_times(path) == pytest.approx([time - source_times[0] for time in source_times], abs=1e-8)
    with av.open(str(path)) as container:
        assert (container.streams.video[0].width, container.streams.video[0].height) == (96, 64)
    data = path.read_bytes()
    atoms, offset = [], 0
    while offset < len(data):
        size, name = struct.unpack_from(">I4s", data, offset)
        assert size >= 8
        atoms.append(name)
        offset += size
    assert atoms.index(b"moov") < atoms.index(b"mdat")
    assert hashlib.sha256(episode.mcap_path.read_bytes()).digest() == source_hash


def test_irregular_frame_gaps_are_not_replaced_by_fixed_fps(tmp_path):
    episode = retimed_episode(tmp_path, [50_000_000, 150_000_000, 400_000_000, 450_000_000])
    path = export_video(episode, TOPIC, tmp_path / "cache")
    assert preview_times(path) == pytest.approx([0, 0.1, 0.35, 0.4], abs=1e-8)


def test_cache_separates_topics_and_reuses_completed_export(tmp_path, monkeypatch):
    wrist = WRIST_TOPICS["left_wrist"]
    episode = synthetic_video(tmp_path, frames=5, cameras={TOPIC: 0, wrist: 7_000_000})
    cache_dir = tmp_path / "cache"
    main = export_video(episode, TOPIC, cache_dir)
    hand = export_video(episode, wrist, cache_dir)
    assert main != hand and main.is_file() and hand.is_file()
    before = main.stat().st_mtime_ns

    def unexpected_decode(*args):
        raise AssertionError("Completed cache must avoid decoding again")

    monkeypatch.setattr("data_citadel.media.video.decode_video", unexpected_decode)
    assert export_video(episode, TOPIC, cache_dir) == main
    assert main.stat().st_mtime_ns == before
    assert len(list(cache_dir.iterdir())) == 2


def test_cache_changes_when_source_stat_changes(tmp_path):
    episode = synthetic_video(tmp_path, frames=5)
    cache_dir = tmp_path / "cache"
    first = export_video(episode, TOPIC, cache_dir)
    stat = episode.mcap_path.stat()
    os.utime(episode.mcap_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = export_video(episode, TOPIC, cache_dir)
    assert first != second
    assert preview_times(first) == preview_times(second)


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_camera_fails_without_cache(tmp_path, empty):
    wrist = WRIST_TOPICS["left_wrist"]
    cameras = {TOPIC: 0, **({wrist: None} if empty else {})}
    episode = synthetic_video(tmp_path, frames=5, cameras=cameras)
    cache_dir = tmp_path / "cache"
    with pytest.raises(MediaError, match="missing_camera_topic"):
        export_video(episode, wrist, cache_dir)
    assert not list(cache_dir.glob("*"))


def test_corrupt_stream_cannot_leave_partial_cache(tmp_path):
    episode = synthetic_video(tmp_path, frames=5, corrupt_topic=TOPIC)
    cache_dir = tmp_path / "cache"
    with pytest.raises(MediaError, match="decode"):
        export_video(episode, TOPIC, cache_dir)
    assert list(cache_dir.iterdir()) == []


def test_negative_first_pts_is_rejected_without_distorting_seek_or_caching(tmp_path):
    episode = retimed_episode(tmp_path, [-50_000_000, 50_000_000, 200_000_000])
    cache_dir = tmp_path / "cache"
    with pytest.raises(MediaError, match="Negative first video timestamp"):
        export_video(episode, TOPIC, cache_dir)
    assert list(cache_dir.iterdir()) == []


@pytest.mark.skipif(not os.environ.get("CITADEL_REAL_MCAP"), reason="Opt-in local real-data smoke test")
@pytest.mark.parametrize(("episode_id", "topic"), [
    ("005b3292336e7a5ba1c4092695b3ce97", TOPIC),
    ("171f0ddc528c299ac484ef996b563f73", WRIST_TOPICS["left_wrist"]),
])
def test_real_mcap_preview_preserves_all_camera_timestamps(tmp_path, episode_id, topic):
    root = Path(os.environ.get("CITADEL_DATASET_ROOT", str(Path(__file__).resolve().parents[2] / "datasets")))
    episode = EpisodeRepository(root).get(episode_id)
    start_ns = inspect_mcap(episode.mcap_path)["start_time_ns"]
    source_times = [timestamp for timestamp, _ in decode_video(episode.mcap_path, topic, start_ns, [])]
    path = export_video(episode, topic, tmp_path / "cache")
    assert preview_times(path) == pytest.approx([time - source_times[0] for time in source_times], abs=1e-8)
