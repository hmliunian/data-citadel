import av
import pytest
from PIL import Image

from citadel.flatbuffer import Schema
from citadel.media import TOPICS, nearest, render


def frames(times, color):
    return [(round(t * 1e9), av.VideoFrame.from_image(Image.new("RGB", (64, 48), color)))
            for t in times]


def test_nearest_uses_time_and_never_repeats_stale_frames():
    assert nearest([0, 100, 205], 200, 10) == 2
    assert nearest([0, 100, 205], 150, 10) is None
    assert nearest([], 100, 10) is None


def test_stitch_preserves_three_views_real_times_and_video_mapping(tmp_path):
    streams = {
        "main": frames([0, 1, 2, 3], "red"),
        "left_wrist": frames([0.05, 1.05, 2.05, 3.05], "green"),
        "right_wrist": frames([0.07, 1.07, 2.07, 3.07], "blue")}
    media = render(tmp_path, "a" * 32, 0, streams,
                   {"interval_s": 1, "tolerance_s": 0.1, "topics": TOPICS}, [])
    assert not media["incomplete"]
    at_one = next(f for f in media["frames"] if f["time_s"] == 1)
    assert [s["time_s"] for s in at_one["sources"].values()] == [1, 1.05, 1.07]
    with Image.open(tmp_path / at_one["path"]) as image:
        assert image.size == (192, 96)
        assert image.getpixel((32, 72))[0] > 240
        assert image.getpixel((96, 72))[1] > 110
        assert image.getpixel((160, 72))[2] > 240
    with av.open(str(tmp_path / media["videos"]["stitched"]["path"])) as video:
        decoded = list(video.decode(video=0))
    assert len(decoded) == len(media["frames"])
    for frame, saved in zip(decoded, media["frames"]):
        assert abs(float(frame.time) - saved["video_time_s"]) < 0.00001
    for view in TOPICS:
        assert any(f["sources"][view] and f["sources"][view]["time_s"] == streams[view][-1][0] / 1e9
                   for f in media["frames"])


def test_gap_is_blank_and_missing_camera_is_not_disguised(tmp_path):
    streams = {"main": frames([0, 1, 2], "red"),
               "left_wrist": frames([0, 2], "green"), "right_wrist": []}
    media = render(tmp_path, "b" * 32, 0, streams,
                   {"interval_s": 1, "tolerance_s": 0.1, "topics": TOPICS}, [])
    assert media["incomplete"]
    target = next(f for f in media["frames"] if f["time_s"] == 1)
    assert target["sources"]["left_wrist"] is None
    assert target["sources"]["right_wrist"] is None
    assert target["sources"]["main"]["time_s"] == 1


def test_non_bfbs_schema_is_rejected():
    with pytest.raises(ValueError, match="schema"):
        Schema(b"not a schema")
