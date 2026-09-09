"""Opt-in, no model calls: validate exported MP4 against actual expert timestamps."""
import os
from pathlib import Path

import av
import pytest

from pilot.data import load_manifest, read_json


@pytest.mark.skipif(not os.getenv("PILOT_RUN_DIR"), reason="Set PILOT_RUN_DIR for actual-media checks")
def test_real_expert_video_time_matches_sample_evidence():
    run = Path(os.environ["PILOT_RUN_DIR"])
    manifest = load_manifest(run)
    for episode_id in manifest["splits"]["experts"]:
        media = read_json(run / "media" / episode_id / "frames.json")
        assert set(media["views"]) == {"main", "left_wrist", "right_wrist"}
        assert not media["warnings"]
        for view, details in media["views"].items():
            with av.open(str(run / details["video_path"])) as video:
                times = [float(f.pts * f.time_base) for f in video.decode(video=0)]
            assert len(times) == details["decoded_frames"]
            assert all(a < b for a, b in zip(times, times[1:]))
            assert times[0] == 0
            assert abs(times[-1] - (details["end_s"] - details["start_s"])) <= 1 / 90000
            for frame in [f for f in media["frames"] if f["view"] == view]:
                assert min(abs(t - frame["video_time_s"]) for t in times) <= 1 / 90000
                assert abs(frame["time_s"] - frame["video_time_s"] - details["start_s"]) < 1e-8
