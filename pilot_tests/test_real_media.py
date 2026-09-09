"""Opt-in, no model calls: validate exported MP4 against actual expert timestamps."""
import os
import json
from pathlib import Path

import av
import pytest

from pilot.data import load_manifest, read_json
from pilot.runner import check_freeze, latest_results
from pilot.model import QwenClient
from pilot.web import create_app
from fastapi.testclient import TestClient


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


@pytest.mark.skipif(not os.getenv("PILOT_RUN_DIR"), reason="Set PILOT_RUN_DIR for actual-artifact checks")
def test_real_run_is_frozen_and_inputs_have_no_source_review_labels():
    run = Path(os.environ["PILOT_RUN_DIR"])
    if not (run / "freeze.json").exists():
        pytest.skip("Holdout has not been opened")
    check_freeze(run, QwenClient(run))
    manifest = load_manifest(run)
    rows = latest_results(run)
    assert len(rows) == 16
    assert all(r["status"] in ("completed", "needs_review") for r in rows)
    requests = list((run / "calls").glob("*/request.json"))
    assert len(requests) == 19
    for path in requests:
        request = read_json(path)
        assert read_json(path.with_name("response.json"))["http_status"] == 200
        assert request["attempt"] == 1
        text = json.dumps(request["messages"], ensure_ascii=False)
        assert "Accepted" not in text and "Denied" not in text
        assert "data:image/" not in text  # Archived image hashes, not payload bytes.
        for record in manifest["episodes"].values():
            for key in ("gt_reason", "reviewer", "review_time"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    assert value not in text


@pytest.mark.skipif(not os.getenv("PILOT_RUN_DIR"), reason="Set PILOT_RUN_DIR for actual-API checks")
def test_real_result_api_and_video_range_use_same_episode():
    run = Path(os.environ["PILOT_RUN_DIR"])
    with TestClient(create_app(run)) as client:
        results = client.get("/api/results").json()
        assert len(results) == 16
        row = next(r for r in results if r["split"] == "holdout")
        episode = client.get("/api/episodes/" + row["episode_id"]).json()
        assert episode["episode_id"] == row["episode_id"]
        assert episode["gt"] == row["gt"]
        assert set(episode["views"]) == {"main", "left_wrist", "right_wrist"}
        for view in episode["views"].values():
            response = client.get(view["url"], headers={"Range": "bytes=0-31"})
            assert response.status_code == 206
            assert len(response.content) == 32
        for frame in episode["frames"]:
            assert client.get(frame["url"]).headers["content-type"] == "image/jpeg"
        assert client.get("/media/" + row["episode_id"] + "/frames.json").status_code == 404
