"""Real HTTP preview contracts without any external model call."""

from io import BytesIO
import os

import av
from fastapi.testclient import TestClient
import pytest

from data_citadel.api import create_app
from data_citadel.settings import Settings


@pytest.mark.skipif(not os.environ.get("CITADEL_REAL_MCAP"), reason="Opt-in local MCAP preview")
def test_three_camera_mp4s_match_frame_links_and_support_seeking(tmp_path, monkeypatch):
    def no_model(_):
        pytest.fail("Preview must not read model credentials")

    monkeypatch.setattr(Settings, "api_key", no_model)
    settings = Settings(artifacts_dir=tmp_path)
    episode_id = "005b3292336e7a5ba1c4092695b3ce97"
    with TestClient(create_app(settings)) as client:
        preview = client.get(f"/v1/episodes/{episode_id}/frames?camera_mode=main_wrist&interval_s=2")
        assert preview.status_code == 200
        data = preview.json()
        assert set(data["videos"]) == {"main", "left_wrist", "right_wrist"}
        for view, source in data["videos"].items():
            response = client.get(source["url"])
            assert response.status_code == 200
            assert response.headers["content-type"] == "video/mp4"
            with av.open(BytesIO(response.content)) as container:
                assert container.streams.video[0].codec_context.name == "h264"
                frames = list(container.decode(video=0))
            times = [float(frame.pts * frame.time_base) for frame in frames]
            assert times[0] == 0
            assert frames[0].format.name == "yuv420p"
            selected = [frame for frame in data["frames"] if frame["view"] == view]
            assert len(frames) > len(selected)
            assert source["start_s"] == selected[0]["timestamp_s"]
            for frame in selected:
                target = frame["timestamp_s"] - source["start_s"]
                assert min(abs(value - target) for value in times) <= 1e-6
            ranged = client.get(source["url"], headers={"Range": "bytes=0-63"})
            assert ranged.status_code == 206
            assert ranged.content == response.content[:64]
