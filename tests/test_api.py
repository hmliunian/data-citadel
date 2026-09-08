import base64
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from data_citadel.api import create_app
from data_citadel.models import Episode, EpisodeNotFound, Frame, ProviderError, ReviewResult, SampledVideo
from data_citadel.settings import Settings

EPISODE_ID = "a" * 32


@pytest.fixture
def web(tmp_path):
    episode = Episode(EPISODE_ID, "A_001", "DL-TEST", "拿起杯子", "private-collector",
                      Path("/private/data/episode.mcap"), Path("/private/sidecar.json"),
                      "correct", "Accepted", {"task.review.deny_reason": "private-label"})

    def get(episode_id):
        if episode_id != EPISODE_ID:
            raise EpisodeNotFound("Unknown episode id")
        return episode

    def review(episode_id, strategy="uniform", camera_mode=None):
        get(episode_id)
        return ReviewResult(episode_id=episode_id, action_id="A_001", task_code="DL-TEST",
                            verdict="uncertain", reason="需要人工确认专家样例",
                            provenance={"camera_mode": camera_mode or "main_wrist"})

    repository = SimpleNamespace(get=get, list_episodes=lambda action=None: [episode],
                                 inventory=lambda: {"total_episodes": 1})
    sampler = SimpleNamespace(sample=lambda *args, **kwargs: SampledVideo(
        [Frame(0.3, b"jpeg-sample")], 3.0, "/camera/test", "uniform"
    ))
    service = SimpleNamespace(repository=repository, sampler=sampler, review=review)
    settings = Settings(artifacts_dir=tmp_path / "artifacts", camera_mode="main_wrist")
    with TestClient(create_app(settings, service)) as client:
        yield client, service, settings


def test_web_and_episode_listing_do_not_expose_private_metadata(web):
    client, _, _ = web
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200
    response = client.get("/v1/episodes")
    assert response.status_code == 200
    assert response.json()[0]["instruction"] == "拿起杯子"
    assert "private" not in response.text
    assert client.get("/v1/inventory").json()["total_episodes"] == 1


def test_review_returns_uncertain_and_saves_result(web):
    client, _, settings = web
    response = client.post("/v1/reviews", json={"episode_id": EPISODE_ID})
    assert response.status_code == 200
    assert response.json()["verdict"] == "uncertain"
    assert response.json()["ground_truth_candidate"] is False
    artifacts = list((settings.artifacts_dir / "reviews").glob("*.json"))
    assert len(artifacts) == 1
    assert ReviewResult.model_validate_json(artifacts[0].read_text()).episode_id == EPISODE_ID


def test_invalid_request_and_unknown_id_have_no_results(web):
    client, _, settings = web
    assert client.post("/v1/reviews", json={"episode_id": "../../key.txt"}).status_code == 422
    assert client.post("/v1/reviews", json={"episode_id": "b" * 32}).status_code == 404
    assert client.post("/v1/reviews", json={"episode_id": EPISODE_ID, "label": "correct"}).status_code == 422
    assert client.post("/v1/reviews", json={"episode_id": EPISODE_ID, "camera_mode": "unknown"}).status_code == 422
    assert not (settings.artifacts_dir / "reviews").exists()


def test_provider_failure_is_operational_not_incorrect(web):
    client, service, settings = web

    def fail(*args, **kwargs):
        raise ProviderError("Qwen request failed")

    service.review = fail
    response = client.post("/v1/reviews", json={"episode_id": EPISODE_ID})
    assert response.status_code == 502
    assert "verdict" not in response.json()
    assert not (settings.artifacts_dir / "reviews").exists()


def test_preview_preserves_timestamps_and_image_bytes(web):
    client, _, _ = web
    response = client.get(f"/v1/episodes/{EPISODE_ID}/frames")
    frame = response.json()["frames"][0]
    assert frame["timestamp_s"] == 0.3
    assert frame["view"] == "main"
    assert response.json()["videos"]["main"] == {
        "url": f"/v1/episodes/{EPISODE_ID}/video?view=main", "start_s": 0.3,
    }
    assert response.json()["camera_mode"] == "main_wrist"
    assert base64.b64decode(frame["image"].split(",")[1]) == b"jpeg-sample"
    assert client.get(f"/v1/episodes/{EPISODE_ID}/frames?interval_s=0").status_code == 422

@pytest.mark.parametrize("mode", ["main", "main_wrist"])
def test_review_and_preview_accept_explicit_camera_mode(web, mode):
    client, _, _ = web
    result = client.post("/v1/reviews", json={"episode_id": EPISODE_ID, "camera_mode": mode})
    assert result.status_code == 200
    assert result.json()["provenance"]["camera_mode"] == mode
    preview = client.get(f"/v1/episodes/{EPISODE_ID}/frames?camera_mode={mode}")
    assert preview.status_code == 200
    assert preview.json()["camera_mode"] == mode
    assert client.get(f"/v1/episodes/{EPISODE_ID}/frames?camera_mode=unknown").status_code == 422

@pytest.mark.parametrize("view", ["main", "left_wrist", "right_wrist"])
def test_video_endpoint_routes_each_view_and_supports_byte_ranges(web, tmp_path, monkeypatch, view):
    from data_citadel.media.sampling import WRIST_TOPICS

    client, _, settings = web
    output = tmp_path / "preview.mp4"
    output.write_bytes(b"test-mp4-content")
    calls = []

    def export(episode, topic, cache_dir):
        calls.append((episode.episode_id, topic, cache_dir))
        return output

    monkeypatch.setattr("data_citadel.api.export_video", export)
    response = client.get(f"/v1/episodes/{EPISODE_ID}/video?view={view}", headers={"Range": "bytes=0-3"})
    assert response.status_code == 206
    assert response.content == b"test"
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-disposition"].startswith("inline")
    topic = settings.camera_topic if view == "main" else WRIST_TOPICS[view]
    assert calls == [(EPISODE_ID, topic, settings.artifacts_dir / "videos")]
    assert client.get(f"/v1/episodes/{EPISODE_ID}/video?view=../../secret").status_code == 422
    assert client.get(f"/v1/episodes/{'b' * 32}/video?view=main").status_code == 404
    assert len(calls) == 1


def test_video_missing_source_is_operational_error(web, monkeypatch):
    from data_citadel.models import MediaError

    client, _, _ = web

    def export(*args):
        raise MediaError("missing_camera_topic")

    monkeypatch.setattr("data_citadel.api.export_video", export)
    response = client.get(f"/v1/episodes/{EPISODE_ID}/video?view=left_wrist")
    assert response.status_code == 422
    assert response.json()["error"] == "MediaError"
    assert "verdict" not in response.json()
