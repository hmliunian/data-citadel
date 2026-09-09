import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pilot.data import digest, read_json, sha256, write_json
from pilot.web import create_app

EXPERT, DEVELOPMENT, HOLDOUT = [f"{number:032x}" for number in (1, 2, 3)]


def result(episode_id, route="A"):
    return {"episode_id": episode_id, "task_code": "DL-TEST", "instruction": "搬运棕色狗",
            "split": "holdout" if episode_id == HOLDOUT else "development", "route": route,
            "created_at": "2026-09-09T01:00:00+00:00",
            "gt": "correct", "gt_status": "Accepted", "gt_reason": None,
            "status": "completed", "label": "correct", "reason": "已完成", "checks": {},
            "evidence": [], "expert_ids": [EXPERT], "reference": {}, "usage": {}}


@pytest.fixture
def run_dir(tmp_path):
    episodes = {episode_id: {**result(episode_id), "quality": "high",
                            "mcap_path": "/private/source/recording.mcap"}
                for episode_id in (EXPERT, DEVELOPMENT, HOLDOUT)}
    manifest = {"episodes": episodes, "sampling": {"interval_s": 2.0},
                "splits": {"experts": [EXPERT], "expert_pool": [EXPERT],
                           "development": [DEVELOPMENT], "holdout": [HOLDOUT]}}
    manifest["snapshot_sha256"] = digest(manifest)
    write_json(tmp_path / "manifest.json", manifest)
    for episode_id in episodes:
        folder = tmp_path / "media" / episode_id
        folder.mkdir(parents=True)
        frames, views = [], {}
        for view in ("main", "left_wrist", "right_wrist"):
            image = folder / f"{view}-00000.jpg"
            video = folder / f"{view}.mp4"
            Image.new("RGB", (8, 6), (45, 80, 99)).save(image)
            video.write_bytes(b"0123456789")
            frames.append({"frame_id": image.stem, "view": view, "topic": f"/camera/{view}",
                           "time_s": 1.25, "source_ns": 1780000001250000000, "video_time_s": 0.5,
                           "path": str(image.relative_to(tmp_path)), "sha256": sha256(image),
                           "width": 8, "height": 6})
            views[view] = {"topic": f"/camera/{view}", "start_s": 0.75, "end_s": 3.0,
                           "video_path": str(video.relative_to(tmp_path)), "decoded_frames": 70,
                           "sampled_frames": 1, "uncovered_targets_s": []}
        write_json(folder / "frames.json", {"episode_id": episode_id, "frames": frames,
                                           "views": views, "warnings": []})
    for route in ("A", "B"):
        write_json(tmp_path / "results" / "development" / f"{route}-{DEVELOPMENT}-12345678.json",
                   result(DEVELOPMENT, route))
    return tmp_path


def test_read_only_page_and_cached_results(run_dir):
    with TestClient(create_app(run_dir)) as client:
        assert client.get("/health").json() == {"status": "ok", "mode": "read_only"}
        page = client.get("/")
        assert page.status_code == 200
        assert "只读缓存页" in page.text
        assert page.text.index('id="gt"') < page.text.index('id="predict"')
        assert client.post("/api/results", json={}).status_code == 405
        assert client.get("/api/results").json() == [result(DEVELOPMENT), result(DEVELOPMENT, "B")]
        response = client.get(f"/api/episodes/{DEVELOPMENT}")
        assert response.status_code == 200
        media = response.json()
        assert media["gt_status"] == "Accepted"
        assert media["expert_ids"] == [EXPERT]
        assert "mcap_path" not in media and "/private/source" not in response.text
        assert media["frames"][0]["time_s"] == 1.25
        assert media["frames"][0]["video_time_s"] == 0.5
        assert media["frames"][0]["source_ns"] == 1780000001250000000
        assert media["frames"][0]["url"] == f"/media/{DEVELOPMENT}/main-00000.jpg"
        assert media["views"]["main"]["url"] == f"/media/{DEVELOPMENT}/main.mp4"
        assert client.get(f"/api/episodes/{EXPERT}").status_code == 200


def test_prepared_images_and_video_ranges_are_served(run_dir):
    with TestClient(create_app(run_dir)) as client:
        image = client.get(f"/media/{DEVELOPMENT}/main-00000.jpg")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.content == (run_dir / "media" / DEVELOPMENT / "main-00000.jpg").read_bytes()
        video = client.get(f"/media/{DEVELOPMENT}/main.mp4", headers={"Range": "bytes=2-4"})
        assert video.status_code == 206 and video.content == b"234"
        assert video.headers["content-type"] == "video/mp4"


def test_unlisted_assets_private_files_and_traversal_are_unavailable(run_dir):
    (run_dir / "media" / DEVELOPMENT / "unlisted.jpg").write_bytes(b"not listed")
    with TestClient(create_app(run_dir)) as client:
        for url in (f"/media/{DEVELOPMENT}/unlisted.jpg", f"/media/{DEVELOPMENT}/frames.json",
                    f"/media/{DEVELOPMENT}/..%2f..%2fmanifest.json", "/manifest.json",
                    "/calls/request.json", "/Qwen-api/qwen_api_key.txt",
                    "/api/episodes/not-an-id", f"/api/episodes/{'f' * 32}"):
            assert client.get(url).status_code == 404


@pytest.mark.parametrize("kind", ["asset", "folder", "manifest"])
def test_symlinked_media_is_rejected(run_dir, kind):
    folder = run_dir / "media" / DEVELOPMENT
    if kind == "folder":
        destination = run_dir / "elsewhere"
        folder.rename(destination)
        folder.symlink_to(destination, target_is_directory=True)
    else:
        path = folder / ("main-00000.jpg" if kind == "asset" else "frames.json")
        destination = run_dir / "private-file"
        path.rename(destination)
        path.symlink_to(destination)
    with TestClient(create_app(run_dir)) as client:
        assert client.get(f"/media/{DEVELOPMENT}/main-00000.jpg").status_code == 404


@pytest.mark.parametrize("relative", ["../../secret.jpg", f"media/{EXPERT}/main-00000.jpg"])
def test_media_manifest_cannot_reference_another_directory(run_dir, relative):
    path = run_dir / "media" / DEVELOPMENT / "frames.json"
    media = read_json(path)
    media["frames"][0]["path"] = relative
    path.write_text(json.dumps(media))
    with TestClient(create_app(run_dir)) as client:
        assert client.get(f"/api/episodes/{DEVELOPMENT}").status_code == 409
        assert client.get(f"/media/{DEVELOPMENT}/main-00000.jpg").status_code == 409


@pytest.mark.parametrize("frozen,saved", [(False, False), (False, True), (True, False), (True, True)])
def test_holdout_requires_both_freeze_and_its_own_saved_result(run_dir, frozen, saved):
    with TestClient(create_app(run_dir)) as client:
        if frozen:
            write_json(run_dir / "freeze.json", {})
        if saved:
            write_json(run_dir / "results" / "holdout" / f"A-{HOLDOUT}-12345678.json", result(HOLDOUT))
        expected = 200 if frozen and saved else 403
        assert client.get(f"/api/episodes/{HOLDOUT}").status_code == expected
        assert client.get(f"/media/{HOLDOUT}/main-00000.jpg").status_code == expected
        visible_ids = {row["episode_id"] for row in client.get("/api/results").json()}
        assert (HOLDOUT in visible_ids) == (frozen and saved)


def test_symlinked_results_cannot_unlock_holdout_or_expose_private_data(run_dir):
    write_json(run_dir / "freeze.json", {})
    private = run_dir / "private-result.json"
    write_json(private, {**result(HOLDOUT), "secret": "PRIVATE_CREDENTIAL"})
    (run_dir / "results" / "holdout").mkdir()
    (run_dir / "results" / "holdout" / f"A-{HOLDOUT}-12345678.json").symlink_to(private)
    with TestClient(create_app(run_dir)) as client:
        response = client.get("/api/results")
        assert "PRIVATE_CREDENTIAL" not in response.text
        assert client.get(f"/api/episodes/{HOLDOUT}").status_code == 403


def test_results_prefer_latest_attempt_and_ignore_incomplete_writes(run_dir):
    latest = {**result(DEVELOPMENT), "created_at": "2026-09-09T02:00:00+00:00", "label": None,
              "status": "needs_review"}
    write_json(run_dir / "results" / "development" / f"A-{DEVELOPMENT}-87654321.json", latest)
    (run_dir / "results" / "development" / "pending.json").write_text('{"episode_id":')
    with TestClient(create_app(run_dir)) as client:
        rows = client.get("/api/results").json()
        assert len(rows) == 3
        assert rows[0] == latest
