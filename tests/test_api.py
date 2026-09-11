import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from citadel.__main__ import main
from citadel.api import create_app


@pytest.fixture
def api(service_case):
    service, _ = service_case
    with TestClient(create_app(service)) as client:
        yield client


def test_window_and_gt_input_rejection(api, service_case):
    service, fake = service_case
    assert api.get("/health").json()["episodes"] == 8
    assert "UMI-T" in api.get("/").text
    assert len(api.get("/episodes").json()) == 6
    episode_id = service.manifest["splits"]["development"][0]
    response = api.post("/reviews", json={"episode_id": episode_id, "gt": "correct"})
    assert response.status_code == 422 and fake.requests == []


def test_media_preview_asset_scope_and_range(api, service_case):
    service, fake = service_case
    episode_id = service.manifest["splits"]["development"][0]
    assert api.get("/media/" + episode_id).status_code == 404
    response = api.post("/media/" + episode_id)
    assert response.status_code == 200
    assert fake.requests == []
    url = response.json()["frames"][0]["url"]
    image = api.get(url)
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    partial = api.get(url, headers={"Range": "bytes=0-9"})
    assert partial.status_code == 206 and partial.content == image.content[:10]
    assert api.get(f"/assets/{episode_id}/media.json").status_code == 404
    assert api.get(f"/assets/{episode_id}/manifest.json").status_code == 404


def test_holdout_preview_stays_closed_before_freeze(api, service_case):
    service, _ = service_case
    episode_id = service.manifest["splits"]["holdout"][0]
    assert api.post("/media/" + episode_id).status_code == 409
    assert api.get("/episodes?split=holdout").status_code == 409
    assert api.post("/freeze").status_code == 409


def test_review_round_trip_and_report(api, service_case):
    service, fake = service_case
    episode_id = service.manifest["splits"]["development"][0]
    first = api.post("/reviews", json={"episode_id": episode_id})
    second = api.post("/reviews", json={"episode_id": episode_id})
    assert first.status_code == second.status_code == 200
    assert first.json()["label"] == "correct" and second.json()["cached"]
    assert len(fake.requests) == 1
    result = api.get("/results/" + episode_id).json()
    assert result["checks"]["object_match"]["state"] == "pass"
    assert result["checks"]["scene_match"]["state"] == "pass"
    assert api.get("/report").json()["counts"]["not_run"] == 5


def test_cli_runs_and_exports_complete_denominator(service_case, monkeypatch, capsys):
    service, client = service_case
    monkeypatch.setattr("citadel.__main__.Service", lambda *args: service)
    assert main(["run", "--limit", "1"]) == 0
    assert len(client.requests) == 1
    capsys.readouterr()
    assert main(["report"]) == 0
    exported = json.loads(capsys.readouterr().out)
    folder = Path(exported["path"])
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and exported["counts"]["not_run"] == 5
