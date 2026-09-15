import json
import time
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from citadel.server.app import create_app
from citadel_client import CitadelClient


@pytest.fixture
def api(runtime_case):
    runtime, _ = runtime_case
    with TestClient(create_app(runtime)) as client:
        yield client


def submit(api, episode_id, kind="reviews", retry=False):
    response = api.post("/api/v1/" + kind,
                        json={"run_id": "test", "episode_id": episode_id, "retry_failed": retry})
    assert response.status_code == 202, response.text
    return response.json()


def wait(api, job_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        response = api.get("/api/v1/jobs/" + job_id)
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError("Job did not finish")


def episode(runtime_case):
    return runtime_case[0].runs["test"].source.manifest["splits"]["development"][0]


def test_window_and_gt_input_rejection(api, runtime_case):
    _, fake = runtime_case
    assert api.get("/api/v1/health").json()["status"] == "ok"
    assert "UMI-T" in api.get("/").text
    assert api.get("/app.js").status_code == 200
    assert len(api.get("/api/v1/episodes?run_id=test").json()) == 6
    response = api.post("/api/v1/reviews",
                        json={"run_id": "test", "episode_id": episode(runtime_case), "gt": "correct"})
    assert response.status_code == 422 and fake.requests == []
    schema = api.get("/openapi.json").json()
    assert "JobView" in schema["components"]["schemas"]


def test_preview_never_calls_model_and_assets_support_ranges(api, runtime_case):
    episode_id = episode(runtime_case)
    assert api.get(f"/api/v1/episodes/{episode_id}/media?run_id=test").status_code == 404
    job = wait(api, submit(api, episode_id, "previews")["job_id"])
    assert job["status"] == "succeeded", job
    value = api.get("/api/v1/jobs/" + job["job_id"] + "/output").json()
    assert "base64," not in json.dumps(value)
    assert "新的" not in value["task"]["messages"][0]["content"]
    assert runtime_case[1].requests == []
    media = api.get(f"/api/v1/episodes/{episode_id}/media?run_id=test").json()
    image = media["frames"][0]
    assert "path" not in image
    assert api.get(image["url"]).status_code == 200
    assert api.get(image["url"], headers={"Range": "bytes=0-9"}).status_code == 206
    assert api.get(f"/api/v1/runs/test/assets/{episode_id}/manifest.json").status_code == 404


def test_holdout_preview_and_freeze_are_gated(api, runtime_case):
    run = runtime_case[0].runs["test"]
    episode_id = run.source.manifest["splits"]["holdout"][0]
    assert api.post("/api/v1/previews", json={"run_id": "test", "episode_id": episode_id}).status_code == 409
    assert api.get("/api/v1/episodes?run_id=test&split=holdout").status_code == 409
    assert api.post("/api/v1/runs/test/freeze").status_code == 409


def test_review_round_trip_idempotency_and_sdk(api, runtime_case):
    episode_id = episode(runtime_case)
    first = submit(api, episode_id)
    second = submit(api, episode_id)
    assert first["job_id"] == second["job_id"]
    job = wait(api, first["job_id"])
    assert job["status"] == "succeeded"
    assert len(runtime_case[1].requests) == 2
    def forward(request):
        response = api.request(request.method, str(request.url), content=request.content,
                               headers=dict(request.headers))
        return httpx.Response(response.status_code, content=response.content)
    transport = httpx.MockTransport(forward)
    with CitadelClient("http://testserver", transport=transport) as client:
        assert client.job(job["job_id"])["status"] == "succeeded"
        result = client.result("test", job["result_id"])
        assert result["checks"]["action"]["state"] == "pass"
        assert result["evidence"][0]["url"].startswith("/api/v1/")
        assert client.report("test")["counts"]["not_run"] == 5


def test_independent_requests_run_concurrently_and_freeze_waits(api, runtime_case):
    runtime, fake = runtime_case
    entered, release, lock = threading.Event(), threading.Event(), threading.Lock()
    original = fake.complete
    active = 0
    def complete(*args, **kwargs):
        nonlocal active
        with lock:
            active += 1
            if active >= 2:
                entered.set()
        assert release.wait(5)
        try:
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1
    fake.complete = complete
    ids = runtime.runs["test"].source.manifest["splits"]["development"][:2]
    jobs = [submit(api, value) for value in ids]
    try:
        assert entered.wait(3), "Separate HTTP requests were serialized"
        assert api.post("/api/v1/runs/test/freeze").status_code == 409
        assert api.get("/api/v1/health").status_code == 200
    finally:
        release.set()
    assert all(wait(api, job["job_id"])["status"] == "succeeded" for job in jobs)
    assert len(fake.requests) == 4


def test_failed_job_requires_explicit_retry(api, runtime_case):
    _, fake = runtime_case
    fake.error = RuntimeError("private error")
    first = wait(api, submit(api, episode(runtime_case))["job_id"])
    assert first["status"] == "failed"
    fake.error = None
    assert submit(api, episode(runtime_case))["job_id"] == first["job_id"]
    retried = submit(api, episode(runtime_case), retry=True)
    assert retried["job_id"] != first["job_id"]
    assert wait(api, retried["job_id"])["status"] == "succeeded"


def test_rejection_is_a_successful_job(api, runtime_case, answer):
    answer["checks"]["action"]["state"] = "fail"
    job = wait(api, submit(api, episode(runtime_case))["job_id"])
    assert job["status"] == "succeeded"
    result = api.get("/api/v1/jobs/" + job["job_id"] + "/output").json()
    assert result["label"] == "incorrect"


def test_batch_records_one_snapshot(api, runtime_case):
    response = api.post("/api/v1/batches", json={"run_id": "test", "limit": 3})
    assert response.status_code == 202
    jobs = response.json()
    assert len(jobs) == 3 and len({j["configuration_sha256"] for j in jobs}) == 1
    assert all(wait(api, j["job_id"])["status"] == "succeeded" for j in jobs)


def test_full_queue_rejects_entire_batch(api, runtime_case):
    runtime, fake = runtime_case
    runtime.jobs.repository.max_pending = 2
    response = api.post("/api/v1/batches", json={"run_id": "test", "limit": 3})
    assert response.status_code == 409
    assert runtime.jobs.repository.active("test") == 0 and fake.requests == []
