import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from citadel.domain.errors import GateError


def first_id(run):
    return run.source.manifest["splits"]["development"][0]


def test_gt_never_sent_and_result_reused(service_case):
    run, client = service_case
    episode_id = next(i for i in run.source.manifest["splits"]["development"]
                      if run.source.records()[i]["gt"] == "incorrect")
    snapshot = run.snapshot()
    first = run.reviews.review(episode_id, snapshot)
    second = run.reviews.review(episode_id, snapshot)
    assert first["label"] == "correct"
    assert run.experiment.decorate(first)["evaluation"]["gt"] == "incorrect"
    assert "evaluation" not in first
    assert not first["cached"] and second["cached"] and len(client.requests) == 2
    for sentinel in ("PRIVATE_GT_REASON", "PRIVATE_REVIEWER", "Accepted", "Denied", '"gt"'):
        assert sentinel not in json.dumps(client.requests)
    report = run.experiment.report("development", snapshot)
    assert report["counts"]["total"] == 6
    assert report["counts"]["not_run"] == 5
    assert report["counts"]["false_accept"] == 1
    assert report["counts"]["coverage"] == 1 / 6


def test_failed_call_requires_explicit_retry(service_case):
    run, client = service_case
    episode_id, snapshot = first_id(run), run.snapshot()
    client.error = RuntimeError("private vendor error")
    first = run.reviews.review(episode_id, snapshot)
    assert first["status"] == "failed" and first["label"] is None
    assert "private vendor error" not in json.dumps(first)
    client.error = None
    assert run.reviews.review(episode_id, snapshot)["status"] == "failed"
    assert len(client.requests) == 1
    assert run.reviews.review(episode_id, snapshot, retry_failed=True)["status"] == "completed"
    assert len(client.requests) == 3


def test_holdout_requires_complete_development_and_unchanged_freeze(service_case):
    run, _ = service_case
    snapshot = run.snapshot()
    holdout = run.source.manifest["splits"]["holdout"][0]
    with pytest.raises(GateError):
        run.authorize(holdout)
    with pytest.raises(GateError):
        run.experiment.freeze(snapshot)
    for episode_id in run.source.manifest["splits"]["development"]:
        assert run.reviews.review(episode_id, snapshot)["status"] == "completed"
    run.experiment.freeze(snapshot)
    assert run.reviews.review(holdout, snapshot)["status"] == "completed"
    path = run.configuration.root / "tasks.json"
    profiles = json.loads(path.read_text())
    profiles["grasp"]["success"] = "changed"
    path.write_text(json.dumps(profiles))
    with pytest.raises(GateError, match="changed"):
        run.authorize(holdout)


def test_prompt_change_versions_results_and_old_snapshot_stays_stable(service_case):
    run, client = service_case
    snapshot = run.snapshot()
    episode_id = first_id(run)
    first = run.reviews.review(episode_id, snapshot)
    path = run.configuration.root / "prompts/review_system.txt"
    path.write_text(path.read_text() + "新的审核说明。")
    changed = run.snapshot()
    second = run.reviews.review(episode_id, changed)
    assert len(client.requests) == 4
    assert first["configuration_sha256"] != second["configuration_sha256"]
    assert "新的审核说明。" in client.requests[2][0]["content"]
    assert run.reviews.review(episode_id, snapshot)["result_id"] == first["result_id"]
    assert len(run.artifacts.history(episode_id)) == 2


def test_same_episode_is_reviewed_once_across_concurrent_callers(service_case):
    run, client = service_case
    episode_id, snapshot = first_id(run), run.snapshot()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run.reviews.review, episode_id, snapshot) for _ in range(2)]
        results = [job.result() for job in jobs]
    assert len({result["result_id"] for result in results}) == 1
    assert len(client.requests) == 2


def test_invalid_response_is_failed_not_business_rejection(service_case, answer):
    run, _ = service_case
    answer["checks"]["action"]["evidence_ids"] = ["V999"]
    result = run.reviews.review(first_id(run), run.snapshot())
    assert result["status"] == "failed" and result["error"]["stage"] == "evidence"
    assert result["label"] is None and "model_call" in result


def test_changed_code_cannot_be_mislabeled_as_loaded_version(service_case, monkeypatch):
    run, client = service_case
    monkeypatch.setattr("citadel.configuration.code_version", lambda: {"domain/decision.py": "changed"})
    with pytest.raises(GateError, match="restart"):
        run.snapshot()
    assert client.requests == []


def test_quality_failure_does_not_release_a_successful_action(service_case):
    run, client = service_case
    original = client.complete
    def complete(*args, quality_only=False, **kwargs):
        if quality_only:
            raise RuntimeError("quality unavailable")
        return original(*args, **kwargs)
    client.complete = complete
    result = run.reviews.review(first_id(run), run.snapshot())
    assert result["status"] == "failed" and result["label"] is None
    assert result["error"]["stage"] == "quality" and "model_call" in result


def test_quality_response_cannot_overwrite_task_checks(service_case):
    run, client = service_case
    original = client.complete
    def complete(*args, quality_only=False, **kwargs):
        response = original(*args, quality_only=quality_only, **kwargs)
        if quality_only:
            response["data"]["checks"] = {"action": {"state": "fail", "evidence_ids": ["V000"]}}
        return response
    client.complete = complete
    result = run.reviews.review(first_id(run), run.snapshot())
    assert result["status"] == "failed" and result["label"] is None
    assert result["error"]["stage"] == "quality"
