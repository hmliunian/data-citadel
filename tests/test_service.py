import json

import pytest

from citadel.service import BusyError, GateError


def test_gt_never_sent_and_result_reused(service_case):
    service, client = service_case
    episode_id = next(i for i in service.manifest["splits"]["development"]
                      if service.manifest["episodes"][i]["gt"] == "incorrect")
    first = service.review(episode_id)
    second = service.review(episode_id)
    assert first["label"] == "correct" and first["evaluation"]["gt"] == "incorrect"
    assert not first["cached"] and second["cached"] and len(client.requests) == 1
    sent = json.dumps(client.requests)
    for sentinel in ("PRIVATE_GT_REASON", "PRIVATE_REVIEWER", "Accepted", "Denied", '"gt"'):
        assert sentinel not in sent
    report = service.report()
    assert report["counts"]["total"] == 6
    assert report["counts"]["not_run"] == 5
    assert report["counts"]["false_accept"] == 1
    assert report["counts"]["coverage"] == 1 / 6


def test_failed_call_requires_explicit_retry(service_case):
    service, client = service_case
    episode_id = service.manifest["splits"]["development"][0]
    client.error = RuntimeError("private vendor error")
    first = service.review(episode_id)
    assert first["status"] == "failed" and first["label"] is None
    assert "private vendor error" not in json.dumps(first)
    client.error = None
    assert service.review(episode_id)["status"] == "failed"
    assert len(client.requests) == 1
    assert service.review(episode_id, retry_failed=True)["status"] == "completed"
    assert len(client.requests) == 2


def test_holdout_requires_complete_development_and_unchanged_freeze(service_case):
    service, client = service_case
    holdout = service.manifest["splits"]["holdout"][0]
    with pytest.raises(GateError):
        service.review(holdout)
    with pytest.raises(GateError):
        service.freeze()
    for episode_id in service.manifest["splits"]["development"]:
        assert service.review(episode_id)["status"] == "completed"
    service.freeze()
    assert service.review(holdout)["status"] == "completed"
    profiles = json.loads(service.profiles_path.read_text())
    profiles["grasp"]["success"] = "changed"
    service.profiles_path.write_text(json.dumps(profiles))
    with pytest.raises(GateError, match="changed"):
        service.review(holdout)


def test_configuration_change_does_not_reuse_old_result(service_case):
    service, client = service_case
    episode_id = service.manifest["splits"]["development"][0]
    first = service.review(episode_id)
    profiles = json.loads(service.profiles_path.read_text())
    profiles["grasp"]["allowed"] = "其他允许的路径"
    service.profiles_path.write_text(json.dumps(profiles))
    second = service.review(episode_id)
    assert len(client.requests) == 2
    assert first["configuration_sha256"] != second["configuration_sha256"]


def test_concurrent_review_rejected_before_model_call(service_case):
    service, client = service_case
    with service.lock(), pytest.raises(BusyError):
        service.review(service.manifest["splits"]["development"][0])
    assert client.requests == []


def test_invalid_response_is_failed_not_business_rejection(service_case, answer):
    service, client = service_case
    answer["checks"]["action"]["evidence_ids"] = ["V999"]
    result = service.review(service.manifest["splits"]["development"][0])
    assert result["status"] == "failed" and result["error"]["stage"] == "evidence"
    assert result["label"] is None and "model_call" in result


def test_batch_is_bounded_resumable_and_exclusive(service_case):
    import threading
    service, client = service_case
    original = client.complete
    lock, ready = threading.Lock(), threading.Event()
    active, peak = 0, 0
    def complete(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
            if active == 2:
                ready.set()
        assert ready.wait(3), "Batch did not run two independent candidates concurrently"
        with pytest.raises(BusyError):
            service.review(service.manifest["splits"]["development"][0])
        value = original(*args, **kwargs)
        with lock:
            active -= 1
        return value
    client.complete = complete
    results = list(service.run(workers=2))
    assert len(results) == 6 and peak == 2
    assert all(r["status"] == "completed" for r in results)
    assert len({r["configuration_sha256"] for r in results}) == 1
    assert all(r["cached"] for r in service.run(workers=2))
    assert len(client.requests) == 6


def test_changed_code_cannot_be_mislabeled_as_loaded_version(service_case, monkeypatch):
    service, client = service_case
    monkeypatch.setattr("citadel.service.code_version", lambda: {"model.py": "changed"})
    with pytest.raises(GateError, match="restart"):
        service.review(service.manifest["splits"]["development"][0])
    assert client.requests == []
