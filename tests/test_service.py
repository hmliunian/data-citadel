import copy
import json
from pathlib import Path

import pytest

from citadel.data import file_hash, fingerprint, prepare, write
from citadel.service import BusyError, GateError, Service


@pytest.fixture
def service_case(tmp_path, dataset, model_case, answer, jpeg):
    _, resources, profile, media = model_case
    work = tmp_path / "run"
    prepare(dataset, work)
    (work / "image.jpg").write_bytes(jpeg)
    profiles = work / "rules.json"
    write(profiles, {"grasp": {**profile, "action_ids": ["A_001"]}})
    resources = {**resources, "task_code": "DL-TEST"}
    resources["sha256"] = fingerprint(resources)
    class FakeClient:
        model, base_url = "fake-qwen", "https://example.invalid/v1"
        def __init__(self):
            self.requests = []
            self.error = None
        def complete(self, request_messages, context):
            self.requests.append(copy.deepcopy(request_messages))
            if self.error:
                raise self.error
            return {"data": copy.deepcopy(answer), "model": self.model, "usage": {}}
    client = FakeClient()
    def load_media(output, source, sampling):
        assert set(source) == {"episode_id", "mcap_path", "mcap_sha256"}
        assert file_hash(Path(source["mcap_path"])) == source["mcap_sha256"]
        result = copy.deepcopy(media)
        result["episode_id"] = source["episode_id"]
        folder = work / "media" / source["episode_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for frame in result["frames"]:
            path = folder / (frame["frame_id"] + ".jpg")
            path.write_bytes(jpeg)
            frame["path"] = str(path.relative_to(work))
        result["videos"] = {}
        if not (folder / "media.json").exists():
            write(folder / "media.json", result)
        return result
    service = Service(work, profiles, client=client,
                      resource_loader=lambda *args: resources, media_loader=load_media)
    return service, client


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
