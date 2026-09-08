"""Exercise real module boundaries with generated MCAP and an in-memory HTTP provider."""

import hashlib
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from data_citadel.api import create_app
from data_citadel.experts import EXPERT_POLICY, ExpertLibrary
from data_citadel.media import VideoSampler
from data_citadel.media.sampling import WRIST_TOPICS
from data_citadel.qwen import QwenClient
from data_citadel.repository import EpisodeRepository
from data_citadel.review.service import ReviewService
from data_citadel.settings import Settings
from test_media import TOPIC, synthetic_video
from test_repository import bundle


@pytest.mark.parametrize("camera_mode", ["main", "main_wrist"])
@pytest.mark.parametrize("model_verdict,strategy,approved,expected,error_types", [
    ("correct", "uniform", True, "correct", []),
    ("uncertain", "uniform", True, "uncertain", []),
    ("incorrect", "uniform", True, "incorrect", ["incomplete_action"]),
    ("correct", "keyframes", True, "uncertain", []),
    ("correct", "uniform", False, "uncertain", []),
    ("incorrect", "uniform", False, "incorrect", ["blurred"]),
])
def test_verified_mcap_experts_qwen_and_api_share_contracts(
    tmp_path, monkeypatch, model_verdict, strategy, approved, expected, error_types, camera_mode,
):
    # Generated fixtures only: this approval cannot affect real expert manifests.
    monkeypatch.setenv("QWEN_API_KEY", "integration-fixture-key")
    cameras = {TOPIC: 0, **dict.fromkeys(WRIST_TOPICS.values(), 0)}
    encoded = synthetic_video(tmp_path, frames=21, moving=True, cameras=cameras).mcap_path.read_bytes()
    dataset = tmp_path / "dataset"
    ids = [f"{index:032x}" for index in range(6)]
    instructions = [f"拿起{object_name}。" for object_name in ("杯子", "书本", "瓶子", "毛巾", "玩具", "苹果")]
    for index, episode_id in enumerate(ids):
        directory = bundle(dataset, episode_id, label="other" if index == 5 else "correct", document={
            "task.action_id": "A_001", "task.task_code": f"private-task-{index}",
            "task.action_text": {"rendered_zh": instructions[index]},
            "task.collector.user": f"private-person-{index}", "task.review.status": "Accepted",
            "task.review.reviewer": f"private-reviewer-{index}",
            "task.review.deny_reason": "private-evaluation-label",
        })
        path = directory / "episode.mcap"
        path.write_bytes(encoded)
        receipt_path = directory / "verification.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][path.name].update(
            size=len(encoded), remote_size=len(encoded), mtime_ns=path.stat().st_mtime_ns,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )
        receipt_path.write_text(json.dumps(receipt))
    manifest = tmp_path / "experts.json"
    assert EXPERT_POLICY == "same_action_v2"
    manifest.write_text(json.dumps({
        "version": "synthetic-integration-v2",
        "groups": [{
            "action_id": "A_001", "approved": approved, "approval_source": "dataset_review",
            "policy": EXPERT_POLICY, "expert_episode_ids": ids[:5],
        }],
    }))
    settings = Settings(
        dataset_root=dataset, experts_path=manifest, artifacts_dir=tmp_path / "artifacts",
        camera_topic=TOPIC, camera_mode=camera_mode,
        model="qwen-vl-max", base_url="https://fixture.invalid/v1",
    )
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.headers["authorization"] == "Bearer integration-fixture-key"
        content = payload["messages"][1]["content"]
        is_task = json.loads(content[0]["text"])["video"] == "EXPERT 1"
        verdict = model_verdict if is_task or not approved else "correct"
        assessment = {
            "verdict": verdict, "reason": "synthetic provider assessment",
            "confidence": 0.99, "complete": verdict != "uncertain",
            "evidence": [{"view": "main", "timestamp_s": 0.0, "description": "start"},
                         {"view": "main", "timestamp_s": 2.0, "description": "completion"}],
        }
        if verdict == "incorrect":
            assessment["findings"] = [{
                "code": "incomplete_action" if is_task else "blurred",
                "reason": "visible fixture problem",
                "evidence": [{"view": "main", "timestamp_s": 2.0, "description": "problem at end"}],
            }]
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(assessment)},
        }]})

    repository = EpisodeRepository(dataset)
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        service = ReviewService(
            repository, VideoSampler(TOPIC, cache_dir=settings.artifacts_dir / "cache"),
            ExpertLibrary(manifest, repository), QwenClient(settings, transport), settings,
        )
        with TestClient(create_app(settings, service)) as web:
            response = web.post("/v1/reviews", json={"episode_id": ids[-1], "strategy": strategy})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["verdict"] == expected
    assert result["ground_truth_candidate"] is (expected == "correct")
    assert result["error_types"] == error_types
    assert result["provenance"]["expert_ids"] == (ids[:5] if approved else [])
    assert ids[-1] not in result["provenance"]["expert_ids"]
    views = ["main"] if camera_mode == "main" else ["main", "left_wrist", "right_wrist"]
    assert result["provenance"]["camera_mode"] == camera_mode
    assert set(result["provenance"]["candidate_views"]) == set(views)
    assert result["provenance"]["candidate_timestamps_s"] == sorted([0.0, 2.0] * len(views))
    assert len(requests) == (2 if approved else 1)
    for index, payload in enumerate(requests):
        content = payload["messages"][1]["content"]
        headers = [json.loads(item["text"]) for item in content
                   if item["type"] == "text" and item["text"].startswith("{")]
        expected_instructions = instructions if index else instructions[-1:]
        assert [header["task_instruction"] for header in headers] == expected_instructions
        assert headers[-1]["video"] == "CANDIDATE"
        task_images = 17 if camera_mode == "main" else 41
        assert sum(item["type"] == "image_url" for item in content) == (task_images if index else 2)
    transmitted = json.dumps(requests)
    for private in ("private-person", "private-reviewer", "private-task", "private-evaluation-label", str(dataset), *ids):
        assert private not in transmitted
    artifacts = list((settings.artifacts_dir / "reviews").glob("*.json"))
    assert len(artifacts) == 1
    assert json.loads(artifacts[0].read_text()) == result
