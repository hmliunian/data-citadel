import json

import httpx
import pytest
from PIL import Image

from pilot.data import sha256
from pilot.model import ModelCallError, QwenClient, decide, frame_parts


def sample(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8)).save(path)
    return {"frames": [{"frame_id": "main-00000", "view": "main", "time_s": 0.2,
                       "path": "frame.jpg", "sha256": sha256(path)}], "warnings": []}


def answer(state="pass"):
    return {"reason": "测试证据", "checks": {key: {"state": state, "evidence_ids": ["C-main-00000"]}
            for key in ("object", "action", "retry_free", "quality")}}


def test_decision_unknown_and_failed_retry(tmp_path):
    media = sample(tmp_path)
    assert decide(answer(), media)["label"] == "correct"
    data = answer()
    data["checks"]["retry_free"]["state"] = "fail"
    assert decide(data, media)["label"] == "incorrect"
    data["checks"]["retry_free"]["state"] = "unknown"
    assert decide(data, media)["status"] == "needs_review"
    assert decide(data, media)["label"] is None


def test_candidate_evidence_is_checked(tmp_path):
    data = answer()
    data["checks"]["action"]["evidence_ids"] = ["E1-main-00000"]
    with pytest.raises(ValueError, match="candidate"):
        decide(data, sample(tmp_path))


def test_frame_input_uses_allowlist_and_never_source_gt(tmp_path):
    media = sample(tmp_path)
    media.update({"gt": "SENTINEL_GT", "quality": "SENTINEL_QUALITY",
                  "reviewer": "SENTINEL_REVIEWER", "deny_reason": "SENTINEL_REASON"})
    text = json.dumps(frame_parts(media, "C", tmp_path))
    assert "SENTINEL" not in text
    assert "C-main-00000" in text
    assert "data:image/jpeg;base64," in text


def test_transport_archives_usage_without_key(tmp_path):
    def respond(request):
        assert request.headers["authorization"] == "Bearer secret-test-key"
        assert json.loads(request.content)["response_format"]["type"] == "json_object"
        return httpx.Response(200, json={"id": "test", "model": "qwen-vl-max",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer())}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    client = QwenClient(tmp_path, api_key="secret-test-key", transport=httpx.MockTransport(respond))
    reply = client.complete([{"role": "user", "content": "Return JSON"}])
    assert reply["data"] == answer()
    assert reply["usage"]["prompt_tokens"] == 10
    assert all("secret-test-key" not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_provider_failure_is_not_semantic_incorrect(tmp_path):
    client = QwenClient(tmp_path, api_key="test",
                        transport=httpx.MockTransport(lambda _: httpx.Response(
                            401, json={"error": {"code": "InvalidApiKey"}})))
    with pytest.raises(ModelCallError, match="401"):
        client.complete([{"role": "user", "content": "JSON"}])
