import copy
import json

import httpx
import pytest

from citadel.data import file_hash, read
from citadel.model import CHECKS, Qwen, decide, messages


@pytest.fixture
def model_case(tmp_path, jpeg):
    work = tmp_path / "work"
    work.mkdir()
    path = work / "image.jpg"
    path.write_bytes(jpeg)
    asset = {"path": "image.jpg", "sha256": file_hash(path)}
    resources = {"steps": [{"action_id": "A_001", "action_text": "拿起指定物体。"}],
                 "images": [{**asset, "id": "OBJECT", "name": "目标",
                             "type": "object", "mime": "image/jpeg"}]}
    media = {"frames": [
        {**asset, "frame_id": f"V{i:03d}", "time_s": float(i),
         "sources": {v: {"time_s": float(i)} for v in ("main", "left_wrist", "right_wrist")}}
        for i in range(4)], "warnings": [], "incomplete": False,
        "signature": {"interval_s": 1.0}}
    profile = {"success": "受控悬空约2秒，末态悬空", "hold_seconds": 2.0,
               "hold_tolerance_s": 0.2, "allowed": "自由路径", "failures": ["失败重试"]}
    return work, resources, profile, media


@pytest.fixture
def answer():
    return {"observations": [
        {"phase": "start", "description": "物体静置", "evidence_ids": ["V000"]},
        {"phase": "hold", "description": "夹持并持续悬空", "evidence_ids": ["V001", "V002", "V003"]},
        {"phase": "end", "description": "末态仍悬空", "evidence_ids": ["V003"]}],
        "checks": {key: {"state": "pass", "evidence_ids": ["V000", "V003"]} for key in CHECKS},
        "hold": {"state": "pass", "evidence_ids": ["V001", "V002", "V003"]},
        "reason": "完成基本抓取且末态悬空。"}


def test_basic_grasp_and_actual_timestamps(model_case, answer):
    _, _, profile, media = model_case
    result = decide(answer, media, profile)
    assert result["label"] == "correct"
    assert result["hold_evidence_span_s"] == 2
    changed = copy.deepcopy(media)
    for frame in changed["frames"]:
        frame["time_s"] *= 0.5
    result = decide(answer, changed, profile)
    assert result["label"] is None and result["status"] == "needs_review"


@pytest.mark.parametrize("check", CHECKS)
def test_each_bad_boundary_rejects(model_case, answer, check):
    _, _, profile, media = model_case
    answer["checks"][check]["state"] = "fail"
    result = decide(answer, media, profile)
    assert result["label"] == "incorrect" and check in result["issues"]


def test_failure_not_erased_by_later_success(model_case, answer):
    _, _, profile, media = model_case
    answer["observations"].insert(1, {
        "phase": "failure", "description": "首次空夹失败", "evidence_ids": ["V001"]})
    assert decide(answer, media, profile)["label"] == "incorrect"


@pytest.mark.parametrize("missing", ["camera", "end", "check"])
def test_insufficient_evidence_needs_review(model_case, answer, missing):
    _, _, profile, media = model_case
    if missing == "camera":
        media["incomplete"] = True
    elif missing == "end":
        answer["observations"].pop()
    else:
        answer["checks"]["scene_match"] = {"state": "unknown", "evidence_ids": []}
    assert decide(answer, media, profile)["status"] == "needs_review"


def test_invented_evidence_rejected(model_case, answer):
    _, _, profile, media = model_case
    answer["checks"]["action"]["evidence_ids"] = ["V999"]
    with pytest.raises(ValueError, match="Evidence"):
        decide(answer, media, profile)


def test_other_atomic_task_does_not_require_hold(model_case, answer):
    _, _, _, media = model_case
    answer["hold"] = None
    assert decide(answer, media, {"success": "擦拭指定区域"})["label"] == "correct"


def test_qwen_native_video_and_safe_trace(model_case, answer):
    work, resources, profile, media = model_case
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "response-id", "model": "test-model",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})
    client = Qwen(work, api_key="test-private-secret", transport=httpx.MockTransport(respond))
    result = client.complete(messages(work, resources, profile, media), {"episode_id": "candidate"})
    parts = sent[0]["messages"][1]["content"]
    assert parts[-1]["type"] == "video" and len(parts[-1]["video"]) == 4
    assert parts[-1]["fps"] == 1.0
    assert "V003" in parts[0]["text"] and "source_times_s" in parts[0]["text"]
    assert result["data"] == answer and result["usage"]["total_tokens"] == 150
    trace = (work / result["call_path"] / "request.json").read_text()
    assert "test-private-secret" not in trace and "base64," not in trace


def test_retry_is_bounded_and_every_attempt_recorded(model_case, monkeypatch):
    work, resources, profile, media = model_case
    monkeypatch.setattr("citadel.model.time.sleep", lambda _: None)
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "rate limited"})
    client = Qwen(work, api_key="secret", transport=httpx.MockTransport(respond))
    with pytest.raises(RuntimeError, match="429"):
        client.complete(messages(work, resources, profile, media))
    assert len(calls) == 2
    assert len(list((work / "calls").glob("*/response.json"))) == 2
    assert sorted(read(p)["attempt"] for p in (work / "calls").glob("*/request.json")) == [1, 2]


def test_truncated_model_output_is_execution_error(model_case):
    work, resources, profile, media = model_case
    client = Qwen(work, api_key="secret", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"choices": [{"finish_reason": "length"}]})))
    with pytest.raises(ValueError, match="incomplete"):
        client.complete(messages(work, resources, profile, media))
