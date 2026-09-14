import copy
import json

import httpx
import pytest

from citadel.data import read
from citadel.model import CHECKS, Qwen, decide, messages


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


def test_stitched_sequence_keeps_timestamps_and_safe_trace(model_case, answer):
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
    assert [p["type"] for p in parts[-8:]] == ["text", "image_url"] * 4
    timing = json.loads(parts[-2]["text"])["candidate_frame"]
    assert timing["frame_id"] == "V003" and timing["time_s"] == 3.0
    assert timing["source_times_s"]["main"] == 3.0
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


def test_empty_main_panel_cannot_prove_visibility_failure(model_case, answer):
    _, _, profile, media = model_case
    media["frames"][0]["sources"]["main"] = None
    answer["checks"]["main_visibility"] = {"state": "fail", "evidence_ids": ["V000"]}
    result = decide(answer, media, profile)
    assert result["label"] is None and result["status"] == "needs_review"
    assert result["checks"]["main_visibility"]["state"] == "unknown"


def test_mid_video_hold_does_not_prove_final_airborne_state(model_case, answer):
    _, _, profile, media = model_case
    answer["hold"]["evidence_ids"] = ["V000", "V001", "V002"]
    assert decide(answer, media, profile)["status"] == "needs_review"


def test_supported_new_qwen_runs_without_hidden_thinking(model_case, answer):
    work, resources, profile, media = model_case
    seen = []
    def respond(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(answer)}}]})
    client = Qwen(work, model="qwen3.5-plus-2026-02-15", api_key="secret",
                  transport=httpx.MockTransport(respond))
    result = client.complete(messages(work, resources, profile, media))
    assert seen[0]["enable_thinking"] is False
    trace = read(work / result["call_path"] / "request.json")
    assert trace["parameters"]["enable_thinking"] is False
    context = json.loads(seen[0]["messages"][1]["content"][0]["text"])
    assert context["camera_ranges"]["main"] == {"first_frame_id": "V000", "last_frame_id": "V003"}
