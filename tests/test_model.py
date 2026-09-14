import copy
import json

import httpx
import pytest

from citadel.infrastructure.files import read
from citadel.domain.models import CHECKS
from citadel.domain.decision import decide
from citadel.infrastructure.qwen import QwenGateway as Qwen
from citadel.application.prompts import PromptBuilder

messages = PromptBuilder().review
quality_messages = PromptBuilder().quality


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
    if check == "image_quality":
        answer["quality_by_camera"]["main"]["state"] = "fail"
    if check == "main_visibility":
        answer["main_visibility_by_frame"]["V001"] = "absent"
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
    monkeypatch.setattr("citadel.infrastructure.qwen.time.sleep", lambda _: None)
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


@pytest.mark.parametrize("model", ["qwen3.5-plus-2026-02-15", "qwen3.8-max-0902"])
def test_supported_new_qwen_runs_without_hidden_thinking(model_case, answer, model):
    work, resources, profile, media = model_case
    seen = []
    def respond(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(answer)}}]})
    client = Qwen(work, model=model, api_key="secret",
                  transport=httpx.MockTransport(respond))
    result = client.complete(messages(work, resources, profile, media))
    assert seen[0]["enable_thinking"] is False
    output = seen[0]["response_format"]
    if model.startswith("qwen3.8-max"):
        assert output["type"] == "json_schema" and output["json_schema"]["strict"]
        schema = output["json_schema"]["schema"]
        checks = schema["properties"]["checks"]
        assert set(checks["properties"]) == set(CHECKS)
        assert set(checks["required"]) == set(CHECKS)
        assert checks["additionalProperties"] is False
        assert "hold" in schema["required"] and "hold" not in checks["properties"]
    else:
        assert output == {"type": "json_object"}
    trace = read(work / result["call_path"] / "request.json")
    assert trace["parameters"]["enable_thinking"] is False
    context = json.loads(seen[0]["messages"][1]["content"][0]["text"])
    assert context["camera_ranges"]["main"] == {"first_frame_id": "V000", "last_frame_id": "V003"}



def test_gripper_extrema_are_bound_to_video_time_without_gt(model_case):
    work, resources, profile, media = model_case
    media["gripper"] = {
        "sha256": "verified-signal-hash", "warnings": ["right_position:missing"],
        "channels": {"left_position": {"samples": [[0.35, 20], [0.39, 100]]}},
        "gt": "private-review-label",
    }
    sent = messages(work, resources, profile, media)
    content = [json.loads(p["text"]) for p in sent[1]["content"] if p["type"] == "text"]
    candidate = next(p["candidate_frame"] for p in content if p.get("candidate_frame", {}).get("frame_id") == "V001")
    assert candidate["gripper_before"] == [[0.3, 0.4, [20, 100], None, None, None, None, None]]
    assert len(next(p["gripper_columns"] for p in content if "gripper_columns" in p)) == 8
    assert "private-review-label" not in json.dumps(sent)
    assert "原始joint_position" in sent[0]["content"]


def test_absent_main_frame_overrides_later_visible_summary(model_case, answer):
    _, _, profile, media = model_case
    answer["main_visibility_by_frame"]["V001"] = "absent"
    result = decide(answer, media, profile)
    assert result["label"] == "incorrect"
    assert result["checks"]["main_visibility"] == {"state": "fail", "evidence_ids": ["V001"]}
    assert "main_visibility" in result["issues"]
    assert result["checks"]["retry_free"]["state"] == "pass"


def test_visibility_requires_every_frame_and_never_invents_technical_blanks(model_case, answer):
    _, _, profile, media = model_case
    del answer["main_visibility_by_frame"]["V001"]
    with pytest.raises(ValueError, match="every candidate frame"):
        decide(answer, media, profile)
    answer["main_visibility_by_frame"]["V001"] = "no_frame"
    result = decide(answer, media, profile)
    assert result["status"] == "needs_review"
    assert result["main_visibility_by_frame"]["V001"] == "uncertain"


def test_technical_blank_is_excluded_from_visibility_failure(model_case, answer):
    _, _, profile, media = model_case
    media["frames"][0]["sources"]["main"] = None
    answer["main_visibility_by_frame"]["V000"] = "absent"
    result = decide(answer, media, profile)
    assert result["label"] == "correct"
    assert result["main_visibility_by_frame"]["V000"] == "no_frame"


def test_response_schema_requires_each_frame_and_respects_camera_presence(model_case, answer):
    work, resources, profile, media = model_case
    media["frames"][0]["sources"]["main"] = None
    answer["main_visibility_by_frame"]["V000"] = "no_frame"
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(answer)}}]})
    Qwen(work, api_key="secret", transport=httpx.MockTransport(respond)).complete(
        messages(work, resources, profile, media))
    schema = sent[0]["response_format"]["json_schema"]["schema"]
    assert "quality_by_camera" not in schema["properties"]
    table = schema["properties"]["main_visibility_by_frame"]
    assert "main_visibility_by_frame" in schema["required"]
    assert table["required"] == ["V000", "V001", "V002", "V003"]
    assert table["additionalProperties"] is False
    assert table["properties"]["V000"]["enum"] == ["no_frame"]
    assert table["properties"]["V001"]["enum"] == ["visible", "absent", "uncertain"]


@pytest.mark.parametrize("camera", ["main", "left_wrist", "right_wrist"])
def test_one_blurred_camera_rejects_despite_clear_other_views(model_case, answer, camera):
    _, _, profile, media = model_case
    answer["quality_by_camera"][camera].update(
        state="fail", evidence_ids=["V001", "V003"], description="持续发虚，轮廓尚可见")
    result = decide(answer, media, profile)
    assert result["label"] == "incorrect"
    assert result["checks"]["image_quality"] == {"state": "fail", "evidence_ids": ["V001", "V003"]}
    assert result["checks"]["action"]["state"] == "pass"
    assert result["checks"]["main_visibility"]["state"] == "pass"


def test_missing_camera_quality_or_unknown_evidence_cannot_pass(model_case, answer):
    _, _, profile, media = model_case
    del answer["quality_by_camera"]["main"]
    with pytest.raises(ValueError, match="all three cameras"):
        decide(answer, media, profile)
    answer["quality_by_camera"]["main"] = {
        "state": "unknown", "evidence_ids": [], "description": "画质证据不足"}
    assert decide(answer, media, profile)["status"] == "needs_review"
    answer["quality_by_camera"]["main"].update(state="fail", evidence_ids=["V999"])
    with pytest.raises(ValueError, match="Evidence"):
        decide(answer, media, profile)


def test_quality_ignores_technical_blank_evidence(model_case, answer):
    _, _, profile, media = model_case
    media["frames"][0]["sources"]["main"] = None
    answer["quality_by_camera"]["main"].update(state="fail", evidence_ids=["V000"])
    result = decide(answer, media, profile)
    assert result["status"] == "needs_review"
    assert result["checks"]["image_quality"]["state"] == "unknown"
    assert result["quality_by_camera"]["main"]["evidence_ids"] == []

def test_conflicting_quality_summary_needs_review(model_case, answer):
    _, _, profile, media = model_case
    answer["checks"]["image_quality"]["state"] = "fail"
    assert decide(answer, media, profile)["status"] == "needs_review"


def test_quality_call_is_separate_from_task_references_and_gripper(model_case, answer):
    work, _, _, media = model_case
    media["gripper"] = {"private_sensor_marker": True}
    request = quality_messages(work, media)
    seen = []
    def respond(sent):
        seen.append(json.loads(sent.content))
        data = {"quality_by_camera": answer["quality_by_camera"]}
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(data)}}]})
    Qwen(work, api_key="secret", transport=httpx.MockTransport(respond)).complete(
        request, quality_only=True)
    schema = seen[0]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["quality_by_camera"]
    assert set(schema["properties"]["quality_by_camera"]["required"]) == {
        "main", "left_wrist", "right_wrist"}
    assert schema["properties"]["quality_by_camera"]["additionalProperties"] is False
    serialized = json.dumps(request)
    for excluded in ("gripper_before", "reference_type", "private_sensor_marker"):
        assert excluded not in serialized
    assert sum(p["type"] == "image_url" for p in request[1]["content"]) == len(media["frames"])
    assert media["gripper"] == {"private_sensor_marker": True}
