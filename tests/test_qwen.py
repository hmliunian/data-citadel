import json
from dataclasses import replace

import httpx
import pytest

from data_citadel.models import Frame, ProviderError, SampledVideo
from data_citadel.qwen import QwenClient
from data_citadel.settings import Settings


@pytest.fixture(autouse=True)
def provider_environment(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test-provider-key")
    monkeypatch.setattr("data_citadel.qwen.time.sleep", lambda _: None)


@pytest.fixture
def video():
    return SampledVideo(
        [Frame(0.0, b"jpeg-a"), Frame(2.0, b"jpeg-b")], 2.0, "/private/camera", "uniform",
        motion={"stationary_intervals": [], "deny_reason": "hidden-label", "collector_id": "hidden-person"},
    )


@pytest.fixture
def multiview(video):
    return replace(video, duration_s=2.01, frames=[
        Frame(0.0, b"main-start"), Frame(0.004, b"left-start", "left_wrist"),
        Frame(0.008, b"right-start", "right_wrist"), Frame(2.0, b"main-end"),
        Frame(2.004, b"left-end", "left_wrist"), Frame(2.008, b"right-end", "right_wrist"),
    ], motion={**video.motion, "gt_label": "private-ground-truth", "reviewer": "private-reviewer",
               "views": {"left_wrist": {"topic": "/private/wrist-topic", "gt": "private-source-label"}}})


def model_response(**updates):
    value = {
        "verdict": "correct", "findings": [], "reason": "画面清晰且可观察到操作完成。",
        "evidence": [{"timestamp_s": 0.0, "description": "起始状态"},
                     {"timestamp_s": 2.0, "description": "动作完成"}],
        "confidence": 0.99, "complete": True, "retry_outcome": "none",
    }
    value.update(updates)
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}


def test_visual_request_contains_only_instructions_frames_and_safe_motion(video):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-provider-key"
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=model_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        result = QwenClient(Settings(), transport).assess_task("拿起杯子", video, [("拿起杯子", video)] * 5)
        assert result.verdict == "correct"
        assert all(item.view == "main" for item in result.evidence)
    payload = captured[0]
    text = json.dumps(payload, ensure_ascii=False)
    for secret in ("hidden-label", "hidden-person", "/private/camera", "test-provider-key", "deny_reason"):
        assert secret not in text
    assert payload["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in payload["messages"][0]["content"]
    parts = payload["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in parts) == 12
    assert "EXPERT 5" in text and "CANDIDATE view=main timestamp_s=2.0" in text


def test_generic_sends_only_main_frames_and_a_quality_only_schema(multiview):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=model_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        result = QwenClient(Settings(max_request_images=2), transport).assess_generic("任务", multiview)
    assert all(item.view == "main" for item in result.evidence)
    payload = captured[0]
    parts = payload["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in parts) == 2
    assert json.loads(parts[0]["text"])["sampled_timestamps_by_view"] == {"main": [0.0, 2.0]}
    assert json.loads(parts[0]["text"])["duration_s"] == multiview.duration_s
    schema = json.loads(payload["messages"][0]["content"].split("JSON Schema:\n")[1])
    assert schema["$defs"]["Finding"]["properties"]["code"]["enum"] == ["blurred", "other"]


def test_generic_http_payload_is_identical_when_wrist_video_ends_later(video):
    combined = replace(
        video, frames=[*video.frames, Frame(2.01, b"later-wrist", "right_wrist")], duration_s=2.01,
        motion={**video.motion, "views": {"main": {"duration_s": video.duration_s}}},
    )
    transmitted = []

    def handler(request):
        transmitted.append(request.content)
        return httpx.Response(200, json=model_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = QwenClient(Settings(), transport)
        client.assess_generic("任务", video)
        client.assess_generic("任务", combined)
    assert len(transmitted) == 2
    assert transmitted[0] == transmitted[1]


def test_multiview_task_sends_all_views_without_gt_or_source_metadata(multiview):
    captured = []
    body = model_response(evidence=[
        {"view": "main", "timestamp_s": 0.0, "description": "过程"},
        {"view": "main", "timestamp_s": 2.0, "description": "完成"},
        {"view": "left_wrist", "timestamp_s": 2.0047, "description": "物体身份细节"},
    ])

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        result = QwenClient(Settings(max_request_images=36), transport).assess_task(
            "候选物体", multiview, [(f"示范物体{i}", multiview) for i in range(5)],
        )
    assert result.evidence[-1].view == "left_wrist" and result.evidence[-1].timestamp_s == 2.004
    payload = captured[0]
    parts = payload["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in parts) == 36
    headers = [json.loads(part["text"]) for part in parts if part["type"] == "text" and part["text"].startswith("{")]
    assert [header["video"] for header in headers] == [*(f"EXPERT {i}" for i in range(1, 6)), "CANDIDATE"]
    assert all(set(header["sampled_timestamps_by_view"]) == {"main", "left_wrist", "right_wrist"} for header in headers)
    transmitted = json.dumps(payload, ensure_ascii=False)
    for private in ("private-ground-truth", "private-reviewer", "private-source-label", "/private/", "hidden-label"):
        assert private not in transmitted
    assert "CANDIDATE view=right_wrist timestamp_s=0.008" in transmitted
    assert headers[-1]["task_instruction"] == "候选物体"
    schema = json.loads(payload["messages"][0]["content"].split("JSON Schema:\n")[1])
    assert "view" in schema["$defs"]["Evidence"]["required"]
    assert schema["$defs"]["Evidence"]["properties"]["view"] == {
        "type": "string", "enum": ["main", "left_wrist", "right_wrist"],
    }


@pytest.mark.parametrize("view,timestamp,message", [
    (None, 0.0, "requires_view"),
    ("left_wrist", 0.0, "not_observed_in_candidate_view"),
    ("right_wrist", 0.004, "not_observed_in_candidate_view"),
])
def test_multiview_evidence_cannot_borrow_another_views_timestamp(multiview, view, timestamp, message):
    body = model_response(evidence=[{"view": view, "timestamp_s": timestamp, "description": "来源必须匹配"}])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        with pytest.raises(ProviderError, match=message):
            QwenClient(Settings(), transport).assess_task("任务", multiview, [("任务", multiview)] * 5)


def test_expert_view_cannot_replace_a_missing_candidate_view(multiview):
    candidate = replace(multiview, frames=[frame for frame in multiview.frames if frame.view != "right_wrist"])
    body = model_response(evidence=[{"view": "right_wrist", "timestamp_s": 0.008, "description": "只在专家出现"}])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        with pytest.raises(ProviderError, match="view_not_observed_in_candidate"):
            QwenClient(Settings(), transport).assess_task("任务", candidate, [("任务", multiview)] * 5)


def test_wrist_finding_is_checked_even_with_valid_main_evidence(multiview):
    body = model_response(
        evidence=[{"view": "main", "timestamp_s": 0.0, "description": "有效候选帧"}],
        findings=[{"code": "content_mismatch", "reason": "错误视角",
                   "evidence": [{"view": "left_wrist", "timestamp_s": 0.008, "description": "右腕的时间"}]}],
    )
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        with pytest.raises(ProviderError, match="not_observed_in_candidate_view"):
            QwenClient(Settings(), transport).assess_task("任务", multiview, [("任务", multiview)] * 5)


def test_all_views_and_five_experts_count_toward_the_request_budget(multiview, monkeypatch):
    monkeypatch.setattr(Settings, "api_key", lambda _: pytest.fail("budget must be checked before credentials"))
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("unexpected network"))) as transport:
        with pytest.raises(ProviderError, match="image_budget_exceeded"):
            QwenClient(Settings(max_request_images=35), transport).assess_task("任务", multiview, [("任务", multiview)] * 5)


@pytest.mark.parametrize("code", ["content_mismatch", "incomplete_action", "annotation_error", "repeated_retry", "data_missing"])
def test_generic_rejects_task_or_structural_classifications(video, code):
    body = model_response(verdict="incorrect", findings=[{
        "code": code, "reason": "越过通用画质职责", "evidence": [{"timestamp_s": 0.0, "description": "可见时间"}],
    }])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        with pytest.raises(ProviderError, match="generic_review_invalid_finding_code"):
            QwenClient(Settings(), transport).assess_generic("任务", video)


@pytest.mark.parametrize("code", ["blurred", "other"])
def test_generic_retains_supported_quality_findings(video, code):
    body = model_response(verdict="incorrect", findings=[{
        "code": code, "reason": "明确画质问题", "evidence": [{"timestamp_s": 0.0, "description": "问题帧"}],
    }])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        result = QwenClient(Settings(), transport).assess_generic("任务", video)
    assert result.verdict == "incorrect" and result.findings[0].code == code


@pytest.mark.parametrize("update", [
    {"unexpected_field": "secret-upstream"}, {"confidence": "0.99"}, {"confidence": 1.1},
    {"verdict": "probably_correct"}, {"complete": "true"}, {"reason": "   "},
    {"evidence": [{"timestamp_s": 0.0, "description": "   "}]},
    {"evidence": [{"timestamp_s": float("inf"), "description": "infinite time"}]},
    {"evidence": [{"view": "invented_camera", "timestamp_s": 0.0, "description": "bad view"}]},
    {"findings": [{"code": "invented_error", "reason": "bad"}]},
])
def test_response_schema_is_strict(video, update):
    transport = httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=model_response(**update))
    ))
    with transport, pytest.raises(ProviderError, match="invalid_or_incomplete") as error:
        QwenClient(Settings(), transport).assess_generic("任务", video)
    assert "secret-upstream" not in str(error.value)


@pytest.mark.parametrize("response", [
    {}, {"choices": []}, {"choices": [None]}, {"choices": [[]]},
    {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "```json\n{}\n```"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": {}}}]},
])
def test_malformed_or_truncated_response_is_operational_error(video, response):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))) as transport:
        with pytest.raises(ProviderError, match="invalid_or_incomplete"):
            QwenClient(Settings(), transport).assess_generic("任务", video)


@pytest.mark.parametrize("timestamp", [None, 2.01])
def test_evidence_must_be_on_candidate_timeline(video, timestamp):
    body = model_response(evidence=[{"timestamp_s": timestamp, "description": "专家画面不能充当待测证据"}])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        with pytest.raises(ProviderError, match="candidate_timeline"):
            QwenClient(Settings(), transport).assess_generic("任务", video)


def test_full_timestamp_display_and_rounded_end_evidence_are_consistent(video):
    timestamp = 2.0000006
    video = replace(video, frames=[video.frames[0], Frame(timestamp, b"last")], duration_s=timestamp)
    body = model_response(evidence=[{"timestamp_s": 2.000001, "description": "完整末帧"}])

    def handler(request):
        assert f"CANDIDATE view=main timestamp_s={timestamp}" in request.content.decode()
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        result = QwenClient(Settings(), transport).assess_generic("任务", video)
    assert result.evidence[0].timestamp_s == timestamp and result.evidence[0].view == "main"


def test_expert_frame_inside_duration_cannot_substitute_for_unseen_candidate_frame(video):
    expert = replace(video, frames=[video.frames[0], Frame(1.0, b"expert-only"), video.frames[1]])
    body = model_response(evidence=[{"timestamp_s": 1.0, "description": "在专家中可见的动作"}])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        with pytest.raises(ProviderError, match="not_observed_in_candidate"):
            QwenClient(Settings(), transport).assess_task("任务", video, [("任务", expert)] * 5)


@pytest.mark.parametrize("stage,timestamp,accepted", [
    ("generic", 0.2508, True), ("generic", 1.7505, True),
    ("generic", 1.0, False), ("task", 0.25, False),
])
def test_only_generic_review_can_cite_motion_interval_boundaries(video, stage, timestamp, accepted):
    video = replace(video, motion={"stationary_intervals": [{"start_s": 0.25, "end_s": 1.75}]})
    body = model_response(evidence=[{"timestamp_s": timestamp, "description": "静止摘要的边界"}])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        client = QwenClient(Settings(), transport)

        def assess():
            if stage == "task":
                return client.assess_task("任务", video, [("任务", video)] * 5)
            return client.assess_generic("任务", video)

        if accepted:
            assert assess().evidence[0].timestamp_s == (0.25 if timestamp < 1 else 1.75)
        else:
            with pytest.raises(ProviderError, match="not_observed_in_candidate"):
                assess()


@pytest.mark.parametrize("timestamp,message", [(20.0, "candidate_timeline"), (1.998, "not_observed_in_candidate")])
def test_finding_timestamps_are_validated_too(video, timestamp, message):
    body = model_response(verdict="incorrect", findings=[{
        "code": "incomplete_action", "reason": "截断",
        "evidence": [{"timestamp_s": timestamp, "description": "时间不对应待测画面"}],
    }])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as transport:
        with pytest.raises(ProviderError, match=message):
            QwenClient(Settings(), transport).assess_generic("任务", video)


@pytest.mark.parametrize("status,attempts", [(429, 3), (503, 3), (401, 1), (400, 1)])
def test_retries_are_bounded_and_provider_body_is_not_exposed(video, status, attempts):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, text="sensitive upstream body test-provider-key", headers={"Retry-After": "0"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        with pytest.raises(ProviderError, match=f"http_{status}") as error:
            QwenClient(Settings(max_retries=2), transport).assess_generic("任务", video)
    assert len(requests) == attempts
    assert "sensitive" not in str(error.value) and "test-provider-key" not in str(error.value)


def test_transient_transport_failure_then_success(video):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("secret transport detail", request=request)
        return httpx.Response(200, json=model_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        assert QwenClient(Settings(), transport).assess_generic("任务", video).verdict == "correct"
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["budget", "text_model", "empty_frames", "backwards_time", "expert_count"])
def test_invalid_inputs_fail_before_credentials_or_network(video, mode, monkeypatch):
    def no_key(_):
        pytest.fail("credentials must not be read for invalid input")

    monkeypatch.setattr(Settings, "api_key", no_key)
    settings = Settings(max_request_images=1) if mode == "budget" else Settings()
    if mode == "text_model":
        settings = replace(settings, model="qwen-max")
    elif mode == "empty_frames":
        video = replace(video, frames=[])
    elif mode == "backwards_time":
        video = replace(video, frames=list(reversed(video.frames)))
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("unexpected network"))) as transport:
        client = QwenClient(settings, transport)
        with pytest.raises(ProviderError):
            if mode == "expert_count":
                client.assess_task("任务", video, [("任务", video)])
            else:
                client.assess_generic("任务", video)
