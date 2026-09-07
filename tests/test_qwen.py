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
        client = QwenClient(Settings(), transport)
        assert client.assess_task("拿起杯子", video, [("拿起杯子", video)] * 5).verdict == "correct"
    payload = captured[0]
    text = json.dumps(payload, ensure_ascii=False)
    for secret in ("hidden-label", "hidden-person", "/private/camera", "test-provider-key", "deny_reason"):
        assert secret not in text
    assert payload["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in payload["messages"][0]["content"]
    parts = payload["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in parts) == 12
    assert "EXPERT 5" in text and "CANDIDATE timestamp_s=2.0" in text


@pytest.mark.parametrize("update", [
    {"unexpected_field": "secret-upstream"}, {"confidence": "0.99"}, {"confidence": 1.1},
    {"verdict": "probably_correct"}, {"complete": "true"}, {"reason": "   "},
    {"evidence": [{"timestamp_s": 0.0, "description": "   "}]},
    {"evidence": [{"timestamp_s": float("inf"), "description": "infinite time"}]},
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
        assert f"CANDIDATE timestamp_s={timestamp}" in request.content.decode()
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        result = QwenClient(Settings(), transport).assess_generic("任务", video)
    assert result.evidence[0].timestamp_s == timestamp


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
