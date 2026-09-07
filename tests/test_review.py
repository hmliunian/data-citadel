from dataclasses import replace
from unittest.mock import Mock

import pytest

from data_citadel.experts import EXPERT_POLICY
from data_citadel.models import Assessment, Episode, Evidence, ExpertError, Finding, Frame, MediaError, ProviderError, SampledVideo
from data_citadel.review.integrity import check_fields, check_integrity
from data_citadel.review.policy import decide
from data_citadel.review.service import ReviewService
from data_citadel.settings import Settings


@pytest.fixture
def episode(tmp_path):
    mcap = tmp_path / "episode.mcap"
    sidecar = tmp_path / "episode.json"
    mcap.write_bytes(b"source fixture")
    sidecar.write_text("{}")
    return Episode("candidate", "A_001", "TASK_1", "拿起杯子", "collector", mcap, sidecar,
                   review_status="Accepted")


def passing(**updates):
    data = {
        "verdict": "correct", "reason": "证据通过", "confidence": 0.99, "complete": True,
        "evidence": [Evidence(timestamp_s=0.0, description="开始"),
                     Evidence(timestamp_s=2.0, description="完成")],
    }
    data.update(updates)
    return Assessment(**data)


def test_only_all_complete_supported_passes_can_admit_ground_truth(episode):
    result = decide(episode, {name: passing() for name in ("integrity", "generic", "task")})
    assert result.verdict == "correct" and result.ground_truth_candidate


@pytest.mark.parametrize("change", [
    {"confidence": 0.94}, {"complete": False}, {"evidence": []},
    {"evidence": [Evidence(timestamp_s=2.0, description="仅最终状态")]},
    {"evidence": [Evidence(description="无时间")]},
    {"verdict": "uncertain"}, {"retry_outcome": "success"},
    {"findings": [Finding(code="blurred", reason="与correct矛盾")]},
])
def test_confidence_completion_evidence_and_consistency_are_required(episode, change):
    assessments = {name: passing() for name in ("integrity", "generic", "task")}
    assessments["task"] = passing(**change)
    result = decide(episode, assessments)
    assert result.verdict == "uncertain" and not result.ground_truth_candidate
    assert result.error_types == []


def test_sampling_warning_prevents_ground_truth(episode):
    result = decide(episode, {name: passing() for name in ("integrity", "generic", "task")},
                    warnings=["max_frames_reached"])
    assert result.verdict == "uncertain" and "max_frames_reached" in result.reason


@pytest.mark.parametrize("outcome,retain", [("success", True), ("failure", False), ("unknown", False), ("none", False)])
def test_retries_preserve_training_flag_only_with_observed_eventual_success(episode, outcome, retain):
    assessment = passing(verdict="incorrect", retry_outcome=outcome, findings=[Finding(
        code="repeated_retry", reason="失败后再次尝试", evidence=passing().evidence,
    )])
    result = decide(episode, {"task": assessment})
    assert result.verdict == "incorrect" and not result.ground_truth_candidate
    assert result.retain_retry_sample is retain
    assert result.retry_outcome == ("unknown" if outcome == "none" else outcome)


@pytest.mark.parametrize("confidence,evidence", [(0.6, [Evidence(timestamp_s=0.0, description="疑似")]), (0.99, [])])
def test_unsupported_negative_is_uncertain_without_inventing_other_category(episode, confidence, evidence):
    result = decide(episode, {"task": passing(verdict="incorrect", confidence=confidence,
        findings=[Finding(code="incomplete_action", reason="不够确定", evidence=evidence)])})
    assert result.verdict == "uncertain" and result.error_types == []


def test_accepted_sample_without_deny_reason_is_not_missing_data(episode):
    assert check_fields(episode).verdict == "correct"
    assert check_integrity(episode, {
        "missing_required_channels": [], "empty_required_channels": [], "message_counts": {"camera": 60},
    }).verdict == "correct"


def test_metadata_and_missing_stream_are_specific_data_missing_evidence(episode):
    assert check_fields(replace(episode, instruction="")).findings[0].code == "data_missing"
    assessment = check_integrity(episode, {
        "missing_required_channels": ["camera"], "empty_required_channels": [], "message_counts": {},
    })
    assert assessment.verdict == "incorrect" and "camera" in assessment.reason
    assert decide(episode, {"integrity": assessment}).error_types == ["data_missing"]


@pytest.fixture
def service(episode, tmp_path, monkeypatch):
    repository = Mock()
    repository.get.return_value = episode
    sampler = Mock()
    sampler.sample.return_value = SampledVideo(
        [Frame(0.0, b"a"), Frame(2.0, b"b")], 2.0, "camera", "uniform",
    )
    experts = Mock(version="test-v1", manifest_sha256="a" * 64)
    experts.resolve.return_value = [
        replace(episode, episode_id=f"expert-{i}", task_code=f"other-task-{i}",
                instruction=f"拿起物体{i}", collector_id=f"other-collector-{i}",
                metadata={"task.review": {"reviewer": f"source-reviewer-{i}",
                                         "review_time": "2026-07-30 09:47:01"}})
        for i in range(5)
    ]
    client = Mock()
    client.assess_generic.return_value = passing()
    client.assess_task.return_value = passing()
    monkeypatch.setattr("data_citadel.review.service.inspect_mcap", lambda *args, **kwargs: {
        "missing_required_channels": [], "empty_required_channels": [], "message_counts": {"camera": 60},
    })
    return ReviewService(repository, sampler, experts, client, Settings(
        experts_path=tmp_path / "absent-experts.json", camera_topic="camera",
    ))


def test_service_keeps_each_expert_instruction_and_local_review_provenance(service):
    result = service.review("candidate", "keyframes")
    assert result.ground_truth_candidate
    calls = service.sampler.sample.call_args_list
    assert calls[0].kwargs == {"strategy": "keyframes", "interval_s": 2.0}
    assert all(call.kwargs == {"strategy": "uniform", "interval_s": 1.0} for call in calls[1:])
    assert len(result.provenance["expert_ids"]) == 5
    assert result.provenance["candidate_timestamps_s"] == [0.0, 2.0]
    assert result.provenance["expert_version"] == "test-v1"
    assert result.provenance["expert_config_sha256"] == "a" * 64
    assert result.provenance["expert_matching_policy"] == EXPERT_POLICY
    records = result.provenance["expert_tasks"]
    assert [item["task_code"] for item in records] == [f"other-task-{i}" for i in range(5)]
    assert [item["source_reviewer"] for item in records] == [f"source-reviewer-{i}" for i in range(5)]
    assert all(item["source_review_status"] == "Accepted" for item in records)
    assert all(item["source_review_time"] == "2026-07-30 09:47:01" for item in records)
    assert "source-reviewer-0" in result.model_dump_json()
    assert service.client.assess_generic.call_args.args[0] == "拿起杯子"
    candidate_instruction, _, expert_videos = service.client.assess_task.call_args.args
    assert candidate_instruction == "拿起杯子"
    assert [text for text, _ in expert_videos] == [f"拿起物体{i}" for i in range(5)]
    assert "source-reviewer" not in repr(service.client.mock_calls)
    assert "Accepted" not in repr(service.client.mock_calls)


@pytest.mark.parametrize("generic_verdict,expected", [
    ("correct", "uncertain"), ("uncertain", "uncertain"), ("incorrect", "incorrect"),
])
def test_no_experts_still_runs_generic_without_leaking_resolution_error(service, generic_verdict, expected):
    service.experts.resolve.side_effect = ExpertError("private-label-or-reviewer")
    findings = [Finding(code="blurred", reason="关键画面模糊", evidence=passing().evidence)]
    service.client.assess_generic.return_value = passing(
        verdict=generic_verdict, findings=findings if generic_verdict == "incorrect" else [],
    )
    result = service.review("candidate")
    assert result.verdict == expected and not result.ground_truth_candidate
    assert result.assessments["task"].verdict == "uncertain"
    assert result.assessments["generic"].verdict == generic_verdict
    assert result.error_types == (["blurred"] if generic_verdict == "incorrect" else [])
    assert result.provenance["candidate_timestamps_s"] == [0.0, 2.0]
    assert result.provenance["expert_ids"] == []
    assert result.provenance["expert_tasks"] == []
    assert "private-label-or-reviewer" not in result.model_dump_json()
    service.sampler.sample.assert_called_once()
    service.client.assess_generic.assert_called_once()
    service.client.assess_task.assert_not_called()


def test_expert_sampling_warning_still_prevents_ground_truth(service):
    video = service.sampler.sample.return_value
    service.sampler.sample.side_effect = [video, replace(video, warnings=["coverage_gap"]), *([video] * 4)]
    result = service.review("candidate")
    assert result.verdict == "uncertain" and not result.ground_truth_candidate
    assert result.provenance["warnings"] == ["expert_1: coverage_gap"]


def test_missing_metadata_stops_before_sampling_or_model(service, episode):
    service.repository.get.return_value = replace(episode, instruction="")
    result = service.review("candidate")
    assert result.verdict == "incorrect" and result.error_types == ["data_missing"]
    service.sampler.sample.assert_not_called()
    service.client.assess_generic.assert_not_called()


@pytest.mark.parametrize("stage,error", [
    ("sampler", MediaError("unsupported schema")),
    ("provider", ProviderError("temporary provider outage")),
])
def test_operational_failures_are_not_negative_training_labels(service, stage, error):
    if stage == "sampler":
        service.sampler.sample.side_effect = error
    else:
        service.client.assess_generic.side_effect = error
    with pytest.raises(type(error)):
        service.review("candidate")


@pytest.mark.parametrize("experts_available,max_images", [(True, 11), (False, 1)])
def test_request_budget_fails_before_any_paid_calls(service, experts_available, max_images):
    if not experts_available:
        service.experts.resolve.side_effect = ExpertError("no experts")
    service.settings = replace(service.settings, max_request_images=max_images)
    with pytest.raises(ProviderError, match="image_budget_exceeded"):
        service.review("candidate")
    service.client.assess_generic.assert_not_called()
    service.client.assess_task.assert_not_called()
