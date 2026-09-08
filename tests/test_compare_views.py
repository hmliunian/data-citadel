import hashlib
import json
from pathlib import Path
from runpy import run_path

import httpx
import pytest

from data_citadel.experts import EXPERT_POLICY
from data_citadel.models import ReviewResult
from data_citadel.settings import Settings
from test_repository import bundle


run_compare = run_path(str(Path(__file__).parents[1] / "scripts" / "compare_views.py"))["run_compare"]


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "dataset"
    expert_ids = [f"{i:032x}" for i in range(5)]
    target_ids = [f"{i:032x}" for i in (100, 101)]
    for episode_id in expert_ids:
        bundle(root, episode_id)
    bundle(root, target_ids[0], label="correct")
    bundle(root, target_ids[1], label="annotation_error")
    path = tmp_path / "experts.json"
    path.write_text(json.dumps({"version": "fixture-v2", "groups": [{
        "action_id": "A_001", "policy": EXPERT_POLICY, "approval_source": "dataset_review",
        "approved": True, "expert_episode_ids": expert_ids,
    }]}))
    settings = Settings(dataset_root=root, experts_path=path, artifacts_dir=tmp_path / "artifacts")
    provenance = {
        "model": "fixture-qwen", "prompt_version": "fixed-prompt", "policy_version": "fixed-policy",
        "strategy": "keyframes", "correct_threshold": 0.95, "candidate_interval_s": 2.0,
        "expert_interval_s": 1.0, "candidate_wrist_interval_s": 2.0,
        "expert_wrist_interval_s": 2.0, "camera_topic": "/head", "max_image_size": 768,
        "max_frames_per_video": 96, "max_request_images": 250,
        "expert_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "expert_ids": expert_ids,
    }
    return settings, target_ids, expert_ids, provenance


def result_body(body, provenance, verdict="correct", errors=()):
    return ReviewResult(
        episode_id=body["episode_id"], action_id="A_001", task_code="task-1", verdict=verdict,
        reason="fixture result", error_types=list(errors), ground_truth_candidate=verdict == "correct",
        provenance={**provenance, "camera_mode": body["camera_mode"], "strategy": body["strategy"]},
    ).model_dump(mode="json")


def test_paired_calls_keep_gt_local_and_use_single_primary_label(experiment, tmp_path):
    settings, ids, _, provenance = experiment
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert set(body) == {"episode_id", "strategy", "camera_mode"}
        assert request.url.path == "/v1/reviews"
        verdict, errors = "correct", ()
        if body["episode_id"] == ids[0] and body["camera_mode"] == "main":
            verdict = "uncertain"
        elif body["episode_id"] == ids[1] and body["camera_mode"] == "main_wrist":
            verdict, errors = "incorrect", ("content_mismatch", "annotation_error")
        return httpx.Response(200, json=result_body(body, provenance, verdict, errors))

    target = tmp_path / "run"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        summary = run_compare(ids, target, settings=settings, client=client, strategy="keyframes")
        assert not client.is_closed
    assert [(r["episode_id"], r["camera_mode"]) for r in requests] == [
        (episode_id, mode) for episode_id in ids for mode in ("main", "main_wrist")
    ]
    assert all(r["strategy"] == "keyframes" for r in requests)
    assert summary["diagnostic_only"] is True
    assert summary["comparable_pairs"] == 2
    assert summary["metrics"]["main"]["false_accepts"] == 1
    assert summary["metrics"]["main"]["uncertain"] == 1
    assert summary["metrics"]["main_wrist"]["false_accepts"] == 0
    assert summary["metrics"]["main_wrist"]["exact_matches"] == 1
    assert summary["results"][1]["multiview_predict"] == "content_mismatch"
    assert json.loads((target / "summary.json").read_text()) == summary
    raw = json.loads((target / f"{ids[1]}.main_wrist.json").read_text())
    assert raw["response"]["error_types"] == ["content_mismatch", "annotation_error"]
    assert raw["http_status"] == 200 and raw["elapsed_s"] >= 0


@pytest.mark.parametrize("failure", ["http", "timeout", "invalid_json"])
def test_call_failures_are_recorded_without_semantic_predictions(experiment, tmp_path, failure):
    settings, ids, _, provenance = experiment

    def respond(request):
        body = json.loads(request.content)
        if body["camera_mode"] == "main":
            if failure == "http":
                return httpx.Response(503, json={"error": "ProviderError"})
            if failure == "timeout":
                raise httpx.ReadTimeout("fixture timeout", request=request)
            return httpx.Response(200, text="not json")
        return httpx.Response(200, json=result_body(body, provenance))

    target = tmp_path / "run"
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        summary = run_compare(ids[:1], target, settings=settings, client=client)
    row = summary["results"][0]
    assert row["main_predict"] is None and row["multiview_predict"] == "correct"
    assert summary["call_failures"] == {"main": 1, "main_wrist": 0}
    assert summary["comparable_pairs"] == 0
    assert summary["metrics"]["main_wrist"]["exact_match_fraction"] is None
    assert json.loads((target / f"{ids[0]}.main.json").read_text())["operational_error"] is not None


@pytest.mark.parametrize("field", ["model", "prompt_version", "expert_config_sha256", "expert_ids"])
def test_changed_settings_do_not_count_as_paired_evidence(experiment, tmp_path, field):
    settings, ids, _, provenance = experiment

    def respond(request):
        body = json.loads(request.content)
        response = result_body(body, provenance)
        if body["camera_mode"] == "main_wrist":
            response["provenance"][field] = [ids[0]] if field == "expert_ids" else "changed"
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        summary = run_compare(ids[:1], tmp_path / "run", settings=settings, client=client)
    assert summary["comparable_pairs"] == 0
    assert any(field in error for error in summary["results"][0]["comparison_errors"])
    assert summary["metrics"]["main"]["exact_match_fraction"] is None


@pytest.mark.parametrize("problem", ["duplicate", "expert", "existing_output"])
def test_preflight_blocks_leakage_and_overwrite_before_requests(experiment, tmp_path, problem):
    settings, ids, expert_ids, _ = experiment
    target = tmp_path / "run"
    if problem == "duplicate":
        ids = [ids[0], ids[0]]
    elif problem == "expert":
        ids = [expert_ids[0]]
    else:
        target.mkdir()
        (target / "keep.txt").write_text("preserve")

    def unexpected_request(request):
        pytest.fail("Preflight must finish before an API request")

    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client:
        with pytest.raises((ValueError, FileExistsError)):
            run_compare(ids, target, settings=settings, client=client)
    if problem == "existing_output":
        assert (target / "keep.txt").read_text() == "preserve"
    else:
        assert not target.exists()
