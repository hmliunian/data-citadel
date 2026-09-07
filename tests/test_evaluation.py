import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from data_citadel.evaluation import evaluate_manifest, score_records
from data_citadel.models import ReviewResult


def record(episode_id, label, verdict="uncertain", error_types=()):
    return {
        "episode_id": episode_id, "action_id": "A_001", "label": label, "cohort": "baseline",
        "prediction": ReviewResult(
            episode_id=episode_id, action_id="A_001", task_code="TASK-CUP", verdict=verdict,
            error_types=list(error_types), reason="test prediction",
        ).model_dump(),
        "operational_error": None,
    }


def test_metrics_expose_false_accepts_uncertainty_and_operational_failures():
    rows = [
        record("true-positive", "correct", "correct"),
        record("false-accept", "blurred", "correct"),
        record("uncertain", "correct"),
        record("retry", "retry_then_success", "incorrect", ["repeated_retry"]),
        {"episode_id": "failed", "label": "correct", "prediction": None,
         "operational_error": {"type": "ProviderError"}},
    ]
    summary = score_records(rows)
    assert summary["metrics"]["correct_precision"] == 0.5
    assert summary["metrics"]["correct_recall"] == 0.5
    assert summary["metrics"]["error_false_accept_rate"] == 0.5
    assert summary["metrics"]["uncertain_rate"] == 0.25
    assert summary["metrics"]["prediction_coverage"] == 0.8
    assert summary["metrics"]["correct_recall_including_operational_errors"] == pytest.approx(1 / 3)
    assert summary["support_by_label"]["correct"] == 3
    assert summary["scored_support_by_label"]["correct"] == 2
    assert summary["operational_error_count"] == 1
    assert summary["confusion"]["retry_then_success"]["retry_then_success"] == 1
    assert summary["confusion"]["correct"]["operational_error"] == 1


def test_no_correct_predictions_is_undefined_precision_and_multilabel_is_explicit():
    rows = [record("multi", "blurred", "incorrect", ["blurred", "content_mismatch"])]
    summary = score_records(rows)
    assert summary["metrics"]["correct_precision"] is None
    assert summary["metrics"]["correct_recall"] is None
    assert summary["confusion"]["blurred"]["blurred"] == 1
    assert summary["confusion"]["blurred"]["content_mismatch"] == 1
    assert summary["metrics"]["exact_label_accuracy"] == 0
    assert all(value is None for value in score_records([])["metrics"].values())


@pytest.fixture
def evaluation_setup(tmp_path):
    rows = [record("test-1", "correct", "correct"), record("test-2", "other")]
    episodes = {
        row["episode_id"]: SimpleNamespace(action_id="A_001", label=row["label"])
        for row in rows
    }
    repository = Mock(get=Mock(side_effect=episodes.__getitem__),
                      list_episodes=Mock(return_value=list(episodes.values())))
    predictions = {row["episode_id"]: row["prediction"] for row in rows}
    service = SimpleNamespace(
        experts=SimpleNamespace(episode_ids={"expert-1"}, version="experts-v1"),
        review=Mock(side_effect=lambda episode_id, *, strategy: predictions[episode_id]),
    )
    manifest = {
        "version": "evaluation-v1", "expert_episode_ids": ["expert-1"],
        "samples": [{key: value for key, value in row.items()
                     if key not in ("prediction", "operational_error")} for row in rows],
    }
    return tmp_path / "manifest.json", tmp_path / "run", repository, service, manifest


def run_evaluation(setup, **kwargs):
    manifest_path, output_dir, repository, service, manifest = setup
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return evaluate_manifest(service, repository, manifest_path, output_dir, **kwargs)


def test_labels_remain_offline_and_results_include_run_and_cohort_coverage(evaluation_setup):
    summary = run_evaluation(evaluation_setup, strategy="keyframes", limit=1)
    service = evaluation_setup[3]
    service.review.assert_called_once_with("test-1", strategy="keyframes")
    assert summary["manifest_samples"] == 2
    assert summary["manifest_coverage"] == 0.5
    assert summary["dataset_available_episodes"] == 2
    assert summary["cohorts"]["baseline"]["scored_episodes"] == 1
    persisted = json.loads((evaluation_setup[1] / "summary.json").read_text())
    assert persisted == summary
    assert len((evaluation_setup[1] / "predictions.jsonl").read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        run_evaluation(evaluation_setup)


@pytest.mark.parametrize("problem", ["duplicate", "declared_overlap", "actual_overlap", "label"])
def test_invalid_split_rejected_before_any_service_call_even_beyond_limit(evaluation_setup, problem):
    manifest, service = evaluation_setup[4], evaluation_setup[3]
    if problem == "duplicate":
        manifest["samples"].append(dict(manifest["samples"][0]))
    elif problem == "declared_overlap":
        manifest["expert_episode_ids"].append("test-2")
    elif problem == "actual_overlap":
        service.experts.episode_ids.add("test-2")
    else:
        manifest["samples"][1]["label"] = "blurred"
    with pytest.raises(ValueError):
        run_evaluation(evaluation_setup, limit=1)
    service.review.assert_not_called()
    assert not evaluation_setup[1].exists()


def test_provider_failure_is_not_a_negative_video_label_and_does_not_leak_secrets(evaluation_setup):
    evaluation_setup[3].review.side_effect = RuntimeError("secret-api-key")
    summary = run_evaluation(evaluation_setup)
    assert summary["operational_error_count"] == 2
    assert summary["scored_episodes"] == 0
    assert summary["metrics"]["correct_precision"] is None
    assert summary["metrics"]["prediction_coverage"] == 0
    assert "secret-api-key" not in (evaluation_setup[1] / "predictions.jsonl").read_text()


def test_duplicate_scoring_records_rejected():
    row = record("same-episode", "correct")
    with pytest.raises(ValueError, match="duplicate"):
        score_records([row, row])


def test_unrecognized_sampling_strategy_is_rejected_before_review(evaluation_setup):
    with pytest.raises(ValueError, match="sampling strategy"):
        run_evaluation(evaluation_setup, strategy="gripper")
    evaluation_setup[3].review.assert_not_called()
