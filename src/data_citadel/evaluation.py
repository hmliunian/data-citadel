"""Offline labels and metrics, isolated from the review service's inputs."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .models import LABELS, ReviewResult


PREDICTED_LABELS = (*LABELS, "uncertain", "unclassified_error", "operational_error")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _prediction_labels(prediction: ReviewResult) -> set[str]:
    if prediction.verdict != "incorrect":
        return {prediction.verdict}
    return {
        "retry_then_success" if label == "repeated_retry" else label
        for label in prediction.error_types
    } or {"unclassified_error"}


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one record per attempted episode, including failed operations.

    Rates use successful predictions as their population; coverage and a second
    recall expose operational failures. Multilabel predictions contribute once
    to each predicted column, so a confusion row may exceed its support.
    """
    support = Counter()
    scored_support = Counter()
    operational_errors = Counter()
    confusion = {label: {column: 0 for column in PREDICTED_LABELS} for label in LABELS}
    ids: set[str] = set()
    true_positive = predicted_correct = false_accept = uncertain = exact = 0
    for record in records:
        episode_id, label = record["episode_id"], record["label"]
        if episode_id in ids or label not in LABELS:
            raise ValueError("Evaluation records contain a duplicate episode or unknown label")
        ids.add(episode_id)
        support[label] += 1
        if record.get("operational_error") is not None:
            if record.get("prediction") is not None:
                raise ValueError("A failed operation cannot also contain a prediction")
            operational_errors[record["operational_error"]["type"]] += 1
            confusion[label]["operational_error"] += 1
            continue
        prediction = ReviewResult.model_validate(record["prediction"])
        if prediction.episode_id != episode_id:
            raise ValueError("Prediction episode does not match its evaluation record")
        scored_support[label] += 1
        predicted_labels = _prediction_labels(prediction)
        for predicted_label in predicted_labels:
            confusion[label][predicted_label] += 1
        exact += predicted_labels == {label}
        if prediction.verdict == "correct":
            predicted_correct += 1
            true_positive += label == "correct"
            false_accept += label != "correct"
        uncertain += prediction.verdict == "uncertain"
    scored = sum(scored_support.values())
    scored_errors = scored - scored_support["correct"]
    return {
        "attempted_episodes": len(records),
        "scored_episodes": scored,
        "operational_error_count": sum(operational_errors.values()),
        "operational_errors_by_type": dict(sorted(operational_errors.items())),
        "support_by_label": {label: support[label] for label in LABELS},
        "scored_support_by_label": {label: scored_support[label] for label in LABELS},
        "metrics": {
            "correct_precision": _ratio(true_positive, predicted_correct),
            "correct_recall": _ratio(true_positive, scored_support["correct"]),
            "error_false_accept_rate": _ratio(false_accept, scored_errors),
            "uncertain_rate": _ratio(uncertain, scored),
            "prediction_coverage": _ratio(scored, len(records)),
            "decisive_coverage": _ratio(scored - uncertain, len(records)),
            "exact_label_accuracy": _ratio(exact, scored),
            "correct_recall_including_operational_errors": _ratio(true_positive, support["correct"]),
        },
        "metric_counts": {
            "true_correct_accepted": true_positive,
            "predicted_correct": predicted_correct,
            "human_errors_accepted": false_accept,
            "scored_human_correct": scored_support["correct"],
            "scored_human_errors": scored_errors,
            "uncertain": uncertain,
        },
        "confusion": confusion,
        "notes": [
            "Rates use scored episodes; operational errors are separate and reduce coverage.",
            "Confusion rows are human labels; columns are predicted labels (possibly multiple).",
            "repeated_retry maps to retry_then_success for category comparison only; neither "
            "the downloaded label nor that mapping verifies the eventual outcome.",
        ],
    }


def evaluate_manifest(
    service: Any,
    repository: Any,
    manifest_path: Path,
    output_dir: Path,
    *,
    strategy: str = "uniform",
    limit: int | None = None,
) -> dict[str, Any]:
    """Validate the whole split before calling review(id, strategy=...).

    The service receives only the ID and sampling strategy. Human labels,
    cohort membership and errors remain in this offline evaluator. Files use
    exclusive creation so an earlier experiment cannot be overwritten.
    """
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Evaluation limit must be a positive integer")
    if strategy not in ("uniform", "keyframes"):
        raise ValueError("Unknown sampling strategy")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest, dict)
            or not isinstance(manifest.get("version"), str)
            or not manifest["version"].strip()):
        raise ValueError("Evaluation manifest requires a nonempty version")
    expert_ids = manifest.get("expert_episode_ids")
    if (not isinstance(expert_ids, list)
            or any(not isinstance(eid, str) or not eid.strip() for eid in expert_ids)
            or len(expert_ids) != len(set(expert_ids))):
        raise ValueError("Evaluation manifest requires unique expert episode IDs")
    library = getattr(service, "experts", None)
    all_expert_ids = set(expert_ids).union(getattr(library, "episode_ids", ()))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Evaluation samples must be a list")
    seen: set[str] = set()
    resolved: list[dict[str, str]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Evaluation samples must be objects")
        episode_id = sample.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip() or episode_id in seen:
            raise ValueError("Evaluation samples require unique, nonempty episode IDs")
        seen.add(episode_id)
        if episode_id in all_expert_ids:
            raise ValueError("Expert/test overlap would leak an evaluation answer")
        if sample.get("label") not in LABELS:
            raise ValueError("Unknown human evaluation label")
        if sample.get("cohort") not in ("baseline", "exploratory"):
            raise ValueError("Every sample needs an explicit baseline or exploratory cohort")
        episode = repository.get(episode_id)
        if sample["label"] != episode.label:
            raise ValueError("Evaluation label differs from the indexed dataset label")
        if sample.get("action_id", episode.action_id) != episode.action_id:
            raise ValueError("Evaluation action differs from the indexed episode")
        resolved.append({
            "episode_id": episode_id,
            "action_id": episode.action_id,
            "label": sample["label"],
            "cohort": sample["cohort"],
        })
    output_dir = Path(output_dir)
    prediction_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    if prediction_path.exists() or summary_path.exists():
        raise FileExistsError("Use a new output directory for each evaluation run")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with prediction_path.open("x", encoding="utf-8") as output:
        for sample in resolved[:limit]:
            record: dict[str, Any] = dict(sample)
            try:
                prediction = ReviewResult.model_validate(
                    service.review(sample["episode_id"], strategy=strategy)
                )
                if (prediction.episode_id != sample["episode_id"]
                        or prediction.action_id != sample["action_id"]):
                    raise ValueError("Review response identifies a different episode")
                record["prediction"] = prediction.model_dump(mode="json")
                record["operational_error"] = None
            except Exception as exc:
                # Provider messages may contain credentials; persist only the type.
                record["prediction"] = None
                record["operational_error"] = {"type": type(exc).__name__}
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            records.append(record)
    summary = score_records(records)
    summary.update({
        "manifest_version": manifest["version"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "expert_version": getattr(library, "version", None),
        "expert_manifest_sha256": getattr(library, "manifest_sha256", None),
        "strategy": strategy,
        "manifest_samples": len(resolved),
        "manifest_coverage": _ratio(len(records), len(resolved)),
        "dataset_available_episodes": len(repository.list_episodes()),
        "cohorts": {
            cohort: score_records([r for r in records if r["cohort"] == cohort])
            for cohort in ("baseline", "exploratory")
        },
        "actions": {
            action: score_records([r for r in records if r["action_id"] == action])
            for action in sorted({sample["action_id"] for sample in resolved})
        },
        "preparation_coverage": manifest.get("coverage"),
    })
    with summary_path.open("x", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return summary
