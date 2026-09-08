"""Compare two camera modes through FastAPI on an explicitly selected diagnostic set."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from time import perf_counter

import httpx

from data_citadel.experts import ExpertLibrary
from data_citadel.models import CitadelError, LABELS, ReviewResult
from data_citadel.repository import EpisodeRepository
from data_citadel.runtime import write_json
from data_citadel.settings import Settings


MODES = ("main", "main_wrist")
FIXED_FIELDS = (
    "model", "prompt_version", "policy_version", "strategy", "correct_threshold",
    "candidate_interval_s", "expert_interval_s", "candidate_wrist_interval_s",
    "expert_wrist_interval_s", "camera_topic", "max_image_size",
    "max_frames_per_video", "max_request_images", "expert_config_sha256", "expert_ids",
)


def primary_label(result: ReviewResult) -> str:
    if result.verdict == "incorrect" and result.error_types:
        return result.error_types[0]
    return result.verdict


def _pair_errors(results, reference_ids, expert_hash):
    if any(mode not in results for mode in MODES):
        return ["One or both API calls failed"]
    left, right = (results[mode].provenance for mode in MODES)
    errors = []
    for field in FIXED_FIELDS:
        if field not in left or field not in right:
            errors.append(f"Missing provenance: {field}")
        elif left[field] != right[field]:
            errors.append(f"Different provenance: {field}")
    for mode in MODES:
        provenance = results[mode].provenance
        if provenance.get("expert_ids") != reference_ids:
            errors.append(f"{mode}: expert IDs differ from locally validated references")
        if provenance.get("expert_config_sha256") != expert_hash:
            errors.append(f"{mode}: expert configuration differs from the local manifest")
    return errors


def _metrics(rows, prediction_key):
    total = len(rows)
    errors = sum(row["gt"] != "correct" for row in rows)
    matches = sum(row[prediction_key] == row["gt"] for row in rows)
    false_accepts = sum(row["gt"] != "correct" and row[prediction_key] == "correct" for row in rows)
    uncertain = sum(row[prediction_key] == "uncertain" for row in rows)
    return {
        "comparable_episodes": total,
        "exact_matches": matches,
        "exact_match_fraction": matches / total if total else None,
        "gt_error_episodes": errors,
        "false_accepts": false_accepts,
        "false_accept_fraction": false_accepts / errors if errors else None,
        "uncertain": uncertain,
        "uncertain_fraction": uncertain / total if total else None,
    }


def run_compare(
    episode_ids,
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8001",
    strategy: str = "uniform",
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Write raw calls and compare only pairs with matching experimental settings.

    GT stays local. An injected client remains owned by its caller. No request
    is retried: each explicitly selected episode receives exactly two calls.
    """
    ids = list(episode_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Provide at least one unique episode ID; duplicates are not allowed")
    if strategy not in ("uniform", "keyframes"):
        raise ValueError("Unknown sampling strategy")
    settings = settings or Settings()
    repository = EpisodeRepository(settings.dataset_root)
    experts = ExpertLibrary(settings.experts_path, repository)
    episodes, references = {}, {}
    for episode_id in ids:
        episode = repository.get(episode_id)
        if episode.label not in LABELS:
            raise ValueError(f"No recognized source GT for episode {episode_id}")
        if episode_id in experts.episode_ids:
            raise ValueError("Expert episodes cannot be used as diagnostic test samples")
        episodes[episode_id] = episode
        references[episode_id] = [expert.episode_id for expert in experts.resolve(episode)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    failures = dict.fromkeys(MODES, 0)
    endpoint = base_url.rstrip("/") + "/v1/reviews"
    with (nullcontext(client) if client is not None else httpx.Client(timeout=900.0)) as api:
        for episode_id in ids:
            results, elapsed, artifacts = {}, {}, {}
            for mode in MODES:
                body = {"episode_id": episode_id, "strategy": strategy, "camera_mode": mode}
                record = {"request": body, "http_status": None, "response": None,
                          "operational_error": None}
                started = perf_counter()
                try:
                    response = api.post(endpoint, json=body)
                    record["http_status"] = response.status_code
                    try:
                        record["response"] = response.json()
                    except ValueError:
                        record["response"] = response.text
                    response.raise_for_status()
                    result = ReviewResult.model_validate(record["response"])
                    if (result.episode_id != episode_id
                            or result.action_id != episodes[episode_id].action_id):
                        raise ValueError("API response identifies a different episode")
                    if (result.provenance.get("camera_mode") != mode
                            or result.provenance.get("strategy") != strategy):
                        raise ValueError("API response does not confirm the requested camera mode/strategy")
                    results[mode] = result
                except Exception as error:
                    record["operational_error"] = {"type": type(error).__name__}
                    failures[mode] += 1
                elapsed[mode] = record["elapsed_s"] = perf_counter() - started
                artifacts[mode] = f"{episode_id}.{mode}.json"
                write_json(output_dir / artifacts[mode], record)
            issues = _pair_errors(results, references[episode_id], experts.manifest_sha256)
            rows.append({
                "episode_id": episode_id,
                "action_id": episodes[episode_id].action_id,
                "gt": episodes[episode_id].label,
                "main_predict": primary_label(results["main"]) if "main" in results else None,
                "multiview_predict": primary_label(results["main_wrist"]) if "main_wrist" in results else None,
                "paired_comparable": not issues,
                "comparison_errors": issues,
                "elapsed_s": elapsed,
                "artifacts": artifacts,
            })
    comparable = [row for row in rows if row["paired_comparable"]]
    summary = {
        "diagnostic_only": True,
        "note": "Explicitly selected diagnostic cases; results do not estimate population accuracy.",
        "strategy": strategy,
        "expert_manifest_sha256": experts.manifest_sha256,
        "requested_episodes": len(ids),
        "comparable_pairs": len(comparable),
        "incomparable_pairs": len(rows) - len(comparable),
        "call_failures": failures,
        "metrics": {
            "main": _metrics(comparable, "main_predict"),
            "main_wrist": _metrics(comparable, "multiview_predict"),
        },
        "results": rows,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", nargs="+", required=True)
    parser.add_argument("--strategy", choices=("uniform", "keyframes"), default="uniform")
    args = parser.parse_args(argv)
    try:
        summary = run_compare(args.episode_ids, args.output_dir,
                              base_url=args.base_url, strategy=args.strategy)
    except (CitadelError, OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary["incomparable_pairs"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
