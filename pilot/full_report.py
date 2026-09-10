"""Population denominators and all-attempt usage for a full-dataset run."""
import csv
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .data import read_json, write_json

LABELS = {"correct", "incorrect"}


def rate(count, total):
    return {"count": count, "total": total, "rate": count / total if total else None}


def metrics(rows):
    states = Counter(row["status"] for row in rows)
    positive = [row for row in rows if row["gt"] == "correct"]
    negative = [row for row in rows if row["gt"] == "incorrect"]
    valid = positive + negative
    predicted = [row for row in valid if row["label"] in LABELS]
    matched = sum(row["label"] == row["gt"] for row in valid)
    attempted = len(rows) - states["pending"]
    return {
        "total": len(rows), "attempted": attempted, "pending": states["pending"],
        "completed": states["completed"], "needs_review": states["needs_review"],
        "failed": states["failed"], "partial": bool(states["pending"]),
        "gt_correct": len(positive), "gt_incorrect": len(negative),
        "gt_valid": len(valid), "gt_unknown": len(rows) - len(valid),
        "gt_agreement": rate(matched, len(valid)),
        "prediction_accuracy": rate(matched, len(predicted)),
        "positive_pass_rate": rate(sum(row["label"] == "correct" for row in positive), len(positive)),
        "predicted_correct_rate": rate(sum(row["label"] == "correct" for row in rows), len(rows)),
        "false_accept": rate(sum(row["label"] == "correct" for row in negative), len(negative)),
        "false_reject": rate(sum(row["label"] == "incorrect" for row in positive), len(positive)),
        "coverage": rate(states["completed"], len(rows)),
        "processing_success": rate(states["completed"] + states["needs_review"], attempted),
    }


def archived_results(run_dir):
    rows, unreadable = [], []
    for path in sorted((run_dir / "results").glob("*/*.json")):
        try:
            row = read_json(path)
            if not isinstance(row, dict):
                raise ValueError("Expected a result object")
        except (OSError, ValueError):
            unreadable.append(str(path.relative_to(run_dir)))
            continue
        rows.append({**row, "result_path": str(path.relative_to(run_dir))})
    return sorted(rows, key=lambda row: (row.get("created_at", ""), row["result_path"])), unreadable


def archived_calls(run_dir, results):
    roles = {}
    for row in results:
        if row.get("review_call"):
            roles[row["review_call"]] = (row.get("route", "unknown"), "review")
        for path in row.get("extra_calls", []):
            roles[path] = ("B", "expert_caption")
    calls = []
    for path in sorted((run_dir / "calls").glob("*/request.json")):
        archives = {}
        for name in ("request", "response", "error"):
            try:
                archives[name] = read_json(path.with_name(name + ".json"))
            except (FileNotFoundError, json.JSONDecodeError):
                archives[name] = {}
        request, response, error = (archives[name] for name in ("request", "response", "error"))
        call_path = str(path.parent.relative_to(run_dir))
        context = request.get("context", {})
        purpose = context.get("purpose", "unknown")
        route = "B" if purpose == "expert_caption" else context.get("route", "unknown")
        route, purpose = roles.get(call_path, (route, purpose))
        if purpose not in ("review", "expert_caption"):
            purpose = "unknown"
        calls.append({
            "call_path": call_path, "context": context, "route": route, "purpose": purpose,
            "task_code": context.get("task_code"), "episode_id": context.get("episode_id"),
            "model": request.get("model"), "base_url": request.get("base_url", ""),
            "attempt": request.get("attempt"), "input_images": request.get("input_images"),
            "http_status": response.get("http_status"), "error": error.get("type"),
            "elapsed_s": response.get("elapsed_s", error.get("elapsed_s")),
            "usage": response.get("body", {}).get("usage") or {},
        })
    return calls


def billing(calls):
    prompt = sum(call["usage"].get("prompt_tokens", 0) for call in calls)
    completion = sum(call["usage"].get("completion_tokens", 0) for call in calls)
    standard = all(call["model"] == "qwen-vl-max"
                   and urlparse(call["base_url"]).hostname == "dashscope.aliyuncs.com" for call in calls)
    return {
        "http_attempts": len(calls),
        "http_responses": sum(call["http_status"] is not None for call in calls),
        "http_errors": sum((call["http_status"] or 0) >= 400 for call in calls),
        "transport_errors": sum(call["error"] is not None for call in calls),
        "incomplete": sum(call["http_status"] is None and call["error"] is None for call in calls),
        "usage_missing": sum(not call["usage"] for call in calls),
        "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": sum(call["usage"].get("total_tokens", call["usage"].get("prompt_tokens", 0)
                                               + call["usage"].get("completion_tokens", 0)) for call in calls),
        "elapsed_s": sum(call["elapsed_s"] or 0 for call in calls),
        "purpose_counts": dict(Counter(call["purpose"] for call in calls)),
        "estimated_cny_before_discounts": (prompt * 1.6 + completion * 4) / 1e6 if standard else None,
    }


def report(run_dir: Path) -> tuple[Path, dict]:
    manifest = read_json(run_dir / "manifest.json")
    tasks, episodes, routes = manifest["tasks"], manifest["episodes"], manifest.get("routes", ["A", "B"])
    candidates, pool = {}, set()
    exclusions = [dict(row) for row in manifest.get("exclusions", [])]
    excluded_ids = {row.get("episode_id") for row in exclusions if row.get("episode_id")}
    for task_code, task in tasks.items():
        for episode_id in task["candidate_ids"]:
            if episode_id in candidates or episode_id not in episodes:
                raise ValueError("Duplicate candidate or missing candidate metadata")
            candidates[episode_id] = task_code
        for episode_id in dict.fromkeys(task["expert_pool"] + task["experts"]):
            pool.add(episode_id)
            if episode_id not in excluded_ids:
                exclusions.append({"episode_id": episode_id, "task_code": task_code,
                                   "reason": "expert_pool", "selected_expert": episode_id in task["experts"]})
                excluded_ids.add(episode_id)
    if set(candidates) & (pool | excluded_ids):
        raise ValueError("Excluded episodes cannot enter the candidate denominator")
    if not routes or len(set(routes)) != len(routes) or set(routes) - {"A", "B"}:
        raise ValueError("Expected unique A/B routes")
    results, unreadable = archived_results(run_dir)
    latest, ignored = {}, []
    for row in results:
        key = row.get("route"), row.get("episode_id")
        if key[0] not in routes or candidates.get(key[1]) != row.get("task_code"):
            ignored.append(row["result_path"])
            continue
        if (row.get("status") not in ("completed", "needs_review", "failed")
                or (row["status"] == "completed") != (row.get("label") in LABELS)):
            raise ValueError("Result status and prediction label disagree")
        latest[key] = row
    rows = []
    for route in routes:
        for episode_id, task_code in candidates.items():
            metadata = episodes[episode_id]
            row = {"status": "pending", "label": None, "reason": "尚未生成审核结果。",
                   **latest.get((route, episode_id), {})}
            gt = metadata.get("gt")
            row.update(episode_id=episode_id, task_code=task_code, route=route,
                       gt=gt if gt in LABELS else {"Accepted": "correct", "Denied": "incorrect"}.get(metadata.get("gt_status")),
                       gt_status=metadata.get("gt_status"), gt_reason=metadata.get("gt_reason"))
            rows.append(row)
    calls = archived_calls(run_dir, results)

    def group(route, task_code=None):
        selected = [row for row in rows if row["route"] == route
                    and (task_code is None or row["task_code"] == task_code)]
        selected_calls = [call for call in calls if call["route"] == route
                          and (task_code is None or call["task_code"] == task_code)]
        return {**metrics(selected), "billing": billing(selected_calls)}

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_inventory": manifest.get("source_inventory", {}),
        "partial": any(row["status"] == "pending" for row in rows) or bool(unreadable),
        "candidate_episodes": len(candidates), "expected_predictions": len(rows), "task_count": len(tasks),
        "routes": {route: {"overall": group(route),
                           "tasks": {task_code: group(route, task_code) for task_code in tasks}} for route in routes},
        "exclusions": {"entries": len(exclusions), "episode_count": len(excluded_ids),
                       "expert_pool_episodes": len(pool),
                       "selected_experts": sum(len(task["experts"]) for task in tasks.values()),
                       "by_reason": dict(Counter(str(row.get("reason", "unspecified")) for row in exclusions)),
                       "by_task": dict(Counter(row.get("task_code", "unknown") for row in exclusions))},
        "archived_result_files": len(results), "superseded_results": len(results) - len(ignored) - len(latest),
        "ignored_result_files": ignored, "unreadable_result_files": unreadable,
        "billing": billing(calls), "calls": calls,
        "unassigned_call_paths": [call["call_path"] for call in calls if call["route"] not in routes],
        "definitions": {
            "gt_agreement": "Matching labels / all candidates with valid GT; pending, review, and failure do not match.",
            "prediction_accuracy": "Matching labels / candidates with both a prediction and valid GT.",
            "positive_pass_rate": "GT-correct candidates predicted correct / all GT-correct candidates.",
            "predicted_correct_rate": "Predicted correct / all candidates; this is not GT agreement.",
            "processing_success": "(Completed predictions + needs_review) / attempted candidates.",
            "billing": "All archived HTTP attempts, including superseded/error/caption calls. Missing usage is unreported, not proof of zero cost. Estimate uses the existing pilot/model.py schedule (CNY 1.6/4 per million prompt/completion tokens), not an invoice.",
        },
    }
    folder = run_dir / "reports" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "summary.json", summary)
    fields = ("episode_id", "task_code", "route", "gt", "gt_status", "gt_reason", "status", "label",
              "reason", "created_at", "model_elapsed_s", "elapsed_s", "review_call", "extra_calls", "result_path")
    with (folder / "predictions.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)
    for name, entries in (("predictions", rows), ("exclusions", exclusions)):
        with (folder / (name + ".jsonl")).open("x", encoding="utf-8") as stream:
            for entry in entries:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return folder, summary
