"""Recompute benchmark scores and costs from saved provider receipts."""
import csv
import io
import json
import math
from pathlib import Path
import statistics

from citadel.application.experiments import counts
from citadel.infrastructure.files import read

RESULT_GLOB = "[0-9a-f]" * 32 + ".json"


def price_usage(usage, tiers):
    if not usage or "prompt_tokens" not in usage or "completion_tokens" not in usage:
        return {"list_cny": None, "cache_adjusted_cny": None, "cached_tokens": None}
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if min(prompt, completion, cached) < 0 or cached > prompt:
        raise ValueError("Invalid usage receipt")
    tier = next((row for row in tiers if prompt <= row[0]), None)
    if tier is None:
        raise ValueError("Usage exceeds published price tiers")
    _, input_rate, output_rate, cached_rate = tier
    ordinary = (prompt * input_rate + completion * output_rate) / 1_000_000
    adjusted = (None if cached and cached_rate is None else
                ((prompt - cached) * input_rate + cached * (cached_rate or 0) +
                 completion * output_rate) / 1_000_000)
    return {"list_cny": ordinary, "cache_adjusted_cny": adjusted, "cached_tokens": cached}


def percentile(values, fraction):
    if not values:
        return None
    data = sorted(values)
    point = (len(data) - 1) * fraction
    low, high = math.floor(point), math.ceil(point)
    return data[low] + (data[high] - data[low]) * (point - low)


def wilson(matched, total):
    z = 1.959963984540054
    proportion = matched / total
    center = (proportion + z * z / (2 * total)) / (1 + z * z / total)
    span = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / (1 + z * z / total)
    return [center - span, center + span]


def call_rows(work, model, price):
    rows = []
    for path in sorted((work / "models" / model / "calls").glob("*/request.json")):
        request = read(path)
        response_path, error_path = path.with_name("response.json"), path.with_name("error.json")
        response = read(response_path) if response_path.exists() else {}
        error = read(error_path) if error_path.exists() else {}
        body = response.get("body", {})
        usage, context = body.get("usage", {}), request.get("context", {})
        choice = (body.get("choices") or [{}])[0]
        rows.append({
            "model": model, "episode_id": context.get("episode_id"), "result_id": context.get("result_id"),
            "stage": context.get("stage", "task"), "attempt": request["attempt"], "call_id": path.parent.name,
            "started_at": request.get("created_at"), "http_status": response.get("status"),
            "error_type": error.get("type"), "returned_model": body.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "reasoning_content_present": bool((choice.get("message") or {}).get("reasoning_content")),
            "request_sha256": request["request_sha256"], "input_images": request["input_images"],
            "response_format": request["parameters"].get("response_format", {}).get("type", "text"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "elapsed_s": response.get("elapsed_s", error.get("elapsed_s")),
            **price_usage(usage, price["tiers"])})
    return sorted(rows, key=lambda row: (row["started_at"] or "", row["call_id"]))


def summarize(work):
    frozen = read(work / "experiment.json")
    manifest = frozen["manifest"]
    split = {episode_id: name for name, ids in manifest["splits"].items() for episode_id in ids}
    summaries, episodes, calls = [], [], []
    for model in frozen["plan"]["models"]:
        receipts = call_rows(work, model, frozen["pricing"]["models"][model])
        calls.extend(receipts)
        results = {path.stem: read(path) for path in (work / "models" / model / "results").glob(RESULT_GLOB)}
        rows, latencies = [], []
        for episode_id, source in manifest["episodes"].items():
            result = results.get(episode_id, {})
            attempts = [row for row in receipts if row["episode_id"] == episode_id]
            api_time = sum(row["elapsed_s"] or 0 for row in attempts)
            row = {"model": model, "episode_id": episode_id, "split": split.get(episode_id, "excluded"),
                   "gt": source["gt"], "gt_reason": source["gt_reason"], "task_code": source["task_code"],
                   "status": result.get("status", "not_run"), "label": result.get("label"),
                   "error_stage": result.get("error", {}).get("stage"), "result_id": result.get("result_id"),
                   "calls": len(attempts), "list_cny": sum(r["list_cny"] or 0 for r in attempts),
                   "unknown_cost_calls": sum(r["list_cny"] is None for r in attempts),
                   "api_elapsed_s": api_time, "wall_elapsed_s": result.get("elapsed_s"),
                   "pacing_s": result.get("pacing_s"), "issues": result.get("issues", [])}
            if result and result["status"] != "failed":
                latencies.append(api_time)
            rows.append(row)
        score = counts(rows)
        positives = [row for row in rows if row["gt"] == "correct"]
        negatives = [row for row in rows if row["gt"] == "incorrect"]
        total_cost = sum(row["list_cny"] or 0 for row in receipts)
        reasons = sorted({row["gt_reason"] or row["gt"] for row in rows})
        summaries.append({
            "model": model, **score, "match_rate": score["matched"] / len(rows),
            "match_rate_wilson95": wilson(score["matched"], len(rows)),
            "positive_pass": sum(row["label"] == "correct" and row["status"] == "completed" for row in positives),
            "positive_total": len(positives),
            "negative_reject": sum(row["label"] == "incorrect" and row["status"] == "completed" for row in negatives),
            "negative_total": len(negatives), "calls": len(receipts),
            "retries": sum(row["attempt"] > 1 for row in receipts),
            **{key: sum(row[key] or 0 for row in receipts)
               for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")},
            "unknown_usage_calls": sum(row["list_cny"] is None for row in receipts),
            "unknown_cache_price_calls": sum(row["list_cny"] is not None and row["cache_adjusted_cny"] is None for row in receipts),
            "list_cny": total_cost,
            "cache_adjusted_known_cny": sum(row["cache_adjusted_cny"] or 0 for row in receipts),
            "mean_cny_per_episode": total_cost / max(1, len(results)),
            "mean_cny_per_matched": total_cost / score["matched"] if score["matched"] else None,
            "valid_latency_n": len(latencies), "api_p50_s": statistics.median(latencies) if latencies else None,
            "api_p95_s": percentile(latencies, .95),
            "returned_models": sorted({row["returned_model"] for row in receipts if row["returned_model"]}),
            "reasoning_response_calls": sum(row["reasoning_content_present"] for row in receipts),
            "response_formats": sorted({row["response_format"] for row in receipts}),
            "by_reason": {reason: counts([row for row in rows if (row["gt_reason"] or row["gt"]) == reason]) for reason in reasons},
            "by_split": {name: counts([row for row in rows if row["split"] == name]) for name in manifest["splits"]},
            "by_stage": {stage: {"calls": len([row for row in receipts if row["stage"] == stage]),
                                 "list_cny": sum(row["list_cny"] or 0 for row in receipts if row["stage"] == stage),
                                 "total_tokens": sum(row["total_tokens"] or 0 for row in receipts if row["stage"] == stage)}
                         for stage in ("task", "quality")}})
        episodes.extend(rows)
    return {"experiment_sha256": frozen["sha256"], "models": summaries}, episodes, calls, frozen


def csv_text(rows):
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    return output.getvalue()


if __name__ == "__main__":
    summary, _, _, _ = summarize(Path("artifacts/benchmark_20260915"))
    print(json.dumps([{key: row[key] for key in
                     ("model", "matched", "failed", "not_run", "calls", "list_cny")} for row in summary["models"]]))
