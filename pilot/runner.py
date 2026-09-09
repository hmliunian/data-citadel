"""One persisted review path, with a development/holdout freeze and counted metrics."""
from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import route_a, route_b
from .data import digest, load_manifest, read_json, sha256, write_json
from .media import prepare_media
from .model import QwenClient, decide

CODE_FILES = ("data.py", "flatbuffer.py", "media.py", "model.py", "route_a.py", "route_b.py",
              "runner.py")


def configuration(run_dir: Path, client) -> dict:
    manifest = load_manifest(run_dir)
    captions = {str(p.relative_to(run_dir)): sha256(p)
                for p in sorted((run_dir / "references" / "route_b").glob("*.json"))}
    return {"manifest_sha256": manifest["snapshot_sha256"],
            "model": client.model, "base_url": client.base_url,
            "code": {name: sha256(Path(__file__).parent / name) for name in CODE_FILES},
            "expert_captions": captions, "expert_ids": manifest["splits"]["experts"],
            "sampling": manifest["sampling"]}


def latest_results(run_dir: Path) -> list[dict]:
    latest = {}
    rows = [read_json(p) for p in (run_dir / "results").glob("*/*.json")]
    for row in sorted(rows, key=lambda r: r["created_at"]):
        latest[row["split"], row["route"], row["episode_id"]] = row
    return list(latest.values())


def check_freeze(run_dir: Path, client):
    saved = read_json(run_dir / "freeze.json")
    if saved["configuration"] != configuration(run_dir, client):
        raise ValueError("Frozen code/model/experts changed; do not reuse the holdout")
    return saved


def freeze(run_dir: Path):
    manifest = load_manifest(run_dir)
    results = latest_results(run_dir)
    for route in ("A", "B"):
        rows = [r for r in results if r["split"] == "development" and r["route"] == route]
        if ({r["episode_id"] for r in rows} != set(manifest["splits"]["development"])
                or any(r["status"] == "failed" for r in rows)):
            raise ValueError("Both development routes must finish before holdout freeze")
    config = configuration(run_dir, QwenClient(run_dir))
    if len(config["expert_captions"]) != len(config["expert_ids"]):
        raise ValueError("Expected one frozen description per expert")
    value = {"created_at": datetime.now(timezone.utc).isoformat(), "configuration": config,
             "configuration_sha256": digest(config),
             "criterion": "Descriptive pilot only; no production performance acceptance threshold.",
             "development_results": {str(p.relative_to(run_dir)): sha256(p)
                                     for p in sorted((run_dir / "results" / "development").glob("*.json"))}}
    write_json(run_dir / "freeze.json", value)
    return value


def review_episode(run_dir: Path, episode_id: str, route: str, *, client=None) -> dict:
    manifest = load_manifest(run_dir)
    if route not in ("A", "B"):
        raise ValueError("Route must be A or B")
    split = next((s for s in ("development", "holdout")
                  if episode_id in manifest["splits"][s]), None)
    if split is None:
        raise ValueError("Only selected non-expert pilot episodes may be reviewed")
    if split == "development" and (run_dir / "freeze.json").exists():
        raise ValueError("Development is closed after freeze")
    client = client or QwenClient(run_dir, context={"episode_id": episode_id,
                                                   "route": route, "split": split})
    if split == "holdout":
        check_freeze(run_dir, client)
    episode = manifest["episodes"][episode_id]
    expert_ids = manifest["splits"]["experts"]
    experts = [manifest["episodes"][i] for i in expert_ids]
    if (episode_id in manifest["splits"]["expert_pool"] or not experts
            or any(e["task_code"] != episode["task_code"] or e["quality"] != "high"
                   or e["gt_status"] != "Accepted" or not e.get("reviewer")
                   or not e.get("review_time") for e in experts)):
        raise ValueError("Invalid same-task, independently reviewed expert set")
    # Source tags/receipts are local provenance only; neither route receives these records.
    dataset = Path(manifest["dataset"])
    for item in [episode, *experts]:
        tags = dataset / "tasks" / item["task_code"] / "api_tags" / (item["episode_id"] + ".json")
        receipt = dataset / "receipts" / (item["episode_id"] + ".json")
        if sha256(tags) != item["tags_sha256"] or sha256(receipt) != item["receipt_sha256"]:
            raise ValueError("Source review metadata changed after selection")
    started = time.monotonic()
    result = {"episode_id": episode_id, "task_code": episode["task_code"],
              "instruction": manifest["instruction"], "split": split, "route": route,
              "created_at": datetime.now(timezone.utc).isoformat(), "expert_ids": expert_ids,
              "configuration": configuration(run_dir, client)}
    try:
        media = prepare_media(run_dir, episode, manifest["sampling"])
        expert_media = [prepare_media(run_dir, e, manifest["sampling"]) for e in experts]
        if not media["frames"] or any(not e["frames"] for e in expert_media):
            result.update(status="needs_review", label=None, reason="缺少可用的候选或专家画面。",
                          checks={}, evidence=[], warnings=media["warnings"])
        else:
            response = {"A": route_a, "B": route_b}[route].review(
                client, run_dir=run_dir, instruction=manifest["instruction"],
                experts=expert_media, candidate=media)
            reply = response["reply"]
            result.update(decide(reply["data"], media))
            result.update(reference=response["reference"], usage=reply["usage"],
                          model=reply["model"], model_elapsed_s=reply["elapsed_s"],
                          review_call=reply["call_path"],
                          extra_calls=[r["call_path"] for r in response["extra_calls"]],
                          estimated_cny_before_discounts=reply["estimated_cny_before_discounts"],
                          request_id=reply["request_id"], request_sha256=reply["request_sha256"])
    except Exception as exc:
        # An execution error is never an incorrect task label. Raw responses stay in calls/.
        result.update(status="failed", label=None, reason=f"{type(exc).__name__}: {exc}",
                      checks={}, evidence=[], warnings=[])
    result.update(gt=episode["gt"], gt_status=episode["gt_status"], gt_reason=episode["gt_reason"],
                  elapsed_s=time.monotonic() - started)
    write_json(run_dir / "results" / split / f"{route}-{episode_id}-{uuid.uuid4().hex[:8]}.json", result)
    return result


def run_split(run_dir: Path, split: str, route: str, *, retry_failed: bool = False):
    if split not in ("development", "holdout"):
        raise ValueError("Unknown split")
    existing = {(r["split"], r["route"], r["episode_id"]): r for r in latest_results(run_dir)}
    for episode_id in load_manifest(run_dir)["splits"][split]:
        previous = existing.get((split, route, episode_id))
        if previous and (previous["status"] != "failed" or not retry_failed):
            yield previous
        else:
            result = review_episode(run_dir, episode_id, route)
            yield result
            if result["status"] == "failed":
                return


def metrics(rows: list[dict]) -> dict:
    def rate(count, total):
        return {"count": count, "total": total, "rate": count / total if total else None}
    positive = [r for r in rows if r["gt"] == "correct"]
    negative = [r for r in rows if r["gt"] == "incorrect"]
    return {"requests": len(rows), "gt_correct": len(positive), "gt_incorrect": len(negative),
            "false_accept": rate(sum(r["label"] == "correct" for r in negative), len(negative)),
            "false_reject": rate(sum(r["label"] == "incorrect" for r in positive), len(positive)),
            "coverage": rate(sum(r["label"] in ("correct", "incorrect") for r in rows), len(rows)),
            "needs_review": sum(r["status"] == "needs_review" for r in rows),
            "failed": sum(r["status"] == "failed" for r in rows),
            "mean_review_latency_s": (sum(r.get("model_elapsed_s", 0) for r in rows) / len(rows)
                                      if rows else None)}


def report(run_dir: Path):
    rows = latest_results(run_dir)
    calls = []
    for path in sorted((run_dir / "calls").glob("*/request.json")):
        request = read_json(path)
        response_path = path.with_name("response.json")
        response = read_json(response_path) if response_path.exists() else {}
        calls.append({"call_path": str(path.parent.relative_to(run_dir)),
                      "context": request.get("context", {}), "model": request["model"],
                      "base_url": request["base_url"], "attempt": request["attempt"],
                      "input_images": request["input_images"],
                      "http_status": response.get("http_status"),
                      "elapsed_s": response.get("elapsed_s"),
                      "usage": response.get("body", {}).get("usage", {})})
    prompt = sum(c["usage"].get("prompt_tokens", 0) for c in calls)
    completion = sum(c["usage"].get("completion_tokens", 0) for c in calls)
    standard_price = all(c["model"] == "qwen-vl-max"
                         and c["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
                         for c in calls)
    summary = {"task_code": load_manifest(run_dir)["task_code"],
               "groups": {f"{s}/{route}": metrics([r for r in rows if r["split"] == s and r["route"] == route])
                          for s in ("development", "holdout") for route in ("A", "B")},
               "billing": {"http_attempts": len(calls), "prompt_tokens": prompt,
                           "completion_tokens": completion,
                           "estimated_cny_before_discounts": ((prompt * 1.6 + completion * 4) / 1e6
                                                               if standard_price else None),
                           "source": "https://help.aliyun.com/zh/model-studio/model-pricing",
                           "note": "Usage from every archived HTTP response, including failed/truncated attempts; estimate is not an invoice."},
               "limitations": ["One task; one negative in each split; no population performance claim.",
                               "Ground truth is the original platform review, not a new adjudication.",
                               "Two-second sampling may omit short events; hand swaps/retries need targeted examples."],
               "calls": calls}
    folder = run_dir / "reports" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                                   + "-" + uuid.uuid4().hex[:6])
    write_json(folder / "summary.json", summary)
    fields = ("episode_id", "task_code", "split", "route", "gt", "label", "status", "reason",
              "model_elapsed_s", "review_call")
    with (folder / "predictions.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (folder / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return folder, summary
