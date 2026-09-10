"""Frozen full-dataset replay using the unchanged A/B policy."""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import shutil
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import route_a, route_b
from .data import DEFAULT_DATASET, digest, load_manifest, read_json, sha256, write_json
from .media import prepare_media
from .model import ModelCallError, QwenClient, decide

FROZEN_FILES = ("data.py", "flatbuffer.py", "media.py", "model.py", "route_a.py", "route_b.py",
                "full.py", "full_data.py")


class Throttle:
    def __init__(self, requests_per_minute):
        self.interval = 60 / requests_per_minute
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self):
        with self.lock:
            delay = max(0, self.next_start - time.monotonic())
            self.next_start = max(time.monotonic(), self.next_start) + self.interval
        if delay:
            time.sleep(delay)


class BatchClient(QwenClient):
    def __init__(self, *args, throttle, **kwargs):
        super().__init__(*args, **kwargs)
        self.throttle = throttle

    def complete(self, messages, **kwargs):
        self.context["purpose"] = ("expert_caption" if route_b.CAPTION_RULES in messages[0]["content"]
                                   else "review")
        self.throttle.wait()
        return super().complete(messages, **kwargs)


def current_configuration(run_dir):
    client = QwenClient(run_dir)
    return {"manifest_sha256": load_manifest(run_dir)["snapshot_sha256"],
            "model": client.model, "base_url": client.base_url,
            "code": {name: sha256(Path(__file__).parent / name) for name in FROZEN_FILES}}


def latest_results(run_dir):
    latest = {}
    for path in (run_dir / "results").glob("*/*.json"):
        row = read_json(path)
        key = row["route"], row["episode_id"]
        if key not in latest or row["created_at"] > latest[key]["created_at"]:
            latest[key] = row
    return latest


class FullRun:
    def __init__(self, run_dir, *, media_workers=4, requests_per_minute=45, client_factory=None):
        self.root = run_dir.resolve()
        self.manifest = load_manifest(self.root)
        if not set(self.manifest.get("routes", ["A", "B"])) <= {"A", "B"}:
            raise ValueError("Unsupported route")
        self.config = current_configuration(self.root)
        frozen_path = self.root / "full-freeze.json"
        if frozen_path.exists():
            if read_json(frozen_path)["configuration"] != self.config:
                raise ValueError("Full-run configuration changed; use a new run directory")
        else:
            write_json(frozen_path, {"created_at": datetime.now(timezone.utc).isoformat(),
                                    "configuration": self.config,
                                    "configuration_sha256": digest(self.config)})
        self.media_guard = threading.Semaphore(media_workers)
        self.throttle = Throttle(requests_per_minute)
        self.client_factory = client_factory or BatchClient
        self.task_locks = {code: threading.Lock() for code in self.manifest["tasks"]}
        self.caption_locks = {code: threading.Lock() for code in self.manifest["tasks"]}
        self.experts = {}
        self.expert_errors = {}
        self.captions_ready = set()
        self.caption_errors = {}
        self.stop = threading.Event()
        self.stop_reason = None

    def stop_run(self, reason):
        self.stop_reason = reason
        self.stop.set()

    def validate_source_metadata(self, episode):
        dataset = Path(self.manifest["dataset"])
        episode_id = episode["episode_id"]
        tags = dataset / "tasks" / episode["task_code"] / "api_tags" / (episode_id + ".json")
        receipt = dataset / "receipts" / (episode_id + ".json")
        if sha256(tags) != episode["tags_sha256"] or sha256(receipt) != episode["receipt_sha256"]:
            raise ValueError("Source review metadata changed after full selection")

    def media(self, episode):
        with self.media_guard:
            if shutil.disk_usage(self.root).free < 8 * 2**30:
                self.stop_run("Less than 8 GiB disk space remains")
                raise OSError(self.stop_reason)
            self.validate_source_metadata(episode)
            return prepare_media(self.root, episode, self.manifest["sampling"])

    def task_experts(self, task_code):
        with self.task_locks[task_code]:
            if task_code in self.expert_errors:
                raise ValueError(self.expert_errors[task_code])
            if task_code not in self.experts:
                task = self.manifest["tasks"][task_code]
                rows = [self.manifest["episodes"][i] for i in task["experts"]]
                if any(r["task_code"] != task_code or r["quality"] != "high"
                       or r["gt_status"] != "Accepted" or not r.get("reviewer")
                       or not r.get("review_time") for r in rows):
                    raise ValueError("Invalid same-task expert approval")
                try:
                    self.experts[task_code] = [self.media(e) for e in rows]
                except Exception as exc:
                    self.expert_errors[task_code] = f"{type(exc).__name__}: {exc}"
                    raise
            return self.experts[task_code]

    def review_b(self, client, task_code, *, instruction, experts, candidate):
        arguments = dict(run_dir=self.root, instruction=instruction, experts=experts, candidate=candidate)
        with self.caption_locks[task_code]:
            if task_code in self.caption_errors:
                raise ModelCallError("Expert caption preparation failed: " + self.caption_errors[task_code])
            if task_code not in self.captions_ready:
                try:
                    return route_b.review(client, **arguments)
                finally:
                    # A candidate failure need not invalidate successfully generated expert captions.
                    ready = all((self.root / "references" / "route_b" / (digest(route_b._signature(
                        client, self.root, instruction, e, f"E{i}")) + ".json")).exists()
                        for i, e in enumerate(experts, 1))
                    if ready:
                        self.captions_ready.add(task_code)
                    else:
                        self.caption_errors[task_code] = "Incomplete cache; explicit retry required"
        return route_b.review(client, **arguments)

    def case(self, episode_id, routes):
        if self.stop.is_set():
            return []
        episode = self.manifest["episodes"][episode_id]
        task_code = episode["task_code"]
        task = self.manifest["tasks"][task_code]
        if episode_id not in task["candidate_ids"] or episode_id in task["expert_pool"]:
            raise ValueError("Episode must be an independent non-expert candidate")
        preparation_error = None
        started = time.monotonic()
        try:
            media = self.media(episode)
            experts = self.task_experts(task_code)
        except Exception as exc:
            preparation_error = f"{type(exc).__name__}: {exc}"
        prepared_s = time.monotonic() - started
        output = []
        for route in routes:
            if self.stop.is_set():
                break
            started = time.monotonic()
            row = {"episode_id": episode_id, "task_code": task_code, "instruction": task["instruction"],
                   "split": "full", "route": route, "expert_ids": task["experts"],
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "configuration_sha256": digest(self.config), "media_prepare_s": prepared_s}
            try:
                if preparation_error:
                    raise ValueError(preparation_error)
                if not experts or not media["frames"] or any(not e["frames"] for e in experts):
                    row.update(status="needs_review", label=None, reason="缺少可用的同任务专家或候选画面。",
                               checks={}, evidence=[], warnings=task.get("warnings", []) + media["warnings"])
                else:
                    client = self.client_factory(self.root, throttle=self.throttle,
                        context={"episode_id": episode_id, "task_code": task_code, "route": route,
                                 "split": "full"})
                    arguments = dict(instruction=task["instruction"], experts=experts, candidate=media)
                    response = (route_a.review(client, run_dir=self.root, **arguments) if route == "A"
                                else self.review_b(client, task_code, **arguments))
                    reply = response["reply"]
                    row.update(decide(reply["data"], media))
                    row.update(reference=response["reference"], usage=reply["usage"], model=reply["model"],
                               model_elapsed_s=reply["elapsed_s"], review_call=reply["call_path"],
                               extra_calls=[r["call_path"] for r in response["extra_calls"]],
                               estimated_cny_before_discounts=reply["estimated_cny_before_discounts"],
                               request_id=reply["request_id"], request_sha256=reply["request_sha256"])
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                row.update(status="failed", label=None, reason=reason, checks={}, evidence=[], warnings=[])
                if isinstance(exc, ModelCallError) and any(s in str(exc) for s in ("HTTP 401", "HTTP 403")):
                    self.stop_run("Authorization failed; stopped further model calls")
            row.update(gt=episode["gt"], gt_status=episode["gt_status"], gt_reason=episode["gt_reason"],
                       elapsed_s=time.monotonic() - started)
            path = self.root / "results" / task_code / f"{route}-{episode_id}-{uuid.uuid4().hex[:8]}.json"
            temporary = path.with_suffix(".tmp")
            write_json(temporary, row)
            temporary.rename(path)
            output.append(row)
        return output


def run(run_dir, *, workers=16, media_workers=4, requests_per_minute=45,
        first_per_task=False, retry_failed=False, max_episodes=None):
    if workers < 1 or media_workers < 1 or requests_per_minute <= 0:
        raise ValueError("Worker counts and rate must be positive")
    run_dir = run_dir.resolve()
    with (run_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = FullRun(run_dir, media_workers=media_workers, requests_per_minute=requests_per_minute)
        previous = latest_results(run_dir)
        groups = [t["candidate_ids"][:1] if first_per_task else t["candidate_ids"]
                  for t in batch.manifest["tasks"].values()]
        jobs = []
        for stripe in itertools.zip_longest(*groups):
            for episode_id in stripe:
                if episode_id is None:
                    continue
                routes = [r for r in batch.manifest.get("routes", ["A", "B"])
                          if (r, episode_id) not in previous
                          or (retry_failed and previous[r, episode_id]["status"] == "failed")]
                if routes:
                    jobs.append((episode_id, routes))
        if max_episodes is not None:
            jobs = jobs[:max_episodes]
        counts = Counter((r["route"], r["status"]) for r in previous.values())
        print(json.dumps({"queued_episodes": len(jobs), "queued_reviews": sum(len(r) for _, r in jobs),
                          "workers": workers, "requests_per_minute": requests_per_minute}), flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(batch.case, episode_id, routes) for episode_id, routes in jobs}
            for future in as_completed(pending):
                for row in future.result():
                    key = row["route"], row["episode_id"]
                    if key in previous:
                        old = previous[key]
                        counts[old["route"], old["status"]] -= 1
                    previous[key] = row
                    counts[row["route"], row["status"]] += 1
                    print(json.dumps({k: row[k] for k in ("task_code", "episode_id", "route", "gt", "label", "status")},
                                     ensure_ascii=False), flush=True)
                if batch.stop.is_set():
                    for job in pending:
                        job.cancel()
                    break
        summary = {"processed": len(previous), "counts": {f"{r}/{s}": n for (r, s), n in counts.items()},
                   "stopped": batch.stop.is_set(), "stop_reason": batch.stop_reason}
        write_json(run_dir / "execution" / (uuid.uuid4().hex + ".json"), summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preparation.add_argument("--experts", type=int, default=3)
    execution = commands.add_parser("run")
    execution.add_argument("--workers", type=int, default=16)
    execution.add_argument("--media-workers", type=int, default=4)
    execution.add_argument("--requests-per-minute", type=float, default=45)
    execution.add_argument("--first-per-task", action="store_true")
    execution.add_argument("--max-episodes", type=int)
    execution.add_argument("--retry-failed", action="store_true")
    commands.add_parser("report")
    args = parser.parse_args()
    if args.command == "prepare":
        from .full_data import prepare
        manifest = prepare(args.dataset, args.run_dir, args.experts)
        print(json.dumps({"tasks": len(manifest["tasks"]), "indexed": len(manifest["episodes"]),
                          "candidates": sum(len(t["candidate_ids"]) for t in manifest["tasks"].values()),
                          "experts": sum(len(t["experts"]) for t in manifest["tasks"].values()),
                          "exclusions": len(manifest["exclusions"])}, ensure_ascii=False, indent=2))
    elif args.command == "run":
        run(args.run_dir, workers=args.workers, media_workers=args.media_workers,
            requests_per_minute=args.requests_per_minute, first_per_task=args.first_per_task,
            max_episodes=args.max_episodes, retry_failed=args.retry_failed)
    else:
        from .full_report import report
        folder, summary = report(args.run_dir)
        print(json.dumps({"report_dir": str(folder), "partial": summary["partial"],
                          "candidate_episodes": summary["candidate_episodes"],
                          "routes": {r: data["overall"] for r, data in summary["routes"].items()},
                          "billing": summary["billing"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
