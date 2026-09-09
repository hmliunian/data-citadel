"""Read-only source indexing and disjoint, reproducible pilot selection."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET = Path("/home/xuran/xuran_projects/data_review/datasets/20260907_afternoon")
ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_records(dataset: Path, task_code: str) -> tuple[dict, list[dict]]:
    if not re.fullmatch(r"DL-[A-Z0-9]+", task_code):
        raise ValueError("Invalid task_code")
    task = next((t for t in read_json(dataset / "task-summary.json")
                 if t["task_code"] == task_code), None)
    if task is None:
        raise ValueError("Task not found")
    instruction = "\n".join(step["action_text"] for step in task["steps"])
    records = []
    for row in read_json(dataset / "tasks" / task_code / "episodes.json"):
        episode_id = row["sample_id"]
        if not ID_PATTERN.fullmatch(episode_id):
            raise ValueError("Invalid episode id")
        tags_path = dataset / "tasks" / task_code / "api_tags" / (episode_id + ".json")
        tags = read_json(tags_path)
        if tags.get("task.task_code") != task_code:
            raise ValueError("Task metadata mismatch")
        receipt_path = dataset / "receipts" / (episode_id + ".json")
        receipt = read_json(receipt_path)
        if (not receipt.get("complete") or receipt["episode"]["id"] != episode_id
                or receipt["episode"]["task_code"] != task_code):
            raise ValueError("Incomplete or mismatched receipt")
        mcaps = [f for f in receipt["files"] if f["relative_path"].endswith(".mcap")]
        if len(mcaps) != 1:
            raise ValueError("Expected exactly one source MCAP")
        file = mcaps[0]
        relative = Path(file["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source path")
        path = dataset / "tasks" / task_code / "data" / episode_id / relative
        if path.stat().st_size != file["size"]:
            raise ValueError("Downloaded MCAP size mismatch")
        records.append({
            "episode_id": episode_id, "task_code": task_code, "instruction": instruction,
            "mcap_path": str(path.resolve()), "mcap_sha256": file["sha256"],
            "tags_sha256": sha256(tags_path), "receipt_sha256": sha256(receipt_path),
            "quality": tags.get("task.review.data.quantify"),
            "gt": {"Accepted": "correct", "Denied": "incorrect"}.get(tags.get("task.review.status")),
            "gt_status": tags.get("task.review.status"),
            "gt_reason": tags.get("task.review.deny_reason"),
        })
    return task, records


def choose_samples(records: list[dict], expert_count: int, seed: str) -> dict:
    if expert_count < 1:
        raise ValueError("At least one expert is needed")
    ordered = sorted(records, key=lambda r: digest([seed, r["episode_id"]]))
    pool = [r for r in ordered if r["quality"] == "high"]
    if len(pool) < expert_count:
        raise ValueError("Not enough high expert candidates")
    expert_hashes = {r["mcap_sha256"] for r in pool}
    seen, accepted, denied, duplicates = set(expert_hashes), [], [], []
    for row in ordered:
        if row in pool:
            continue
        if row["mcap_sha256"] in seen:
            duplicates.append(row["episode_id"])
            continue
        seen.add(row["mcap_sha256"])
        if row["gt"] == "correct":
            accepted.append(row)
        elif row["gt"] == "incorrect":
            denied.append(row)
    if len(accepted) < 6 or len(denied) < 2:
        raise ValueError("This pilot needs six non-expert positives and two negatives")
    experts = pool[:expert_count]
    if len({r["mcap_sha256"] for r in experts}) != len(experts):
        raise ValueError("Duplicate expert recordings")
    return {
        "expert_pool": [r["episode_id"] for r in pool],
        "experts": [r["episode_id"] for r in experts],
        "development": [r["episode_id"] for r in accepted[:3] + denied[:1]],
        "holdout": [r["episode_id"] for r in accepted[3:6] + denied[1:2]],
        "duplicate_exclusions": duplicates,
    }


def prepare(dataset: Path, run_dir: Path, task_code: str, expert_count: int = 3):
    dataset, run_dir = dataset.resolve(), run_dir.resolve()
    if run_dir == dataset or dataset in run_dir.parents:
        raise ValueError("Artifacts must be outside the read-only dataset")
    task, records = source_records(dataset, task_code)
    splits = choose_samples(records, expert_count, seed="citadel-pilot-20260909-v1")
    used = set(splits["expert_pool"] + splits["development"] + splits["holdout"])
    manifest = {
        "version": "pilot-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "task_code": task_code, "instruction": records[0]["instruction"],
        "sampling": {"interval_s": 2.0, "interior_tolerance_s": 0.10, "include_endpoints": True},
        "task_counts": {"total": len(records), "gt": task["review_status_counts"],
                        "quality": task["quality_counts"]},
        "splits": splits,
        "episodes": {r["episode_id"]: r for r in records if r["episode_id"] in used},
        "selection_note": "Stratified by GT; exact MCAP hashes deduplicated. No visual inspection used.",
    }
    manifest["snapshot_sha256"] = digest(manifest)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def load_manifest(run_dir: Path):
    manifest = read_json(run_dir / "manifest.json")
    original = {k: v for k, v in manifest.items() if k != "snapshot_sha256"}
    if digest(original) != manifest["snapshot_sha256"]:
        raise ValueError("Pilot manifest changed after selection")
    return manifest
