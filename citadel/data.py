"""Source inventory, immutable experiment split and small JSON helpers."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ID = re.compile(r"[0-9a-f]{32}")
TASK = re.compile(r"DL-[A-Z0-9]+")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def inventory(dataset: Path):
    records = {}
    for index in sorted((dataset / "tasks").glob("*/episodes.json")):
        code = index.parent.name
        if not TASK.fullmatch(code):
            raise ValueError("Invalid task code")
        for item in read(index):
            episode_id = item["sample_id"]
            if not ID.fullmatch(episode_id) or episode_id in records:
                raise ValueError("Invalid or duplicate episode ID")
            tags_path = index.parent / "api_tags" / (episode_id + ".json")
            receipt_path = dataset / "receipts" / (episode_id + ".json")
            tags, receipt = read(tags_path), read(receipt_path)
            if (tags.get("task.task_code") != code or not receipt.get("complete")
                    or receipt["episode"]["id"] != episode_id
                    or receipt["episode"]["task_code"] != code):
                raise ValueError("Source metadata or download receipt mismatch")
            entries = [f for f in receipt["files"] if f["relative_path"].endswith(".mcap")]
            if len(entries) != 1:
                raise ValueError("Expected one MCAP per episode")
            entry = entries[0]
            folder = (index.parent / "data" / episode_id).resolve()
            source = (folder / entry["relative_path"]).resolve()
            if not source.is_relative_to(folder) or source.stat().st_size != entry["size"]:
                raise ValueError("Invalid source MCAP path or size")
            records[episode_id] = {
                "episode_id": episode_id, "task_code": code, "mcap_path": str(source),
                "mcap_sha256": entry["sha256"], "tags_sha256": file_hash(tags_path),
                "receipt_sha256": file_hash(receipt_path),
                "gt": {"Accepted": "correct", "Denied": "incorrect"}.get(tags.get("task.review.status")),
                "gt_reason": tags.get("task.review.deny_reason"),
            }
    if not records:
        raise ValueError("Dataset contains no episodes")
    return records


def prepare(dataset: Path, work: Path, holdout_fraction: float = 0.3):
    dataset, work = dataset.resolve(), work.resolve()
    if work.is_relative_to(dataset):
        raise ValueError("Output must be outside the read-only dataset")
    if not 0 < holdout_fraction < 1:
        raise ValueError("Holdout fraction must be between zero and one")
    records = inventory(dataset)
    groups, hashes, excluded = defaultdict(list), {}, []
    for episode_id in sorted(records, key=lambda i: fingerprint(["citadel-v1", i])):
        row = records[episode_id]
        previous = hashes.get(row["mcap_sha256"])
        if previous:
            if (records[previous]["gt"], records[previous]["task_code"]) != (row["gt"], row["task_code"]):
                raise ValueError("Duplicate MCAP has conflicting task or GT")
            excluded.append({"episode_id": episode_id, "duplicate_of": previous})
            continue
        hashes[row["mcap_sha256"]] = episode_id
        groups[(row["task_code"], row["gt"], row["gt_reason"])].append(episode_id)
    splits = {"development": [], "holdout": []}
    for (_, gt, _), ids in groups.items():
        count = min(len(ids) - 1, max(1, round(len(ids) * holdout_fraction))) if gt else 0
        splits["holdout"].extend(ids[:count])
        splits["development"].extend(ids[count:])
    manifest = {"created_at": now(), "dataset": str(dataset), "episodes": records,
                "splits": splits, "duplicate_exclusions": excluded,
                "sampling": {"interval_s": 1.0, "tolerance_s": 0.1}}
    manifest["sha256"] = fingerprint(manifest)
    work.mkdir(parents=True, exist_ok=False)
    write(work / "manifest.json", manifest)
    return manifest


def load_manifest(work: Path):
    data = read(work / "manifest.json")
    if fingerprint({k: v for k, v in data.items() if k != "sha256"}) != data["sha256"]:
        raise ValueError("Manifest changed after preparation")
    return data


def profile_for(resources: dict, profiles: dict):
    actions = {s["action_id"] for s in resources["steps"]}
    matches = [(name, profile) for name, profile in profiles.items()
               if actions and actions <= set(profile["action_ids"])
               and (not profile.get("task_codes") or resources.get("task_code") in profile["task_codes"])]
    if len(matches) != 1:
        raise ValueError("Task actions need exactly one configured atomic-task profile")
    name, profile = matches[0]
    if not all(profile.get(k) for k in ("success", "allowed", "failures")):
        raise ValueError("Task profile needs success, allowed variation and failure rules")
    hold, tolerance = profile.get("hold_seconds"), profile.get("hold_tolerance_s", 0)
    if hold is not None and not (0 <= tolerance < hold):
        raise ValueError("Invalid hold duration/tolerance")
    return {"name": name, **profile}
