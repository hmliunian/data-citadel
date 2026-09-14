"""Read-only source inventory and reproducible experiment manifests."""
import re
from collections import defaultdict
from pathlib import Path

from .files import file_hash, fingerprint, now, read, write

ID = re.compile(r"[0-9a-f]{32}")
TASK = re.compile(r"DL-[A-Z0-9]+")


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


