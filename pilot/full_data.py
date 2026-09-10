"""Immutable full-inventory selection without decoding media or changing source labels."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .data import digest, read_json, source_records, write_json

SELECTION_SEED = "citadel-pilot-20260909-v1"


def _blocked(run_dir: Path, inventory: dict, errors: list[dict]):
    path = run_dir / "prepare_errors.json"
    write_json(path, {"status": "blocked", "source_inventory": inventory, "source_errors": errors})
    raise ValueError(f"Full snapshot blocked by {len(errors)} source error(s); see {path}")


def prepare(dataset: Path, run_dir: Path, expert_count: int = 3) -> dict:
    dataset, run_dir = dataset.resolve(), run_dir.resolve()
    if run_dir == dataset or dataset in run_dir.parents:
        raise ValueError("Artifacts must be outside the read-only dataset")
    if type(expert_count) is not int or expert_count < 1:
        raise ValueError("At least one expert is required")
    run_dir.mkdir(parents=True, exist_ok=False)
    inventory = {"task_count": None, "episode_count": None,
                 "indexed_task_count": 0, "indexed_episode_count": 0}
    try:
        summaries = read_json(dataset / "task-summary.json")
        if not isinstance(summaries, list) or not summaries:
            raise ValueError("Expected a non-empty task inventory")
        codes = [task["task_code"] for task in summaries]
        if len(set(codes)) != len(codes):
            raise ValueError("Duplicate task_code in source inventory")
        if any(type(t["episode_count"]) is not int or t["episode_count"] < 0 for t in summaries):
            raise ValueError("Invalid source episode count")
        inventory.update(task_count=len(summaries),
                         episode_count=sum(t["episode_count"] for t in summaries))
    except Exception as exc:
        _blocked(run_dir, inventory, [{"task_code": None, "scope": "inventory",
                                      "type": type(exc).__name__, "message": str(exc)}])

    tasks, episodes, records_by_task, errors = {}, {}, {}, []
    for summary in sorted(summaries, key=lambda t: t["task_code"]):
        code = summary["task_code"]
        try:
            task, records = source_records(dataset, code)
            if task["task_code"] != code or any(r["task_code"] != code for r in records):
                raise ValueError("Source records cross task codes")
            if len(records) != summary["episode_count"]:
                raise ValueError(f"Inventory count {summary['episode_count']} != indexed {len(records)}")
            ids = [r["episode_id"] for r in records]
            if len(set(ids)) != len(ids) or any(episode_id in episodes for episode_id in ids):
                raise ValueError("Duplicate episode ID in source inventory")
            instruction = "\n".join(step["action_text"] for step in task["steps"])
            if not instruction.strip() or any(r["instruction"] != instruction for r in records):
                raise ValueError("Missing or inconsistent task instruction")
            ordered = sorted(records, key=lambda r: digest([SELECTION_SEED, r["episode_id"]]))
            records_by_task[code] = ordered
            episodes.update((r["episode_id"], r) for r in ordered)
            tasks[code] = {
                "instruction": instruction, "experts": [], "expert_pool": [],
                "candidate_ids": [], "warnings": [],
                "source_counts": {"total": len(records), "expected_total": summary["episode_count"],
                                  "gt": dict(Counter(r["gt_status"] or "unknown" for r in records)),
                                  "quality": dict(Counter(r["quality"] or "unknown" for r in records))},
            }
        except Exception as exc:
            errors.append({"task_code": code, "scope": "task", "expected_episodes": summary["episode_count"],
                           "type": type(exc).__name__, "message": str(exc)})
    inventory.update(indexed_task_count=len(tasks), indexed_episode_count=len(episodes))
    if errors:
        _blocked(run_dir, inventory, errors)

    groups = defaultdict(list)
    ordered = sorted(episodes.values(), key=lambda r: digest([SELECTION_SEED, r["episode_id"]]))
    for row in ordered:
        groups[row["mcap_sha256"]].append(row)
    conflicts = []
    conflict_hashes = set()
    for mcap_hash, group in groups.items():
        if {r["gt"] for r in group if r["gt"] is not None} == {"correct", "incorrect"}:
            conflict_hashes.add(mcap_hash)
            conflicts.append({"mcap_sha256": mcap_hash, "episode_ids": [r["episode_id"] for r in group],
                              "source_labels": [{k: r[k] for k in ("episode_id", "task_code", "gt", "gt_status")}
                                                for r in group]})

    pool_hashes = defaultdict(list)
    for code, records in records_by_task.items():
        pool = [r for r in records if r["quality"] == "high"]
        tasks[code]["expert_pool"] = [r["episode_id"] for r in pool]
        eligible, seen = [], set()
        for row in pool:
            mcap_hash = row["mcap_sha256"]
            pool_hashes[mcap_hash].append(row["episode_id"])
            if (row["gt_status"] == "Accepted" and mcap_hash not in conflict_hashes
                    and all(isinstance(row.get(k), str) and row[k].strip()
                            for k in ("reviewer", "review_time")) and mcap_hash not in seen):
                eligible.append(row["episode_id"])
                seen.add(mcap_hash)
        tasks[code]["source_counts"]["eligible_experts"] = len(eligible)
        if len(eligible) >= expert_count:
            tasks[code]["experts"] = eligible[:expert_count]
        else:
            tasks[code]["warnings"].append(f"insufficient_eligible_experts:{len(eligible)}/{expert_count}")

    exclusions, seen_candidates = [], {}
    for row in ordered:
        episode_id, code, mcap_hash = row["episode_id"], row["task_code"], row["mcap_sha256"]
        exclusion = {"episode_id": episode_id, "task_code": code, "mcap_sha256": mcap_hash}
        if mcap_hash in conflict_hashes:
            exclusion.update(reason="conflicting_gt",
                             related_episode_ids=[r["episode_id"] for r in groups[mcap_hash]])
        elif row["quality"] == "high":
            exclusion["reason"] = "expert_pool"
        elif mcap_hash in pool_hashes:
            exclusion.update(reason="expert_duplicate_mcap", related_episode_ids=pool_hashes[mcap_hash])
        elif mcap_hash in seen_candidates:
            exclusion.update(reason="candidate_duplicate_mcap", kept_episode_id=seen_candidates[mcap_hash])
        else:
            tasks[code]["candidate_ids"].append(episode_id)
            seen_candidates[mcap_hash] = episode_id
            continue
        exclusions.append(exclusion)
    for code, task in tasks.items():
        task["source_counts"]["candidates"] = len(task["candidate_ids"])
        task["source_counts"]["exclusions"] = dict(Counter(e["reason"] for e in exclusions
                                                         if e["task_code"] == code))
        if task["source_counts"]["exclusions"].get("conflicting_gt"):
            task["warnings"].append("conflicting_gt_excluded")
    manifest = {
        "version": "full-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "routes": ["A", "B"], "expert_count": expert_count,
        "sampling": {"interval_s": 2.0, "interior_tolerance_s": 0.10, "include_endpoints": True},
        "tasks": tasks, "episodes": episodes, "exclusions": exclusions, "gt_conflicts": conflicts,
        "source_inventory": inventory, "source_errors": [],
        "counts": {"experts": sum(len(t["experts"]) for t in tasks.values()),
                   "expert_pool": sum(len(t["expert_pool"]) for t in tasks.values()),
                   "candidates": sum(len(t["candidate_ids"]) for t in tasks.values()),
                   "excluded": len(exclusions), "gt_conflict_groups": len(conflicts)},
    }
    manifest["snapshot_sha256"] = digest(manifest)
    write_json(run_dir / "manifest.json", manifest)
    return manifest
