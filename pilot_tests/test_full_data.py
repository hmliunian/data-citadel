from collections import Counter

import pytest

from pilot import full_data
from pilot.data import choose_samples, digest, load_manifest, read_json, sha256, source_records, write_json


def rows(base=0):
    return [{"episode_id": f"{base + i:032x}", "quality": "high" if i < 3 else "medium",
             "gt_status": "Accepted" if i < 9 else "Denied", "reviewer": "original-reviewer",
             "review_time": "2026-09-09T12:00:00+08:00", "payload": f"source-{base + i}"}
            for i in range(11)]


def make_dataset(tmp_path, groups):
    dataset = tmp_path / "dataset"
    summaries = []
    for code, records in groups.items():
        instruction = "搬运 " + code + " 的任务物体"
        for row in records:
            episode_id = row["episode_id"]
            path = dataset / "tasks" / code / "data" / episode_id / "episode.mcap"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(row["payload"].encode())
            tags = {"task.task_code": code, "task.review.data.quantify": row["quality"],
                    "task.review.status": row["gt_status"], "task.review.deny_reason": row.get("gt_reason"),
                    "task.review.reviewer": row["reviewer"], "task.review.review_time": row["review_time"]}
            write_json(dataset / "tasks" / code / "api_tags" / (episode_id + ".json"), tags)
            write_json(dataset / "receipts" / (episode_id + ".json"), {
                "complete": True, "episode": {"id": episode_id, "task_code": code},
                "files": [{"relative_path": "episode.mcap", "size": path.stat().st_size,
                           "sha256": sha256(path)}],
            })
        write_json(dataset / "tasks" / code / "episodes.json",
                   [{"sample_id": r["episode_id"]} for r in records])
        summaries.append({"task_code": code, "episode_count": len(records),
                          "steps": [{"action_text": instruction}],
                          "review_status_counts": dict(Counter(r["gt_status"] for r in records)),
                          "quality_counts": dict(Counter(r["quality"] for r in records))})
    write_json(dataset / "task-summary.json", summaries)
    return dataset


def test_full_snapshot_keeps_all_candidates_and_original_expert_order(tmp_path):
    dataset = make_dataset(tmp_path, {"DL-A": rows(), "DL-B": rows(100)})
    run = tmp_path / "full"
    manifest = full_data.prepare(dataset, run)
    assert load_manifest(run) == manifest
    assert manifest["version"] == "full-v1" and manifest["routes"] == ["A", "B"]
    assert manifest["counts"] == {"experts": 6, "expert_pool": 6, "candidates": 16,
                                  "excluded": 6, "gt_conflict_groups": 0}
    assert manifest["source_inventory"] == {"task_count": 2, "episode_count": 22,
                                            "indexed_task_count": 2, "indexed_episode_count": 22}
    for code, task in manifest["tasks"].items():
        _, original = source_records(dataset, code)
        assert task["experts"] == choose_samples(original, 3, full_data.SELECTION_SEED)["experts"]
        assert set(task["expert_pool"]).isdisjoint(task["candidate_ids"])
        assert all(manifest["episodes"][i]["task_code"] == code for i in task["experts"])
        assert {r["episode_id"]: r for r in original} == {
            i: r for i, r in manifest["episodes"].items() if r["task_code"] == code}
    saved = (run / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        full_data.prepare(dataset, run)
    assert (run / "manifest.json").read_bytes() == saved


def test_expert_count_is_configurable_but_entire_high_pool_is_reserved(tmp_path):
    dataset = make_dataset(tmp_path, {"DL-A": rows()})
    result = full_data.prepare(dataset, tmp_path / "full", expert_count=1)
    task = result["tasks"]["DL-A"]
    assert len(task["experts"]) == 1 and len(task["expert_pool"]) == 3
    assert len(task["candidate_ids"]) == 8
    assert len(result["exclusions"]) == 3


@pytest.mark.parametrize("field,value", [("gt_status", "Denied"), ("reviewer", None),
                                        ("reviewer", "  "), ("review_time", "")])
def test_unavailable_experts_do_not_drop_the_task_candidates(tmp_path, field, value):
    records = rows()
    records[0][field] = value
    dataset = make_dataset(tmp_path, {"DL-A": records})
    result = full_data.prepare(dataset, tmp_path / "full")
    task = result["tasks"]["DL-A"]
    assert task["experts"] == []
    assert task["warnings"] == ["insufficient_eligible_experts:2/3"]
    assert len(task["candidate_ids"]) == 8
    assert records[0]["episode_id"] in task["expert_pool"]
    assert result["source_errors"] == []


def test_duplicate_experts_are_not_counted_twice(tmp_path):
    records = rows()
    records[1]["payload"] = records[0]["payload"]
    dataset = make_dataset(tmp_path, {"DL-A": records})
    result = full_data.prepare(dataset, tmp_path / "full")
    assert result["tasks"]["DL-A"]["experts"] == []
    assert result["tasks"]["DL-A"]["source_counts"]["eligible_experts"] == 2


def test_global_duplicate_exclusions_have_per_id_reasons(tmp_path):
    first, second = rows(), rows(100)
    second[3]["payload"] = first[0]["payload"]
    second[4]["payload"] = first[4]["payload"]
    dataset = make_dataset(tmp_path, {"DL-A": first, "DL-B": second})
    result = full_data.prepare(dataset, tmp_path / "full")
    exclusions = {e["episode_id"]: e for e in result["exclusions"]}
    assert exclusions[second[3]["episode_id"]]["reason"] == "expert_duplicate_mcap"
    duplicates = [first[4]["episode_id"], second[4]["episode_id"]]
    kept, excluded = sorted(duplicates, key=lambda i: digest([full_data.SELECTION_SEED, i]))
    assert exclusions[excluded]["reason"] == "candidate_duplicate_mcap"
    assert exclusions[excluded]["kept_episode_id"] == kept
    candidate_ids = [i for t in result["tasks"].values() for i in t["candidate_ids"]]
    assert kept in candidate_ids and excluded not in candidate_ids
    assert result["counts"]["candidates"] == 14
    assert len(candidate_ids) + len(exclusions) == len(result["episodes"])


@pytest.mark.parametrize("with_expert", [False, True])
def test_conflicting_ground_truth_is_preserved_and_quarantined(tmp_path, with_expert):
    records = rows()
    original = records[0 if with_expert else 3]
    records[9]["payload"] = original["payload"]
    dataset = make_dataset(tmp_path, {"DL-A": records})
    result = full_data.prepare(dataset, tmp_path / "full")
    ids = {original["episode_id"], records[9]["episode_id"]}
    assert set(result["gt_conflicts"][0]["episode_ids"]) == ids
    assert {result["episodes"][i]["gt"] for i in ids} == {"correct", "incorrect"}
    assert all(e["reason"] == "conflicting_gt" for e in result["exclusions"] if e["episode_id"] in ids)
    task = result["tasks"]["DL-A"]
    assert ids.isdisjoint(task["candidate_ids"] + task["experts"])
    assert "conflicting_gt_excluded" in task["warnings"]
    if with_expert:
        assert task["experts"] == []


def test_source_error_blocks_full_manifest_and_records_inventory_scope(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path, {"DL-A": rows(), "DL-B": rows(100)})
    original = full_data.source_records

    def broken(dataset, code):
        if code == "DL-A":
            raise RuntimeError("Unreadable source receipt")
        return original(dataset, code)

    monkeypatch.setattr(full_data, "source_records", broken)
    run = tmp_path / "full"
    with pytest.raises(ValueError, match="Full snapshot blocked"):
        full_data.prepare(dataset, run)
    assert not (run / "manifest.json").exists()
    diagnosis = read_json(run / "prepare_errors.json")
    assert diagnosis["source_inventory"]["episode_count"] == 22
    assert diagnosis["source_inventory"]["indexed_episode_count"] == 11
    assert diagnosis["source_errors"] == [{"task_code": "DL-A", "scope": "task",
        "expected_episodes": 11, "type": "RuntimeError", "message": "Unreadable source receipt"}]


@pytest.mark.parametrize("corruption", ["count", "cross_task", "duplicate_id", "instruction"])
def test_inconsistent_source_inventory_cannot_be_marked_complete(tmp_path, monkeypatch, corruption):
    dataset = make_dataset(tmp_path, {"DL-A": rows()})
    original = full_data.source_records

    def changed(dataset, code):
        task, records = original(dataset, code)
        if corruption == "count":
            records.pop()
        elif corruption == "cross_task":
            records[0]["task_code"] = "DL-OTHER"
        elif corruption == "duplicate_id":
            records[1]["episode_id"] = records[0]["episode_id"]
        elif corruption == "instruction":
            records[0]["instruction"] = "另一个任务"
        return task, records

    monkeypatch.setattr(full_data, "source_records", changed)
    run = tmp_path / "full"
    with pytest.raises(ValueError, match="Full snapshot blocked"):
        full_data.prepare(dataset, run)
    assert not (run / "manifest.json").exists()
    assert read_json(run / "prepare_errors.json")["source_errors"][0]["task_code"] == "DL-A"


def test_unreadable_top_inventory_gets_a_readable_diagnostic(tmp_path):
    dataset = tmp_path / "missing-dataset"
    run = tmp_path / "full"
    with pytest.raises(ValueError, match="Full snapshot blocked"):
        full_data.prepare(dataset, run)
    diagnostic = read_json(run / "prepare_errors.json")
    assert diagnostic["source_inventory"]["episode_count"] is None
    assert diagnostic["source_errors"][0]["scope"] == "inventory"
    assert not (run / "manifest.json").exists()


def test_snapshot_cannot_be_written_into_source_dataset(tmp_path):
    dataset = make_dataset(tmp_path, {"DL-A": rows()})
    with pytest.raises(ValueError, match="read-only"):
        full_data.prepare(dataset, dataset / "new-run")
    assert not (dataset / "new-run").exists()


def test_unknown_gt_and_quality_remain_unmodified_candidates(tmp_path):
    records = rows()
    records[-1].update(gt_status=None, quality=None)
    dataset = make_dataset(tmp_path, {"DL-A": records})
    result = full_data.prepare(dataset, tmp_path / "full")
    episode_id = records[-1]["episode_id"]
    task = result["tasks"]["DL-A"]
    assert episode_id in task["candidate_ids"]
    assert result["episodes"][episode_id]["gt"] is None
    assert result["episodes"][episode_id]["gt_status"] is None
    assert result["episodes"][episode_id]["quality"] is None
    assert task["source_counts"]["gt"]["unknown"] == 1
    assert task["source_counts"]["quality"]["unknown"] == 1
    assert load_manifest(tmp_path / "full") == result


@pytest.mark.parametrize("expert_count", [0, -1, True, 1.5])
def test_invalid_expert_count_does_not_create_artifact(tmp_path, expert_count):
    dataset = make_dataset(tmp_path, {"DL-A": rows()})
    run = tmp_path / "full"
    with pytest.raises(ValueError, match="At least one expert"):
        full_data.prepare(dataset, run, expert_count=expert_count)
    assert not run.exists()
