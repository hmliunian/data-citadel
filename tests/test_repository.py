import hashlib
import json
from pathlib import Path

import pytest

from data_citadel.models import DatasetError, EpisodeNotFound
from data_citadel.repository import EpisodeRepository, dotted_get, reviewed_correct


def bundle(root: Path, episode_id="a" * 32, label="correct", document=None):
    directory = root / "atomic" / "A_001" / label / episode_id
    directory.mkdir(parents=True)
    (directory / "episode.mcap").write_bytes(b"verified-test-mcap")
    (directory / f"{episode_id}.json").write_text(json.dumps(document or {
        "task.action_id": "A_001", "task.task_code": "task-1",
        "task.action_text": {"rendered_zh": "把杯子拿起来"},
        "task.collector.user": "private-person", "task.review.status": "Accepted",
        "task.review.reviewer": "private-reviewer",
        "task.review.deny_reason": "private-evaluation-label",
    }))
    files = {}
    for path in directory.iterdir():
        stat = path.stat()
        files[path.name] = {"name": path.name, "size": stat.st_size, "remote_size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "verification.json").write_text(json.dumps({
        "version": 2, "action_id": "A_001", "episode_uuid": episode_id, "stratum": label,
        "mcap_doctor": "passed", "files": files,
    }))
    return directory


def test_verified_index_adapts_mixed_keys_without_exposing_collectors(tmp_path):
    bundle(tmp_path)
    repository = EpisodeRepository(tmp_path)
    episode = repository.get("a" * 32)
    assert episode.instruction == "把杯子拿起来"
    assert episode.label == "correct"
    inventory = repository.inventory()
    assert inventory["total_episodes"] == 1
    assert inventory["category_counts"]["retry_then_success"] == 0
    assert inventory["expert_availability"][0]["collector_group_counts"] == [1]
    assert "private-person" not in json.dumps(inventory)
    assert "private-evaluation-label" not in json.dumps(inventory)


def test_missing_receipt_is_ignored_but_changed_verified_file_is_rejected(tmp_path):
    incomplete = tmp_path / "atomic" / "A_001" / "correct" / ("b" * 32)
    incomplete.mkdir(parents=True)
    (incomplete / "episode.mcap").write_bytes(b"partial")
    assert EpisodeRepository(tmp_path).list_episodes() == []
    directory = bundle(tmp_path)
    (directory / "episode.mcap").write_bytes(b"changed")
    with pytest.raises(DatasetError, match="changed"):
        EpisodeRepository(tmp_path).list_episodes()


def test_duplicate_identity_and_traversal_are_rejected(tmp_path):
    bundle(tmp_path)
    bundle(tmp_path, label="other")
    with pytest.raises(DatasetError, match="Duplicate episode"):
        EpisodeRepository(tmp_path).list_episodes()
    with pytest.raises(EpisodeNotFound):
        EpisodeRepository(tmp_path).get("../../secret")


def test_symlink_escape_is_rejected(tmp_path):
    dataset, outside = tmp_path / "dataset", tmp_path / "outside"
    outside.mkdir()
    directory = bundle(dataset)
    (outside / "episode.mcap").write_bytes(b"private")
    (directory / "episode.mcap").unlink()
    (directory / "episode.mcap").symlink_to(outside / "episode.mcap")
    with pytest.raises(DatasetError, match="escapes"):
        EpisodeRepository(dataset).list_episodes()


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '[]', '{'])
def test_invalid_json_is_rejected_even_with_matching_receipt(tmp_path, raw):
    directory = bundle(tmp_path)
    receipt_path = directory / "verification.json"
    receipt_path.write_text(raw)
    with pytest.raises(DatasetError):
        EpisodeRepository(tmp_path).list_episodes()


def test_conflicting_dotted_metadata_is_rejected():
    assert dotted_get({"task": {"action_text.rendered_zh": "task"}},
                      "task.action_text.rendered_zh") == "task"
    with pytest.raises(DatasetError, match="Conflicting"):
        dotted_get({"task.action_id": "A_001", "task": {"action_id": "A_002"}}, "task.action_id")


def test_inventory_counts_one_action_across_tasks_objects_and_collectors(tmp_path):
    for index in range(10):
        bundle(tmp_path, f"{index:032x}", document={
            "task.action_id": "A_001", "task.task_code": f"task-{index % 2}",
            "task.action_text": {"rendered_zh": f"拿起物体 {index}"},
            "task.collector.user": f"private-collector-{index % 2}",
            "task.review.status": "Accepted", "task.review.reviewer": "private-reviewer",
        })
    inventory = EpisodeRepository(tmp_path).inventory()
    assert inventory["task_grouping"] == "action_id"
    assert inventory["action_count"] == 1
    assert inventory["actions"]["A_001"]["correct"] == 10
    assert inventory["expert_availability"] == [{
        "action_id": "A_001", "reviewed_correct_count": 10, "collector_count": 2,
        "collector_group_counts": [5, 5], "expert_candidates": 5, "correct_test_candidates": 5,
    }]
    assert "private-" not in json.dumps(inventory)


@pytest.mark.parametrize("reviewer,status,expected", [
    ("original-reviewer", "Accepted", True),
    ("original-reviewer", "accepted", True),
    ("", "Accepted", False),
    (None, "Accepted", False),
    ("original-reviewer", "Denied", False),
])
def test_source_approval_needs_accepted_and_a_recorded_reviewer(tmp_path, reviewer, status, expected):
    bundle(tmp_path, document={
        "task.action_id": "A_001", "task.task_code": "task",
        "task.action_text": {"rendered_zh": "拿起杯子"},
        "task.collector.user": "collector",
        "task": {"review": {"status": status, "reviewer": reviewer}},
    })
    repository = EpisodeRepository(tmp_path)
    assert reviewed_correct(repository.get("a" * 32)) is expected
    availability = repository.inventory()["expert_availability"][0]
    assert availability["reviewed_correct_count"] == int(expected)
