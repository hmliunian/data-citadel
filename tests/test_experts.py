from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from data_citadel.experts import EXPERT_POLICY, ExpertLibrary, prepare_experiment
from data_citadel.models import ACTIONS, LABELS, Episode, EpisodeNotFound, ExpertError


def episode(episode_id, **changes):
    return replace(Episode(
        episode_id=episode_id, action_id="A_001", task_code="TASK-CUP",
        instruction="Pick up the cup", collector_id="collector-1",
        mcap_path=Path("unused.mcap"), sidecar_path=Path("unused.json"),
        label="correct", review_status="Accepted",
        metadata={"task.review.reviewer": "original-reviewer"},
    ), **changes)


@pytest.fixture
def registered(tmp_path):
    video = tmp_path / "expert.mcap"
    video.write_bytes(b"video fixture")
    episodes = {
        f"expert-{i}": episode(f"expert-{i}", mcap_path=video, task_code=f"TASK-{i}",
                               instruction=f"Pick up object {i}", collector_id=f"collector-{i}")
        for i in range(5)
    }
    repository = Mock()
    repository.get.side_effect = episodes.__getitem__
    manifest = {
        "version": "reviewed-v2",
        "groups": [{
            "action_id": "A_001",
            "approved": True, "reviewer": "human-reviewer", "policy": EXPERT_POLICY,
            "expert_episode_ids": list(episodes),
        }],
    }
    return tmp_path / "experts.json", repository, episodes, manifest


def library(registered):
    path, repository, _, manifest = registered
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return ExpertLibrary(path, repository)


def test_approved_experts_share_action_across_objects_tasks_and_collectors(registered):
    experts = library(registered)
    result = experts.resolve(episode("target", task_code="NEW-TASK", collector_id="new-collector"))
    assert len(result) == 5
    assert len({e.task_code for e in result}) == 5
    assert len({e.instruction for e in result}) == 5
    assert len({e.collector_id for e in result}) == 5
    assert experts.episode_ids == {e.episode_id for e in result}
    assert experts.version == "reviewed-v2"
    expected_hash = hashlib.sha256(registered[0].read_bytes()).hexdigest()
    registered[0].write_text("{}", encoding="utf-8")
    assert experts.manifest_sha256 == expected_hash
    assert len(experts.resolve(episode("another-target"))) == 5


@pytest.mark.parametrize("source", ["manual", "dataset_review"])
def test_explicitly_disabled_group_stays_disabled(registered, source):
    registered[3]["groups"][0].update(approved=False, reviewer=None, approval_source=source)
    with pytest.raises(ExpertError, match="approved"):
        library(registered).resolve(episode("target"))


@pytest.mark.parametrize("change", [
    {"action_id": "A_002"}, {"label": "other"}, {"instruction": " "},
    {"mcap_path": Path("missing-expert.mcap")},
])
def test_experts_require_correct_action_valid_instruction_and_media(registered, change):
    registered[2]["expert-0"] = replace(registered[2]["expert-0"], **change)
    with pytest.raises(ExpertError):
        library(registered).resolve(episode("target"))


def test_empty_expert_media_is_rejected(registered):
    registered[2]["expert-0"].mcap_path.write_bytes(b"")
    with pytest.raises(ExpertError, match="video is unavailable"):
        library(registered).resolve(episode("target"))


def test_target_cannot_be_expert_or_a_different_action(registered):
    experts = library(registered)
    with pytest.raises(ExpertError, match="own expert"):
        experts.resolve(episode("expert-0"))
    with pytest.raises(ExpertError, match="action_id"):
        experts.resolve(episode("target", action_id="A_002"))


@pytest.mark.parametrize("change", [
    {"reviewer": None}, {"approved": "true"}, {"approval_source": "automatic_model"},
    {"expert_episode_ids": ["expert-0"] * 5},
])
def test_ambiguous_or_invalid_approval_is_rejected(registered, change):
    registered[3]["groups"][0].update(change)
    with pytest.raises(ExpertError):
        library(registered)


def test_old_policy_requires_preparing_a_new_manifest(registered):
    registered[3]["groups"][0]["policy"] = "same_task_same_collector_v1"
    with pytest.raises(ExpertError, match="prepare"):
        library(registered)


def test_action_id_is_the_only_group_identity(registered):
    group = registered[3]["groups"][0]
    group.update(task_code="legacy-task-1", collector_id="legacy-collector-1")
    registered[3]["groups"].append({
        **group, "task_code": "legacy-task-2", "collector_id": "legacy-collector-2",
        "expert_episode_ids": [f"another-{i}" for i in range(5)],
    })
    with pytest.raises(ExpertError, match="unique action_id"):
        library(registered)


def test_missing_or_insufficient_experts_are_expert_errors(registered):
    registered[1].get.side_effect = EpisodeNotFound("removed")
    with pytest.raises(ExpertError, match="unavailable"):
        library(registered).resolve(episode("target"))
    registered[3]["groups"][0]["expert_episode_ids"].pop()
    with pytest.raises(ExpertError, match="five"):
        library(registered).resolve(episode("target"))


def test_dataset_approval_uses_original_reviewers_without_new_group_reviewer(registered):
    group = registered[3]["groups"][0]
    group["approval_source"] = "dataset_review"
    del group["reviewer"]
    assert len(library(registered).resolve(episode("target"))) == 5


@pytest.mark.parametrize("change", [
    {"review_status": "Rejected"}, {"metadata": {}},
    {"metadata": {"task.review.reviewer": " "}}, {"label": "other"},
])
def test_dataset_approval_rechecks_actual_source_records(registered, change):
    registered[3]["groups"][0]["approval_source"] = "dataset_review"
    experts = library(registered)
    registered[2]["expert-0"] = replace(registered[2]["expert-0"], **change)
    with pytest.raises(ExpertError, match="review source"):
        experts.resolve(episode("target"))


def test_prepare_ten_actions_reuses_source_reviews_and_keeps_five_plus_five_disjoint():
    episodes = [
        episode(f"{action}-{label}-{i:02}", action_id=action, label=label,
                task_code=f"TASK-{i}", instruction=f"Use object {i}",
                collector_id=f"collector-{i % 3}")
        for action in ACTIONS for label in LABELS
        for i in range(10 if label == "correct" else 5)
    ]
    repository = Mock(list_episodes=Mock(return_value=list(reversed(episodes))))
    prepared = prepare_experiment(repository)
    experts = prepared["evaluation"]["expert_episode_ids"]
    samples = prepared["evaluation"]["samples"]
    tests = [sample["episode_id"] for sample in samples]
    assert len(experts) == 50
    assert len(tests) == 400
    assert sum(sample["label"] == "correct" for sample in samples) == 50
    assert not set(experts).intersection(tests)
    assert len(prepared["report"]["actions"]) == 10
    assert prepared["report"]["baseline_ready"] is True
    assert prepared["report"]["approval_source"] == "dataset_review"
    for group in prepared["experts"]["groups"]:
        assert group["approved"] is True
        assert group["policy"] == "same_action_v2"
        assert group["approval_source"] == "dataset_review"
        assert "reviewer" not in group
        assert group["expert_episode_ids"] == [f"{group['action_id']}-correct-{i:02}" for i in range(5)]
    assert all(report["correct_candidate_count"] == 10 and report["collector_count"] == 3
               for report in prepared["report"]["actions"].values())


def test_prepare_pools_task_codes_and_collectors_but_not_unaccepted_sources():
    episodes = [episode(f"same-{i}") for i in range(6)]
    episodes += [episode(f"collector-{i}", collector_id="collector-2", task_code="TASK-BOWL",
                         instruction="Pick up the bowl") for i in range(4)]
    episodes += [episode(f"rejected-{i}", review_status="Rejected") for i in range(10)]
    episodes += [episode(f"bowl-{i}", task_code="TASK-BOWL", label="blurred") for i in range(5)]
    prepared = prepare_experiment(Mock(list_episodes=Mock(return_value=episodes)))
    report = prepared["report"]["actions"]["A_001"]
    assert report["correct_candidate_count"] == 10
    assert report["collector_count"] == 2
    assert report["shortfalls_by_label"]["correct"] == 0
    assert report["shortfalls_by_label"]["blurred"] == 0
    assert not report["baseline_ready"]
    assert all(sample["cohort"] == "baseline"
               for sample in prepared["evaluation"]["samples"] if sample["label"] == "blurred")
    assert prepared["report"]["actions"]["A_002"]["expert_shortfall"] == 5


def test_prepare_requires_a_recorded_source_reviewer():
    episodes = [episode(f"reviewed-{i}") for i in range(6)]
    episodes += [episode(f"unreviewed-{i}", metadata={}) for i in range(4)]
    report = prepare_experiment(Mock(list_episodes=Mock(return_value=episodes)))["report"]
    assert report["actions"]["A_001"]["correct_candidate_count"] == 6
    assert report["actions"]["A_001"]["shortfalls_by_label"]["correct"] == 4


def test_missing_manifest_is_unconfigured_and_resolve_still_requires_approval(tmp_path):
    experts = ExpertLibrary(tmp_path / "config" / "experts.json", Mock())
    assert experts.version == "unconfigured"
    assert experts.manifest_sha256 is None
    assert experts.groups == []
    assert experts.episode_ids == frozenset()
    with pytest.raises(ExpertError, match="approved"):
        experts.resolve(episode("target"))


@pytest.mark.parametrize("contents", ["{", "{}", "[]"])
def test_existing_malformed_manifest_is_not_treated_as_unconfigured(tmp_path, contents):
    path = tmp_path / "experts.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ExpertError):
        ExpertLibrary(path, Mock())
