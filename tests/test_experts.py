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
    ), **changes)


@pytest.fixture
def registered(tmp_path):
    video = tmp_path / "expert.mcap"
    video.write_bytes(b"video fixture")
    episodes = {f"expert-{i}": episode(f"expert-{i}", mcap_path=video) for i in range(5)}
    repository = Mock()
    repository.get.side_effect = episodes.__getitem__
    manifest = {
        "version": "reviewed-v1",
        "groups": [{
            "action_id": "A_001", "task_code": "TASK-CUP", "collector_id": "collector-1",
            "approved": True, "reviewer": "human-reviewer", "policy": EXPERT_POLICY,
            "expert_episode_ids": list(episodes),
        }],
    }
    return tmp_path / "experts.json", repository, episodes, manifest


def library(registered):
    path, repository, _, manifest = registered
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return ExpertLibrary(path, repository)


def test_approved_experts_match_concrete_task_and_can_review_another_collector(registered):
    experts = library(registered)
    result = experts.resolve(episode("target", collector_id="collector-2"))
    assert len(result) == 5
    assert experts.episode_ids == {e.episode_id for e in result}
    assert experts.version == "reviewed-v1"
    expected_hash = hashlib.sha256(registered[0].read_bytes()).hexdigest()
    registered[0].write_text("{}", encoding="utf-8")
    assert experts.manifest_sha256 == expected_hash
    assert len(experts.resolve(episode("another-target"))) == 5


def test_downloaded_accepted_status_does_not_grant_human_approval(registered):
    registered[3]["groups"][0].update(approved=False, reviewer=None)
    with pytest.raises(ExpertError, match="approved"):
        library(registered).resolve(episode("target"))


@pytest.mark.parametrize("change", [
    {"task_code": "TASK-BOWL"}, {"action_id": "A_002"},
    {"collector_id": "another-collector"}, {"label": "other"},
    {"instruction": "Pick up the bowl"}, {"mcap_path": Path("missing-expert.mcap")},
])
def test_expert_metadata_and_media_must_match_registration(registered, change):
    registered[2]["expert-0"] = replace(registered[2]["expert-0"], **change)
    with pytest.raises(ExpertError):
        library(registered).resolve(episode("target"))


def test_target_cannot_be_expert_or_a_different_concrete_task(registered):
    experts = library(registered)
    with pytest.raises(ExpertError, match="own expert"):
        experts.resolve(episode("expert-0"))
    with pytest.raises(ExpertError, match="concrete task"):
        experts.resolve(episode("target", task_code="TASK-BOWL"))


@pytest.mark.parametrize("change", [
    {"reviewer": None}, {"approved": "true"}, {"policy": "any_action"},
    {"expert_episode_ids": ["expert-0"] * 5},
])
def test_ambiguous_or_invalid_approval_is_rejected(registered, change):
    registered[3]["groups"][0].update(change)
    with pytest.raises(ExpertError):
        library(registered)


def test_missing_or_insufficient_experts_are_expert_errors(registered):
    registered[1].get.side_effect = EpisodeNotFound("removed")
    with pytest.raises(ExpertError, match="unavailable"):
        library(registered).resolve(episode("target"))
    registered[3]["groups"][0]["expert_episode_ids"].pop()
    with pytest.raises(ExpertError, match="five"):
        library(registered).resolve(episode("target"))


def test_prepare_all_ten_actions_disjoint_but_unapproved():
    episodes = [
        episode(f"{action}-{label}-{i:02}", action_id=action, label=label)
        for action in ACTIONS for label in LABELS
        for i in range(10 if label == "correct" else 5)
    ]
    prepared = prepare_experiment(Mock(list_episodes=Mock(return_value=episodes)))
    experts = prepared["evaluation"]["expert_episode_ids"]
    tests = [sample["episode_id"] for sample in prepared["evaluation"]["samples"]]
    assert len(experts) == 50
    assert len(tests) == 400
    assert not set(experts).intersection(tests)
    assert len(prepared["report"]["actions"]) == 10
    assert all(group["approved"] is False for group in prepared["experts"]["groups"])
    assert prepared["report"]["baseline_ready"] is False
    assert all(report["baseline_candidate_complete"]
               for report in prepared["report"]["actions"].values())


def test_shortfalls_never_borrow_from_other_collectors_or_tasks():
    episodes = [episode(f"same-{i}") for i in range(6)]
    episodes += [episode(f"collector-{i}", collector_id="collector-2") for i in range(4)]
    episodes += [episode(f"rejected-{i}", review_status="Rejected") for i in range(10)]
    episodes += [episode(f"bowl-{i}", task_code="TASK-BOWL", label="blurred") for i in range(5)]
    prepared = prepare_experiment(Mock(list_episodes=Mock(return_value=episodes)))
    report = prepared["report"]["actions"]["A_001"]
    assert report["strict_group_correct_count"] == 6
    assert report["shortfalls_by_label"]["correct"] == 4
    assert report["shortfalls_by_label"]["blurred"] == 5
    assert not report["baseline_candidate_complete"]
    assert all(sample["cohort"] == "exploratory"
               for sample in prepared["evaluation"]["samples"] if sample["label"] == "blurred")
    assert prepared["report"]["actions"]["A_002"]["expert_shortfall"] == 5


def test_inconsistent_instructions_do_not_complete_a_candidate_group():
    episodes = [episode(f"cup-{i}") for i in range(5)]
    episodes += [episode(f"bowl-{i}", instruction="Pick up the bowl") for i in range(5)]
    report = prepare_experiment(Mock(list_episodes=Mock(return_value=episodes)))["report"]
    assert report["actions"]["A_001"]["strict_group_correct_count"] == 5
    assert report["actions"]["A_001"]["shortfalls_by_label"]["correct"] == 5


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
