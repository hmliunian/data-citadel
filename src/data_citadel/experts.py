"""Approved expert references and disjoint experiments grouped by atomic action."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import ACTIONS, LABELS, CitadelError, Episode, ExpertError
from .repository import reviewed_correct

if TYPE_CHECKING:
    from .repository import EpisodeRepository


EXPERT_POLICY = "same_action_v2"


class ExpertLibrary:
    """Five approved references teach an action across objects and collectors."""

    def __init__(self, path: Path, repository: EpisodeRepository):
        self.path = Path(path)
        self.repository = repository
        try:
            manifest_bytes = self.path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except FileNotFoundError:
            self.version = "unconfigured"
            self.manifest_sha256 = None
            self.groups = []
            self.episode_ids = frozenset()
            return
        except (OSError, ValueError) as exc:
            raise ExpertError("Cannot read expert manifest") from exc
        if not isinstance(manifest, dict) or not _text(manifest.get("version")):
            raise ExpertError("Expert manifest needs a nonempty version")
        groups = manifest.get("groups")
        if not isinstance(groups, list) or any(not isinstance(g, dict) for g in groups):
            raise ExpertError("Expert manifest groups must be a list of objects")
        used: set[str] = set()
        actions: set[str] = set()
        for group in groups:
            if group.get("policy") != EXPERT_POLICY:
                raise ExpertError(
                    f"Unsupported expert policy; run data-citadel prepare to use {EXPERT_POLICY}"
                )
            action = group.get("action_id")
            if not _text(action) or action in actions:
                raise ExpertError("Expert groups need unique action_id values")
            actions.add(action)
            ids = group.get("expert_episode_ids")
            if not isinstance(ids, list) or any(not _text(eid) for eid in ids):
                raise ExpertError("expert_episode_ids must be a list of nonempty IDs")
            if len(set(ids)) != len(ids) or used.intersection(ids):
                raise ExpertError("Duplicate expert episode ID")
            if len(ids) > 5:
                raise ExpertError("Each expert group may contain at most five candidates")
            used.update(ids)
            if not isinstance(group.get("approved"), bool):
                raise ExpertError("Expert groups need an explicit boolean approved field")
            source = group.get("approval_source", "manual")
            if source not in ("manual", "dataset_review"):
                raise ExpertError("Unknown expert approval source")
            if source == "manual" and group["approved"] and not _text(group.get("reviewer")):
                raise ExpertError("Manually approved experts require a named human reviewer")
        self.version: str = manifest["version"]
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.groups: list[dict[str, Any]] = groups
        self.episode_ids = frozenset(used)

    def resolve(self, episode: Episode) -> list[Episode]:
        matches = [group for group in self.groups
                   if group["action_id"] == episode.action_id and group["approved"]]
        if len(matches) != 1:
            raise ExpertError("Need one approved expert group for this action_id")
        group = matches[0]
        ids = group["expert_episode_ids"]
        if len(ids) != 5:
            raise ExpertError("Exactly five approved expert episodes are required")
        if episode.episode_id in ids:
            raise ExpertError("The target episode cannot be one of its own expert references")
        experts: list[Episode] = []
        for episode_id in ids:
            try:
                expert = self.repository.get(episode_id)
            except (CitadelError, LookupError, ValueError) as exc:
                raise ExpertError(f"Expert episode is unavailable: {episode_id}") from exc
            if expert.action_id != episode.action_id:
                raise ExpertError("Expert episode does not match the registered action_id")
            if not _text(expert.instruction):
                raise ExpertError("Expert instruction is missing")
            correct = (reviewed_correct(expert) if group.get("approval_source") == "dataset_review"
                       else expert.label == "correct")
            if not correct:
                raise ExpertError("Expert episode lacks the required correct review source")
            if not expert.mcap_path.is_file() or expert.mcap_path.stat().st_size == 0:
                raise ExpertError(f"Expert video is unavailable: {episode_id}")
            experts.append(expert)
        return experts


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def prepare_experiment(
    repository: EpisodeRepository,
    *,
    expert_count: int = 5,
    correct_test_count: int = 5,
    error_test_count: int = 5,
    version: str = "v2",
    include_exploratory: bool = True,
) -> dict[str, Any]:
    """Reuse recorded human approvals; split each action by stable episode ID.

    The first five reviewed correct episodes are experts, the next five are
    correct tests. Objects, task codes and collectors do not split an action.
    """
    if type(expert_count) is not int or expert_count != 5:
        raise ValueError("The review protocol requires exactly five expert references")
    if any(type(n) is not int or n < 1 for n in (correct_test_count, error_test_count)):
        raise ValueError("Test counts must be positive integers")
    if not _text(version):
        raise ValueError("Experiment version must be nonempty")
    episodes = sorted(repository.list_episodes(), key=lambda episode: episode.episode_id)
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("Duplicate episode IDs in repository")
    actions = sorted(set(ACTIONS).union(episode.action_id for episode in episodes))
    groups: list[dict[str, Any]] = []
    samples: list[dict[str, str]] = []
    reports: dict[str, Any] = {}
    all_expert_ids: set[str] = set()
    for action_id in actions:
        action_episodes = [episode for episode in episodes if episode.action_id == action_id]
        candidates = [episode for episode in action_episodes if reviewed_correct(episode)]
        experts = candidates[:expert_count]
        expert_ids = {episode.episode_id for episode in experts}
        all_expert_ids.update(expert_ids)
        baseline = candidates[expert_count:expert_count + correct_test_count]
        for label in LABELS:
            if label != "correct":
                baseline.extend([episode for episode in action_episodes
                                 if episode.label == label][:error_test_count])
        baseline_ids = {episode.episode_id for episode in baseline}
        for episode in action_episodes:
            if episode.episode_id in expert_ids or episode.label not in LABELS:
                continue
            is_baseline = episode.episode_id in baseline_ids
            if is_baseline or include_exploratory:
                samples.append({
                    "episode_id": episode.episode_id,
                    "action_id": action_id,
                    "label": episode.label,
                    "cohort": "baseline" if is_baseline else "exploratory",
                })
        if experts:
            groups.append({
                "action_id": action_id,
                "collector_ids": sorted({e.collector_id for e in experts if _text(e.collector_id)}),
                "approved": len(experts) == expert_count,
                "approval_source": "dataset_review",
                "policy": EXPERT_POLICY,
                "expert_episode_ids": [episode.episode_id for episode in experts],
            })
        selected = Counter(episode.label for episode in baseline)
        available = Counter(episode.label for episode in action_episodes)
        deficits = {
            label: max(0, (correct_test_count if label == "correct" else error_test_count)
                       - selected[label])
            for label in LABELS
        }
        complete = len(experts) == expert_count and not any(deficits.values())
        reports[action_id] = {
            "available_episodes": len(action_episodes),
            "available_by_label": {label: available[label] for label in LABELS},
            "unlabeled_episodes": sum(e.label not in LABELS for e in action_episodes),
            "correct_candidate_count": len(candidates),
            "collector_count": len({e.collector_id for e in candidates if _text(e.collector_id)}),
            "expert_candidates": len(experts),
            "expert_shortfall": expert_count - len(experts),
            "baseline_selected_by_label": {label: selected[label] for label in LABELS},
            "shortfalls_by_label": deficits,
            "baseline_candidate_complete": complete,
            "baseline_ready": complete,
        }
    report = {
        "scope_actions": actions,
        "available_episodes": len(episodes),
        "expert_candidates": len(all_expert_ids),
        "evaluation_samples": len(samples),
        "baseline_samples": sum(sample["cohort"] == "baseline" for sample in samples),
        "exploratory_samples": sum(sample["cohort"] == "exploratory" for sample in samples),
        "baseline_ready": all(report["baseline_ready"] for report in reports.values()),
        "approval_source": "dataset_review",
        "approval_note": "Uses existing Accepted reviews and recorded reviewers; no second review is asserted.",
        "actions": reports,
    }
    return {
        "experts": {"version": version, "groups": groups},
        "evaluation": {
            "version": version,
            "expert_episode_ids": sorted(all_expert_ids),
            "samples": samples,
            "coverage": report,
            "label_notes": {
                "retry_then_success": "Downloaded category; the label alone does not verify final success.",
            },
        },
        "report": report,
    }
