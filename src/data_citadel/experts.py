"""Explicit human approval and reproducible, disjoint experiment candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import ACTIONS, LABELS, CitadelError, Episode, ExpertError

if TYPE_CHECKING:
    from .repository import EpisodeRepository


EXPERT_POLICY = "same_task_same_collector_v1"


class ExpertLibrary:
    """A candidate file is usable only after a reviewer explicitly approves it.

    Experts must share a concrete task and collector. A target may have a
    different collector, but may never have a different concrete task.
    """

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
        keys: set[tuple[str, str, str]] = set()
        for group in groups:
            key = tuple(group.get(k) for k in ("action_id", "task_code", "collector_id"))
            if any(not _text(part) for part in key) or key in keys:
                raise ExpertError("Expert groups need unique action/task/collector keys")
            keys.add(key)
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
            if group.get("policy") != EXPERT_POLICY:
                raise ExpertError(f"Expert policy must be {EXPERT_POLICY}")
            if group["approved"] and not _text(group.get("reviewer")):
                raise ExpertError("Approved experts require a named human reviewer")
        self.version: str = manifest["version"]
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.groups: list[dict[str, Any]] = groups
        self.episode_ids = frozenset(used)

    def resolve(self, episode: Episode) -> list[Episode]:
        matches = [
            group for group in self.groups
            if group["action_id"] == episode.action_id
            and group["task_code"] == episode.task_code
            and group["approved"]
        ]
        same_collector = [g for g in matches if g["collector_id"] == episode.collector_id]
        if same_collector:
            matches = same_collector
        if len(matches) != 1:
            raise ExpertError("Need exactly one approved expert group for this concrete task")
        group = matches[0]
        ids = group["expert_episode_ids"]
        if len(ids) != 5:
            raise ExpertError("Exactly five human-confirmed expert episodes are required")
        if episode.episode_id in ids:
            raise ExpertError("The target episode cannot be one of its own expert references")
        experts: list[Episode] = []
        for episode_id in ids:
            try:
                expert = self.repository.get(episode_id)
            except (CitadelError, LookupError, ValueError) as exc:
                raise ExpertError(f"Expert episode is unavailable: {episode_id}") from exc
            if (
                expert.action_id != group["action_id"]
                or expert.task_code != group["task_code"]
                or expert.collector_id != group["collector_id"]
            ):
                raise ExpertError("Expert episode does not match the registered task and collector")
            if not _text(expert.instruction) or _instruction(expert) != _instruction(episode):
                raise ExpertError("Expert and target instructions differ for the registered task")
            if expert.label != "correct":
                raise ExpertError("Expert episode conflicts with the registered correct label")
            if not expert.mcap_path.is_file():
                raise ExpertError(f"Expert video is unavailable: {episode_id}")
            experts.append(expert)
        return experts


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _instruction(episode: Episode) -> str:
    return " ".join(episode.instruction.split())


def prepare_experiment(
    repository: EpisodeRepository,
    *,
    expert_count: int = 5,
    correct_test_count: int = 5,
    error_test_count: int = 5,
    version: str = "v1",
    include_exploratory: bool = True,
) -> dict[str, Any]:
    """Return manifests and deficits; never turn downloaded labels into approval.

    Select the largest Accepted correct group per action using a deterministic
    task/collector/instruction/episode sort. Correct baseline tests share that
    group; error baseline tests share its task. Other labels are exploratory.
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
        correct_groups: dict[tuple[str, str, str], list[Episode]] = defaultdict(list)
        for episode in action_episodes:
            if (
                episode.label == "correct"
                and (episode.review_status or "").casefold() == "accepted"
                and _text(episode.task_code)
                and _text(episode.collector_id)
                and _text(episode.instruction)
            ):
                key = (episode.task_code, episode.collector_id, _instruction(episode))
                correct_groups[key].append(episode)
        task_code = collector_id = instruction = None
        candidates: list[Episode] = []
        if correct_groups:
            key = min(correct_groups, key=lambda key: (-len(correct_groups[key]), key))
            task_code, collector_id, instruction = key
            candidates = correct_groups[key]
        experts = candidates[:expert_count]
        expert_ids = {episode.episode_id for episode in experts}
        all_expert_ids.update(expert_ids)
        baseline = candidates[expert_count:expert_count + correct_test_count]
        for label in LABELS:
            if label != "correct" and task_code is not None:
                baseline.extend([
                    episode for episode in action_episodes
                    if episode.task_code == task_code and episode.label == label
                ][:error_test_count])
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
                "task_code": task_code,
                "collector_id": collector_id,
                "instruction": instruction,
                "approved": False,
                "reviewer": None,
                "policy": EXPERT_POLICY,
                "expert_episode_ids": [episode.episode_id for episode in experts],
                "candidate_source": "dataset label correct and review status Accepted",
            })
        selected = Counter(episode.label for episode in baseline)
        available = Counter(episode.label for episode in action_episodes)
        deficits = {
            label: max(0, (correct_test_count if label == "correct" else error_test_count)
                       - selected[label])
            for label in LABELS
        }
        reports[action_id] = {
            "available_episodes": len(action_episodes),
            "available_by_label": {label: available[label] for label in LABELS},
            "unlabeled_episodes": sum(e.label not in LABELS for e in action_episodes),
            "selected_task_code": task_code,
            "selected_collector_id": collector_id,
            "strict_group_correct_count": len(candidates),
            "expert_candidates": len(experts),
            "expert_shortfall": expert_count - len(experts),
            "baseline_selected_by_label": {label: selected[label] for label in LABELS},
            "shortfalls_by_label": deficits,
            "baseline_candidate_complete": len(experts) == expert_count and not any(deficits.values()),
            "baseline_ready": False,
        }
    report = {
        "scope_actions": actions,
        "available_episodes": len(episodes),
        "expert_candidates": len(all_expert_ids),
        "evaluation_samples": len(samples),
        "baseline_samples": sum(sample["cohort"] == "baseline" for sample in samples),
        "exploratory_samples": sum(sample["cohort"] == "exploratory" for sample in samples),
        "baseline_ready": False,
        "approval_required": "A human must verify all five expert videos and fill approved/reviewer.",
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
