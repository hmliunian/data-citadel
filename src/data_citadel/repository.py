"""Read-only index of complete, locally verified episode bundles."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .models import LABELS, DatasetError, Episode, EpisodeNotFound


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_ACTION = re.compile(r"A_[0-9]{3}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def dotted_get(document: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Resolve nested, dotted, or mixed keys, rejecting conflicting encodings."""
    values = []
    if name in document:
        values.append(document[name])
    for index, char in enumerate(name):
        if char == "." and isinstance(document.get(name[:index]), Mapping):
            marker = object()
            value = dotted_get(document[name[:index]], name[index + 1 :], marker)
            if value is not marker:
                values.append(value)
    if not values:
        return default
    if any(value != values[0] for value in values[1:]):
        raise DatasetError(f"Conflicting representations of metadata field {name}")
    return values[0]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw, object_pairs_hook=_unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  ValueError(f"Non-finite JSON number: {value}")))
    except (OSError, UnicodeError, ValueError) as exc:
        raise DatasetError(f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(document, dict):
        raise DatasetError(f"Expected JSON object in {path.name}")
    return document, raw


class EpisodeRepository:
    """Index receipt-backed bundles; labels are exclusively evaluation metadata.

    Unfinished directories without a receipt are ignored. A malformed receipt or
    a changed file is an error, rather than silently reducing evaluation coverage.
    MCAP content was hashed by the downloader; startup checks its size/mtime and
    the receipt, and hashes the small sidecar again. Full MCAP CRC validation is
    performed by the media reader when consuming messages.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise DatasetError("Dataset directory does not exist")
        self._episodes: dict[str, Episode] | None = None

    def _secure(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise DatasetError("Dataset path escapes the configured root")
        current = path
        while current != self.root:
            if current.is_symlink():
                raise DatasetError("Symbolic links are not supported inside the dataset")
            if current.parent == current:
                raise DatasetError("Dataset path escapes the configured root")
            current = current.parent
        return resolved

    def _load(self) -> dict[str, Episode]:
        if self._episodes is not None:
            return self._episodes
        atomic_root = self.root / "atomic" if (self.root / "atomic").is_dir() else self.root
        result: dict[str, Episode] = {}
        for receipt_path in sorted(atomic_root.glob("*/*/*/verification.json")):
            self._secure(receipt_path)
            action, label, episode_id = receipt_path.parent.relative_to(atomic_root).parts
            if not _ACTION.fullmatch(action) or not _IDENTIFIER.fullmatch(episode_id):
                raise DatasetError("Invalid action or episode directory identifier")
            if label not in LABELS:
                raise DatasetError(f"Unknown evaluation category: {label}")
            if episode_id in result:
                raise DatasetError(f"Duplicate episode id: {episode_id}")
            receipt, _ = _read_json(receipt_path)
            if (receipt.get("version") not in (1, 2)
                    or receipt.get("episode_uuid") != episode_id
                    or receipt.get("action_id") != action
                    or receipt.get("stratum") != label
                    or receipt.get("mcap_doctor") != "passed"):
                raise DatasetError(f"Invalid verification receipt for {episode_id}")
            files = receipt.get("files")
            if not isinstance(files, dict):
                raise DatasetError(f"Missing verified files for {episode_id}")
            sidecar = receipt_path.parent / f"{episode_id}.json"
            mcap_path = receipt_path.parent / "episode.mcap"
            for path in (sidecar, mcap_path):
                self._secure(path)
                entry = files.get(path.name)
                if not path.is_file() or not isinstance(entry, dict):
                    raise DatasetError(f"Incomplete verified bundle: {episode_id}")
                stat = path.stat()
                if (entry.get("name") != path.name or stat.st_size <= 0
                        or stat.st_size != entry.get("size")
                        or stat.st_size != entry.get("remote_size")
                        or stat.st_mtime_ns != entry.get("mtime_ns")
                        or not isinstance(entry.get("sha256"), str)
                        or not _SHA256.fullmatch(entry["sha256"])):
                    raise DatasetError(f"Verified file has changed: {episode_id}/{path.name}")
            document, raw = _read_json(sidecar)
            if hashlib.sha256(raw).hexdigest() != files[sidecar.name]["sha256"]:
                raise DatasetError(f"Sidecar checksum mismatch: {episode_id}")
            actual_action = dotted_get(document, "task.action_id")
            if actual_action is not None and actual_action != action:
                raise DatasetError(f"Action identity mismatch for {episode_id}")

            def text_field(name: str) -> str:
                value = dotted_get(document, name, "")
                return value if isinstance(value, str) else ""

            result[episode_id] = Episode(
                episode_id=episode_id, action_id=action,
                task_code=text_field("task.task_code"),
                instruction=text_field("task.action_text.rendered_zh")
                or text_field("task.action_text.rendered_en"),
                collector_id=text_field("task.collector.user"),
                mcap_path=mcap_path, sidecar_path=sidecar,
                label=label, review_status=text_field("task.review.status") or None,
                metadata=document,
            )
        self._episodes = result
        return result

    def list_episodes(self, action_id: str | None = None) -> list[Episode]:
        if action_id is not None and not _ACTION.fullmatch(action_id):
            raise DatasetError("Invalid action id")
        return [episode for episode in self._load().values()
                if action_id is None or episode.action_id == action_id]

    def get(self, episode_id: str) -> Episode:
        if not isinstance(episode_id, str) or not _IDENTIFIER.fullmatch(episode_id):
            raise EpisodeNotFound("Invalid episode id")
        try:
            return self._load()[episode_id]
        except KeyError as exc:
            raise EpisodeNotFound(f"Unknown episode id: {episode_id}") from exc

    def inventory(self) -> dict[str, Any]:
        episodes = self.list_episodes()
        counts: dict[str, Counter] = defaultdict(Counter)
        groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
        total = Counter()
        for episode in episodes:
            counts[episode.action_id][episode.label] += 1
            total[episode.label] += 1
            if (episode.label == "correct" and episode.review_status == "Accepted"
                    and episode.task_code and episode.collector_id):
                groups[(episode.action_id, episode.task_code)][episode.collector_id] += 1
        return {
            "total_episodes": len(episodes), "action_count": len(counts),
            "category_counts": {label: total[label] for label in LABELS},
            "actions": {action: {label: count[label] for label in LABELS}
                        for action, count in sorted(counts.items())},
            "expert_availability": [
                {"action_id": action, "task_code": task,
                 "collector_group_counts": sorted(collectors.values(), reverse=True),
                 "max_same_collector_correct": max(collectors.values()),
                 "groups_with_10_correct": sum(count >= 10 for count in collectors.values())}
                for (action, task), collectors in sorted(groups.items())
            ],
            "expert_candidates_are_human_verified": False,
        }
