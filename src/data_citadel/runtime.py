"""Shared setup and artifact output for the CLI and web interface."""

import json
from pathlib import Path
from uuid import uuid4

from .experts import ExpertLibrary
from .media.sampling import VideoSampler
from .models import Episode, ReviewResult
from .qwen import QwenClient
from .repository import EpisodeRepository
from .review.service import ReviewService
from .settings import Settings


def build_sampler(settings: Settings) -> VideoSampler:
    return VideoSampler(
        camera_topic=settings.camera_topic, max_frames=settings.max_frames,
        max_image_size=settings.max_image_size, cache_dir=settings.artifacts_dir / "cache",
    )


def build_service(settings: Settings) -> ReviewService:
    repository = EpisodeRepository(settings.dataset_root)
    return ReviewService(
        repository,
        build_sampler(settings),
        ExpertLibrary(settings.experts_path, repository),
        QwenClient(settings),
        settings,
    )


def episode_summary(episode: Episode) -> dict:
    return {
        "episode_id": episode.episode_id,
        "action_id": episode.action_id,
        "task_code": episode.task_code,
        "instruction": episode.instruction,
        "reference_label": episode.label,
    }


def save_review(result: ReviewResult, artifacts_dir: Path) -> Path:
    directory = artifacts_dir / "reviews"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(result.model_dump_json(indent=2))
    return path


def write_json(path: Path, value: dict) -> None:
    """Create an artifact without overwriting an earlier experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
