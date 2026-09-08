"""Episode review orchestration. Infrastructure errors remain operational errors."""

from __future__ import annotations

from datetime import datetime, timezone

from .. import __version__
from ..experts import EXPERT_POLICY
from ..media.mcap_reader import inspect_mcap
from ..media.sampling import WRIST_INTERVAL_S
from ..models import Assessment, ExpertError, ProviderError, ReviewResult, SampledVideo
from ..prompts import PROMPT_VERSION
from ..repository import dotted_get
from ..settings import Settings
from .integrity import check_fields, check_integrity
from .policy import POLICY_VERSION, decide


class ReviewService:
    def __init__(self, repository, sampler, experts, client, settings: Settings):
        self.repository = repository
        self.sampler = sampler
        self.experts = experts
        self.client = client
        self.settings = settings

    def review(
        self, episode_id: str, strategy: str = "uniform", camera_mode: str | None = None,
    ) -> ReviewResult:
        episode = self.repository.get(episode_id)
        camera_mode = self.settings.camera_mode if camera_mode is None else camera_mode
        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine_version": __version__,
            "prompt_version": PROMPT_VERSION,
            "policy_version": POLICY_VERSION,
            "model": self.settings.model,
            "strategy": strategy,
            "camera_mode": camera_mode,
            "candidate_interval_s": 2.0,
            "expert_interval_s": 1.0,
            "candidate_wrist_interval_s": WRIST_INTERVAL_S,
            "expert_wrist_interval_s": WRIST_INTERVAL_S,
            "camera_topic": self.settings.camera_topic,
            "max_image_size": self.settings.max_image_size,
            "max_frames_per_video": self.settings.max_frames,
            "max_request_images": self.settings.max_request_images,
            "correct_threshold": self.settings.correct_threshold,
            "expert_matching_policy": EXPERT_POLICY,
            "expert_version": self.experts.version,
            "expert_config_sha256": self.experts.manifest_sha256,
            "expert_ids": [],
            "expert_tasks": [],
        }
        assessments = {"integrity": check_fields(episode)}
        if assessments["integrity"].verdict == "incorrect":
            return self._result(episode, assessments, provenance)
        inspection = inspect_mcap(
            episode.mcap_path, required_topics=[self.settings.camera_topic]
        )
        assessments["integrity"] = check_integrity(episode, inspection)
        if assessments["integrity"].verdict != "correct":
            return self._result(episode, assessments, provenance)

        candidate = self.sampler.sample(
            episode, strategy=strategy, interval_s=2.0, camera_mode=camera_mode,
        )
        expert_episodes = []
        try:
            expert_episodes = self.experts.resolve(episode)
        except ExpertError:
            assessments["task"] = Assessment(
                verdict="uncertain", reason="没有可用的五条独立、人工批准且同原子动作的专家样例。",
                confidence=0.0,
            )

        provenance["expert_ids"] = [item.episode_id for item in expert_episodes]
        provenance["expert_tasks"] = [
            {"episode_id": item.episode_id, "action_id": item.action_id,
             "task_code": item.task_code, "instruction": item.instruction,
             "source_review_status": item.review_status,
             "source_reviewer": dotted_get(item.metadata, "task.review.reviewer"),
             "source_review_time": dotted_get(item.metadata, "task.review.review_time")}
            for item in expert_episodes
        ]
        expert_videos = [
            (item.instruction, self.sampler.sample(
                item, strategy="uniform", interval_s=1.0, camera_mode=camera_mode,
            ))
            for item in expert_episodes
        ]
        warnings = [*candidate.warnings]
        warnings.extend(
            f"expert_{index}: {warning}"
            for index, (_, video) in enumerate(expert_videos, 1) for warning in video.warnings
        )
        provenance["candidate_timestamps_s"] = [item.timestamp_s for item in candidate.frames]
        provenance["candidate_duration_s"] = candidate.duration_s
        provenance["candidate_motion"] = candidate.motion
        provenance["candidate_views"] = _view_sources(candidate)
        provenance["expert_timestamps_s"] = {
            item.episode_id: [frame.timestamp_s for frame in video.frames]
            for item, (_, video) in zip(expert_episodes, expert_videos, strict=True)
        }
        provenance["expert_views"] = {
            item.episode_id: _view_sources(video)
            for item, (_, video) in zip(expert_episodes, expert_videos, strict=True)
        }
        provenance["warnings"] = warnings
        # Check the largest actual request before spending on the generic call.
        total_frames = (len(candidate.frames) + sum(len(video.frames) for _, video in expert_videos)
                        if expert_videos else sum(frame.view == "main" for frame in candidate.frames))
        if total_frames > self.settings.max_request_images:
            raise ProviderError("image_budget_exceeded: raise max_request_images or select shorter clips")
        assessments["generic"] = self.client.assess_generic(episode.instruction, candidate)
        if expert_videos:
            assessments["task"] = self.client.assess_task(episode.instruction, candidate, expert_videos)
        return self._result(episode, assessments, provenance, warnings)

    def _result(self, episode, assessments, provenance, warnings=None) -> ReviewResult:
        return decide(
            episode, assessments, threshold=self.settings.correct_threshold,
            warnings=warnings, provenance=provenance,
        )


def _view_sources(video: SampledVideo) -> dict:
    sources = video.motion.get("views", {})
    return {
        view: {
            "topic": sources.get(view, {}).get("topic", video.camera_topic if view == "main" else None),
            "timestamps_s": [frame.timestamp_s for frame in video.frames if frame.view == view],
        }
        for view in sorted({frame.view for frame in video.frames} | sources.keys())
    }
