"""Episode review orchestration. Infrastructure errors remain operational errors."""

from __future__ import annotations

from datetime import datetime, timezone

from .. import __version__
from ..media.mcap_reader import inspect_mcap
from ..models import Assessment, ExpertError, ProviderError, ReviewResult
from ..prompts import PROMPT_VERSION
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

    def review(self, episode_id: str, strategy: str = "uniform") -> ReviewResult:
        episode = self.repository.get(episode_id)
        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine_version": __version__,
            "prompt_version": PROMPT_VERSION,
            "policy_version": POLICY_VERSION,
            "model": self.settings.model,
            "strategy": strategy,
            "candidate_interval_s": 2.0,
            "expert_interval_s": 1.0,
            "camera_topic": self.settings.camera_topic,
            "max_image_size": self.settings.max_image_size,
            "max_frames_per_video": self.settings.max_frames,
            "max_request_images": self.settings.max_request_images,
            "correct_threshold": self.settings.correct_threshold,
            "expert_ids": [],
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

        try:
            expert_episodes = self.experts.resolve(episode)
        except ExpertError:
            assessments["task"] = Assessment(
                verdict="uncertain", reason="没有可用的五条独立、人工批准且任务匹配的专家样例。",
                confidence=0.0,
            )
            return self._result(episode, assessments, provenance)

        provenance["expert_ids"] = [item.episode_id for item in expert_episodes]
        provenance["expert_version"] = self.experts.version
        provenance["expert_config_sha256"] = self.experts.manifest_sha256
        candidate = self.sampler.sample(episode, strategy=strategy, interval_s=2.0)
        expert_videos = [
            (item.instruction, self.sampler.sample(item, strategy="uniform", interval_s=1.0))
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
        provenance["expert_timestamps_s"] = {
            item.episode_id: [frame.timestamp_s for frame in video.frames]
            for item, (_, video) in zip(expert_episodes, expert_videos, strict=True)
        }
        provenance["warnings"] = warnings
        # Check the larger request first to avoid spending on an unusable review.
        total_frames = len(candidate.frames) + sum(len(video.frames) for _, video in expert_videos)
        if total_frames > self.settings.max_request_images:
            raise ProviderError("image_budget_exceeded: raise max_request_images or select shorter clips")
        assessments["generic"] = self.client.assess_generic(episode.instruction, candidate)
        assessments["task"] = self.client.assess_task(episode.instruction, candidate, expert_videos)
        return self._result(episode, assessments, provenance, warnings)

    def _result(self, episode, assessments, provenance, warnings=None) -> ReviewResult:
        return decide(
            episode, assessments, threshold=self.settings.correct_threshold,
            warnings=warnings, provenance=provenance,
        )
