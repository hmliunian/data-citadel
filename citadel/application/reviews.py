"""One episode per review; scheduling and GT evaluation live elsewhere."""
import uuid
from datetime import datetime, timezone
from typing import Callable

from citadel.configuration import ConfigSnapshot, PromptBundle
from citadel.domain.decision import decide
from citadel.domain.tasks import profile_for
from .prompts import PromptBuilder
from .ports import MediaPreparer, ModelGateway, TaskResources


class ReviewPipeline:
    def __init__(self, gateway: ModelGateway, prompts: PromptBuilder):
        self.gateway, self.prompts = gateway, prompts

    def execute(self, work, resources, profile, media, context, progress, result):
        progress("model")
        response = self.gateway.complete(self.prompts.review(work, resources, profile, media), context)
        result["model_call"] = {k: v for k, v in response.items() if k != "data"}
        progress("quality")
        quality = self.gateway.complete(self.prompts.quality(work, media),
                                        {**context, "stage": "quality"}, quality_only=True)
        result["quality_call"] = {k: v for k, v in quality.items() if k != "data"}
        if set(quality["data"]) != {"quality_by_camera"}:
            raise ValueError("Quality response must contain only camera quality")
        progress("evidence")
        result.update(decide({**response["data"], **quality["data"]}, media, profile))


class ReviewService:
    def __init__(self, source, artifacts, resources: TaskResources, media: MediaPreparer,
                 gateway_factory: Callable[[ConfigSnapshot], ModelGateway], experiment):
        self.source, self.artifacts, self.resources = source, artifacts, resources
        self.media, self.gateway_factory, self.experiment = media, gateway_factory, experiment

    def prepare(self, episode_id, snapshot, progress):
        progress("source")
        source = self.source.source(episode_id)
        self.experiment.gate(self.experiment.split_of(episode_id), snapshot)
        progress("resources")
        resources = self.resources.get(source.task_code)
        profile = profile_for(resources, snapshot.data["profiles"])
        progress("media")
        media = self.media.prepare(source, snapshot.data["sampling"])
        return source, resources, profile, media

    def review(self, episode_id, snapshot, retry_failed=False, progress=lambda stage: None):
        with self.artifacts.episode_lock(episode_id):
            stage = "source"
            def advance(value):
                nonlocal stage
                stage = value
                progress(value)
            result = {"result_id": uuid.uuid4().hex, "episode_id": episode_id,
                      "configuration_sha256": snapshot.sha256,
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "status": "failed", "label": None}
            self.artifacts.save_snapshot(snapshot)
            try:
                source, resources, profile, media = self.prepare(episode_id, snapshot, advance)
                result.update(task_code=source.task_code, split=self.experiment.split_of(episode_id),
                              resources_sha256=resources["sha256"], task_profile=profile["name"],
                              media_path=f"media/{episode_id}/media.json")
                existing = self.artifacts.latest(episode_id, snapshot.sha256)
                if existing and not (retry_failed and existing["status"] == "failed"):
                    if existing.get("resources_sha256") not in (None, resources["sha256"]):
                        raise ValueError("References changed; prepare a new run")
                    return {**existing, "cached": True}
                pipeline = ReviewPipeline(self.gateway_factory(snapshot),
                                          PromptBuilder(PromptBundle(**snapshot.data["prompts"]), self.artifacts.image))
                context = {k: result[k] for k in
                           ("result_id", "episode_id", "task_code", "split", "configuration_sha256")}
                pipeline.execute(self.artifacts.work, resources, profile, media, context, advance, result)
            except Exception as exc:
                result.update(reason=f"处理失败：{stage} / {type(exc).__name__}",
                              error={"stage": stage, "type": type(exc).__name__})
            self.artifacts.save_result(result)
            return {**result, "cached": False}

    def preview(self, episode_id, snapshot, progress=lambda stage: None):
        with self.artifacts.episode_lock(episode_id):
            _, resources, profile, media = self.prepare(episode_id, snapshot, progress)
            self.artifacts.save_snapshot(snapshot)
            builder = PromptBuilder(PromptBundle(**snapshot.data["prompts"]), self.artifacts.image)
            gateway = self.gateway_factory(snapshot)
            return {"episode_id": episode_id, "configuration_sha256": snapshot.sha256,
                    "configuration": snapshot.data,
                    "task": gateway.preview(builder.review(self.artifacts.work, resources, profile, media)),
                    "quality": gateway.preview(builder.quality(self.artifacts.work, media), quality_only=True)}
