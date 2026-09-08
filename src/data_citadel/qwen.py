"""Synchronous Qwen-VL adapter with bounded input and strict output validation."""

from __future__ import annotations

import base64
from dataclasses import replace
import json
import math
import time
from collections.abc import Sequence
from typing import Any, get_args

import httpx

from .models import Assessment, CameraView, ProviderError, SampledVideo
from .prompts import COMMON, GENERIC, TASK
from .settings import Settings

GENERIC_CODES = ("blurred", "other")


class QwenClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.request_timeout_s)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def assess_generic(self, instruction: str, video: SampledVideo) -> Assessment:
        source = video.motion.get("views", {}).get("main", {})
        main = replace(
            video, frames=[frame for frame in video.frames if frame.view == "main"],
            duration_s=source.get("duration_s", video.duration_s),
        )
        return self._assess(instruction, main, (), GENERIC)

    def assess_task(
        self, instruction: str, video: SampledVideo,
        experts: Sequence[tuple[str, SampledVideo]],
    ) -> Assessment:
        if len(experts) != 5:
            raise ProviderError("exactly_five_expert_videos_required")
        return self._assess(instruction, video, experts, TASK)

    def _assess(
        self, instruction: str, video: SampledVideo,
        experts: Sequence[tuple[str, SampledVideo]], prompt: str,
    ) -> Assessment:
        videos = [video, *(expert for _, expert in experts)]
        if "vl" not in self.settings.model.lower():
            raise ProviderError("a_vision_capable_qwen_vl_model_is_required")
        if sum(len(item.frames) for item in videos) > self.settings.max_request_images:
            raise ProviderError("image_budget_exceeded: raise max_request_images or select shorter clips")
        for item in videos:
            self._validate_video(item)
        multiview = any(frame.view != "main" for item in videos for frame in item.frames)
        schema = Assessment.model_json_schema()
        if multiview:
            evidence_schema = schema["$defs"]["Evidence"]
            evidence_schema["required"].append("view")
            evidence_schema["properties"]["view"] = {"type": "string", "enum": list(get_args(CameraView))}
        if prompt == GENERIC:
            schema["$defs"]["Finding"]["properties"]["code"]["enum"] = list(GENERIC_CODES)
            schema["properties"]["retry_outcome"]["enum"] = ["none"]

        content: list[dict[str, Any]] = []
        for index, (expert_instruction, expert) in enumerate(experts, 1):
            content.extend(self._video_content(f"EXPERT {index}", expert_instruction, expert))
        content.extend(self._video_content("CANDIDATE", instruction, video))
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": COMMON + prompt + "\nJSON Schema:\n"
                 + json.dumps(schema, ensure_ascii=False)},
                {"role": "user", "content": content},
            ],
            # Qwen-VL-Max supports JSON object mode, not strict JSON Schema mode.
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 4096,
        }
        response = self._request(payload)
        try:
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("incomplete model response")
            raw = choice["message"]["content"]
            if not isinstance(raw, str):
                raise ValueError("content must be JSON text")
            assessment = Assessment.model_validate_json(raw, strict=True)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise ProviderError("invalid_or_incomplete_model_response") from None

        observed = {
            view: {frame.timestamp_s for frame in video.frames if frame.view == view}
            for view in {frame.view for frame in video.frames}
        }
        if prompt == GENERIC:
            for interval in video.motion.get("stationary_intervals", []):
                observed["main"].update(
                    value for key in ("start_s", "end_s")
                    if isinstance(value := interval.get(key), (int, float))
                    and math.isfinite(value) and 0 <= value <= video.duration_s
                )
        evidence = [*assessment.evidence]
        for finding in assessment.findings:
            evidence.extend(finding.evidence)
        for item in evidence:
            if (
                item.timestamp_s is None or not math.isfinite(item.timestamp_s)
                or not 0 <= item.timestamp_s <= video.duration_s + 1e-3
            ):
                raise ProviderError("model_evidence_outside_candidate_timeline")
            if item.view is None and multiview:
                raise ProviderError("multiview_model_evidence_requires_view")
            view = item.view or "main"
            if view not in observed:
                raise ProviderError("model_evidence_view_not_observed_in_candidate")
            nearest = min(observed[view], key=lambda value: abs(value - item.timestamp_s))
            if abs(nearest - item.timestamp_s) > 1e-3:
                raise ProviderError("model_evidence_not_observed_in_candidate_view")
            # Normalize millisecond rounding only within the declared source view.
            item.view, item.timestamp_s = view, nearest
        if prompt == GENERIC:
            if any(finding.code not in GENERIC_CODES for finding in assessment.findings):
                raise ProviderError("generic_review_invalid_finding_code")
            if assessment.retry_outcome != "none":
                raise ProviderError("generic_review_must_not_classify_retry_outcome")
        return assessment

    @staticmethod
    def _validate_video(video: SampledVideo) -> None:
        if not video.frames or not math.isfinite(video.duration_s) or video.duration_s < 0:
            raise ProviderError("invalid_sampled_video")
        if not any(frame.view == "main" for frame in video.frames):
            raise ProviderError("main_camera_frames_required")
        previous = 0.0
        for frame in video.frames:
            if (
                frame.view not in get_args(CameraView) or not math.isfinite(frame.timestamp_s)
                or not previous <= frame.timestamp_s <= video.duration_s or not frame.jpeg
            ):
                raise ProviderError("invalid_sampled_video")
            previous = frame.timestamp_s

    @staticmethod
    def _video_content(name: str, instruction: str, video: SampledVideo) -> list[dict[str, Any]]:
        # Explicit allowlist: never serialize Episode, metadata, paths or labels.
        header: dict[str, Any] = {
            "video": name, "task_instruction": instruction, "duration_s": video.duration_s,
            "sampled_timestamps_by_view": {
                view: [frame.timestamp_s for frame in video.frames if frame.view == view]
                for view in sorted({frame.view for frame in video.frames})
            },
        }
        if name == "CANDIDATE":
            header["motion_view"] = "main"
            header["motion"] = {
                key: video.motion[key]
                for key in (
                    "analyzed_frames", "coverage_s", "max_gap_s", "threshold_mean_abs_diff",
                    "stationary_intervals", "max_stationary_duration_s",
                )
                if key in video.motion
            }
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(header, ensure_ascii=False)}
        ]
        for frame in video.frames:
            content.append({"type": "text", "text":
                            f"{name} view={frame.view} timestamp_s={frame.timestamp_s}"})
            encoded = base64.b64encode(frame.jpeg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        return content

    def _request(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.settings.api_key()}"}
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        for attempt in range(self.settings.max_retries + 1):
            response = None
            try:
                response = self._client.post(
                    url, headers=headers, json=payload, timeout=self.settings.request_timeout_s
                )
            except httpx.TransportError:
                if attempt == self.settings.max_retries:
                    raise ProviderError("model_provider_transport_error") from None
            else:
                if 200 <= response.status_code < 300:
                    return response
                retryable = response.status_code in (408, 429) or response.status_code >= 500
                if not retryable or attempt == self.settings.max_retries:
                    raise ProviderError(f"model_provider_http_{response.status_code}") from None
            delay = min(2.0**attempt, 5.0)
            if response is not None:
                try:
                    retry_after = float(response.headers.get("retry-after", delay))
                    if math.isfinite(retry_after):
                        delay = min(max(retry_after, 0.0), 5.0)
                except ValueError:
                    pass
            time.sleep(delay)
        raise ProviderError("model_provider_retry_limit")
