"""Review data contracts, independent of HTTP, storage and media libraries."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CAMERAS = ("main", "left_wrist", "right_wrist")
CHECKS = ("object_match", "scene_match", "main_visibility", "image_quality",
          "action", "retry_free", "completeness")


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["pass", "fail", "unknown"]
    evidence_ids: list[str]


class CameraQuality(Check):
    description: str = Field(min_length=1)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: Literal["start", "action", "hold", "release", "failure", "end", "uncertain"]
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Observation] = Field(min_length=1)
    main_visibility_by_frame: dict[str, Literal["visible", "absent", "uncertain", "no_frame"]]
    checks: dict[str, Check]
    hold: Check | None
    reason: str = Field(min_length=1)


class QualityReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quality_by_camera: dict[str, CameraQuality]


