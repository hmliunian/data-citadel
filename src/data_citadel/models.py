"""Shared contracts. Evaluation labels never belong in model prompts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["correct", "incorrect", "uncertain"]
CameraMode = Literal["main", "main_wrist"]
CameraView = Literal["main", "left_wrist", "right_wrist"]
ErrorType = Literal[
    "data_missing", "blurred", "content_mismatch", "incomplete_action",
    "repeated_retry", "annotation_error", "other",
]
LABELS = (
    "correct", "data_missing", "blurred", "content_mismatch", "incomplete_action",
    "retry_then_success", "annotation_error", "other",
)
ACTIONS = (
    "A_001", "A_002", "A_003", "A_004", "A_005", "A_009", "A_011", "A_012",
    "A_013", "A_015",
)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    action_id: str
    task_code: str
    instruction: str
    collector_id: str
    mcap_path: Path
    sidecar_path: Path
    label: str | None = None
    review_status: str | None = None
    metadata: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Frame:
    timestamp_s: float
    jpeg: bytes = field(repr=False)
    view: CameraView = "main"


@dataclass(frozen=True)
class SampledVideo:
    frames: list[Frame]
    duration_s: float
    camera_topic: str
    strategy: str
    warnings: list[str] = field(default_factory=list)
    motion: dict = field(default_factory=dict)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Evidence(StrictModel):
    view: CameraView | None = None
    timestamp_s: float | None = Field(default=None, ge=0)
    description: str = Field(min_length=1)


class Finding(StrictModel):
    code: ErrorType
    reason: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)


class Assessment(StrictModel):
    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    complete: bool = False
    retry_outcome: Literal["none", "success", "failure", "unknown"] = "none"


class ReviewResult(StrictModel):
    episode_id: str
    action_id: str
    task_code: str
    verdict: Verdict
    error_types: list[ErrorType] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    reason: str
    ground_truth_candidate: bool = False
    retain_retry_sample: bool = False
    retry_outcome: Literal["none", "success", "failure", "unknown"] = "none"
    assessments: dict[str, Assessment] = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)


class CitadelError(Exception):
    """An operational error; never automatically a semantic negative label."""


class EpisodeNotFound(CitadelError):
    pass


class DatasetError(CitadelError):
    pass


class MediaError(CitadelError):
    pass


class ProviderError(CitadelError):
    pass


class ExpertError(CitadelError):
    pass
