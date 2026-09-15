"""Public HTTP contracts; clients need no server imports."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1)
    episode_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    retry_failed: bool = False


class BatchSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    split: Literal["development", "holdout"] = "development"
    limit: int | None = Field(default=None, ge=1)
    retry_failed: bool = False


class JobView(BaseModel):
    job_id: str
    run_id: str
    episode_id: str
    kind: Literal["review", "preview"]
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str
    configuration_sha256: str
    created_ns: int
    updated_ns: int
    result_id: str | None = None
    error: dict | None = None


class EpisodeView(BaseModel):
    episode_id: str
    task_code: str
    split: str
    status: str
    label: str | None
    result_id: str | None
    gt: str | None
    gt_reason: str | None


class ResultView(BaseModel):
    model_config = ConfigDict(extra="allow")
    result_id: str
    episode_id: str
    configuration_sha256: str
    status: Literal["completed", "needs_review", "failed"]
    label: Literal["correct", "incorrect"] | None
    reason: str
