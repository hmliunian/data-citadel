"""A high confidence model label alone cannot admit an episode as ground truth."""

from __future__ import annotations

from ..models import Assessment, Episode, Finding, ReviewResult

POLICY_VERSION = "strict-atomic-v2-main-evidence"


def decide(
    episode: Episode,
    assessments: dict[str, Assessment],
    *,
    threshold: float = 0.95,
    warnings: list[str] | None = None,
    provenance: dict | None = None,
) -> ReviewResult:
    warnings = warnings or []
    confirmed: list[Finding] = []
    for stage, assessment in assessments.items():
        if assessment.verdict == "incorrect" and assessment.confidence >= threshold:
            for finding in assessment.findings:
                if finding.evidence and (
                    stage == "integrity"
                    or all(item.timestamp_s is not None for item in finding.evidence)
                ):
                    confirmed.append(finding)

    required = ("integrity", "generic", "task")
    all_pass = not warnings and all(
        stage in assessments and _passes(assessments[stage], threshold, stage)
        for stage in required
    )
    if confirmed:
        verdict = "incorrect"
        reason = "；".join(dict.fromkeys(finding.reason for finding in confirmed))
    elif all_pass:
        verdict = "correct"
        reason = "完整性、通用质量与任务完成检查均通过，证据满足当前严格策略。"
    else:
        verdict = "uncertain"
        reasons = [
            assessment.reason for stage, assessment in assessments.items()
            if not _passes(assessment, threshold, stage)
        ]
        reasons.extend(warnings)
        absent = [stage for stage in required if stage not in assessments]
        if absent:
            reasons.append("尚未完成审核项：" + ", ".join(absent))
        reason = "；".join(dict.fromkeys(reasons)) or "证据不足，无法确认全部要求通过。"

    codes = list(dict.fromkeys(finding.code for finding in confirmed))
    task = assessments.get("task")
    retry_outcome = "none"
    retain_retry = False
    if "repeated_retry" in codes:
        retry_outcome = task.retry_outcome if task and task.retry_outcome != "none" else "unknown"
        retain_retry = bool(
            task and task.complete and task.confidence >= threshold
            and retry_outcome == "success"
            and len({
                item.timestamp_s for finding in task.findings
                if finding.code == "repeated_retry" for item in finding.evidence
                if item.timestamp_s is not None
            }) >= 2
        )
    return ReviewResult(
        episode_id=episode.episode_id, action_id=episode.action_id, task_code=episode.task_code,
        verdict=verdict, error_types=codes, findings=confirmed, reason=reason,
        ground_truth_candidate=verdict == "correct", retain_retry_sample=retain_retry,
        retry_outcome=retry_outcome, assessments=assessments, provenance=provenance or {},
    )


def _passes(assessment: Assessment, threshold: float, stage: str) -> bool:
    if not (
        assessment.verdict == "correct" and assessment.complete
        and assessment.confidence >= threshold and not assessment.findings
        and assessment.retry_outcome == "none" and assessment.evidence
    ):
        return False
    if stage == "integrity":
        return True
    timestamps = {item.timestamp_s for item in assessment.evidence}
    if None in timestamps:
        return False
    if stage == "task":
        # Wrist close-ups cannot replace main-view process and completion evidence.
        timestamps = {item.timestamp_s for item in assessment.evidence
                      if item.view in (None, "main")}
    return len(timestamps) >= (2 if stage == "task" else 1)
