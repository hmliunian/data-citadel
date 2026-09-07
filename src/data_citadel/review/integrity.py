"""Deterministic checks distinguish missing source data from operational failures."""

from __future__ import annotations

from ..models import Assessment, Episode, Evidence, Finding


def check_fields(episode: Episode) -> Assessment:
    missing = [
        name for name in ("episode_id", "action_id", "task_code", "instruction", "collector_id")
        if not getattr(episode, name, "").strip()
    ]
    missing.extend(
        name for name in ("mcap_path", "sidecar_path")
        if not getattr(episode, name).is_file()
    )
    if missing:
        return _missing("必需字段或文件缺失：" + ", ".join(missing))
    return Assessment(
        verdict="correct", reason="必需元数据与文件存在。", confidence=1.0, complete=True,
        evidence=[Evidence(description="已检查 episode/action/task/instruction/collector 及两个源文件。")],
    )


def check_integrity(episode: Episode, inspection: dict) -> Assessment:
    fields = check_fields(episode)
    if fields.verdict != "correct":
        return fields
    missing = inspection.get("missing_required_channels", [])
    empty = inspection.get("empty_required_channels", [])
    if missing or empty:
        return _missing(f"MCAP 必需流缺失 {missing}；必需流无消息 {empty}。")
    if not all(
        key in inspection
        for key in ("missing_required_channels", "empty_required_channels", "message_counts")
    ):
        return Assessment(
            verdict="uncertain", reason="未获得完整的数据流检查结果。", confidence=0.0,
        )
    return Assessment(
        verdict="correct", reason="必需元数据、文件和主相机消息均存在。",
        confidence=1.0, complete=True,
        evidence=[*fields.evidence, Evidence(description="MCAP 必需流存在且消息数大于零。")],
    )


def _missing(reason: str) -> Assessment:
    evidence = [Evidence(description=reason)]
    return Assessment(
        verdict="incorrect", reason=reason, confidence=1.0, complete=True,
        findings=[Finding(code="data_missing", reason=reason, evidence=evidence)],
        evidence=evidence,
    )
