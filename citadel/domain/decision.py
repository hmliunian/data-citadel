"""Validate temporal evidence and aggregate the business verdict."""
from .models import CAMERAS, CHECKS, Check, QualityReview, Review


def decide(data, media, profile):
    data = dict(data)
    quality_by_camera = QualityReview.model_validate({
        "quality_by_camera": data.pop("quality_by_camera", None)}).quality_by_camera
    parsed = Review.model_validate(data)
    if set(parsed.checks) != set(CHECKS):
        raise ValueError("Model must return all seven checks")
    if set(quality_by_camera) != set(CAMERAS):
        raise ValueError("Quality must cover all three cameras")
    lookup = {f["frame_id"]: f for f in media["frames"]}
    frame_visibility = parsed.main_visibility_by_frame
    if set(frame_visibility) != set(lookup):
        raise ValueError("Visibility must cover every candidate frame")
    used = set(lookup)
    for item in [*parsed.observations, *parsed.checks.values(), *quality_by_camera.values(),
                 *([parsed.hold] if parsed.hold else [])]:
        ids = item.evidence_ids
        if (len(set(ids)) != len(ids) or any(i not in lookup for i in ids)
                or isinstance(item, Check) and item.state != "unknown" and not ids):
            raise ValueError("Evidence must reference supplied candidate frames")
    warnings = list(media["warnings"])
    visibility = parsed.checks["main_visibility"]
    if visibility.state != "unknown" and not any(
            lookup[i]["sources"].get("main") for i in visibility.evidence_ids):
        visibility.state = "unknown"
        warnings.append("main_visibility:no_valid_main_evidence")
    valid_main = [i for i, frame in lookup.items() if frame["sources"].get("main")]
    for frame_id in lookup:
        if frame_id not in valid_main:
            frame_visibility[frame_id] = "no_frame"
        elif frame_visibility[frame_id] == "no_frame":
            frame_visibility[frame_id] = "uncertain"
            warnings.append(frame_id + ":main_camera_is_present")
    absent = [i for i in valid_main if frame_visibility[i] == "absent"]
    unclear = [i for i in valid_main if frame_visibility[i] == "uncertain"]
    if absent:
        visibility.state, visibility.evidence_ids = "fail", absent
    elif unclear or not valid_main or visibility.state != "pass":
        visibility.state, visibility.evidence_ids = "unknown", unclear or valid_main
    else:
        visibility.evidence_ids = valid_main
    for view, quality in quality_by_camera.items():
        quality.evidence_ids = [i for i in quality.evidence_ids if lookup[i]["sources"].get(view)]
        if quality.state != "unknown" and not quality.evidence_ids:
            quality.state = "unknown"
            warnings.append(view + ":no_valid_quality_evidence")
    qualities = list(quality_by_camera.values())
    bad_quality = [q for q in qualities if q.state == "fail"]
    quality_state = "fail" if bad_quality else (
        "unknown" if any(q.state == "unknown" for q in qualities) else "pass")
    if quality_state == "pass" and parsed.checks["image_quality"].state != "pass":
        quality_state = "unknown"
    parsed.checks["image_quality"] = Check(state=quality_state, evidence_ids=sorted({
        i for q in (bad_quality or qualities) for i in q.evidence_ids}))
    issues = [key for key, check in parsed.checks.items() if check.state == "fail"]
    if any(o.phase == "failure" for o in parsed.observations):
        issues.append("observed_failure")
    uncertain = media["incomplete"] or any(c.state == "unknown" for c in parsed.checks.values())
    first, last = media["frames"][0]["frame_id"], media["frames"][-1]["frame_id"]
    endpoints = (any(o.phase == "start" and first in o.evidence_ids for o in parsed.observations)
                 and any(o.phase == "end" and last in o.evidence_ids for o in parsed.observations))
    uncertain |= not endpoints
    hold_span = None
    if profile.get("hold_seconds") is not None:
        if parsed.hold and parsed.hold.state == "fail":
            issues.append("hold")
        if not parsed.hold or parsed.hold.state == "unknown":
            uncertain = True
        elif parsed.hold.state == "pass":
            times = [lookup[i]["time_s"] for i in parsed.hold.evidence_ids]
            hold_span = max(times) - min(times)
            uncertain |= hold_span < profile["hold_seconds"] - profile.get("hold_tolerance_s", 0)
            uncertain |= max(times) < media["frames"][-1]["time_s"] - media["signature"].get("tolerance_s", 0.1)
    label = "incorrect" if issues else (None if uncertain else "correct")
    reason = ("主镜头目标出画；" if absent else "") + parsed.reason
    if bad_quality:
        names = {"main": "主镜头", "left_wrist": "左腕", "right_wrist": "右腕"}
        reason = "画质不合格：" + "；".join(
            names[v] + "：" + q.description for v, q in quality_by_camera.items()
            if q.state == "fail") + "；" + reason
    if label is None:
        reason += "；时间、首尾或采集证据仍不足，需复核。"
    if "observed_failure" in issues:
        reason = "过程包含明确失败；" + reason
    return {"status": "needs_review" if label is None else "completed", "label": label,
            "reason": reason, "issues": issues, "checks": {k: v.model_dump() for k, v in parsed.checks.items()},
            "main_visibility_by_frame": frame_visibility,
            "quality_by_camera": {k: v.model_dump() for k, v in quality_by_camera.items()},
            "observations": [o.model_dump() for o in parsed.observations],
            "hold": parsed.hold.model_dump() if parsed.hold else None,
            "hold_evidence_span_s": hold_span, "evidence": [lookup[i] for i in sorted(used)],
            "warnings": warnings}
