"""Build complete, timestamped multimodal requests from a prompt snapshot."""
import json
from pathlib import Path

from citadel.configuration import PromptBundle
from citadel.infrastructure.resources import image_input
from citadel.infrastructure.mcap.sensors import TOPICS as GRIPPER_CHANNELS, intervals


class PromptBuilder:
    def __init__(self, prompts: PromptBundle | None = None):
        self.prompts = prompts or PromptBundle.load()

    def review(self, work: Path, resources: dict, profile: dict, media: dict):
        timeline = [{"frame_id": f["frame_id"], "time_s": f["time_s"],
                     "source_times_s": {v: item["time_s"] if item else None
                                        for v, item in f["sources"].items()}} for f in media["frames"]]
        camera_ranges = {}
        for view in media["frames"][0]["sources"]:
            ids = [f["frame_id"] for f in media["frames"] if f["sources"].get(view)]
            camera_ranges[view] = {"first_frame_id": ids[0], "last_frame_id": ids[-1]} if ids else None
        content = [{"type": "text", "text": json.dumps({
            "instruction": resources["steps"], "task_rules": profile,
            "first_frame_id": timeline[0]["frame_id"], "last_frame_id": timeline[-1]["frame_id"],
            "frame_count": len(timeline), "camera_ranges": camera_ranges,
            "media_warnings": media["warnings"],
        }, ensure_ascii=False)}]
        for item in resources["images"]:
            content.extend([
                {"type": "text", "text": json.dumps(
                    {"reference_type": item["type"], "name": item["name"], "id": item["id"]},
                    ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": image_input(work, item)}}])
        gripper = media.get("gripper")
        if gripper:
            content.insert(1, {"type": "text", "text": json.dumps({
                "gripper_columns": ["start_s", "end_s", *GRIPPER_CHANNELS],
                "gripper_warnings": gripper["warnings"],
                "gripper_sha256": gripper["sha256"],
            }, ensure_ascii=False)})
            for timing, samples in zip(timeline, intervals(gripper, media["frames"])):
                timing["gripper_before"] = samples
        for frame, timing in zip(media["frames"], timeline):
            content.extend([
                {"type": "text", "text": json.dumps({"candidate_frame": timing}, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": image_input(work, frame)}}])
        return [{"role": "system", "content": self.prompts.review_system + ("\n" + self.prompts.gripper if gripper else "")},
                {"role": "user", "content": content}]

    def quality(self, work: Path, media: dict):
        request = self.review(work, {"steps": [], "images": []}, {}, {**media, "gripper": None})
        request[0]["content"] = self.prompts.quality
        return request
