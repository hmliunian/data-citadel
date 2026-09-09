import base64
import json
from collections import Counter

import pytest
from PIL import Image

from pilot.data import sha256
from pilot.model import REVIEW_RULES
from pilot.route_a import review


class FakeClient:
    def __init__(self):
        self.calls = []
        self.reply = {"data": {"reason": "fake response"}, "usage": {"total_tokens": 17}}

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def make_media(tmp_path):
    def create(number):
        episode_id = f"{number:032x}"
        folder = tmp_path / "media" / episode_id
        folder.mkdir(parents=True)
        frames, views = [], {}
        for view_number, view in enumerate(("main", "left_wrist", "right_wrist")):
            topic = f"/camera/{view}/video"
            times = (number + 0.125, number + 2.375)
            for index, time_s in enumerate(times):
                frame_id = f"{view}-{index:05d}"
                path = folder / (frame_id + ".jpg")
                Image.new("RGB", (8, 6), (number, view_number * 80, index * 100)).save(path)
                frames.append({
                    "frame_id": frame_id, "view": view, "topic": topic,
                    "time_s": time_s, "source_ns": 1780000000000000000 + int(time_s * 10**9),
                    "video_time_s": time_s - times[0], "path": str(path.relative_to(tmp_path)),
                    "sha256": sha256(path), "width": 8, "height": 6,
                })
            views[view] = {"topic": topic, "start_s": times[0], "end_s": times[-1],
                           "video_path": str((folder / (view + ".mp4")).relative_to(tmp_path)),
                           "decoded_frames": 70, "sampled_frames": 2, "uncovered_targets_s": []}
        return {"episode_id": episode_id, "frames": frames, "views": views, "warnings": [],
                "episode_start_ns": 1780000000000000000,
                "signature": {"mcap_sha256": f"{number:064x}", "sampler": "test-sampler",
                              "interval_s": 2.0, "include_endpoints": True},
                "gt": "GT_SECRET", "gt_reason": "PRIVATE_REASON"}
    return create


@pytest.mark.parametrize("expert_count", [1, 3, 4])
def test_joint_review_preserves_each_sequence_and_provenance(tmp_path, make_media, expert_count):
    experts = [make_media(number) for number in range(1, expert_count + 1)]
    candidate = make_media(99)
    client = FakeClient()
    instruction = "搬运棕色戴蝴蝶结狗到指定位置。"

    result = review(client, run_dir=tmp_path, instruction=instruction,
                    experts=experts, candidate=candidate)

    assert result["reply"] is client.reply
    assert result["extra_calls"] == []
    assert len(client.calls) == 1
    messages = client.calls[0]
    assert messages[0] == {"role": "system", "content": REVIEW_RULES}
    assert [message["role"] for message in messages] == ["system", "user"]
    parts = messages[1]["content"]
    text = "\n".join(part["text"] for part in parts if part["type"] == "text")
    assert instruction in text
    media = experts + [candidate]
    prefixes = [f"E{i}" for i in range(1, expert_count + 1)] + ["C"]
    descriptions = [json.loads(part["text"]) for part in parts
                    if part["type"] == "text" and part["text"].startswith("{")]
    expected_frames = [(prefix, frame) for source, prefix in zip(media, prefixes)
                       for frame in source["frames"]]
    assert len(descriptions) == len(expected_frames)
    for description, (prefix, frame) in zip(descriptions, expected_frames):
        assert description["frame_id"] == f"{prefix}-{frame['frame_id']}"
        assert description["time_s"] == frame["time_s"]
        assert description["view"] == frame["view"]
    actual_images = Counter(base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                            for part in parts if part["type"] == "image_url")
    expected_images = Counter((tmp_path / frame["path"]).read_bytes()
                              for source in media for frame in source["frames"])
    assert actual_images == expected_images
    serialized = json.dumps(messages, ensure_ascii=False)
    assert "GT_SECRET" not in serialized
    assert "PRIVATE_REASON" not in serialized
    reference = result["reference"]
    assert reference["route"] == "A"
    assert [entry["episode_id"] for entry in reference["experts"]] == [
        expert["episode_id"] for expert in experts]
    assert [entry["signature"] for entry in reference["experts"]] == [
        expert["signature"] for expert in experts]
    assert reference["candidate"]["episode_id"] == candidate["episode_id"]
    assert reference["candidate"]["signature"] == candidate["signature"]


def test_no_experts_does_not_call_model(tmp_path):
    client = FakeClient()
    with pytest.raises(ValueError, match="at least one expert"):
        review(client, run_dir=tmp_path, instruction="搬运", experts=[], candidate={})
    assert client.calls == []


def test_provider_failure_is_not_a_business_prediction(tmp_path, make_media):
    class BrokenClient:
        def complete(self, messages):
            raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        review(BrokenClient(), run_dir=tmp_path, instruction="搬运",
               experts=[make_media(1)], candidate=make_media(99))
