import copy
import json

import pytest
from PIL import Image

from pilot import route_b
from pilot.data import digest, read_json, sha256

INSTRUCTION = "搬运 棕色戴蝴蝶结狗"
REVIEW = {"reason": "采样不足以确认全过程", "checks": {
    key: {"state": "unknown", "evidence_ids": ["C-main-00000"]}
    for key in ("object", "action", "retry_free", "quality")
}}


class FakeClient:
    model = "fake-vl"
    base_url = "https://model.invalid/v1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.budgets = []

    def complete(self, messages, max_tokens=3000):
        self.calls.append(copy.deepcopy(messages))
        self.budgets.append(max_tokens)
        return {"data": self.responses.pop(0), "usage": {"total_tokens": 11},
                "model": self.model, "request_id": f"fake-{len(self.calls)}", "elapsed_s": 0.1,
                "input_images": sum(p["type"] == "image_url" for m in messages
                                    if isinstance(m["content"], list) for p in m["content"]),
                "request_sha256": digest(messages), "raw": {"fake": True}}


def media(tmp_path, episode_id, color="brown", frame_count=2):
    frames = []
    folder = tmp_path / "media" / episode_id
    folder.mkdir(parents=True)
    for index in range(frame_count):
        time_s = 0.2 if index == 0 else 2 * index + 0.03
        path = folder / f"main-{index:05d}.jpg"
        Image.new("RGB", (8, 8), color=color).save(path)
        frames.append({
            "frame_id": path.stem, "view": "main", "topic": "/camera/test",
            "time_s": time_s, "source_ns": 1_000_000_000 + round(time_s * 1e9),
            "video_time_s": time_s - 0.2, "path": str(path.relative_to(tmp_path)),
            "sha256": sha256(path), "width": 8, "height": 8,
        })
    return {"episode_id": episode_id, "frames": frames, "views": {}, "warnings": [],
            "signature": {"mcap_sha256": "mcap-" + episode_id, "sampler": "test-v1",
                          "interval_s": 2.0, "include_endpoints": True}}


def caption(expert, prefix="E1"):
    ids = [prefix + "-" + f["frame_id"] for f in expert["frames"]]
    return {
        "observations": [{"frame_id": frame_id, "visible": "棕色狗玩具可见，动作状态待确认"}
                         for frame_id in reversed(ids)],
        "completion_criteria": [{"instruction_quote": INSTRUCTION,
                                 "observation": "尚不能确认完成搬运", "evidence_ids": ids}],
        "uncertainties": [{"description": "相邻采样之间的过程未直接观察", "evidence_ids": ids}],
    }


def test_reuses_immutable_captions_and_only_sends_candidate_images(tmp_path):
    experts = [media(tmp_path, "expert1"), media(tmp_path, "expert2", "blue")]
    candidate = media(tmp_path, "candidate", "green")
    for item in experts + [candidate]:
        item.update({"gt": "SECRET_GT", "gt_status": "SECRET_STATUS",
                     "reviewer": "SECRET_REVIEWER", "gt_reason": "SECRET_DENY_REASON",
                     "quality": "SECRET_QUANTIFY", "tags": {"task.review.status": "SECRET_TAG"}})
        item["frames"][0]["gt"] = "SECRET_FRAME_GT"
    client = FakeClient([caption(experts[0]), caption(experts[1], "E2"), REVIEW, REVIEW])
    result = route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                            experts=experts, candidate=candidate)
    assert len(client.calls) == 3
    assert len(result["extra_calls"]) == 2
    assert result["reply"]["input_images"] == 2
    assert all(call["input_images"] == 2 for call in result["extra_calls"])
    reference = result["reference"]
    paths = [tmp_path / r["cache_path"] for r in reference["experts"]]
    saved_bytes = [p.read_bytes() for p in paths]
    assert reference["comparison_policy"] == route_b.TASK_POLICY
    observation = reference["experts"][0]["caption"]["observations"][0]
    assert observation["frame_id"] == "E1-main-00000"
    for key in ("view", "topic", "time_s", "source_ns", "video_time_s"):
        assert observation[key] == experts[0]["frames"][0][key]
    texts = json.dumps(client.calls, ensure_ascii=False)
    assert "SECRET_" not in texts
    main_parts = client.calls[2][1]["content"]
    assert "expert_descriptions" in main_parts[0]["text"]
    assert "E1-main-00000" in main_parts[0]["text"]
    assert "E2-main-00000" in main_parts[0]["text"]
    expert_urls = {p["image_url"]["url"] for call in client.calls[:2]
                   for p in call[1]["content"] if p["type"] == "image_url"}
    candidate_urls = {p["image_url"]["url"] for p in main_parts if p["type"] == "image_url"}
    assert candidate_urls and candidate_urls.isdisjoint(expert_urls)
    repeated = route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                              experts=experts, candidate=candidate)
    assert len(client.calls) == 4
    assert repeated["extra_calls"] == []
    assert all(r["cache_hit"] for r in repeated["reference"]["experts"])
    assert [p.read_bytes() for p in paths] == saved_bytes
    assert repeated["reply"]["data"] == REVIEW


@pytest.mark.parametrize("change", ["model", "base_url", "prompt", "schema", "instruction",
                                    "sampling", "image", "time", "budget"])
def test_cache_invalidates_changed_inputs(tmp_path, monkeypatch, change):
    expert, candidate = media(tmp_path, "expert"), media(tmp_path, "candidate", "green")
    client = FakeClient([caption(expert), REVIEW])
    first = route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                           experts=[expert], candidate=candidate)
    instruction = INSTRUCTION
    if change == "model":
        client.model = "fake-vl-next"
    elif change == "base_url":
        client.base_url = "https://other.invalid/v1"
    elif change == "prompt":
        monkeypatch.setattr(route_b, "CAPTION_RULES", route_b.CAPTION_RULES + "\n更谨慎描述。")
    elif change == "schema":
        monkeypatch.setattr(route_b, "CAPTION_VERSION", "expert-sequence-v2")
    elif change == "instruction":
        instruction += "，完成后松开夹爪"
    elif change == "sampling":
        expert["signature"]["interval_s"] = 1.0
    elif change == "image":
        path = tmp_path / expert["frames"][0]["path"]
        Image.new("RGB", (8, 8), color="red").save(path)
        expert["frames"][0]["sha256"] = sha256(path)
    elif change == "time":
        expert["frames"][0]["time_s"] = 0.25
    elif change == "budget":
        monkeypatch.setattr(route_b, "CAPTION_MAX_TOKENS", 2500)
    client.responses.extend([caption(expert), REVIEW])
    second = route_b.review(client, run_dir=tmp_path, instruction=instruction,
                            experts=[expert], candidate=candidate)
    assert len(client.calls) == 4 and len(second["extra_calls"]) == 1
    old, new = (r["reference"]["experts"][0] for r in (first, second))
    assert old["cache_key"] != new["cache_key"]
    assert (tmp_path / old["cache_path"]).exists()
    assert (tmp_path / new["cache_path"]).exists()
    if change == "budget":
        assert client.budgets == [3000, 3000, 2500, 3000]


@pytest.mark.parametrize("frame_count,budget", [(71, 8100), (72, 8192), (110, 8192)])
def test_caption_budget_respects_model_limit_and_matches_cache(tmp_path, frame_count, budget):
    expert = media(tmp_path, "expert", frame_count=frame_count)
    candidate = media(tmp_path, "candidate", "green")
    client = FakeClient([caption(expert), REVIEW])
    result = route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                            experts=[expert], candidate=candidate)
    assert client.budgets == [budget, 3000]
    cached = read_json(tmp_path / result["reference"]["experts"][0]["cache_path"])
    assert cached["signature"]["max_tokens"] == budget


@pytest.mark.parametrize("corruption", ["unknown_frame", "missing_frame", "duplicate_frame",
                                       "criterion_reference", "uncertainty_reference",
                                       "invented_constraint", "extra_field", "empty_observation"])
def test_invalid_caption_is_rejected_before_candidate_review(tmp_path, corruption):
    expert, candidate = media(tmp_path, "expert"), media(tmp_path, "candidate", "green")
    data = caption(expert)
    if corruption == "unknown_frame":
        data["observations"][0]["frame_id"] = "E1-not-provided"
    elif corruption == "missing_frame":
        data["observations"].pop()
    elif corruption == "duplicate_frame":
        data["observations"].append(copy.deepcopy(data["observations"][0]))
    elif corruption == "criterion_reference":
        data["completion_criteria"][0]["evidence_ids"] = ["C-main-00000"]
    elif corruption == "uncertainty_reference":
        data["uncertainties"][0]["evidence_ids"] = ["E2-main-00000"]
    elif corruption == "invented_constraint":
        data["completion_criteria"][0]["instruction_quote"] = "必须放进左边红色篮子"
    elif corruption == "extra_field":
        data["verdict"] = "correct"
    elif corruption == "empty_observation":
        data["observations"][0]["visible"] = " "
    client = FakeClient([data, REVIEW])
    with pytest.raises(ValueError, match="[Ee]xpert"):
        route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                       experts=[expert], candidate=candidate)
    assert len(client.calls) == 1
    assert not list(tmp_path.glob("references/route_b/*.json"))


def test_cache_hit_still_validates_evidence_and_actual_images(tmp_path):
    expert, candidate = media(tmp_path, "expert"), media(tmp_path, "candidate", "green")
    client = FakeClient([caption(expert), REVIEW])
    result = route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                            experts=[expert], candidate=candidate)
    cache_path = tmp_path / result["reference"]["experts"][0]["cache_path"]
    saved = read_json(cache_path)
    saved["reply"]["data"]["observations"][0]["frame_id"] = "E1-forged"
    cache_path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                       experts=[expert], candidate=candidate)
    Image.new("RGB", (8, 8), color="red").save(tmp_path / expert["frames"][0]["path"])
    with pytest.raises(ValueError, match="hash"):
        route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                       experts=[expert], candidate=candidate)
    assert len(client.calls) == 2


@pytest.mark.parametrize("experts_kind", ["empty", "duplicate", "candidate"])
def test_expert_independence_is_checked_without_model_calls(tmp_path, experts_kind):
    expert, candidate = media(tmp_path, "expert"), media(tmp_path, "candidate")
    experts = {"empty": [], "duplicate": [expert, expert], "candidate": [candidate]}[experts_kind]
    client = FakeClient([])
    with pytest.raises(ValueError):
        route_b.review(client, run_dir=tmp_path, instruction=INSTRUCTION,
                       experts=experts, candidate=candidate)
    assert client.calls == []
