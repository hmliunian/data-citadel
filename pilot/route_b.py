"""Review candidates against cached, frame-grounded expert descriptions."""
from __future__ import annotations

import json
from pathlib import Path

from .data import digest, read_json, sha256, write_json
from .model import REVIEW_RULES, TASK_POLICY, frame_parts

CAPTION_VERSION = "expert-sequence-v1"
CAPTION_RULES = """你正在记录一条专家示范，当前没有待测样本。依据任务指令和所提供的
按时间排列的三路采样图像，描述可见的物体、颜色、动作阶段及最终状态。一次理解整条序列。
每个提供的 frame_id 必须恰好有一条简短 visible 观察，不能遗漏或重复；遮挡或看不清要明确说。
不同相机同一时间可互相补充，但不能把另一个视角的内容当成本帧直接可见。
completion_criteria 只能引用任务指令中连续、逐字一致的原文 instruction_quote，
observation 只说明专家中可见的完成情况。不得根据专家习惯添加指令未规定的目标位置、
路线、用手、速度或其他强制条件。允许的动作差异遵循上述共同审核政策。
每条完成情况及不确定性用 evidence_ids 引用所提供的专家帧；没有可见证据时可以为空，
但必须明确无法确认。观察之间的空白时段不能当成连续动作证据；最终成功不能证明没有失败重试。
不输出正确/错误判定，不输出错误类别。只输出以下 JSON，不添加其他字段：
{"observations":[{"frame_id":"提供的完整帧 ID","visible":"本帧可见事实或无法确认"}],
 "completion_criteria":[{"instruction_quote":"指令原文","observation":"观察到的完成情况或无法确认",
                         "evidence_ids":["提供的完整帧 ID"]}],
 "uncertainties":[{"description":"遮挡、采样空白等不确定性","evidence_ids":[]}]}
"""
REVIEW_CONTEXT = """专家描述是带来源的视觉观察，可能存在遗漏；它不增加任务指令中的约束。
结合全部专家描述理解动作阶段，再独立审核候选图像。不要要求专家与候选同秒对齐。
审核 checks 的 evidence_ids 只能引用本次候选的 C- 帧 ID；专家帧不能代替候选证据。
"""
FRAME_FIELDS = ("frame_id", "view", "topic", "time_s", "source_ns", "video_time_s",
                "sha256", "width", "height")


def _signature(client, run_dir: Path, instruction: str, expert: dict, prefix: str) -> dict:
    frames = []
    for frame in expert["frames"]:
        path = (run_dir / frame["path"]).resolve()
        if not path.is_relative_to(run_dir.resolve()) or sha256(path) != frame["sha256"]:
            raise ValueError("Expert image path or hash does not match its media manifest")
        frames.append({key: frame[key] for key in FRAME_FIELDS})
    if not frames or len({f["frame_id"] for f in frames}) != len(frames):
        raise ValueError("Expert needs non-empty, unique sampled frame IDs")
    return {
        "caption_version": CAPTION_VERSION,
        "prompt_sha256": digest([TASK_POLICY, CAPTION_RULES]),
        "model": client.model, "base_url": str(client.base_url),
        "instruction": instruction, "episode_id": expert["episode_id"], "prefix": prefix,
        "media_signature": expert["signature"], "frames": frames,
        "warnings": expert.get("warnings", []), "views": expert.get("views", {}),
    }


def _validate_caption(data: dict, expert: dict, prefix: str, instruction: str) -> dict:
    required = {"observations", "completion_criteria", "uncertainties"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("Invalid expert caption fields")
    if any(not isinstance(data[key], list) for key in required):
        raise ValueError("Expert caption fields must be lists")
    frames = {f"{prefix}-{f['frame_id']}": f for f in expert["frames"]}
    observed = {}
    for item in data["observations"]:
        if (not isinstance(item, dict) or set(item) != {"frame_id", "visible"}
                or not isinstance(item["frame_id"], str) or item["frame_id"] not in frames
                or item["frame_id"] in observed
                or not isinstance(item["visible"], str) or not item["visible"].strip()):
            raise ValueError("Invalid or duplicate expert observation evidence")
        observed[item["frame_id"]] = item["visible"]
    if set(observed) != set(frames):
        raise ValueError("Expert observations must cover every provided frame")
    if not data["completion_criteria"]:
        raise ValueError("Expert caption needs instruction-grounded completion criteria")
    for field, text_fields in (("completion_criteria", {"instruction_quote", "observation"}),
                               ("uncertainties", {"description"})):
        for item in data[field]:
            if (not isinstance(item, dict) or set(item) != text_fields | {"evidence_ids"}
                    or any(not isinstance(item[key], str) or not item[key].strip()
                           for key in text_fields)):
                raise ValueError("Invalid expert caption detail fields")
            evidence = item["evidence_ids"]
            if (not isinstance(evidence, list)
                    or any(not isinstance(e, str) or e not in frames for e in evidence)
                    or len(set(evidence)) != len(evidence)):
                raise ValueError("Invalid expert caption detail evidence")
            if field == "completion_criteria" and item["instruction_quote"] not in instruction:
                raise ValueError("Expert completion criterion must quote the task instruction")
    return {
        "observations": [
            {"frame_id": frame_id, "visible": observed[frame_id],
             **{key: frame[key] for key in ("view", "topic", "time_s", "source_ns", "video_time_s")}}
            for frame_id, frame in frames.items()
        ],
        "completion_criteria": data["completion_criteria"],
        "uncertainties": data["uncertainties"],
    }


def review(client, *, run_dir: Path, instruction: str, experts: list[dict], candidate: dict) -> dict:
    if not instruction.strip() or not experts:
        raise ValueError("Route B needs an instruction and at least one expert")
    episode_ids = [expert["episode_id"] for expert in experts]
    if len(set(episode_ids)) != len(episode_ids) or candidate["episode_id"] in episode_ids:
        raise ValueError("Experts must be distinct from each other and the candidate")
    references, extra_calls = [], []
    for number, expert in enumerate(experts, 1):
        prefix = f"E{number}"
        signature = _signature(client, run_dir, instruction, expert, prefix)
        cache_key = digest(signature)
        cache_path = run_dir / "references" / "route_b" / (cache_key + ".json")
        cache_hit = cache_path.exists()
        if cache_hit:
            saved = read_json(cache_path)
        else:
            messages = [
                {"role": "system", "content": TASK_POLICY + "\n" + CAPTION_RULES},
                {"role": "user", "content": [
                    {"type": "text", "text": "任务指令：\n" + instruction},
                    *frame_parts(expert, prefix, run_dir),
                ]},
            ]
            reply = client.complete(messages, max_tokens=max(3000, min(12000,
                                    100 * len(expert["frames"]) + 1000)))
            saved = {"signature": signature, "reply": reply}
            extra_calls.append(reply)
        if saved.get("signature") != signature:
            raise ValueError("Expert caption cache signature mismatch")
        reply = saved["reply"]
        caption = _validate_caption(reply["data"], expert, prefix, instruction)
        if not cache_hit:
            write_json(cache_path, saved)
        references.append({
            "episode_id": expert["episode_id"], "prefix": prefix, "cache_key": cache_key,
            "cache_path": str(cache_path.relative_to(run_dir)), "cache_hit": cache_hit,
            "caption": caption, "caption_request_id": reply["request_id"],
            "caption_request_sha256": reply["request_sha256"], "model": reply["model"],
        })
    expert_text = [{"expert": r["prefix"], "episode_id": r["episode_id"], "caption": r["caption"]}
                   for r in references]
    reply = client.complete([
        {"role": "system", "content": REVIEW_RULES + "\n" + REVIEW_CONTEXT},
        {"role": "user", "content": [
            {"type": "text", "text": json.dumps({"instruction": instruction,
                                                  "expert_descriptions": expert_text},
                                                 ensure_ascii=False)},
            *frame_parts(candidate, "C", run_dir),
        ]},
    ])
    return {
        "reply": reply, "extra_calls": extra_calls,
        "reference": {"route": "B", "caption_version": CAPTION_VERSION,
                      "comparison_policy": TASK_POLICY,
                      "review_prompt_sha256": digest([REVIEW_RULES, REVIEW_CONTEXT]),
                      "experts": references},
    }
