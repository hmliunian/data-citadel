"""Review one candidate with the original expert image sequences."""
from pathlib import Path

from .model import REVIEW_RULES, frame_parts


def review(client, *, run_dir: Path, instruction: str, experts: list[dict],
           candidate: dict) -> dict:
    if not experts:
        raise ValueError("Route A requires at least one expert")
    content = [{"type": "text", "text": (
        "路线 A：先阅读下面相互独立的专家示范，再审核唯一的待测序列 C。\n"
        f"任务指令：\n{instruction}\n"
        "专家用于理解同一任务的动作阶段、操作内容和完成状态。"
        "各序列保留自己的实际时间；按过程理解，不要求专家与待测在同一秒的画面或动作一致。"
        "专家中的成功状态不能作为待测已完成的证据。最终只审核 C，证据只引用 C- 开头的帧编号。"
    )}]
    references = []
    for index, expert in enumerate(experts, 1):
        prefix = f"E{index}"
        content.append({"type": "text", "text": f"专家示范 {prefix}（仅供任务参照）"})
        content.extend(frame_parts(expert, prefix, run_dir))
        references.append({"prefix": prefix, "episode_id": expert["episode_id"],
                           "signature": expert["signature"]})
    content.append({"type": "text", "text": "待测序列 C（唯一审核对象）"})
    content.extend(frame_parts(candidate, "C", run_dir))
    reply = client.complete([{"role": "system", "content": REVIEW_RULES},
                             {"role": "user", "content": content}])
    return {"reply": reply, "reference": {
        "route": "A", "experts": references,
        "candidate": {"prefix": "C", "episode_id": candidate["episode_id"],
                      "signature": candidate["signature"]},
    }, "extra_calls": []}
