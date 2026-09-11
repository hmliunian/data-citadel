import io

import pytest
from PIL import Image

from citadel.data import file_hash, write
from citadel.model import CHECKS


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    ids = [f"{i:032x}" for i in range(8)]
    for i, episode_id in enumerate(ids):
        directory = root / "tasks/DL-TEST/data" / episode_id
        directory.mkdir(parents=True)
        source = directory / "episode.mcap"
        source.write_bytes(f"source-{i}".encode())
        write(root / "tasks/DL-TEST/api_tags" / (episode_id + ".json"), {
            "task.task_code": "DL-TEST",
            "task.review.status": "Accepted" if i < 4 else "Denied",
            "task.review.deny_reason": None if i < 4 else "PRIVATE_GT_REASON",
            "task.review.reviewer": "PRIVATE_REVIEWER"})
        write(root / "receipts" / (episode_id + ".json"), {
            "complete": True, "episode": {"id": episode_id, "task_code": "DL-TEST"},
            "files": [{"relative_path": "episode.mcap", "size": source.stat().st_size,
                       "sha256": file_hash(source)}]})
    write(root / "tasks/DL-TEST/episodes.json", [{"sample_id": i} for i in ids])
    return root


@pytest.fixture
def jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(stream, format="JPEG")
    return stream.getvalue()


@pytest.fixture
def model_case(tmp_path, jpeg):
    work = tmp_path / "work"
    work.mkdir()
    path = work / "image.jpg"
    path.write_bytes(jpeg)
    asset = {"path": "image.jpg", "sha256": file_hash(path)}
    resources = {"steps": [{"action_id": "A_001", "action_text": "拿起指定物体。"}],
                 "images": [{**asset, "id": "OBJECT", "name": "目标",
                             "type": "object", "mime": "image/jpeg"}]}
    media = {"frames": [
        {**asset, "frame_id": f"V{i:03d}", "time_s": float(i),
         "sources": {v: {"time_s": float(i)} for v in ("main", "left_wrist", "right_wrist")}}
        for i in range(4)], "warnings": [], "incomplete": False,
        "signature": {"interval_s": 1.0}}
    profile = {"success": "受控悬空约2秒，末态悬空", "hold_seconds": 2.0,
               "hold_tolerance_s": 0.2, "allowed": "自由路径", "failures": ["失败重试"]}
    return work, resources, profile, media


@pytest.fixture
def answer():
    return {"observations": [
        {"phase": "start", "description": "物体静置", "evidence_ids": ["V000"]},
        {"phase": "hold", "description": "夹持并持续悬空", "evidence_ids": ["V001", "V002", "V003"]},
        {"phase": "end", "description": "末态仍悬空", "evidence_ids": ["V003"]}],
        "checks": {key: {"state": "pass", "evidence_ids": ["V000", "V003"]} for key in CHECKS},
        "hold": {"state": "pass", "evidence_ids": ["V001", "V002", "V003"]},
        "reason": "完成基本抓取且末态悬空。"}
