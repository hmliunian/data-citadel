import io
import copy
from pathlib import Path

import pytest
from PIL import Image

from citadel.infrastructure.files import file_hash, fingerprint, write
from citadel.infrastructure.datasets import prepare
from citadel.domain.models import CHECKS
from citadel.service import Service


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
        "quality_by_camera": {v: {"state": "pass", "evidence_ids": ["V000", "V003"],
                                  "description": "操作区边缘清晰"}
                              for v in ("main", "left_wrist", "right_wrist")},
        "main_visibility_by_frame": {f"V{i:03d}": "visible" for i in range(4)},
        "checks": {key: {"state": "pass", "evidence_ids": ["V000", "V003"]} for key in CHECKS},
        "hold": {"state": "pass", "evidence_ids": ["V001", "V002", "V003"]},
        "reason": "完成基本抓取且末态悬空。"}


@pytest.fixture
def service_case(tmp_path, dataset, model_case, answer, jpeg):
    _, resources, profile, media = model_case
    work = tmp_path / "run"
    prepare(dataset, work)
    (work / "image.jpg").write_bytes(jpeg)
    profiles = work / "rules.json"
    write(profiles, {"grasp": {**profile, "action_ids": ["A_001"]}})
    resources = {**resources, "task_code": "DL-TEST"}
    resources["sha256"] = fingerprint(resources)
    class FakeClient:
        model, base_url = "fake-qwen", "https://example.invalid/v1"
        def __init__(self):
            self.requests = []
            self.error = None
        def complete(self, request_messages, context, *, quality_only=False):
            self.requests.append(copy.deepcopy(request_messages))
            if self.error:
                raise self.error
            data = copy.deepcopy(answer)
            if quality_only:
                data = {"quality_by_camera": data["quality_by_camera"]}
            else:
                data.pop("quality_by_camera")
            return {"data": data, "model": self.model, "usage": {}}
    client = FakeClient()
    def load_media(output, source, sampling):
        assert set(source) == {"episode_id", "mcap_path", "mcap_sha256"}
        assert file_hash(Path(source["mcap_path"])) == source["mcap_sha256"]
        result = copy.deepcopy(media)
        result["episode_id"] = source["episode_id"]
        folder = work / "media" / source["episode_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for frame in result["frames"]:
            path = folder / (frame["frame_id"] + ".jpg")
            path.write_bytes(jpeg)
            frame["path"] = str(path.relative_to(work))
        result["videos"] = {}
        if not (folder / "media.json").exists():
            write(folder / "media.json", result)
        return result
    service = Service(work, profiles, client=client,
                      resource_loader=lambda *args: resources, media_loader=load_media)
    return service, client
