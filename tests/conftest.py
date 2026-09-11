import io

import pytest
from PIL import Image

from citadel.data import file_hash, write


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
