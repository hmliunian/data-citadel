from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pilot import full
from pilot.data import digest, read_json, sha256, write_json


@pytest.fixture
def batch(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "run"
    ids = [f"{i:032x}" for i in range(3)]
    episodes = {}
    for episode_id in ids:
        tags = source / "tasks" / "DL-TEST" / "api_tags" / (episode_id + ".json")
        receipt = source / "receipts" / (episode_id + ".json")
        write_json(tags, {"secret": "GT_SENTINEL"})
        write_json(receipt, {"secret": "REVIEWER_SENTINEL"})
        episodes[episode_id] = {"episode_id": episode_id, "task_code": "DL-TEST", "quality": "high",
                                "gt_status": "Accepted", "gt": "correct", "gt_reason": None,
                                "reviewer": "REVIEWER_SENTINEL", "review_time": "source-time",
                                "tags_sha256": sha256(tags), "receipt_sha256": sha256(receipt)}
    manifest = {"dataset": str(source), "sampling": {}, "routes": ["A", "B"],
                "episodes": episodes, "tasks": {"DL-TEST": {"instruction": "搬运棕色狗",
                "expert_pool": ids[:1], "experts": ids[:1], "candidate_ids": ids[1:], "warnings": []}}}
    manifest["snapshot_sha256"] = digest(manifest)
    write_json(root / "manifest.json", manifest)
    monkeypatch.setattr(full, "FROZEN_FILES", ("model.py", "route_a.py", "route_b.py"))
    monkeypatch.setattr(full.FullRun, "media", lambda self, episode: {
        "episode_id": episode["episode_id"], "frames": [{"frame_id": "main-00000", "view": "main"}],
        "warnings": []})
    def review(client, **kwargs):
        assert "GT_SENTINEL" not in str(kwargs)
        assert "REVIEWER_SENTINEL" not in str(kwargs)
        return {"reply": {"data": {"reason": "完成", "checks": {
                    k: {"state": "pass", "evidence_ids": ["C-main-00000"]}
                    for k in ("object", "action", "retry_free", "quality")}},
                    "usage": {}, "model": client.model, "elapsed_s": 1, "call_path": "calls/fake",
                    "request_id": "fake", "request_sha256": "fake",
                    "estimated_cny_before_discounts": 0}, "reference": {}, "extra_calls": []}
    monkeypatch.setattr(full.route_a, "review", review)
    monkeypatch.setattr(full.route_b, "review", review)
    monkeypatch.setattr(full.route_b, "_signature", lambda *a: {"fixture": True})
    write_json(root / "references" / "route_b" / (digest({"fixture": True}) + ".json"), {})
    return root, ids


def test_full_routes_share_the_decision_and_do_not_pass_gt(batch):
    root, ids = batch
    result = full.FullRun(root).case(ids[1], ["A", "B"])
    assert [r["label"] for r in result] == ["correct", "correct"]
    assert all(r["gt"] == "correct" for r in result)
    assert len(full.latest_results(root)) == 2
    assert not list((root / "results").glob("*/*.tmp"))


def test_resume_does_not_reissue_completed_requests(batch, monkeypatch):
    root, ids = batch
    full.run(root, workers=2, first_per_task=True)
    assert len(full.latest_results(root)) == 2
    full.run(root, workers=2)
    assert len(full.latest_results(root)) == 4
    monkeypatch.setattr(full.FullRun, "case", lambda *a: pytest.fail("Repeated paid request"))
    full.run(root, workers=2)
    assert len(list((root / "results").glob("*/*.json"))) == 4


def test_execution_error_is_not_incorrect(batch, monkeypatch):
    root, ids = batch
    def fail(*args, **kwargs):
        raise full.ModelCallError("Qwen HTTP 400: input_limit")
    monkeypatch.setattr(full.route_a, "review", fail)
    rows = full.FullRun(root).case(ids[1], ["A", "B"])
    assert rows[0]["status"] == "failed" and rows[0]["label"] is None
    assert rows[1]["label"] == "correct"


def test_missing_experts_requires_review_without_model_calls(batch, monkeypatch):
    root, ids = batch
    job = full.FullRun(root)
    job.manifest["tasks"]["DL-TEST"]["experts"] = []
    monkeypatch.setattr(full.route_a, "review", lambda *a, **k: pytest.fail("Unexpected model call"))
    assert job.case(ids[1], ["A"])[0]["status"] == "needs_review"


def test_expert_self_review_is_rejected(batch):
    root, ids = batch
    with pytest.raises(ValueError, match="non-expert"):
        full.FullRun(root).case(ids[0], ["A"])


def test_changed_model_cannot_resume_frozen_replay(batch, monkeypatch):
    root, _ = batch
    full.FullRun(root)
    monkeypatch.setenv("QWEN_MODEL", "other-model")
    with pytest.raises(ValueError, match="configuration changed"):
        full.FullRun(root)


def test_first_caption_generation_is_serialized(batch, monkeypatch):
    root, ids = batch
    job = full.FullRun(root)
    active, overlapping = [], []
    original = full.route_b.review
    def review(*args, **kwargs):
        if "DL-TEST" not in job.captions_ready:
            if active:
                overlapping.append(True)
            active.append(True)
            try:
                return original(*args, **kwargs)
            finally:
                active.pop()
        return original(*args, **kwargs)
    monkeypatch.setattr(full.route_b, "review", review)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda episode_id: job.case(episode_id, ["B"]), ids[1:]))
    assert not overlapping
    assert all(row[0]["label"] == "correct" for row in rows)


def test_original_policy_hashes_are_recorded(batch):
    root, _ = batch
    full.FullRun(root)
    config = read_json(root / "full-freeze.json")["configuration"]
    assert config["code"]["model.py"] == sha256(Path(full.__file__).with_name("model.py"))
