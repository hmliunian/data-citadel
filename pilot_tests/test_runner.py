from types import SimpleNamespace

import pytest

from pilot import runner
from pilot.data import digest, read_json, sha256, write_json


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    dataset, run = tmp_path / "source", tmp_path / "run"
    ids = [f"{i:032x}" for i in range(3)]
    episodes = {}
    for episode_id in ids:
        tags = dataset / "tasks" / "DL-TEST" / "api_tags" / (episode_id + ".json")
        receipt = dataset / "receipts" / (episode_id + ".json")
        write_json(tags, {"private": "GT_SENTINEL"})
        write_json(receipt, {"private": "REVIEWER_SENTINEL"})
        episodes[episode_id] = {"episode_id": episode_id, "task_code": "DL-TEST", "quality": "high",
                                "gt_status": "Accepted", "reviewer": "REVIEWER_SENTINEL",
                                "review_time": "2026-09-09", "gt": "correct", "gt_reason": None,
                                "tags_sha256": sha256(tags), "receipt_sha256": sha256(receipt)}
    manifest = {"dataset": str(dataset), "task_code": "DL-TEST", "instruction": "搬运棕色狗",
                "sampling": {}, "episodes": episodes,
                "splits": {"expert_pool": ids[:1], "experts": ids[:1],
                           "development": ids[1:2], "holdout": ids[2:]}}
    manifest["snapshot_sha256"] = digest(manifest)
    write_json(run / "manifest.json", manifest)
    media = {"frames": [{"frame_id": "main-00000", "view": "main", "time_s": 0.1}],
             "warnings": []}
    monkeypatch.setattr(runner, "prepare_media", lambda r, e, s: {**media, "episode_id": e["episode_id"]})
    def review(client, **kwargs):
        assert "GT_SENTINEL" not in str(kwargs)
        assert "REVIEWER_SENTINEL" not in str(kwargs)
        return {"reply": {"data": {"reason": "可见任务完成", "checks": {
                    k: {"state": "pass", "evidence_ids": ["C-main-00000"]}
                    for k in ("object", "action", "retry_free", "quality")}},
                    "usage": {}, "model": client.model, "elapsed_s": 1, "call_path": "calls/fake",
                    "request_id": "fake", "request_sha256": "fake",
                    "estimated_cny_before_discounts": 0}, "reference": {}, "extra_calls": []}
    monkeypatch.setattr(runner.route_a, "review", review)
    monkeypatch.setattr(runner.route_b, "review", review)
    return run, ids


def test_review_adds_gt_only_after_shared_prediction(prepared):
    run, ids = prepared
    result = runner.review_episode(run, ids[1], "A")
    assert result["label"] == "correct"
    assert result["gt"] == "correct"
    assert runner.latest_results(run) == [result]


def test_execution_error_is_not_a_negative_label(prepared, monkeypatch):
    run, ids = prepared
    def broken(*args, **kwargs):
        raise RuntimeError("fake API failure")
    monkeypatch.setattr(runner.route_a, "review", broken)
    result = runner.review_episode(run, ids[1], "A")
    assert result["status"] == "failed" and result["label"] is None
    assert "fake API failure" in result["reason"]


def test_holdout_cannot_decode_before_freeze(prepared, monkeypatch):
    run, ids = prepared
    monkeypatch.setattr(runner, "prepare_media", lambda *a: pytest.fail("held-out media was opened"))
    with pytest.raises(FileNotFoundError):
        runner.review_episode(run, ids[2], "A")
    with pytest.raises(ValueError, match="Both development"):
        runner.freeze(run)


def test_freeze_opens_holdout_and_rejects_development_or_changed_model(prepared):
    run, ids = prepared
    for route in ("A", "B"):
        runner.review_episode(run, ids[1], route)
    write_json(run / "references" / "route_b" / "fake.json", {})
    runner.freeze(run)
    assert runner.review_episode(run, ids[2], "A")["label"] == "correct"
    with pytest.raises(ValueError, match="closed"):
        runner.review_episode(run, ids[1], "A")
    changed = SimpleNamespace(model="different-model", base_url="https://different.invalid")
    with pytest.raises(ValueError, match="Frozen"):
        runner.review_episode(run, ids[2], "A", client=changed)
    assert read_json(run / "freeze.json")["configuration"]["expert_ids"] == ids[:1]


def test_self_expert_is_rejected(prepared):
    run, ids = prepared
    with pytest.raises(ValueError, match="non-expert"):
        runner.review_episode(run, ids[0], "A")


def test_rates_include_abstentions_and_failures_in_denominators():
    result = runner.metrics([
        {"gt": "incorrect", "label": "correct", "status": "completed"},
        {"gt": "incorrect", "label": None, "status": "failed"},
        {"gt": "correct", "label": None, "status": "needs_review"},
        {"gt": "correct", "label": "incorrect", "status": "completed"},
    ])
    assert result["false_accept"] == {"count": 1, "total": 2, "rate": 0.5}
    assert result["false_reject"] == {"count": 1, "total": 2, "rate": 0.5}
    assert result["coverage"] == {"count": 2, "total": 4, "rate": 0.5}
    assert result["failed"] == result["needs_review"] == 1
    assert runner.metrics([])["false_accept"]["rate"] is None


def test_exports_are_append_only(prepared):
    run, ids = prepared
    runner.review_episode(run, ids[1], "A")
    folder1, summary = runner.report(run)
    folder2, _ = runner.report(run)
    assert folder1 != folder2
    assert (folder1 / "predictions.csv").exists()
    assert (folder2 / "predictions.jsonl").exists()
    assert summary["groups"]["development/A"]["requests"] == 1
