import csv
import json

import pytest

from citadel.infrastructure.files import fingerprint, read, write
from scripts.benchmark_report import audit_receipts, publish, schema_failure


@pytest.fixture
def report_case(tmp_path):
    work = tmp_path / "experiment"
    ids = ["a" * 32, "b" * 32]
    messages = [{"role": "system", "content": "Return JSON."}]
    audit = {episode_id: {"frames": 4, "task_messages_sha256": fingerprint(messages),
                          "quality_messages_sha256": fingerprint(messages)} for episode_id in ids}
    plan = {"models": ["fake"], "output": "artifacts/experiments/fake", "temperature": 0,
            "max_tokens": 5000, "timeout_s": 180, "attempts": 2}
    frozen = {"created_at": "2026-09-15T00:00:00+00:00", "git_commit": "test", "purpose": "test",
              "plan": plan, "plan_sha256": fingerprint(plan), "runner_sha256": "test",
              "configuration": {}, "input_audit": audit, "resources": {}, "model_settings": {},
              "manifest": {"splits": {"development": ids[:1], "holdout": ids[1:]},
                           "episodes": {episode_id: {"gt": "correct" if i == 0 else "incorrect",
                                        "gt_reason": None if i == 0 else "PRIVATE_GT_REASON", "task_code": "DL-TEST"}
                                        for i, episode_id in enumerate(ids)}},
              "pricing": {"models": {"fake": {"source": "https://example.invalid", "tiers": [[1000000, 1, 2, .2]]}}}}
    frozen["sha256"] = fingerprint(frozen)
    write(work / "experiment.json", frozen)
    for episode_id in ids:
        write(work / "inputs" / (episode_id + ".json"), {"task": messages, "quality": messages})
        write(work / "models/fake/results" / (episode_id + ".json"),
              {"status": "completed", "label": frozen["manifest"]["episodes"][episode_id]["gt"],
               "configuration_sha256": frozen["sha256"], "input_sha256": fingerprint(audit[episode_id]),
               "result_id": episode_id, "finished_at": "2026-09-15T00:01:00+00:00", "elapsed_s": 4})
        for stage in ("task", "quality"):
            folder = work / "models/fake/calls" / (episode_id + "-" + stage)
            write(folder / "request.json", {"attempt": 1, "model": "fake", "input_images": 0,
                  "created_at": "2026-09-15T00:00:30+00:00", "request_sha256": "test", "parameters": {},
                  "messages": messages, "context": {"episode_id": episode_id, "result_id": episode_id,
                      "stage": stage, "configuration_sha256": frozen["sha256"]}})
            write(folder / "response.json", {"status": 200, "elapsed_s": 2, "body": {
                "model": "fake", "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}})
    return work, frozen


def test_publish_reconciles_receipts_and_is_reproducible(tmp_path, report_case):
    work, _ = report_case
    output = tmp_path / "docs"
    summary = publish(work, output)
    assert summary["audit"]["requests_checked"] == 4
    assert summary["models"][0]["matched"] == 2
    assert summary["models"][0]["list_cny"] == pytest.approx(.0008)
    assert summary["models"][0]["api_p50_s"] == 4
    assert len(list(csv.DictReader((output / "calls.csv").open()))) == 4
    assert "PRIVATE_GT_REASON" not in (output / "calls.csv").read_text()
    assert "PRIVATE_GT_REASON" in (output / "episodes.csv").read_text()
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert publish(work, output) == summary
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    (output / "README.md").write_text("reviewed content")
    with pytest.raises(ValueError, match="Published document differs"):
        publish(work, output)


def test_receipt_audit_rejects_changed_model_messages(report_case):
    work, frozen = report_case
    path = next((work / "models/fake/calls").glob("*/request.json"))
    request = read(path)
    request["messages"][0]["content"] = "Different prompt"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="Request differs"):
        audit_receipts(work, frozen)


def test_publish_rejects_partial_scope(tmp_path, report_case):
    work, _ = report_case
    next((work / "models/fake/results").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="still running"):
        publish(work, tmp_path / "docs")


def test_schema_diagnosis_keeps_extra_fields_and_does_not_rescore(tmp_path, answer):
    answer.pop("quality_by_camera")
    answer["hold"]["duration_s"] = 2
    folder = tmp_path / "models/fake/calls/example"
    write(folder / "response.json", {"body": {"choices": [{"message": {"content": json.dumps(answer)}}]}})
    result = {"status": "failed", "label": None, "error": {"stage": "evidence", "type": "ValidationError"},
              "model_call": {"call_path": "calls/example"}}
    assert schema_failure(tmp_path, "fake", result) == "model_call: hold.duration_s / extra_forbidden"
    assert result["status"] == "failed" and result["label"] is None
