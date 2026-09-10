import csv
import json

import pytest

from pilot.data import read_json, write_json
from pilot.full_report import report

P1, P2, N1, UNKNOWN, P3, N2, E1, E2, DUPLICATE = [f"{n:032x}" for n in range(9)]
TASK_OF = {P1: "DL-T1", P2: "DL-T1", N1: "DL-T1", UNKNOWN: "DL-T1", P3: "DL-T2", N2: "DL-T2",
           E1: "DL-T1", E2: "DL-T2", DUPLICATE: "DL-T1"}


@pytest.fixture
def run_dir(tmp_path):
    episodes = {episode_id: {"task_code": task, "gt": "incorrect" if episode_id in (N1, N2) else "correct"}
                for episode_id, task in TASK_OF.items()}
    episodes[UNKNOWN]["gt"] = None
    for row in episodes.values():
        row["gt_status"] = {"correct": "Accepted", "incorrect": "Denied"}.get(row["gt"])
    write_json(tmp_path / "manifest.json", {
        "routes": ["A", "B"], "episodes": episodes,
        "tasks": {"DL-T1": {"candidate_ids": [P1, P2, N1, UNKNOWN], "experts": [E1], "expert_pool": [E1]},
                  "DL-T2": {"candidate_ids": [P3, N2], "experts": [E2], "expert_pool": [E2]}},
        "exclusions": [{"episode_id": DUPLICATE, "task_code": "DL-T1", "reason": "duplicate"}],
        "source_inventory": {"episodes": 9, "tasks": 2},
    })
    return tmp_path


def result(run_dir, episode_id, *, route="A", status="completed", label="correct", tick=1, **extra):
    row = {"episode_id": episode_id, "task_code": TASK_OF[episode_id], "route": route,
           "gt": "incorrect", "status": status, "label": label, "reason": "测试结果",
           "created_at": f"2026-09-09T00:00:{tick:02d}+00:00", **extra}
    write_json(run_dir / "results" / row["task_code"] / f"{route}-{episode_id}-{tick:08d}.json", row)


def test_full_denominators_and_unknown_gt_are_distinct_from_prediction_accuracy(run_dir):
    result(run_dir, P1)
    result(run_dir, P2, label="incorrect")
    result(run_dir, N1)
    result(run_dir, UNKNOWN)
    result(run_dir, N2, status="needs_review", label=None)
    folder, summary = report(run_dir)
    assert summary["source_inventory"] == {"episodes": 9, "tasks": 2}
    a = summary["routes"]["A"]["overall"]
    assert [a[key] for key in ("total", "attempted", "pending", "completed", "needs_review", "failed")] == [6, 5, 1, 4, 1, 0]
    assert [a[key] for key in ("gt_correct", "gt_incorrect", "gt_valid", "gt_unknown")] == [3, 2, 5, 1]
    assert a["gt_agreement"] == {"count": 1, "total": 5, "rate": 0.2}
    assert a["prediction_accuracy"] == {"count": 1, "total": 3, "rate": 1 / 3}
    assert a["positive_pass_rate"]["rate"] == 1 / 3
    assert a["predicted_correct_rate"]["rate"] == 0.5
    assert a["false_accept"] == {"count": 1, "total": 2, "rate": 0.5}
    assert a["false_reject"]["rate"] == 1 / 3
    assert a["coverage"]["rate"] == 4 / 6
    assert a["processing_success"] == {"count": 5, "total": 5, "rate": 1}
    assert summary["partial"] and a["partial"]
    assert summary["routes"]["B"]["overall"]["pending"] == 6
    assert summary["routes"]["B"]["tasks"]["DL-T2"]["pending"] == 2
    exported = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert len(exported) == 12 and sum(row["status"] == "pending" for row in exported) == 7
    assert next(row for row in exported if row["episode_id"] == P1 and row["route"] == "A")["gt"] == "correct"
    with (folder / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 12


@pytest.mark.parametrize("status,successes", [("failed", 0), ("needs_review", 6)])
def test_all_failures_or_reviews_do_not_become_incorrect_predictions(run_dir, status, successes):
    for route in ("A", "B"):
        for episode_id in (P1, P2, N1, UNKNOWN, P3, N2):
            result(run_dir, episode_id, route=route, status=status, label=None)
    _, summary = report(run_dir)
    assert not summary["partial"]
    for route in summary["routes"].values():
        counts = route["overall"]
        assert counts[status] == 6 and counts["completed"] == 0
        assert counts["gt_agreement"] == {"count": 0, "total": 5, "rate": 0}
        assert counts["prediction_accuracy"]["rate"] is None
        assert counts["false_accept"]["count"] == counts["false_reject"]["count"] == 0
        assert counts["processing_success"] == {"count": successes, "total": 6, "rate": successes / 6}


def test_latest_result_per_route_and_exclusions_never_expand_population(run_dir):
    result(run_dir, P1, status="failed", label=None)
    result(run_dir, P1, tick=2)
    result(run_dir, P1, route="B")
    result(run_dir, P1, route="B", tick=3, label="incorrect")
    result(run_dir, E1)
    result(run_dir, DUPLICATE)
    folder, summary = report(run_dir)
    assert summary["superseded_results"] == 2 and len(summary["ignored_result_files"]) == 2
    assert summary["routes"]["A"]["overall"]["gt_agreement"]["count"] == 1
    assert summary["routes"]["B"]["overall"]["false_reject"]["count"] == 1
    for route in summary["routes"].values():
        assert route["overall"]["total"] == 6 and route["tasks"]["DL-T2"]["pending"] == 2
    assert summary["exclusions"]["by_reason"] == {"duplicate": 1, "expert_pool": 2}
    excluded = [json.loads(line) for line in (folder / "exclusions.jsonl").read_text().splitlines()]
    assert {row["episode_id"] for row in excluded} == {E1, E2, DUPLICATE}
    before = (folder / "summary.json").read_bytes()
    second, _ = report(run_dir)
    assert second != folder and (folder / "summary.json").read_bytes() == before


def call(run_dir, name, *, route="A", purpose="review", tokens=100, status=200, error=False):
    write_json(run_dir / "calls" / name / "request.json", {
        "context": {"route": route, "task_code": "DL-T1", "episode_id": P1, "purpose": purpose},
        "model": "qwen-vl-max", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "attempt": 1, "input_images": 8})
    if status is not None:
        write_json(run_dir / "calls" / name / "response.json", {"http_status": status, "elapsed_s": 2.0,
                   "body": {"usage": {"prompt_tokens": tokens, "completion_tokens": tokens // 10}}})
    if error:
        write_json(run_dir / "calls" / name / "error.json", {"type": "ReadTimeout", "elapsed_s": 5.0})


def test_all_http_attempts_errors_and_b_captions_are_billed_once(run_dir):
    call(run_dir, "a-first", status=429)
    call(run_dir, "a-second", tokens=200)
    call(run_dir, "b-caption", route="B", purpose="expert_caption", tokens=300)
    call(run_dir, "b-inline", route="B", tokens=400)
    call(run_dir, "b-review", route="B", tokens=500)
    call(run_dir, "transport-error", status=None, error=True)
    call(run_dir, "in-flight", route="B", purpose="", status=None)
    result(run_dir, P1, status="failed", label=None, review_call="calls/a-first")
    result(run_dir, P1, tick=2, review_call="calls/a-second", usage={"prompt_tokens": 999999})
    result(run_dir, P1, route="B", review_call="calls/b-review", extra_calls=["calls/b-inline"])
    _, summary = report(run_dir)
    usage = summary["billing"]
    assert usage["http_attempts"] == 7 and usage["http_errors"] == 1
    assert usage["transport_errors"] == 1 and usage["incomplete"] == 1 and usage["usage_missing"] == 2
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (1500, 150, 1650)
    assert usage["estimated_cny_before_discounts"] == pytest.approx((1500 * 1.6 + 150 * 4) / 1e6)
    b = summary["routes"]["B"]["overall"]["billing"]
    assert b["prompt_tokens"] == 1200
    assert b["purpose_counts"] == {"expert_caption": 2, "review": 1, "unknown": 1}
    assert summary["routes"]["B"]["tasks"]["DL-T2"]["billing"]["http_attempts"] == 0


def test_empty_population_and_excluded_overlap(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    manifest["tasks"]["DL-T1"]["candidate_ids"].append(E1)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Excluded episodes"):
        report(run_dir)
    for task in manifest["tasks"].values():
        task["candidate_ids"] = []
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    _, summary = report(run_dir)
    for route in summary["routes"].values():
        counts = route["overall"]
        assert counts["total"] == 0
        assert all(value["rate"] is None for value in counts.values() if isinstance(value, dict) and "rate" in value)


def test_partially_written_http_response_keeps_attempt_incomplete(run_dir):
    call(run_dir, "writing", route="B", status=None)
    (run_dir / "calls" / "writing" / "response.json").write_text('{"http_status": 200,')
    _, summary = report(run_dir)
    assert summary["billing"]["http_attempts"] == 1
    assert summary["billing"]["incomplete"] == 1
    assert summary["billing"]["usage_missing"] == 1
    assert summary["routes"]["B"]["overall"]["billing"]["http_attempts"] == 1
