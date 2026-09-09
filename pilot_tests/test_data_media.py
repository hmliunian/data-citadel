import pytest

from pilot.data import choose_samples
from pilot.media import sample_indices


def records():
    result = []
    for i in range(12):
        result.append({"episode_id": f"{i:032x}", "task_code": "DL-TEST",
                       "mcap_sha256": f"hash-{i}", "quality": "high" if i < 3 else "medium",
                       "gt_status": "Accepted" if i < 10 else "Denied",
                       "reviewer": "source-reviewer", "review_time": "2026-09-09",
                       "gt": "incorrect" if i >= 10 else "correct"})
    return result


def test_splits_keep_expert_pool_and_evaluation_independent():
    result = choose_samples(records(), 2, "fixed")
    groups = [set(result[k]) for k in ("expert_pool", "development", "holdout")]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    assert [len(g) for g in groups] == [3, 4, 4]
    assert choose_samples(records(), 2, "fixed") == result


def test_duplicate_expert_recording_cannot_be_evaluated():
    rows = records()
    rows[3]["mcap_sha256"] = rows[0]["mcap_sha256"]
    result = choose_samples(rows, 2, "fixed")
    assert rows[3]["episode_id"] in result["duplicate_exclusions"]
    assert rows[3]["episode_id"] not in result["development"] + result["holdout"]


def test_sampling_uses_real_times_and_keeps_endpoints():
    times = [0.15, 0.3, 1.99, 2.03, 3.99, 4.04, 5.4]
    indices, gaps = sample_indices(times)
    assert indices == [0, 2, 4, 6]
    assert gaps == []
    indices, gaps = sample_indices([0.1, 0.2, 3.9, 4.0, 5.0])
    assert indices == [0, 3, 4]
    assert gaps == [2.0]


def test_invalid_timestamps_are_rejected():
    with pytest.raises(ValueError):
        sample_indices([0.2, 0.1])


def test_high_without_review_is_reserved_but_not_selected():
    rows = records()
    rows[0]["reviewer"] = None
    result = choose_samples(rows, 2, "fixed")
    assert rows[0]["episode_id"] in result["expert_pool"]
    assert rows[0]["episode_id"] not in result["experts"]
    with pytest.raises(ValueError, match="original review records"):
        choose_samples(rows, 3, "fixed")


def test_cross_task_experts_are_rejected():
    rows = records()
    rows[0]["task_code"] = "DL-OTHER"
    with pytest.raises(ValueError, match="task codes"):
        choose_samples(rows, 2, "fixed")
