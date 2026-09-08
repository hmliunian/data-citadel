import pytest

from data_citadel.models import Episode, Evidence
from data_citadel.review.policy import decide
from test_review import passing


@pytest.mark.parametrize("views,expected", [
    (["left_wrist", "right_wrist"], "uncertain"),
    (["main", "left_wrist"], "uncertain"),
    (["main", "main"], "correct"),
    ([None, None], "correct"),
])
def test_task_needs_main_view_process_and_completion(tmp_path, views, expected):
    episode = Episode("candidate", "A_001", "task", "拿起杯子", "collector",
                      tmp_path / "episode.mcap", tmp_path / "episode.json", label="correct")
    evidence = [Evidence(view=view, timestamp_s=float(index), description="可见状态")
                for index, view in enumerate(views)]
    assessments = {stage: passing() for stage in ("integrity", "generic", "task")}
    assessments["task"] = passing(evidence=evidence)
    result = decide(episode, assessments)
    assert result.verdict == expected
    assert result.ground_truth_candidate is (expected == "correct")
