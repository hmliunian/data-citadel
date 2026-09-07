import json

from data_citadel.cli import main
from data_citadel.experts import ExpertLibrary
from data_citadel.repository import EpisodeRepository
from test_repository import bundle


def test_first_run_inventory_and_prepare_need_no_experts_or_key(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "datasets"
    bundle(dataset)
    # A broken expert configuration must not disable unrelated data commands.
    broken = tmp_path / "experts.json"
    broken.write_text("not json")
    monkeypatch.setenv("QWEN_API_KEY_FILE", str(tmp_path / "absent-key"))
    args = ["--dataset-root", str(dataset), "--experts", str(broken)]
    assert main([*args, "inventory"]) == 0
    assert json.loads(capsys.readouterr().out)["total_episodes"] == 1
    target = tmp_path / "config"
    assert main([*args, "prepare", "--output-dir", str(target)]) == 0
    capsys.readouterr()
    experts = json.loads((target / "experts.json").read_text())
    assert all(group["approved"] is False for group in experts["groups"])
    before = (target / "experts.json").read_bytes()
    assert main([*args, "prepare", "--output-dir", str(target)]) == 1
    assert (target / "experts.json").read_bytes() == before


def test_prepare_reuses_source_reviews_without_splitting_objects(tmp_path, capsys):
    dataset = tmp_path / "datasets"
    for index in range(10):
        bundle(dataset, f"{index:032x}", document={
            "task.action_id": "A_001", "task.task_code": f"source-task-{index}",
            "task.action_text": {"rendered_zh": f"拿起物体 {index}"},
            "task.collector.user": f"collector-{index % 2}",
            "task.review.status": "Accepted", "task.review.reviewer": "source-reviewer",
        })
    target = tmp_path / "config"
    assert main(["--dataset-root", str(dataset), "prepare", "--output-dir", str(target)]) == 0
    report = json.loads(capsys.readouterr().out)["report"]["actions"]["A_001"]
    assert report["correct_candidate_count"] == 10
    assert report["expert_candidates"] == report["baseline_selected_by_label"]["correct"] == 5
    repository = EpisodeRepository(dataset)
    experts = ExpertLibrary(target / "experts.json", repository)
    references = experts.resolve(repository.get(f"{9:032x}"))
    assert [e.episode_id for e in references] == [f"{i:032x}" for i in range(5)]
    assert experts.groups[0]["approval_source"] == "dataset_review"
    assert experts.groups[0]["approved"] is True
    evaluation = json.loads((target / "evaluation.json").read_text())
    assert {s["episode_id"] for s in evaluation["samples"]} == {f"{i:032x}" for i in range(5, 10)}
