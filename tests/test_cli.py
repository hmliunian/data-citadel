import json

from data_citadel.cli import main
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
