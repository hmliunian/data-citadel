import copy
import json
from pathlib import Path

import pytest

from citadel.domain.errors import BusyError
from citadel.infrastructure.execution import LocalExecutor
from citadel.infrastructure.files import file_hash, fingerprint, read, write
from citadel.infrastructure.resources import fetch
from citadel.infrastructure.storage import ArtifactRepository
from scripts.benchmark import Benchmark, PacedGateway
from scripts.benchmark_report import price_usage, summarize


@pytest.fixture
def benchmark_setup(tmp_path, service_case, model_case, jpeg, monkeypatch):
    run, fake = service_case
    _, references, _, template = model_case
    pricing = tmp_path / "pricing.json"
    write(pricing, {"models": {"fake": {"tiers": [[1_000_000, 1, 2, .2]]}}})
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        f'dataset = {json.dumps(run.source.manifest["dataset"])}\n'
        f'output = {json.dumps(str(tmp_path / "benchmark"))}\n'
        f'config_root = {json.dumps(str(run.configuration.root))}\n'
        f'pricing = {json.dumps(str(pricing))}\n'
        'models = ["fake"]\nresource_base = "https://example.invalid"\n'
        'base_url = "https://example.invalid/v1"\ndefault_tpm = 1000000\nmodel_tpm = {}\n')
    prepared = []
    def load_resources(code, work, base):
        folder = work / "resources" / code
        if not (folder / "task.json").exists():
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "object.jpg").write_bytes(jpeg)
            value = copy.deepcopy(references)
            value["task_code"] = code
            value["images"][0]["path"] = str((folder / "object.jpg").relative_to(work))
            value["sha256"] = fingerprint(value)
            write(folder / "task.json", value)
        return fetch(code, work, base)
    def load_media(work, source, sampling):
        assert set(source) == {"episode_id", "mcap_path", "mcap_sha256"}
        assert file_hash(Path(source["mcap_path"])) == source["mcap_sha256"]
        episode_id = source["episode_id"]
        folder = work / "media" / episode_id
        if not (folder / "media.json").exists():
            prepared.append(episode_id)
            folder.mkdir(parents=True, exist_ok=True)
            media = copy.deepcopy(template)
            media.update(episode_id=episode_id, videos={},
                         signature={**sampling, "mcap_sha256": source["mcap_sha256"]})
            for frame in media["frames"]:
                target = folder / (frame["frame_id"] + ".jpg")
                target.write_bytes(jpeg)
                frame["path"] = str(target.relative_to(work))
            write(folder / "media.json", media)
            gripper = {"warnings": [], "channels": {}}
            write(folder / "gripper.json", {**gripper, "sha256": fingerprint(gripper)})
        return ArtifactRepository(work).media(episode_id)
    monkeypatch.setattr("scripts.benchmark.fetch", load_resources)
    monkeypatch.setattr("scripts.benchmark.prepare_media", load_media)
    return Benchmark(plan_path), fake, prepared


@pytest.fixture
def benchmark_case(benchmark_setup):
    benchmark, fake, _ = benchmark_setup
    frozen = benchmark.prepare()
    return benchmark, fake, sorted(frozen["input_audit"])[0], frozen


def test_benchmark_prepares_without_legacy_cache_and_reopens(benchmark_setup, service_case):
    benchmark, fake, prepared = benchmark_setup
    assert not benchmark.work.exists()
    frozen = benchmark.prepare()
    assert fake.requests == []
    assert benchmark.inputs.is_relative_to(benchmark.work)
    assert frozen["manifest"]["splits"] == service_case[0].source.manifest["splits"]
    assert set(prepared) == set(frozen["input_audit"])
    assert "PRIVATE_GT_REASON" not in (benchmark.work / "inputs" / (prepared[0] + ".json")).read_text()
    reopened = Benchmark(benchmark.plan_path)
    assert reopened.prepare() == frozen == read(benchmark.work / "experiment.json")
    assert len(prepared) == len(frozen["input_audit"])
    result = reopened.run_one("fake", prepared[0], frozen, PacedGateway(fake, 1_000_000))
    assert result["status"] == "completed" and len(fake.requests) == 2


def test_benchmark_resumes_preparation_without_overwriting_inputs(benchmark_setup, monkeypatch):
    benchmark, fake, prepared = benchmark_setup
    from scripts import benchmark as module
    loader = module.prepare_media
    def interrupt(work, source, sampling):
        if len(prepared) == 1:
            raise RuntimeError("interrupted preparation")
        return loader(work, source, sampling)
    monkeypatch.setattr(module, "prepare_media", interrupt)
    with pytest.raises(RuntimeError, match="interrupted preparation"):
        benchmark.prepare()
    assert not (benchmark.work / "experiment.json").exists()
    first = benchmark.work / "inputs" / (prepared[0] + ".json")
    original = first.read_bytes()
    monkeypatch.setattr(module, "prepare_media", loader)
    reopened = Benchmark(benchmark.plan_path)
    frozen = reopened.prepare()
    assert len(prepared) == len(set(prepared)) == len(frozen["input_audit"])
    assert first.read_bytes() == original and fake.requests == []


def test_interrupted_preparation_cannot_mix_configurations(benchmark_setup, monkeypatch):
    benchmark, fake, prepared = benchmark_setup
    def interrupt(*args):
        raise RuntimeError("interrupted")
    monkeypatch.setattr("scripts.benchmark.prepare_media", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        benchmark.prepare()
    prompt = benchmark.config.root / "prompts/quality.txt"
    prompt.write_text(prompt.read_text() + "\nchanged")
    with pytest.raises(ValueError, match="Frozen content changed"):
        Benchmark(benchmark.plan_path).prepare()
    assert not prepared and fake.requests == []


def test_benchmark_reuses_results_and_separates_model_inputs_from_gt(benchmark_case):
    benchmark, fake, episode_id, frozen = benchmark_case
    gateway = PacedGateway(fake, 1_000_000)
    result = benchmark.run_one("fake", episode_id, frozen, gateway)
    cached = benchmark.run_one("fake", episode_id, frozen, gateway)
    assert result["status"] == "completed" and cached["result_id"] == result["result_id"]
    assert len(fake.requests) == 2
    assert "PRIVATE_GT_REASON" not in json.dumps(fake.requests)
    second = benchmark.run_one("another-model", episode_id, frozen, gateway)
    assert second["status"] == "completed"
    assert fake.requests[:2] == fake.requests[2:]


def test_benchmark_does_not_replay_an_interrupted_paid_request(benchmark_case):
    benchmark, fake, episode_id, frozen = benchmark_case
    write(benchmark.work / "models/fake/results" / (episode_id + ".started.json"), {"status": "started"})
    with pytest.raises(RuntimeError, match="receipt inspection"):
        benchmark.run_one("fake", episode_id, frozen, PacedGateway(fake, 1_000_000))
    assert fake.requests == []


@pytest.mark.parametrize("changed", ["audit", "frame", "reference", "gripper", "media", "mcap", "receipt"])
def test_benchmark_rejects_changed_inputs_before_calling_model(benchmark_case, changed):
    benchmark, fake, episode_id, frozen = benchmark_case
    folder = benchmark.inputs / "media" / episode_id
    if changed == "audit":
        frozen["input_audit"][episode_id]["task_messages_sha256"] = "another-input"
    elif changed == "frame":
        (folder / "V000.jpg").write_bytes(b"changed frame")
    elif changed == "reference":
        (benchmark.inputs / "resources/DL-TEST/object.jpg").write_bytes(b"changed reference")
    elif changed == "gripper":
        # A valid new receipt still differs from the frozen input.
        value = {"channels": {}, "warnings": ["changed"]}
        (folder / "gripper.json").write_text(json.dumps({**value, "sha256": fingerprint(value)}))
    elif changed == "media":
        path = folder / "media.json"
        value = read(path)
        value["warnings"] = ["changed"]
        path.write_text(json.dumps(value))
    elif changed == "mcap":
        benchmark.source.source(episode_id).mcap_path.write_bytes(b"changed source")
    else:
        path = benchmark.dataset / "receipts" / (episode_id + ".json")
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        benchmark.run_one("fake", episode_id, frozen, PacedGateway(fake, 1_000_000))
    assert fake.requests == []
    assert not (benchmark.work / "models/fake/results" / (episode_id + ".started.json")).exists()


@pytest.mark.parametrize("changed", ["plan", "prompt"])
def test_benchmark_rejects_changed_frozen_protocol(benchmark_case, changed):
    benchmark, fake, _, _ = benchmark_case
    path = (benchmark.plan_path if changed == "plan" else
            benchmark.config.root / "prompts/quality.txt")
    path.write_text(path.read_text() + "\n# changed")
    with pytest.raises(ValueError, match="changed"):
        Benchmark(benchmark.plan_path).validate()
    assert fake.requests == []


def test_benchmark_guards_preparation_and_run_with_the_same_lock(benchmark_case):
    benchmark, fake, _, _ = benchmark_case
    guard = LocalExecutor(benchmark.work, 1)
    try:
        for action in (benchmark.prepare, benchmark.run):
            with pytest.raises(BusyError, match="already has a scheduler"):
                action()
    finally:
        guard.close()
    assert fake.requests == []


def test_benchmark_refuses_output_inside_source_dataset(benchmark_setup):
    benchmark, _, _ = benchmark_setup
    plan = benchmark.plan_path
    plan.write_text(plan.read_text().replace(str(benchmark.work), str(benchmark.dataset / "output")))
    with pytest.raises(ValueError, match="read-only dataset"):
        Benchmark(plan)
    assert not (benchmark.dataset / "output").exists()


def test_cost_selects_one_tier_per_request_and_cache_is_not_double_counted():
    tiers = [[32768, 1, 10, .2], [131072, 1.5, 15, .3]]
    first = price_usage({"prompt_tokens": 32768, "completion_tokens": 100,
                         "prompt_tokens_details": {"cached_tokens": 10000}}, tiers)
    assert first["list_cny"] == pytest.approx(.033768)
    assert first["cache_adjusted_cny"] == pytest.approx(.025768)
    second = price_usage({"prompt_tokens": 32769, "completion_tokens": 100}, tiers)
    assert second["list_cny"] == pytest.approx((32769 * 1.5 + 100 * 15) / 1e6)
    assert price_usage({}, tiers)["list_cny"] is None


def test_failed_attempt_costs_and_unfinished_episodes_remain_in_report(tmp_path):
    model, episode_id = "fake", "a" * 32
    frozen = {"sha256": "test", "plan": {"models": [model]}, "manifest": {
        "splits": {"development": [episode_id, "b" * 32], "holdout": []},
        "episodes": {episode_id: {"gt": "incorrect", "gt_reason": "blur", "task_code": "DL-TEST"},
                     "b" * 32: {"gt": "correct", "gt_reason": None, "task_code": "DL-TEST"}}},
        "pricing": {"models": {model: {"tiers": [[1000000, 1, 2, .2]]}}}}
    write(tmp_path / "experiment.json", frozen)
    for i, usage in enumerate((None, {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})):
        folder = tmp_path / "models" / model / "calls" / str(i)
        write(folder / "request.json", {"attempt": i + 1, "model": model, "input_images": 1,
              "request_sha256": "same", "parameters": {}, "context": {"episode_id": episode_id}})
        write(folder / "response.json", {"status": 429 if usage is None else 200, "elapsed_s": 2,
                                         "body": {"usage": usage or {}}})
    write(tmp_path / "models" / model / "results" / (episode_id + ".json"),
          {"status": "failed", "label": None, "error": {"stage": "evidence"}})
    summary, _, _, _ = summarize(tmp_path)
    result = summary["models"][0]
    assert result["total"] == 2 and result["failed"] == 1 and result["not_run"] == 1
    assert result["matched"] == 0 and result["retries"] == 1 and result["unknown_usage_calls"] == 1
    assert result["list_cny"] == pytest.approx(.0002)
