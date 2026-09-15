import copy
import json

import pytest

from citadel.application.prompts import PromptBuilder
from citadel.configuration import PromptBundle
from citadel.domain.tasks import profile_for
from citadel.infrastructure.files import fingerprint, write
from citadel.infrastructure.resources import image_input
from scripts.benchmark import Benchmark, PacedGateway
from scripts.benchmark_report import price_usage, summarize


def test_prepared_inputs_survive_config_snapshot_round_trip(tmp_path, service_case, monkeypatch):
    run, fake = service_case
    benchmark = Benchmark.__new__(Benchmark)
    benchmark.work, benchmark.inputs = tmp_path / "benchmark", run.artifacts.work
    benchmark.source, benchmark.config, benchmark.pricing = run.source, run.configuration, {}
    benchmark.plan = {"models": ["fake"], "resource_base": "https://example.invalid"}
    benchmark.plan_path = tmp_path / "plan.toml"
    benchmark.plan_path.write_text('models = ["fake"]')
    media = {}
    for episode_id in run.source.records():
        source = run.source.source(episode_id)
        item = run.reviews.media.prepare(source, run.source.manifest["sampling"])
        item["signature"] = {**run.source.manifest["sampling"], "mcap_sha256": source.mcap_sha256}
        item["gripper"] = {"sha256": "empty", "warnings": [], "channels": {}}
        media[episode_id] = item
    benchmark.media = lambda episode_id: copy.deepcopy(media[episode_id])
    monkeypatch.setattr("scripts.benchmark.fetch", lambda *args: run.resources.get("DL-TEST"))
    frozen = benchmark.prepare()
    # Loading the saved snapshot is what a later, separate run process does.
    saved = json.loads((benchmark.work / "experiment.json").read_text())
    assert saved == frozen
    result = benchmark.run_one("fake", sorted(media)[0], saved, PacedGateway(fake, 1_000_000))
    assert result["status"] == "completed" and len(fake.requests) == 2


@pytest.fixture
def benchmark_case(tmp_path, service_case):
    run, fake = service_case
    benchmark = Benchmark.__new__(Benchmark)
    benchmark.work = tmp_path / "benchmark"
    benchmark.inputs = run.artifacts.work
    benchmark.source = run.source
    episode_id = run.source.manifest["splits"]["development"][0]
    media = run.reviews.media.prepare(run.source.source(episode_id), run.snapshot().data["sampling"])
    resources = run.resources.get("DL-TEST")
    write(benchmark.inputs / "resources/DL-TEST/task.json", resources)
    benchmark.media = lambda _: copy.deepcopy(media)
    snapshot = run.snapshot().data
    builder = PromptBuilder(PromptBundle(**snapshot["prompts"]), image_input)
    profile = profile_for(resources, snapshot["profiles"])
    audit = {"task_messages_sha256": fingerprint(builder.review(benchmark.inputs, resources, profile, media)),
             "quality_messages_sha256": fingerprint(builder.quality(benchmark.inputs, media))}
    frozen = {"sha256": "test-version", "configuration": snapshot, "input_audit": {episode_id: audit}}
    return benchmark, fake, episode_id, frozen


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


def test_benchmark_rejects_changed_inputs_before_calling_model(benchmark_case):
    benchmark, fake, episode_id, frozen = benchmark_case
    frozen["input_audit"][episode_id]["task_messages_sha256"] = "another-input"
    with pytest.raises(ValueError, match="input changed"):
        benchmark.run_one("fake", episode_id, frozen, PacedGateway(fake, 1_000_000))
    assert fake.requests == []


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
