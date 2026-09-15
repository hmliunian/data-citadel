"""Run the unchanged review pipeline against fixed, verified regression inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
import tomllib
import uuid

from citadel.application.prompts import PromptBuilder
from citadel.application.reviews import ReviewPipeline
from citadel.configuration import Configuration, ModelSettings, PROJECT_ROOT, PromptBundle, code_version
from citadel.domain.tasks import profile_for
from citadel.infrastructure.execution import LocalExecutor
from citadel.infrastructure.files import file_hash, fingerprint, now, read, write
from citadel.infrastructure.mcap.sensors import read_gripper
from citadel.infrastructure.qwen import QwenGateway, safe_messages
from citadel.infrastructure.resources import fetch, image_input
from citadel.infrastructure.storage import EpisodeRepository


def load_plan(path):
    return tomllib.loads(path.read_text())


def frozen_write(path, value):
    try:
        write(path, value)
    except FileExistsError:
        if read(path) != value:
            raise ValueError("Frozen benchmark content changed: " + str(path))


class PacedGateway:
    """One worker per model; pacing is excluded from HTTP latency metrics."""
    def __init__(self, gateway, tokens_per_minute):
        self.gateway = gateway
        self.rate = tokens_per_minute * 0.8 / 60
        self.next_start = 0.0
        self.wait_s = 0.0

    def complete(self, messages, context, *, quality_only=False):
        pause = max(0, self.next_start - time.monotonic())
        time.sleep(pause)
        self.wait_s += pause
        started = time.monotonic()
        value = self.gateway.complete(messages, context, quality_only=quality_only)
        tokens = value.get("usage", {}).get("total_tokens", 0)
        self.next_start = started + tokens / self.rate
        return value


class Benchmark:
    def __init__(self, plan_path):
        self.plan_path, self.plan = plan_path, load_plan(plan_path)
        self.work = PROJECT_ROOT / self.plan["output"]
        self.inputs = PROJECT_ROOT / self.plan["input_run"]
        self.source = EpisodeRepository(self.inputs)
        self.config = Configuration(PROJECT_ROOT / self.plan["config_root"])
        self.pricing = read(PROJECT_ROOT / self.plan["pricing"])
        self.work.mkdir(parents=True, exist_ok=True)

    def settings(self, model):
        values = {key: self.plan[key] for key in ModelSettings.model_fields if key in self.plan}
        return ModelSettings(model=model, **values)

    def media(self, episode_id):
        folder = self.inputs / "media" / episode_id
        return {**read(folder / "media.json"), "gripper": read_gripper(folder / "gripper.json")}

    def prepare(self):
        """Read source/cache files and freeze fresh resources; no model requests."""
        marker = self.work / "experiment.json"
        if marker.exists():
            return self.validate()
        models = self.plan["models"]
        if len(set(models)) != len(models) or not models:
            raise ValueError("Benchmark needs distinct model IDs")
        resources = {}
        for code in sorted({row["task_code"] for row in self.source.records().values()}):
            fresh = fetch(code, self.work / "fresh_resources", self.plan["resource_base"])
            cached = fetch(code, self.inputs)
            if fresh["sha256"] != cached["sha256"]:
                raise ValueError("Current references differ from the input cache; prepare a new input run")
            resources[code] = cached["sha256"]
        snapshot = self.config.snapshot(self.source.manifest, self.plan["resource_base"]).data
        snapshot.pop("model")
        builder = PromptBuilder(PromptBundle(**snapshot["prompts"]), image_input)
        audit = {}
        for episode_id in sorted(self.source.records()):
            source = self.source.source(episode_id)
            media = self.media(episode_id)
            if media["signature"]["mcap_sha256"] != source.mcap_sha256:
                raise ValueError("Media belongs to another source")
            if file_hash(source.mcap_path) != source.mcap_sha256:
                raise ValueError("Source MCAP changed")
            if any(media["signature"][key] != value for key, value in self.source.manifest["sampling"].items()):
                raise ValueError("Sampling differs")
            # Freeze only media inputs actually sent to Qwen, plus the full sensor receipt.
            for frame in media["frames"]:
                image_input(self.inputs, frame)
            references = fetch(source.task_code, self.inputs)
            profile = profile_for(references, snapshot["profiles"])
            task = builder.review(self.inputs, references, profile, media)
            quality = builder.quality(self.inputs, media)
            audit[episode_id] = {
                "mcap_sha256": source.mcap_sha256, "resources_sha256": references["sha256"],
                "media_sha256": file_hash(self.inputs / "media" / episode_id / "media.json"),
                "gripper_sha256": media["gripper"]["sha256"],
                "frames": len(media["frames"]),
                "task_messages_sha256": fingerprint(task),
                "quality_messages_sha256": fingerprint(quality)}
            frozen_write(self.work / "inputs" / (episode_id + ".json"),
                         {"task": safe_messages(task), "quality": safe_messages(quality), **audit[episode_id]})
            print(json.dumps({"prepared": len(audit), "total": len(self.source.records())}), flush=True)
        experiment = {
            "created_at": now(), "purpose": "fixed_known_data_regression",
            "plan": self.plan, "plan_sha256": file_hash(self.plan_path),
            "runner_sha256": file_hash(Path(__file__)), "configuration": snapshot,
            "manifest": self.source.manifest, "input_audit": audit,
            "resources": resources, "pricing": self.pricing,
            "model_settings": {model: self.settings(model).model_dump() for model in models}}
        experiment["sha256"] = fingerprint(experiment)
        frozen_write(marker, experiment)
        for folder in ("citadel", "config"):
            shutil.copytree(PROJECT_ROOT / folder, self.work / "code" / folder,
                            ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        shutil.copy2(__file__, self.work / "code" / "benchmark.py")
        return experiment

    def validate(self):
        value = read(self.work / "experiment.json")
        if fingerprint({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]:
            raise ValueError("Benchmark manifest changed")
        if (value["plan_sha256"] != file_hash(self.plan_path)
                or value["runner_sha256"] != file_hash(Path(__file__))
                or value["configuration"]["code"] != code_version()
                or value["manifest"]["sha256"] != self.source.manifest["sha256"]):
            raise ValueError("Benchmark code, plan or source manifest changed")
        current = self.config.snapshot(self.source.manifest, self.plan["resource_base"]).data
        current.pop("model")
        if current != value["configuration"]:
            raise ValueError("Benchmark prompts or task rules changed")
        return value

    def run_one(self, model, episode_id, frozen, gateway):
        output = self.work / "models" / model / "results" / (episode_id + ".json")
        if output.exists():
            return read(output)
        started_path = output.with_suffix(".started.json")
        result = {"result_id": uuid.uuid4().hex, "episode_id": episode_id, "model": model,
                  "configuration_sha256": frozen["sha256"], "created_at": now(),
                  "status": "failed", "label": None}
        if started_path.exists():
            raise RuntimeError("Interrupted episode requires receipt inspection before resuming: " + episode_id)
        source = self.source.source(episode_id)
        media, references = self.media(episode_id), fetch(source.task_code, self.inputs)
        builder = PromptBuilder(PromptBundle(**frozen["configuration"]["prompts"]), image_input)
        profile = profile_for(references, frozen["configuration"]["profiles"])
        audit = frozen["input_audit"][episode_id]
        for name, value in (("task", builder.review(self.inputs, references, profile, media)),
                            ("quality", builder.quality(self.inputs, media))):
            if fingerprint(value) != audit[name + "_messages_sha256"]:
                raise ValueError("Model input changed")
        write(started_path, result)
        stage, started, waited = "source", time.monotonic(), gateway.wait_s
        def progress(value):
            nonlocal stage
            stage = value
        context = {**result, "task_code": source.task_code, "stage": "task"}
        # GT is joined only by the report. The frozen original 68/30 split is preserved.
        try:
            ReviewPipeline(gateway, builder).execute(self.inputs, references, profile, media,
                                                      context, progress, result)
        except Exception as exc:
            result.update(reason="处理失败：" + stage + " / " + type(exc).__name__,
                          error={"stage": stage, "type": type(exc).__name__})
        result.update(elapsed_s=time.monotonic() - started, pacing_s=gateway.wait_s - waited,
                      finished_at=now(), input_sha256=fingerprint(audit))
        write(output, result)
        return result

    def run(self, limit=None):
        frozen = self.validate()
        ids = sorted(self.source.records())[:limit]
        def worker(model):
            output = self.work / "models" / model
            gateway = PacedGateway(QwenGateway(output, model=model, base_url=self.plan["base_url"],
                                               settings=self.settings(model)),
                                   self.plan["model_tpm"].get(model, self.plan["default_tpm"]))
            for episode_id in ids:
                result = self.run_one(model, episode_id, frozen, gateway)
                print(json.dumps({"model": model, "episode_id": episode_id, "status": result["status"],
                                  "completed": len(list((output / "results").glob("[0-9a-f]" * 32 + ".json")))},
                                 ensure_ascii=False), flush=True)
        # A second process must not duplicate chargeable calls in this benchmark.
        guard = LocalExecutor(self.work, 1)
        try:
            with ThreadPoolExecutor(max_workers=len(self.plan["models"])) as pool:
                futures = [pool.submit(worker, model) for model in self.plan["models"]]
                for future in as_completed(futures):
                    future.result()
        finally:
            guard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "config/benchmark.toml")
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    benchmark = Benchmark(args.plan)
    if args.command == "prepare":
        value = benchmark.prepare()
        print(json.dumps({"configuration_sha256": value["sha256"], "episodes": len(value["input_audit"])}))
    else:
        benchmark.run(args.limit)


if __name__ == "__main__":
    main()
