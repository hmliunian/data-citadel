"""One review path for the CLI and FastAPI, with immutable results."""
from __future__ import annotations

import fcntl
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from .infrastructure.datasets import ID, load_manifest
from .infrastructure.files import file_hash, fingerprint, now, read, write
from .domain.tasks import profile_for
from .domain.decision import decide
from .application.prompts import PromptBuilder
from .configuration import PromptBundle
from .infrastructure.mcap.media import prepare_media
from .infrastructure.qwen import QwenGateway
from .infrastructure.resources import DEFAULT_BASE, fetch


class BusyError(RuntimeError):
    pass


class GateError(RuntimeError):
    pass


def code_version():
    return {str(p.relative_to(Path(__file__).parent)): file_hash(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))}


LOADED_CODE = code_version()


class Service:
    def __init__(self, work: Path, profiles: Path, *, client=None,
                 resource_loader=fetch, media_loader=prepare_media, resource_base=DEFAULT_BASE):
        self.work, self.profiles_path = work.resolve(), profiles.resolve()
        self.manifest = load_manifest(self.work)
        self.client = client or QwenGateway(self.work)
        self.resource_loader, self.media_loader = resource_loader, media_loader
        self.resource_base = resource_base

    @contextmanager
    def lock(self):
        with (self.work / ".lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BusyError("Another review or freeze is running") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def configuration(self):
        manifest = load_manifest(self.work)
        if manifest["sha256"] != self.manifest["sha256"]:
            raise ValueError("Manifest was replaced; open a new service")
        profiles = read(self.profiles_path)
        current_code = code_version()
        if current_code != LOADED_CODE:
            raise GateError("Code changed; restart the service before reviewing")
        config = {
            "manifest_sha256": manifest["sha256"], "profiles": profiles,
            "code": current_code,
            "model": self.client.model, "base_url": self.client.base_url,
            "resource_base": self.resource_base, "prompts": PromptBundle.load().as_dict(),
        }
        return config, fingerprint(config)

    def split_of(self, episode_id):
        if not ID.fullmatch(episode_id) or episode_id not in self.manifest["episodes"]:
            raise KeyError("Unknown episode ID")
        for split, ids in self.manifest["splits"].items():
            if episode_id in ids:
                return split
        raise ValueError("Duplicate episode is excluded from this run")

    def gate(self, split, signature):
        if split not in self.manifest["splits"]:
            raise ValueError("Split must be development or holdout")
        if split == "holdout":
            path = self.work / "freeze.json"
            if not path.exists():
                raise GateError("Freeze after development before opening holdout")
            frozen = read(path)
            if frozen["configuration_sha256"] != signature:
                raise GateError("Frozen code, prompt or configuration changed; use a new run")
            for code, expected in frozen["resources"].items():
                resource = self.resource_loader(code, self.work, self.resource_base)
                if resource["sha256"] != expected:
                    raise GateError("Frozen task references changed")

    def latest(self, episode_id, signature):
        folder = self.work / "results" / signature / episode_id
        paths = sorted(folder.glob("*.json"))
        return read(paths[-1]) if paths else None

    def episodes(self, split="development"):
        _, signature = self.configuration()
        self.gate(split, signature)
        output = []
        for episode_id in self.manifest["splits"][split]:
            row = self.manifest["episodes"][episode_id]
            result = self.latest(episode_id, signature)
            output.append({
                "episode_id": episode_id, "task_code": row["task_code"],
                "gt": row["gt"], "gt_reason": row["gt_reason"],
                "status": result["status"] if result else "not_run",
                "label": result.get("label") if result else None,
            })
        return output

    def get_result(self, episode_id):
        _, signature = self.configuration()
        self.gate(self.split_of(episode_id), signature)
        result = self.latest(episode_id, signature)
        if result is None:
            raise KeyError("Episode has not been reviewed under this configuration")
        return result

    def review(self, episode_id, retry_failed=False):
        with self.lock():
            config, signature = self.configuration()
            return self._review(episode_id, retry_failed, config, signature)

    def run(self, split="development", *, workers=3, limit=None, retry_failed=False):
        if not 1 <= workers <= 8 or limit is not None and limit < 1:
            raise ValueError("Use 1 to 8 workers and a positive limit")
        with self.lock():
            config, signature = self.configuration()
            self.gate(split, signature)
            ids = self.manifest["splits"][split][:limit]
            # Cache shared reference images once before independent episode workers start.
            codes = {self.manifest["episodes"][i]["task_code"] for i in ids}
            for code in sorted(codes):
                resources = self.resource_loader(code, self.work, self.resource_base)
                profile_for(resources, config["profiles"])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._review, i, retry_failed, config, signature) for i in ids]
                for future in as_completed(futures):
                    yield future.result()

    def _review(self, episode_id, retry_failed, config, signature):
        split = self.split_of(episode_id)
        self.gate(split, signature)
        episode = self.manifest["episodes"][episode_id]
        existing = self.latest(episode_id, signature)
        result = {
            "result_id": uuid.uuid4().hex, "episode_id": episode_id,
            "task_code": episode["task_code"], "split": split,
            "configuration_sha256": signature, "created_at": now(),
            "status": "failed", "label": None,
        }
        stage = "source"
        try:
            task = Path(self.manifest["dataset"]) / "tasks" / episode["task_code"]
            paths = {"tags_sha256": task / "api_tags" / (episode_id + ".json"),
                     "receipt_sha256": Path(self.manifest["dataset"]) / "receipts" / (episode_id + ".json")}
            if any(file_hash(path) != episode[key] for key, path in paths.items()):
                raise ValueError("Source metadata changed")
            stage = "resources"
            resources = self.resource_loader(episode["task_code"], self.work, self.resource_base)
            profile = profile_for(resources, config["profiles"])
            stage = "media"
            source = {k: episode[k] for k in ("episode_id", "mcap_path", "mcap_sha256")}
            media = self.media_loader(self.work, source, self.manifest["sampling"])
            result.update(resources_sha256=resources["sha256"], task_profile=profile["name"],
                          media_path=f"media/{episode_id}/media.json")
            if existing and not (retry_failed and existing["status"] == "failed"):
                if existing.get("resources_sha256") not in (None, resources["sha256"]):
                    raise ValueError("References changed; prepare a new run")
                return {**existing, "cached": True}
            stage = "model"
            context = {k: result[k] for k in (
                "result_id", "episode_id", "task_code", "split", "configuration_sha256")}
            builder = PromptBuilder(PromptBundle(**config["prompts"]))
            response = self.client.complete(builder.review(self.work, resources, profile, media), context)
            result["model_call"] = {k: v for k, v in response.items() if k != "data"}
            stage = "quality"
            quality = self.client.complete(builder.quality(self.work, media),
                                           {**context, "stage": stage}, quality_only=True)
            result["quality_call"] = {k: v for k, v in quality.items() if k != "data"}
            if set(quality["data"]) != {"quality_by_camera"}:
                raise ValueError("Quality response must contain only camera quality")
            stage = "evidence"
            result.update(decide({**response["data"], **quality["data"]}, media, profile))
        except Exception as exc:
            # Responses are retained separately; do not expose URLs, keys or vendor error bodies.
            result.update(reason=f"处理失败：{stage} / {type(exc).__name__}",
                          error={"stage": stage, "type": type(exc).__name__})
        result["evaluation"] = {"gt": episode["gt"], "gt_reason": episode["gt_reason"]}
        write(self.work / "results" / signature / episode_id /
              f"{time.time_ns()}-{result['result_id']}.json", result)
        return {**result, "cached": False}

    def freeze(self):
        with self.lock():
            config, signature = self.configuration()
            if (self.work / "freeze.json").exists():
                self.gate("holdout", signature)
                return read(self.work / "freeze.json")
            for episode_id in self.manifest["splits"]["development"]:
                result = self.latest(episode_id, signature)
                if not result or result["status"] == "failed":
                    raise GateError("Review all development episodes and resolve execution errors first")
            codes = {e["task_code"] for e in self.manifest["episodes"].values()}
            resources = {code: self.resource_loader(code, self.work, self.resource_base)["sha256"]
                         for code in sorted(codes)}
            frozen = {"created_at": now(), "configuration_sha256": signature,
                      "configuration": config, "resources": resources}
            write(self.work / "freeze.json", frozen)
            return frozen

    def report(self, split="development"):
        rows = self.episodes(split)
        _, signature = self.configuration()
        def counts(items):
            complete = [r for r in items if r["status"] == "completed" and r["gt"]]
            matched = sum(r["label"] == r["gt"] for r in complete)
            labeled = sum(bool(r["gt"]) for r in items)
            return {
                "total": len(items), "labeled": labeled,
                **{s: sum(r["status"] == s for r in items)
                   for s in ("completed", "needs_review", "failed", "not_run")},
                "matched": matched,
                "false_accept": sum(r["gt"] == "incorrect" and r["label"] == "correct" for r in complete),
                "false_reject": sum(r["gt"] == "correct" and r["label"] == "incorrect" for r in complete),
                "coverage": len(complete) / labeled if labeled else None,
                "accuracy_on_completed": matched / len(complete) if complete else None,
            }
        usage = {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0,
                 "total_tokens": 0, "attempts_without_usage": 0}
        for request in (self.work / "calls").glob("*/request.json"):
            context = read(request).get("context", {})
            if context.get("configuration_sha256") != signature or context.get("split") != split:
                continue
            usage["attempts"] += 1
            response = request.parent / "response.json"
            tokens = read(response).get("body", {}).get("usage", {}) if response.exists() else {}
            if not tokens:
                usage["attempts_without_usage"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += tokens.get(key, 0)
        return {
            "split": split, "configuration_sha256": signature, "counts": counts(rows),
            "by_gt_reason": {reason: counts([r for r in rows if (r["gt_reason"] or r["gt"] or "unlabeled") == reason])
                             for reason in sorted({r["gt_reason"] or r["gt"] or "unlabeled" for r in rows})},
            "by_task": {code: counts([r for r in rows if r["task_code"] == code])
                        for code in sorted({r["task_code"] for r in rows})},
            "usage": usage, "episodes": rows,
        }
