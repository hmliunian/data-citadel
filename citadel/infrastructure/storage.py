"""Read-only episodes and immutable, atomically published artifacts."""
from contextlib import contextmanager
from pathlib import Path
import re
import threading
import time

from citadel.domain.models import EpisodeInput
from .datasets import ID, load_manifest
from .files import file_hash, now, read, write, write_frozen
from .mcap.media import prepare_media
from .mcap.sensors import read_gripper
from .resources import fetch, image_input


class EpisodeRepository:
    def __init__(self, work: Path):
        self.work = work
        self.manifest = load_manifest(work)

    def records(self):
        if load_manifest(self.work)["sha256"] != self.manifest["sha256"]:
            raise ValueError("Manifest changed; open a new run")
        return self.manifest["episodes"]

    def source(self, episode_id):
        if not ID.fullmatch(episode_id) or episode_id not in self.records():
            raise KeyError("Unknown episode ID")
        row = self.manifest["episodes"][episode_id]
        dataset = Path(self.manifest["dataset"])
        paths = {"tags_sha256": dataset / "tasks" / row["task_code"] / "api_tags" / (episode_id + ".json"),
                 "receipt_sha256": dataset / "receipts" / (episode_id + ".json")}
        if any(file_hash(path) != row[key] for key, path in paths.items()):
            raise ValueError("Source metadata changed")
        return EpisodeInput.model_validate({key: row[key] for key in EpisodeInput.model_fields})


class ArtifactRepository:
    def __init__(self, work: Path):
        self.work = work.resolve()
        self._locks = {}
        self._guard = threading.Lock()

    @contextmanager
    def episode_lock(self, episode_id):
        with self._guard:
            lock = self._locks.setdefault(episode_id, threading.Lock())
        with lock:
            yield

    def latest(self, episode_id, signature):
        paths = sorted((self.work / "results" / signature / episode_id).glob("*.json"))
        return read(paths[-1]) if paths else None

    def save_result(self, result):
        path = (self.work / "results" / result["configuration_sha256"] / result["episode_id"] /
                f"{time.time_ns()}-{result['result_id']}.json")
        write(path, result)

    def result(self, result_id):
        if not ID.fullmatch(result_id):
            raise KeyError("Unknown result ID")
        paths = list((self.work / "results").glob(f"*/*/*-{result_id}.json"))
        if len(paths) != 1:
            raise KeyError("Unknown result ID")
        return read(paths[0])

    def history(self, episode_id):
        if not ID.fullmatch(episode_id):
            raise KeyError("Unknown episode ID")
        return sorted((read(path) for path in (self.work / "results").glob(f"*/{episode_id}/*.json")),
                      key=lambda value: value["created_at"])

    def save_snapshot(self, snapshot):
        path = self.work / "snapshots" / (snapshot.sha256 + ".json")
        write_frozen(path, snapshot.data)

    def media(self, episode_id):
        if not ID.fullmatch(episode_id):
            raise KeyError("Unknown episode ID")
        path = self.work / "media" / episode_id / "media.json"
        if not path.exists():
            raise KeyError("Prepare this episode's media first")
        media = read(path)
        signals = path.with_name("gripper.json")
        if signals.exists():
            media["gripper"] = read_gripper(signals)
        return media

    def asset(self, episode_id, filename):
        media = self.media(episode_id)
        assets = media["frames"] + list(media["videos"].values())
        item = next((a for a in assets if Path(a["path"]).name == filename), None)
        if item is None:
            raise KeyError("Unknown media asset")
        path = (self.work / item["path"]).resolve()
        if not path.is_relative_to(self.work / "media" / episode_id) or file_hash(path) != item["sha256"]:
            raise ValueError("Media asset changed")
        return path

    def traces(self, result):
        output = {"attempts": []}
        paths = sorted((self.work / "calls").glob("*/request.json"), key=lambda p: p.stat().st_mtime_ns)
        for path in paths:
            request = read(path)
            if request.get("context", {}).get("result_id") == result["result_id"]:
                output["attempts"].append({"call_path": str(path.parent.relative_to(self.work)),
                    **{p.stem: read(p) for p in (path, path.with_name("response.json"), path.with_name("error.json"))
                       if p.exists()}})
        for name in ("model_call", "quality_call"):
            call = result.get(name, {}).get("call_path")
            if not call:
                continue
            if not re.fullmatch(r"calls/[0-9a-f]{32}", call):
                raise ValueError("Invalid model trace reference")
            folder = self.work / call
            output[name] = {p.stem: read(p) for p in
                            (folder / "request.json", folder / "response.json", folder / "error.json")
                            if p.exists()}
        return output

    def frozen(self):
        path = self.work / "freeze.json"
        return read(path) if path.exists() else None

    def freeze(self, value):
        write(self.work / "freeze.json", {"created_at": now(), **value})
        return self.frozen()

    def usage(self, signature, split):
        usage = dict.fromkeys(("attempts", "prompt_tokens", "completion_tokens", "total_tokens",
                               "attempts_without_usage"), 0)
        for path in (self.work / "calls").glob("*/request.json"):
            context = read(path).get("context", {})
            if context.get("configuration_sha256") != signature or context.get("split") != split:
                continue
            usage["attempts"] += 1
            response = path.with_name("response.json")
            tokens = read(response).get("body", {}).get("usage", {}) if response.exists() else {}
            usage["attempts_without_usage"] += not bool(tokens)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += tokens.get(key, 0)
        return usage

    def image(self, work, item):
        return image_input(work, item)


class CachedTaskResources:
    def __init__(self, work, base_url, loader=fetch):
        self.work, self.base_url, self.loader = work, base_url, loader
        self._lock = threading.Lock()

    def get(self, task_code):
        with self._lock:
            return self.loader(task_code, self.work, self.base_url)


class McapMediaPreparer:
    def __init__(self, work, limiter, loader=prepare_media):
        self.work, self.limiter, self.loader = work, limiter, loader

    def prepare(self, source, sampling):
        with self.limiter:
            return self.loader(self.work, source.media_source(), sampling)
