"""Composition root: create concrete adapters and inject application dependencies."""
from pathlib import Path
import re
import threading

from .application.experiments import ExperimentService
from .application.jobs import JobService
from .application.reviews import ReviewService
from .application.runtime import RunContext, Runtime
from .configuration import CONFIG_ROOT, Configuration, ModelSettings, server_configuration
from .infrastructure.datasets import prepare
from .infrastructure.execution import LocalExecutor
from .infrastructure.jobs import JobRepository
from .infrastructure.qwen import QwenGateway
from .infrastructure.storage import ArtifactRepository, CachedTaskResources, EpisodeRepository, McapMediaPreparer


def build_run(run_id, work, configuration, resource_base, media_limiter, *,
              resource_loader=None, media_loader=None, gateway_factory=None):
    source, artifacts = EpisodeRepository(work), ArtifactRepository(work)
    resources = CachedTaskResources(work, resource_base, **({"loader": resource_loader} if resource_loader else {}))
    media = McapMediaPreparer(work, media_limiter, **({"loader": media_loader} if media_loader else {}))
    experiment = ExperimentService(source, artifacts, resources)
    factory = gateway_factory or (lambda snapshot: QwenGateway(
        work, settings=ModelSettings.model_validate(snapshot.data["model"]),
        model=snapshot.data["model"]["model"], base_url=snapshot.data["model"]["base_url"]))
    reviews = ReviewService(source, artifacts, resources, media, factory, experiment)
    return RunContext(run_id, configuration, source, artifacts, resources, experiment, reviews, threading.Lock())


def build_runtime(config_root=CONFIG_ROOT, *, settings=None, runs=None):
    if settings is None:
        settings, definitions = server_configuration(config_root)
    else:
        definitions = {}
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    executor = LocalExecutor(settings.work_dir, settings.workers)
    try:
        if runs is None:
            configuration = Configuration(config_root)
            limiter = threading.BoundedSemaphore(settings.media_workers)
            runs = {}
            for run_id, item in definitions.items():
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
                    raise ValueError("Invalid configured run ID")
                work = settings.work_dir / "runs" / run_id
                if not (work / "manifest.json").exists():
                    prepare(Path(item["dataset"]), work)
                runs[run_id] = build_run(run_id, work, configuration, settings.resource_base, limiter)
        runtime = Runtime(settings, runs)
        runtime.jobs = JobService(runtime, JobRepository(settings.work_dir / "jobs.sqlite3",
                                                        settings.max_pending), executor)
        return runtime
    except Exception:
        executor.close()
        raise
