"""Run-scoped dependencies; the HTTP server can host several experiments."""
from dataclasses import dataclass
import threading

from citadel.configuration import Configuration, ServerSettings
from .experiments import ExperimentService
from .reviews import ReviewService
from .ports import TaskResources


@dataclass
class RunContext:
    run_id: str
    configuration: Configuration
    source: object
    artifacts: object
    resources: TaskResources
    experiment: ExperimentService
    reviews: ReviewService
    guard: threading.Lock

    def snapshot(self):
        return self.configuration.snapshot(self.source.manifest, self.resources.base_url)

    def authorize(self, episode_id, snapshot=None):
        self.experiment.gate(self.experiment.split_of(episode_id), snapshot or self.snapshot())


@dataclass
class Runtime:
    settings: ServerSettings
    runs: dict[str, RunContext]
    jobs: object = None

    def run(self, run_id):
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError("Unknown run ID") from exc
