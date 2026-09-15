"""Replaceable external capabilities; application code uses these contracts."""
from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Protocol

from citadel.domain.models import EpisodeInput


class ModelGateway(Protocol):
    def complete(self, messages: list, context: dict, *, quality_only: bool = False) -> dict: ...
    def preview(self, messages: list, *, quality_only: bool = False) -> dict: ...


class TaskResources(Protocol):
    base_url: str
    def get(self, task_code: str) -> dict: ...


class MediaPreparer(Protocol):
    def prepare(self, source: EpisodeInput, sampling: dict) -> dict: ...


class Executor(Protocol):
    def submit(self, function: Callable, *args) -> Future: ...
    def close(self) -> None: ...


class ImageLoader(Protocol):
    def __call__(self, work: Path, image: dict) -> str: ...
