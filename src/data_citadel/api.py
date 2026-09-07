"""Local web interface and typed episode review API."""

import base64
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import Field

from . import __version__
from .models import (
    CitadelError, DatasetError, EpisodeNotFound, ExpertError, MediaError,
    ProviderError, ReviewResult, StrictModel,
)
from .runtime import build_service, episode_summary, save_review
from .settings import Settings


class ReviewRequest(StrictModel):
    episode_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    strategy: Literal["uniform", "keyframes"] = "uniform"


def create_app(settings: Settings | None = None, service=None) -> FastAPI:
    settings = settings or Settings()
    owns_service = service is None
    service = service or build_service(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owns_service:
            service.client.close()

    app = FastAPI(title="Data Citadel", version=__version__, lifespan=lifespan)
    app.state.service = service

    @app.exception_handler(CitadelError)
    async def citadel_error(request: Request, error: CitadelError):
        codes = {EpisodeNotFound: 404, DatasetError: 422, ExpertError: 409,
                 MediaError: 422, ProviderError: 502}
        return JSONResponse(
            status_code=codes.get(type(error), 500),
            content={"error": type(error).__name__, "detail": str(error)},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return files("data_citadel").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/v1/inventory")
    def inventory():
        return service.repository.inventory()

    @app.get("/v1/episodes")
    def episodes(action_id: str | None = Query(default=None, pattern=r"^A_\d{3}$")):
        return [episode_summary(item) for item in service.repository.list_episodes(action_id)]

    @app.get("/v1/episodes/{episode_id}/frames")
    def preview(
        episode_id: str,
        strategy: Literal["uniform", "keyframes"] = "uniform",
        interval_s: float = Query(default=1.0, ge=0.1, le=10.0),
    ):
        episode = service.repository.get(episode_id)
        video = service.sampler.sample(episode, strategy=strategy, interval_s=interval_s)
        return {
            "duration_s": video.duration_s, "strategy": video.strategy,
            "warnings": video.warnings, "motion": video.motion,
            "frames": [{
                "timestamp_s": frame.timestamp_s,
                "image": "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii"),
            } for frame in video.frames],
        }

    @app.post("/v1/reviews", response_model=ReviewResult)
    def review(body: ReviewRequest):
        result = service.review(body.episode_id, strategy=body.strategy)
        save_review(result, settings.artifacts_dir)
        return result

    return app
