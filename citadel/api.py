"""FastAPI interface and an independent atomic-task test window."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .data import file_hash, read
from .service import BusyError, GateError, Service


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    retry_failed: bool = False


def create_app(service: Service):
    app = FastAPI(title="UMI-T 原子任务测试", version="1.0.0")

    @app.exception_handler(KeyError)
    async def missing(_, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid(_, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(BusyError)
    @app.exception_handler(GateError)
    async def conflict(_, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    def media(episode_id):
        _, signature = service.configuration()
        service.gate(service.split_of(episode_id), signature)
        path = service.work / "media" / episode_id / "media.json"
        if not path.exists():
            raise KeyError("Prepare this episode's video first")
        result = read(path)
        for asset in result["frames"] + list(result["videos"].values()):
            asset["url"] = f"/assets/{episode_id}/{Path(asset['path']).name}"
        return result

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(Path(__file__).with_name("web.html"))

    @app.get("/health")
    def health():
        return {"status": "ok", "episodes": len(service.manifest["episodes"]),
                "model": service.client.model}

    @app.get("/settings")
    def settings():
        config, signature = service.configuration()
        return {"profiles": config["profiles"], "sampling": service.manifest["sampling"],
                "configuration_sha256": signature}

    @app.get("/episodes")
    def episodes(split: str = "development"):
        return service.episodes(split)

    @app.post("/reviews")
    def review(request: ReviewRequest):
        return service.review(request.episode_id, request.retry_failed)

    @app.get("/results/{episode_id}")
    def result(episode_id: str):
        return service.get_result(episode_id)

    @app.get("/media/{episode_id}")
    def get_media(episode_id: str):
        return media(episode_id)

    @app.post("/media/{episode_id}")
    def prepare_video(episode_id: str):
        with service.lock():
            _, signature = service.configuration()
            service.gate(service.split_of(episode_id), signature)
            episode = service.manifest["episodes"][episode_id]
            service.media_loader(service.work, {
                k: episode[k] for k in ("episode_id", "mcap_path", "mcap_sha256")},
                service.manifest["sampling"])
        return media(episode_id)

    @app.get("/assets/{episode_id}/{filename}")
    def asset(episode_id: str, filename: str):
        prepared = media(episode_id)
        allowed = prepared["frames"] + list(prepared["videos"].values())
        item = next((a for a in allowed if Path(a["path"]).name == filename), None)
        if item is None:
            raise HTTPException(404, "Unknown video/frame asset")
        path = (service.work / item["path"]).resolve()
        if (not path.is_relative_to((service.work / "media" / episode_id).resolve())
                or file_hash(path) != item["sha256"]):
            raise HTTPException(409, "Media asset changed")
        return FileResponse(path)

    @app.get("/report")
    def report(split: str = "development"):
        return service.report(split)

    @app.post("/freeze")
    def freeze():
        return service.freeze()

    return app
