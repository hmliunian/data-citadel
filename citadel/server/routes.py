"""Translate HTTP requests into application calls and public asset references."""
from pathlib import PurePosixPath
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from citadel.configuration import ConfigSnapshot
from .schemas import BatchSubmission, EpisodeView, JobView, ResultView, Submission


def job_view(job):
    return {key: job[key] for key in
            ("job_id", "run_id", "episode_id", "kind", "status", "stage", "created_ns", "updated_ns", "error")} | {
                "configuration_sha256": ConfigSnapshot.create(job["snapshot"]).sha256,
                "result_id": (job.get("result") or {}).get("result_id")}


def public_assets(value, run_id):
    value = dict(value)
    episode_id = value["episode_id"]
    def asset(item):
        return {key: v for key, v in item.items() if key != "path"} | {
            "url": f"/api/v1/runs/{run_id}/assets/{episode_id}/{PurePosixPath(item['path']).name}"}
    for key in ("frames", "evidence"):
        if key in value:
            value[key] = [asset(item) for item in value[key]]
    if "videos" in value:
        value["videos"] = {key: asset(item) for key, item in value["videos"].items()}
    value.pop("media_path", None)
    return value


def create_router(runtime):
    router = APIRouter(prefix="/api/v1")

    @router.get("/health")
    def health():
        return {"status": "ok", "runs": len(runtime.runs), "workers": runtime.settings.workers}

    @router.get("/runs")
    def runs():
        return [{"run_id": key, "episodes": len(run.source.records()),
                 "splits": {split: len(ids) for split, ids in run.source.manifest["splits"].items()}}
                for key, run in runtime.runs.items()]

    @router.get("/settings")
    def settings(run_id: str):
        snapshot = runtime.run(run_id).snapshot()
        return {**snapshot.data, "configuration_sha256": snapshot.sha256}

    @router.get("/episodes", response_model=list[EpisodeView])
    def episodes(run_id: str, split: str = "development"):
        run = runtime.run(run_id)
        return run.experiment.episodes(split, run.snapshot())

    @router.post("/reviews", status_code=202, response_model=JobView)
    def review(request: Submission):
        return job_view(runtime.jobs.submit(request.run_id, request.episode_id,
                                            retry_failed=request.retry_failed))

    @router.post("/previews", status_code=202, response_model=JobView)
    def preview(request: Submission):
        return job_view(runtime.jobs.submit(request.run_id, request.episode_id, "preview",
                                            request.retry_failed))

    @router.post("/batches", status_code=202, response_model=list[JobView])
    def batch(request: BatchSubmission):
        return [job_view(job) for job in runtime.jobs.batch(request.run_id, request.split,
                                                           request.limit, request.retry_failed)]

    @router.get("/jobs/{job_id}", response_model=JobView)
    def job(job_id: str):
        return job_view(runtime.jobs.get(job_id))

    @router.get("/jobs/{job_id}/output")
    def output(job_id: str):
        job = runtime.jobs.get(job_id)
        if job["result"] is None:
            raise HTTPException(409, "Job has no output yet")
        result = job["result"]
        if job["kind"] == "review":
            run = runtime.run(job["run_id"])
            result = public_assets(run.experiment.decorate(result), job["run_id"])
        return result

    @router.get("/results/{result_id}", response_model=ResultView)
    def result(result_id: str, run_id: str):
        run = runtime.run(run_id)
        value = run.artifacts.result(result_id)
        run.authorize(value["episode_id"])
        return public_assets(run.experiment.decorate(value), run_id)

    @router.get("/results/{result_id}/trace")
    def trace(result_id: str, run_id: str):
        run = runtime.run(run_id)
        value = run.artifacts.result(result_id)
        run.authorize(value["episode_id"])
        return run.artifacts.traces(value)

    @router.get("/episodes/{episode_id}/history", response_model=list[ResultView])
    def history(episode_id: str, run_id: str):
        run = runtime.run(run_id)
        run.authorize(episode_id)
        return [public_assets(run.experiment.decorate(value), run_id)
                for value in run.artifacts.history(episode_id)]

    @router.get("/episodes/{episode_id}/media")
    def media(episode_id: str, run_id: str):
        run = runtime.run(run_id)
        run.authorize(episode_id)
        return public_assets(run.artifacts.media(episode_id), run_id)

    @router.get("/runs/{run_id}/assets/{episode_id}/{filename}")
    def asset(run_id: str, episode_id: str, filename: str):
        run = runtime.run(run_id)
        run.authorize(episode_id)
        return FileResponse(run.artifacts.asset(episode_id, filename))

    @router.get("/reports")
    def report(run_id: str, split: str = "development"):
        run = runtime.run(run_id)
        return run.experiment.report(split, run.snapshot())

    @router.post("/runs/{run_id}/freeze")
    def freeze(run_id: str):
        return runtime.jobs.freeze(run_id)

    return router
