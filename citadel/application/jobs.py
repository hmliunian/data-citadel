"""Submit immutable review jobs independently from HTTP request lifetimes."""
from citadel.configuration import ConfigSnapshot
from citadel.domain.errors import GateError
from .ports import Executor


class JobService:
    def __init__(self, runtime, repository, executor: Executor):
        self.runtime, self.repository, self.executor = runtime, repository, executor
        self.started = False

    def start(self):
        if not self.started:
            self.started = True
            for job_id in self.repository.recover():
                self.executor.submit(self._execute, job_id)

    def close(self):
        self.executor.close()

    def submit(self, run_id, episode_id, kind="review", retry_failed=False):
        if kind not in ("review", "preview"):
            raise ValueError("Unknown job kind")
        run = self.runtime.run(run_id)
        with run.guard:
            snapshot = run.snapshot()
            run.authorize(episode_id, snapshot)
            run.artifacts.save_snapshot(snapshot)
            job, created = self.repository.create(run_id, episode_id, kind, snapshot, retry_failed)
            if created:
                self.executor.submit(self._execute, job["job_id"])
            return job

    def batch(self, run_id, split="development", limit=None, retry_failed=False):
        run = self.runtime.run(run_id)
        with run.guard:
            snapshot = run.snapshot()
            rows = run.experiment.episodes(split, snapshot)[:limit]
            run.artifacts.save_snapshot(snapshot)
            entries = self.repository.create_many(run_id, [row["episode_id"] for row in rows],
                                                  "review", snapshot, retry_failed)
            for job, created in entries:
                if created:
                    self.executor.submit(self._execute, job["job_id"])
            return [job for job, _ in entries]

    def get(self, job_id):
        job = self.repository.get(job_id)
        self.runtime.run(job["run_id"]).authorize(job["episode_id"])
        return job

    def freeze(self, run_id):
        run = self.runtime.run(run_id)
        with run.guard:
            if self.repository.active(run_id):
                raise GateError("Wait for this run's active jobs before freezing")
            return run.experiment.freeze(run.snapshot())

    def _execute(self, job_id):
        job = self.repository.get(job_id)
        self.repository.update(job_id, status="running", stage="configuration")
        try:
            run = self.runtime.run(job["run_id"])
            snapshot = ConfigSnapshot.create(job["snapshot"])
            if snapshot.data["code"] != run.configuration.loaded_code:
                raise GateError("Queued job uses another code version; submit a new configuration")
            if snapshot.data["manifest_sha256"] != run.source.manifest["sha256"]:
                raise GateError("Queued job uses another dataset manifest")
            def progress(stage):
                self.repository.update(job_id, stage=stage)
            if job["kind"] == "review":
                result = run.reviews.review(job["episode_id"], snapshot, job["retry_failed"], progress)
                failed = result["status"] == "failed"
            else:
                result = run.reviews.preview(job["episode_id"], snapshot, progress)
                failed = False
            self.repository.update(job_id, status="failed" if failed else "succeeded",
                                   stage=result.get("error", {}).get("stage", "done"), result=result,
                                   error=result.get("error"))
        except Exception as exc:
            self.repository.update(job_id, status="failed",
                                   error={"type": type(exc).__name__, "message": "Task processing failed"})
