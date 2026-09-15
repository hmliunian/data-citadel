import time

import pytest

from citadel.bootstrap import build_runtime
from citadel.domain.errors import BusyError
from citadel.infrastructure.execution import LocalExecutor
from citadel.infrastructure.jobs import JobRepository


def test_restart_resumes_queued_but_never_replays_interrupted_calls(runtime_case):
    runtime, fake = runtime_case
    run = runtime.runs["test"]
    ids = run.source.manifest["splits"]["development"]
    repo = runtime.jobs.repository
    queued, _ = repo.create("test", ids[0], "review", run.snapshot(), False)
    interrupted, _ = repo.create("test", ids[1], "review", run.snapshot(), False)
    repo.update(interrupted["job_id"], status="running", stage="model")
    runtime.jobs.close()
    restarted = build_runtime(settings=runtime.settings, runs=runtime.runs)
    try:
        restarted.jobs.start()
        deadline = time.monotonic() + 5
        while repo.get(queued["job_id"])["status"] in ("queued", "running"):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert repo.get(queued["job_id"])["status"] == "succeeded"
        stopped = repo.get(interrupted["job_id"])
        assert stopped["status"] == "failed" and stopped["error"]["type"] == "Interrupted"
        assert len(fake.requests) == 2
        assert restarted.jobs.submit("test", ids[1])["job_id"] == stopped["job_id"]
    finally:
        restarted.jobs.close()


def test_queued_job_keeps_submitted_prompt_after_config_edit(runtime_case):
    runtime, fake = runtime_case
    run = runtime.runs["test"]
    snapshot = run.snapshot()
    job, _ = runtime.jobs.repository.create("test", run.source.manifest["splits"]["development"][0],
                                           "review", snapshot, False)
    path = run.configuration.root / "prompts/review_system.txt"
    path.write_text("SUBSEQUENT_CONFIG_EDIT")
    runtime.jobs._execute(job["job_id"])
    assert runtime.jobs.repository.get(job["job_id"])["status"] == "succeeded"
    assert fake.requests[0][0]["content"].startswith(snapshot.data["prompts"]["review_system"])
    assert "SUBSEQUENT_CONFIG_EDIT" not in str(fake.requests)


def test_two_schedulers_cannot_share_a_work_directory(tmp_path):
    first = LocalExecutor(tmp_path, 1)
    try:
        with pytest.raises(BusyError):
            LocalExecutor(tmp_path, 1)
    finally:
        first.close()
    LocalExecutor(tmp_path, 1).close()


def test_deduplication_survives_new_repository_connection(service_case, tmp_path):
    run, _ = service_case
    path = tmp_path / "jobs.sqlite3"
    ids, snapshot = run.source.manifest["splits"]["development"], run.snapshot()
    first, created = JobRepository(path).create("test", ids[0], "review", snapshot, False)
    second, created_again = JobRepository(path).create("test", ids[0], "review", snapshot, False)
    assert created and not created_again and first["job_id"] == second["job_id"]
