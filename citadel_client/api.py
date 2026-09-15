"""Small synchronous SDK for scripts and the CLI."""
import time
import httpx


class CitadelClient:
    def __init__(self, base_url="http://127.0.0.1:8770", *, transport=None, timeout=30):
        self.http = httpx.Client(base_url=base_url.rstrip("/") + "/api/v1/",
                                 transport=transport, timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.http.close()

    def request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def runs(self):
        return self.request("GET", "runs")

    def episodes(self, run_id, split="development"):
        return self.request("GET", "episodes", params={"run_id": run_id, "split": split})

    def submit(self, run_id, episode_id, *, preview=False, retry_failed=False):
        return self.request("POST", "previews" if preview else "reviews",
                            json={"run_id": run_id, "episode_id": episode_id, "retry_failed": retry_failed})

    def batch(self, run_id, split="development", *, limit=None, retry_failed=False):
        return self.request("POST", "batches",
                            json={"run_id": run_id, "split": split, "limit": limit, "retry_failed": retry_failed})

    def job(self, job_id):
        return self.request("GET", f"jobs/{job_id}")

    def wait(self, job_id, *, timeout=900, interval=0.5):
        deadline = time.monotonic() + timeout
        while True:
            job = self.job(job_id)
            if job["status"] in ("succeeded", "failed"):
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {job_id} is still running; query it again later")
            time.sleep(min(interval, max(0, deadline - time.monotonic())))

    def output(self, job_id):
        return self.request("GET", f"jobs/{job_id}/output")

    def result(self, run_id, result_id):
        return self.request("GET", f"results/{result_id}", params={"run_id": run_id})

    def report(self, run_id, split="development"):
        return self.request("GET", "reports", params={"run_id": run_id, "split": split})

    def freeze(self, run_id):
        return self.request("POST", f"runs/{run_id}/freeze")
