import subprocess
import sys

import httpx
import pytest

from scripts.dev import stop_server, wait_ready


def test_readiness_does_not_accept_an_unrelated_server_after_child_exit(monkeypatch):
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    process.wait()
    def forbidden(*args, **kwargs):
        raise AssertionError("Exited child must be checked before polling another server")
    monkeypatch.setattr(httpx.Client, "get", forbidden)
    with pytest.raises(RuntimeError, match="code 7"):
        wait_ready("http://127.0.0.1:1", process)


def test_stop_terminates_only_the_owned_server():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        stop_server(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
