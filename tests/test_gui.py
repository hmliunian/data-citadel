"""Optional Firefox integration test against an HTTP server and fake Qwen."""
import base64
import json
import os
import shutil
import socket
import subprocess
import threading
import time

import av
import httpx
from PIL import Image
import pytest
import uvicorn

from citadel.domain.signals import CHANNELS
from citadel.infrastructure.files import fingerprint, read, write
from citadel.infrastructure.mcap.media import TOPICS, render
from citadel.server.app import create_app
from scripts.dev import stop_server, wait_ready

pytestmark = pytest.mark.skipif(os.getenv("CITADEL_BROWSER_TEST") != "1",
                                reason="Run just test-gui with Firefox and geckodriver installed")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def browser(tmp_path):
    driver_path = shutil.which("geckodriver")
    assert driver_path, "Install Firefox and geckodriver for just test-gui"
    port = free_port()
    session = None
    with (tmp_path / "geckodriver.log").open("w") as log:
        driver = subprocess.Popen([driver_path, "--host", "127.0.0.1", "--port", str(port),
                                   "--profile-root", str(tmp_path)],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=45, trust_env=False) as client:
        try:
            for _ in range(60):
                try:
                    if client.get("/status").is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            response = client.post("/session", json={"capabilities": {"alwaysMatch": {
                "browserName": "firefox", "moz:firefoxOptions": {"args": ["-headless"]}}}})
            response.raise_for_status()
            session = "/session/" + response.json()["value"]["sessionId"]
            def command(path, value=None, method="POST"):
                response = client.request(method, session + path, **({"json": value or {}} if method == "POST" else {}))
                response.raise_for_status()
                return response.json()["value"]
            def script(code, *args):
                return command("/execute/sync", {"script": code, "args": args})
            def until(code):
                for _ in range(80):
                    value = script(code)
                    if value:
                        return value
                    time.sleep(0.2)
                pytest.fail("Browser timed out: " + code + " / " + str(script(
                    "return document.getElementById('notice')?.textContent")))
            command("/window/rect", {"width": 1440, "height": 1200})
            yield command, script, until
            (tmp_path / "gui.png").write_bytes(base64.b64decode(command("/screenshot", method="GET")))
        finally:
            if session:
                client.delete(session)
            stop_server(driver)


def test_preview_review_refresh_history_and_video_sync(runtime_case, browser, tmp_path):
    runtime, fake = runtime_case
    run = runtime.runs["test"]

    def media_loader(work, source, sampling):
        path = work / "media" / source["episode_id"] / "media.json"
        if not path.exists():
            streams = {name: [(i * 10**9, av.VideoFrame.from_image(
                Image.new("RGB", (320, 180), color))) for i in range(4)]
                for name, color in zip(TOPICS, ("red", "green", "blue"))}
            render(work, source["episode_id"], 0, streams, {**sampling, "topics": TOPICS}, [])
            gripper = {"channels": {name: {"samples": [[i / 10, 1 + i % 5] for i in range(31)]}
                                   for name in CHANNELS}, "warnings": []}
            gripper["sha256"] = fingerprint(gripper)
            write(path.with_name("gripper.json"), gripper)
        return {**read(path), "gripper": read(path.with_name("gripper.json"))}
    run.reviews.media.loader = media_loader
    entered, release = threading.Event(), threading.Event()
    original = fake.complete
    def complete(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    fake.complete = complete
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(create_app(runtime), log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    command, script, until = browser
    try:
        wait_ready(url)
        command("/url", {"url": url})
        until("return document.querySelectorAll('#episode option').length === 6")
        script("document.getElementById('prepare').click()")
        until("return document.getElementById('preview').textContent.includes('configuration_sha256') && !document.getElementById('review').disabled")
        assert fake.requests == []
        preview = script("return document.getElementById('preview').textContent")
        assert "PRIVATE_GT_REASON" not in preview and "base64," not in preview
        until("return [...document.querySelectorAll('video')].every(v => v.readyState >= 2 && !v.error)")
        assert script("return !document.getElementById('signals').hidden")

        script("document.getElementById('review').click()")
        assert entered.wait(3)
        until("return new URL(location.href).searchParams.has('job')")
        command("/refresh")
        until("return document.getElementById('review').disabled && document.getElementById('notice').textContent.includes('任务审核')")
        release.set()
        until("return document.getElementById('conclusion').textContent === '通过' && !document.getElementById('review').disabled")
        assert len(fake.requests) == 2
        assert "未审核" not in script("return document.getElementById('episode').selectedOptions[0].textContent")
        assert script("return document.getElementById('versions').hidden")
        until("return document.querySelectorAll('#history option').length === 1")
        until("return [...document.querySelectorAll('video')].every(v => v.readyState >= 2)")
        script("document.getElementById('frame-V002').click()")
        until("return [...document.querySelectorAll('video')].every(v => Math.abs(v.currentTime - 2) < .1)")
        script("const c=document.getElementById('gripper'), r=c.getBoundingClientRect(); c.dispatchEvent(new MouseEvent('click', {clientX:r.left+r.width*(70+555)/1200}));")
        until("return [...document.querySelectorAll('video')].every(v => Math.abs(v.currentTime - 1.5) < .1)")
        selected = script("return document.getElementById('episode').value")
        script("const e=document.getElementById('episode');e.selectedIndex=1;e.dispatchEvent(new Event('change'));")
        until("return document.getElementById('result').hidden")
        script("const e=document.getElementById('episode');e.value=arguments[0];e.dispatchEvent(new Event('change'));", selected)
        until("return !document.getElementById('result').hidden && document.getElementById('conclusion').textContent === '通过'")

        prompt = run.configuration.root / "prompts/review_system.txt"
        prompt.write_text(prompt.read_text() + "\nBROWSER_VERSION_TEST")
        command("/refresh")
        until("return document.querySelectorAll('#history option').length === 1 && document.getElementById('result').hidden")
        script("document.getElementById('review').click()")
        until("return document.querySelectorAll('#history option').length === 2 && !document.getElementById('review').disabled")
        assert len(fake.requests) == 4
        script("document.getElementById('history').selectedIndex=0;document.getElementById('compare').click()")
        versions = script("return ['current-version','past-version'].map(id=>JSON.parse(document.getElementById(id).textContent).configuration_sha256)")
        assert versions[0] != versions[1]
        (tmp_path / "browser-check.json").write_text(json.dumps({
            "fake_model_calls": len(fake.requests), "versions": versions,
            "checks": ["preview_without_model", "resume_on_refresh", "four_video_seek",
                       "gripper_seek", "switch_episode", "history_comparison"]}, indent=2))
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
