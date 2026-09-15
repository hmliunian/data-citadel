"""Own one development server and open its HTTP GUI after readiness."""
import argparse
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from citadel.configuration import CONFIG_ROOT, PROJECT_ROOT, server_configuration


def browser_url(settings):
    host = settings.host if settings.host not in ("0.0.0.0", "::") else "127.0.0.1"
    if ":" in host:
        host = "[" + host + "]"
    return f"http://{host}:{settings.port}"


def wait_ready(url, process=None, timeout=30):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=1, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"Server exited with code {process.returncode}")
            try:
                response = client.get(url.rstrip("/") + "/api/v1/health")
                if response.is_success and response.json().get("status") == "ok":
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.2)
    raise TimeoutError("Server did not become ready: " + url)


def stop_server(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--url", help="URL of an already running server (client mode)")
    parser.add_argument("--client-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=30)
    args = parser.parse_args(argv)
    settings, _ = server_configuration(args.config)
    url = args.url or browser_url(settings)
    process = None

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if not args.client_only:
            target = urlsplit(browser_url(settings))
            try:
                with socket.create_connection((target.hostname, target.port), timeout=0.5):
                    raise RuntimeError("Port already in use; connect with just client or choose another server port")
            except (ConnectionRefusedError, TimeoutError, OSError):
                pass
            process = subprocess.Popen(
                [sys.executable, "-m", "citadel", "--config", str(args.config.resolve()), "serve"],
                cwd=PROJECT_ROOT, start_new_session=True)
        wait_ready(url, process, args.startup_timeout)
        print(f"GUI: {url}/\nAPI: {url}/docs", flush=True)
        if not args.no_browser and not webbrowser.open(url):
            print("No browser opened; use the GUI URL above.", file=sys.stderr)
        return process.wait() if process else 0
    except KeyboardInterrupt:
        return 0
    finally:
        if process:
            stop_server(process)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
