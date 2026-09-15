import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from citadel.infrastructure.resources import fetch


@pytest.mark.parametrize("host", ["127.0.0.1", "172.100.11.189"])
def test_registered_task_api_bypasses_environment_proxy(tmp_path, jpeg, monkeypatch, host):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path.endswith("/resources"):
                self.wfile.write(json.dumps({"code": 200, "data": {
                    "task_code": "DL-TEST", "steps": [{"action_id": "A", "action_text": "Do task"}],
                    "images": [{"type": kind, "url": base + "/image", "id": kind, "name": kind}
                               for kind in ("object", "scene")]}}).encode())
            else:
                self.wfile.write(jpeg)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://{host}:{server.server_port}"
    getaddrinfo = socket.getaddrinfo
    def resolve(name, port, *args, **kwargs):
        if name in (host, host.encode()):
            name = "127.0.0.1"
        return getaddrinfo(name, port, *args, **kwargs)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(variable, "http://127.0.0.1:1")
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(variable, "")
    try:
        result = fetch("DL-TEST", tmp_path, base)
        assert result["task_code"] == "DL-TEST" and len(result["images"]) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
