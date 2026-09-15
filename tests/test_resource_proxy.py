import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from citadel.infrastructure.resources import fetch


def test_private_task_api_bypasses_environment_proxy(tmp_path, jpeg, monkeypatch):
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
    base = f"http://127.0.0.1:{server.server_port}"
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
