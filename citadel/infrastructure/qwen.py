"""Qwen transport and redacted call receipts."""
import copy
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import httpx

from citadel.configuration import PROJECT_ROOT
from citadel.domain.models import CAMERAS, CHECKS, QualityReview, Review
from .files import fingerprint, write


def safe_messages(value):
    clean = copy.deepcopy(value)
    def hidden(url):
        return {"sha256": hashlib.sha256(url.encode()).hexdigest(), "encoded_length": len(url)}
    for message in clean:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "image_url":
                    part["image_url"] = hidden(part["image_url"]["url"])
    return clean


class QwenGateway:
    def __init__(self, work: Path, *, model=None, base_url=None, api_key=None, transport=None):
        self.work, self.api_key, self.transport = work, api_key, transport
        self.model = model or os.getenv("QWEN_MODEL", "qwen3.8-max-0902")
        self.base_url = (base_url or os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")

    def complete(self, request_messages, context=None, *, quality_only=False):
        key = self.api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            file = os.getenv("QWEN_API_KEY_FILE")
            path = Path(file) if file else PROJECT_ROOT / "Qwen-api/qwen_api_key.txt"
            key = path.read_text().strip()
        if not key or any(c.isspace() for c in key):
            raise ValueError("Qwen key must be a single nonempty token")
        count, frame_states = 0, {}
        for message in request_messages:
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part["type"] == "image_url":
                        count += 1
                    elif part["type"] == "text":
                        frame = json.loads(part["text"]).get("candidate_frame")
                        if frame:
                            frame_states[frame["frame_id"]] = (
                                ["visible", "absent", "uncertain"]
                                if frame["source_times_s"].get("main") is not None else ["no_frame"])
        if count > 250:
            raise ValueError("Total image input exceeds this workflow's 250-image limit")
        payload = {"model": self.model, "messages": request_messages, "temperature": 0,
                   "max_tokens": 5000, "response_format": {"type": "json_object"}}
        if self.model.startswith(("qwen3.8-max", "qwen3.5-plus", "qwen3-vl-plus", "qwen3-vl-flash")):
            payload["enable_thinking"] = False
        if self.model.startswith("qwen3.8-max"):
            schema = (QualityReview if quality_only else Review).model_json_schema()
            if quality_only:
                schema["properties"]["quality_by_camera"].update(
                    properties={name: {"$ref": "#/$defs/CameraQuality"} for name in CAMERAS},
                    required=list(CAMERAS), additionalProperties=False)
            else:
                schema["properties"]["checks"].update(
                    properties={name: {"$ref": "#/$defs/Check"} for name in CHECKS},
                    required=list(CHECKS), additionalProperties=False)
                schema["properties"]["main_visibility_by_frame"] = {
                    "type": "object", "properties": {
                        frame_id: {"type": "string", "enum": states}
                        for frame_id, states in frame_states.items()},
                    "required": list(frame_states), "additionalProperties": False}
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "atomic_task_review", "strict": True, "schema": schema}}
        with httpx.Client(timeout=httpx.Timeout(180, connect=15), transport=self.transport) as client:
            for attempt in range(2):
                folder = self.work / "calls" / uuid.uuid4().hex
                write(folder / "request.json", {
                    "model": self.model, "base_url": self.base_url, "attempt": attempt + 1,
                    "context": context or {}, "request_sha256": fingerprint(payload),
                    "input_images": count, "messages": safe_messages(request_messages),
                    "parameters": {k: v for k, v in payload.items() if k != "messages"}})
                start = time.monotonic()
                try:
                    response = client.post(self.base_url + "/chat/completions", json=payload,
                                           headers={"Authorization": "Bearer " + key})
                except httpx.TransportError as exc:
                    write(folder / "error.json", {"type": type(exc).__name__})
                    if attempt == 0:
                        continue
                    raise RuntimeError("Qwen transport failure") from exc
                try:
                    raw = json.loads(response.text.replace(key, "[redacted]"))
                except ValueError:
                    raw = {"invalid_json": True}
                elapsed = time.monotonic() - start
                write(folder / "response.json", {"status": response.status_code,
                                                 "elapsed_s": elapsed, "body": raw})
                if response.status_code >= 400:
                    if attempt == 0 and response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(1)
                        continue
                    raise RuntimeError(f"Qwen HTTP {response.status_code}")
                choice = raw.get("choices", [{}])[0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Qwen response is incomplete")
                return {"data": json.loads(choice["message"]["content"]),
                        "model": raw.get("model", self.model), "usage": raw.get("usage", {}),
                        "elapsed_s": elapsed, "call_path": str(folder.relative_to(self.work)),
                        "request_id": raw.get("id")}
        raise RuntimeError("Qwen returned no result")


