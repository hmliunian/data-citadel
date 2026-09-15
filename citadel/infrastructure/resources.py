"""Fetch current task instructions and reference images without GT."""
from __future__ import annotations

import base64
import io
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image

from .datasets import TASK
from .files import file_hash, fingerprint, read, write

DEFAULT_BASE = "http://172.100.11.189:8001"


def image_input(root: Path, item: dict):
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != item["sha256"]:
        raise ValueError("Image path or content changed")
    encoded = base64.b64encode(path.read_bytes()).decode()
    return "data:" + item.get("mime", "image/jpeg") + ";base64," + encoded


def fetch(code: str, work: Path, base_url: str = DEFAULT_BASE, *, client=None):
    if not TASK.fullmatch(code):
        raise ValueError("Invalid task code")
    cache = work / "resources" / code / "task.json"
    if cache.exists():
        saved = read(cache)
        if (saved["task_code"] != code or
                fingerprint({k: v for k, v in saved.items() if k != "sha256"}) != saved["sha256"]):
            raise ValueError("Cached task mismatch")
        for item in saved["images"]:
            image_input(work, item)
        return saved
    if client is None:
        endpoint = urlsplit(base_url)
        mounts = {}
        try:
            if ipaddress.ip_address(endpoint.hostname).is_private:
                mounts[f"{endpoint.scheme}://{endpoint.netloc}"] = httpx.HTTPTransport()
        except ValueError:
            pass
        with httpx.Client(timeout=20, follow_redirects=True, mounts=mounts) as connection:
            return fetch(code, work, base_url, client=connection)
    response = client.get(base_url.rstrip("/") + f"/api/collection_tasks/code/{code}/resources")
    response.raise_for_status()
    body = response.json()
    task = body.get("data") or {}
    if (body.get("code") != 200 or task.get("task_code") != code or not task.get("steps")
            or any(not s.get("action_text") or not s.get("action_id") for s in task["steps"])):
        raise ValueError("Task resources contain no valid matching instruction")
    references = [i for i in task.get("images", []) if i.get("type") in
                  ("object", "scene", "execution_location")]
    if not {"object", "scene"} <= {i["type"] for i in references}:
        raise ValueError("Task needs both object and scene references")
    images = []
    for index, item in enumerate(references):
        picture = client.get(item["url"])
        picture.raise_for_status()
        with Image.open(io.BytesIO(picture.content)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Unsupported reference image format")
            mime = Image.MIME[image.format]
            image.verify()
        folder = cache.parent
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"reference-{index}.img"
        path.write_bytes(picture.content)
        images.append({k: item.get(k) for k in ("id", "name", "type")} | {
            "path": str(path.relative_to(work)), "sha256": file_hash(path), "mime": mime})
    saved = {"task_code": code, "task_name": task.get("task_name"),
             "steps": task["steps"], "images": images}
    saved["sha256"] = fingerprint(saved)
    write(cache, saved)
    return saved
