"""Read-only local viewer for existing pilot results and prepared media."""
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .data import ID_PATTERN, load_manifest, read_json

ASSET_PATTERN = re.compile(r"[A-Za-z0-9_-]+\.(jpg|mp4)")


def create_app(run_dir: Path) -> FastAPI:
    root = run_dir.resolve()
    manifest = load_manifest(root)
    app = FastAPI(title="Data Citadel cache viewer", docs_url=None, redoc_url=None, openapi_url=None)

    def saved_results():
        rows = []
        for path in sorted((root / "results").glob("*/*.json")):
            if path.is_symlink() or path.resolve().parent.parent != root / "results":
                continue
            try:
                row = read_json(path)
            except (OSError, ValueError):
                continue
            episode_id = row.get("episode_id") if isinstance(row, dict) else None
            if (isinstance(episode_id, str) and ID_PATTERN.fullmatch(episode_id)
                    and episode_id in manifest["episodes"]):
                rows.append(row)
        return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)

    def is_visible(episode_id, rows):
        freeze = root / "freeze.json"
        return episode_id not in manifest["splits"]["holdout"] or (
            freeze.is_file() and not freeze.is_symlink()
            and any(row["episode_id"] == episode_id for row in rows))

    def require_episode(episode_id):
        if not ID_PATTERN.fullmatch(episode_id) or episode_id not in manifest["episodes"]:
            raise HTTPException(404, "Unknown episode")
        if not is_visible(episode_id, saved_results()):
            raise HTTPException(403, "Holdout is locked until freeze and a saved result")
        return manifest["episodes"][episode_id]

    def media_manifest(episode_id):
        require_episode(episode_id)
        folder = root / "media" / episode_id
        path = folder / "frames.json"
        if path.is_symlink() or path.resolve().parent != folder or not path.is_file():
            raise HTTPException(404, "Prepared media is unavailable")
        media = read_json(path)
        if media["episode_id"] != episode_id:
            raise HTTPException(409, "Media episode mismatch")
        return media

    def asset_url(episode_id, relative, suffix):
        name = Path(relative).name
        if (not ASSET_PATTERN.fullmatch(name) or Path(name).suffix != suffix
                or relative != f"media/{episode_id}/{name}"):
            raise HTTPException(409, "Invalid prepared media path")
        return f"/media/{episode_id}/{name}"

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "read_only"}

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("viewer.html"), media_type="text/html")

    @app.get("/api/results")
    def results():
        rows = saved_results()
        return [row for row in rows if is_visible(row["episode_id"], rows)]

    @app.get("/api/episodes/{episode_id}")
    def episode(episode_id: str):
        media = media_manifest(episode_id)
        record = manifest["episodes"][episode_id]
        result = {key: record.get(key) for key in (
            "episode_id", "task_code", "instruction", "quality", "gt", "gt_status", "gt_reason")}
        result["expert_ids"] = manifest["splits"]["experts"]
        result["sampling"] = manifest["sampling"]
        result["warnings"] = media.get("warnings", [])
        result["frames"] = [{**{key: frame[key] for key in (
            "frame_id", "view", "topic", "time_s", "source_ns", "video_time_s", "path",
            "sha256", "width", "height")},
            "url": asset_url(episode_id, frame["path"], ".jpg")} for frame in media["frames"]]
        result["views"] = {view: {**{key: data[key] for key in (
            "topic", "video_path", "start_s", "end_s", "decoded_frames", "sampled_frames",
            "uncovered_targets_s")}, "url": asset_url(episode_id, data["video_path"], ".mp4")}
            for view, data in media["views"].items()}
        return result

    @app.get("/media/{episode_id}/{asset_name}")
    def asset(episode_id: str, asset_name: str):
        media = media_manifest(episode_id)
        if not ASSET_PATTERN.fullmatch(asset_name):
            raise HTTPException(404, "Unknown media asset")
        urls = {asset_url(episode_id, frame["path"], ".jpg") for frame in media["frames"]}
        urls.update(asset_url(episode_id, view["video_path"], ".mp4")
                    for view in media["views"].values())
        folder = root / "media" / episode_id
        path = folder / asset_name
        if (f"/media/{episode_id}/{asset_name}" not in urls or path.is_symlink()
                or path.resolve().parent != folder or not path.is_file()):
            raise HTTPException(404, "Unknown media asset")
        return FileResponse(path, media_type="image/jpeg" if path.suffix == ".jpg" else "video/mp4",
                            headers={"X-Content-Type-Options": "nosniff"})

    return app
