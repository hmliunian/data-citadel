"""Explicit configuration; credentials are read only when an API call is made."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import CameraMode, ProviderError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    dataset_root: Path = field(default_factory=lambda: Path(os.environ.get(
        "DATASET_ROOT", str(PROJECT_ROOT.parent / "datasets")
    )))
    artifacts_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "CITADEL_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")
    )))
    experts_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "CITADEL_EXPERTS_PATH", str(PROJECT_ROOT / "config" / "experts.json")
    )))
    model: str = field(default_factory=lambda: os.environ.get("QWEN_MODEL", "qwen-vl-max"))
    base_url: str = field(default_factory=lambda: os.environ.get(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ))
    camera_topic: str = field(default_factory=lambda: os.environ.get(
        "CITADEL_CAMERA_TOPIC", "/camera/coracam_head/left_h264/video"
    ))
    camera_mode: CameraMode = field(default_factory=lambda: os.environ.get(
        "CITADEL_CAMERA_MODE", "main"
    ))
    request_timeout_s: float = 120.0
    max_retries: int = 2
    max_frames: int = 96
    max_image_size: int = 768
    max_request_images: int = 250
    correct_threshold: float = 0.95

    def api_key(self) -> str:
        direct = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if direct and direct.strip():
            return direct.strip()
        specified = os.environ.get("QWEN_API_KEY_FILE")
        candidates = [Path(specified)] if specified else [
            PROJECT_ROOT / "Qwen-api" / "qwen_api_key.txt",
            PROJECT_ROOT.parent / "dataset_checker" / "Qwen-api" / "qwen_api_key.txt",
        ]
        for path in candidates:
            if path.is_file():
                value = path.read_text().strip()
                if value:
                    return value
        raise ProviderError("未配置 Qwen 凭据：请设置 QWEN_API_KEY 或 QWEN_API_KEY_FILE")
