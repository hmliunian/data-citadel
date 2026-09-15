"""Load prompt text independently from executable code."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tomllib

from pydantic import BaseModel, ConfigDict, Field

from citadel.domain.errors import GateError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"


@dataclass(frozen=True)
class PromptBundle:
    review_system: str
    gripper: str
    quality: str

    @classmethod
    def load(cls, root: Path = CONFIG_ROOT):
        values = {name: (root / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
                  for name in cls.__dataclass_fields__}
        if any(not text.strip() for text in values.values()):
            raise ValueError("Prompt files must not be empty")
        return cls(**values)

    def as_dict(self):
        return asdict(self)


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = "qwen3.8-max-0902"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0
    enable_thinking: bool = False
    max_tokens: int = Field(default=5000, ge=1)
    timeout_s: float = Field(default=180, gt=0)
    connect_timeout_s: float = Field(default=15, gt=0)
    attempts: int = Field(default=2, ge=1, le=5)
    max_images: int = Field(default=250, ge=1, le=250)


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    host: str = "127.0.0.1"
    port: int = Field(default=8770, ge=1, le=65535)
    work_dir: Path = Path("artifacts/server")
    workers: int = Field(default=3, ge=1, le=8)
    media_workers: int = Field(default=1, ge=1, le=8)
    max_pending: int = Field(default=128, ge=1)
    resource_base: str = "http://172.100.11.189:8001"


@dataclass(frozen=True)
class ConfigSnapshot:
    """Canonical JSON makes nested configuration immutable as well."""
    payload: str

    @classmethod
    def create(cls, value: dict):
        return cls(json.dumps(value, ensure_ascii=False, sort_keys=True))

    @property
    def data(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()


def code_version(root: Path = PROJECT_ROOT / "citadel"):
    paths = [root / "configuration.py", root / "application/prompts.py",
             root / "application/reviews.py"]
    for folder in ("domain", "infrastructure"):
        paths.extend((root / folder).rglob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.exists() and p.name != "__init__.py"}


class Configuration:
    def __init__(self, root: Path = CONFIG_ROOT):
        self.root = root.resolve()
        self.loaded_code = code_version()

    def model(self):
        value = tomllib.loads((self.root / "models.toml").read_text())
        for field, variable in (("model", "QWEN_MODEL"), ("base_url", "QWEN_BASE_URL")):
            if os.getenv(variable):
                value[field] = os.environ[variable]
        return ModelSettings.model_validate(value)

    def snapshot(self, manifest: dict, resource_base: str):
        current = code_version()
        if current != self.loaded_code:
            raise GateError("Code changed; restart the server before reviewing")
        return ConfigSnapshot.create({
            "manifest_sha256": manifest["sha256"],
            "profiles": json.loads((self.root / "tasks.json").read_text()),
            "prompts": PromptBundle.load(self.root).as_dict(),
            "model": self.model().model_dump(), "code": current,
            "sampling": manifest["sampling"], "resource_base": resource_base,
        })


def server_configuration(root: Path = CONFIG_ROOT):
    data = tomllib.loads((root / "server.toml").read_text())
    settings = ServerSettings.model_validate(data["server"])
    if not settings.work_dir.is_absolute():
        settings = settings.model_copy(update={"work_dir": PROJECT_ROOT / settings.work_dir})
    return settings, data.get("runs", {})
