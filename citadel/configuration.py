"""Load prompt text independently from executable code."""
from dataclasses import asdict, dataclass
from pathlib import Path

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
