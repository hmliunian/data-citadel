import subprocess

import pytest

from scripts.check_rules import check, main


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("artifacts/\n.cache/\n")
    return tmp_path


def source(repo, path, content):
    path = repo / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize(("path", "code"), [
    ("citadel/domain/rules.py", "from ..infrastructure.storage import ArtifactRepository"),
    ("citadel/domain/rules.py", "from citadel import infrastructure as storage"),
    ("citadel/domain/rules.py", "def run():\n    import httpx"),
    ("citadel/application/review.py", "from .. import bootstrap"),
    ("citadel/application/review.py", "import sqlite3"),
    ("citadel/infrastructure/store.py", "from ..application.reviews import ReviewService"),
    ("citadel/server/routes.py", "from citadel.infrastructure import storage"),
    ("citadel_client/api.py", "from citadel.domain.models import Review"),
    ("citadel/domain/rules.py", "from tests import conftest"),
    ("citadel/bootstrap.py", "from scripts.dev import main"),
])
def test_rejects_dependencies_across_layer_boundaries(repo, path, code):
    source(repo, path, code)
    errors = check(repo)
    assert len(errors) == 1
    assert errors[0].startswith(path + ":")


def test_accepts_protocols_relative_imports_and_entry_points(repo):
    files = {
        "citadel/domain/rules.py": "from .models import Review\nfrom pydantic import BaseModel\nfrom functools import cache",
        "citadel/application/review.py": "from ..domain import rules\nfrom citadel.configuration import Configuration",
        "citadel/infrastructure/store.py": "from ..application.ports import ModelGateway\nimport httpx",
        "citadel/server/routes.py": "from ..application.reviews import ReviewService\nfrom fastapi import FastAPI",
        "citadel/bootstrap.py": "from .infrastructure import storage\nfrom .application import reviews",
        "citadel/__main__.py": "from .bootstrap import build_runtime",
        "citadel_client/api.py": "from . import errors\nimport httpx",
        "scripts/experiment.py": "from citadel.bootstrap import build_runtime\nwork = 'artifacts/experiments/run'",
        "tests/test_review.py": "from scripts.experiment import main\nfrom citadel import bootstrap",
        "gui/app.js": "import './api.js';",
    }
    for path, code in files.items():
        source(repo, path, code)
    assert check(repo) == []


@pytest.mark.parametrize("code", [
    "import artifacts.quality_v2.replay",
    "from artifacts import baseline_server",
    "from .source_snapshot import model",
    "from _cache import legacy",
])
def test_rejects_artifact_imports_even_in_tools_and_reports_line(repo, code):
    source(repo, "scripts/experiment.py", "# experiment\n" + code)
    assert check(repo)[0].startswith("scripts/experiment.py:2: 不得导入")


@pytest.mark.parametrize("path", [
    "probe.py", "experiments/run.py", "config/job.py", "citadel/legacy/model.py",
    "citadel/helpers.py", "gui/server.py", "config/ui.js", "citadel/configuration/extra.py",
])
def test_rejects_misplaced_source(repo, path):
    source(repo, path, "")
    assert check(repo)[0].startswith(path + ":1:")


def test_ignores_local_artifacts_but_rejects_force_tracked_source(repo):
    legacy = source(repo, "artifacts/old/source_snapshot/model.py", "invalid historical code")
    source(repo, ".cache/scratch/probe.py", "invalid temporary code")
    assert check(repo) == []
    subprocess.run(["git", "add", "-f", str(legacy)], cwd=repo, check=True)
    assert check(repo)[0].startswith("artifacts/old/source_snapshot/model.py:1:")


def test_rejects_source_links_and_returns_failing_status(repo, capsys):
    legacy = source(repo, "artifacts/old.py", "import json")
    path = repo / "scripts" / "old.py"
    path.parent.mkdir()
    path.symlink_to(legacy)
    assert main(repo) == 1
    assert "scripts/old.py:1:" in capsys.readouterr().out
    path.unlink()
    assert main(repo) == 0


def test_deleted_tracked_source_is_not_scanned(repo):
    path = source(repo, "scripts/old.py", "import artifacts.old")
    subprocess.run(["git", "add", str(path)], cwd=repo, check=True)
    path.unlink()
    assert check(repo) == []


def test_syntax_error_has_file_and_line(repo):
    source(repo, "scripts/broken.py", "def broken(\n")
    assert check(repo)[0].startswith("scripts/broken.py:1: Python 语法错误")
