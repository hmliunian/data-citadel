set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments
root := justfile_directory()
export PATH := root / ".tools/bin" + ":" + env_var("PATH")
export UV_CACHE_DIR := root / ".cache/uv"
export UV_PYTHON_INSTALL_DIR := root / ".tools/python"
export UV_PYTHON_BIN_DIR := root / ".tools/bin"
config := env_var_or_default("CITADEL_CONFIG", root / "config")
url := env_var_or_default("CITADEL_URL", "")

# Show available commands.
default:
    @just --list

# Install Python dependencies exactly from uv.lock.
setup *args:
    uv sync --locked "$@"

# Start the API server and serve the browser GUI.
server *args:
    uv run --locked python -m citadel --config "{{config}}" serve "$@"

# Open the GUI for a running server; --no-browser only prints its URL.
client *args:
    uv run --locked python scripts/dev.py --config "{{config}}" --url "{{url}}" --client-only "$@"

# One command: sync environment, start server, wait for health, open GUI.
dev *args: setup
    uv run --locked python scripts/dev.py --config "{{config}}" "$@"

# Independent HTTP CLI, e.g. just cli runs.
cli *args:
    uv run --locked python -m citadel_client "$@"

# Unit and API tests use fake Qwen clients.
test *args:
    uv run --locked python -m pytest -q --basetemp artifacts/tests "$@"

# Optional real Firefox UI smoke test; no real Qwen requests.
test-gui:
    CITADEL_BROWSER_TEST=1 uv run --locked python -m pytest -q tests/test_gui.py --basetemp artifacts/gui-tests

# Static checks.
check:
    uv run --locked python -m ruff check citadel citadel_client scripts tests
    git diff --check
