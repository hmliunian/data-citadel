#!/usr/bin/env bash
# Project-local tool installation; shell profiles and global Python stay untouched.
set -euo pipefail
citadel_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
citadel_bin="$citadel_root/.tools/bin"
citadel_install="$citadel_root/.tools/installers"
mkdir -p "$citadel_bin" "$citadel_install"
export PATH="$citadel_bin:$PATH"
export UV_CACHE_DIR="$citadel_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$citadel_root/.tools/python"
export UV_PYTHON_BIN_DIR="$citadel_bin"

if ! command -v uv >/dev/null; then
  curl --proto '=https' --tlsv1.2 -fLsS --retry 3 \
    https://releases.astral.sh/github/uv/releases/download/0.12.13/uv-installer.sh \
    -o "$citadel_install/uv.sh"
  UV_UNMANAGED_INSTALL="$citadel_bin" sh "$citadel_install/uv.sh"
fi
if ! command -v just >/dev/null; then
  curl --proto '=https' --tlsv1.2 -fLsS --retry 3 \
    https://just.systems/install.sh -o "$citadel_install/just.sh"
  bash "$citadel_install/just.sh" --tag 1.58.0 --to "$citadel_bin"
fi
uv --version
just --version
if [[ "${1:-}" != "--tools-only" ]]; then
  cd "$citadel_root"
  just setup
fi
printf '\nTools are ready. In your project shell:\n  export PATH="%s:$PATH"\n  just dev\n' "$citadel_bin"
