#!/usr/bin/env bash
# Idempotent Cloud Agent setup for haiku-judge-evals.
# Creates a project virtualenv at .venv and installs the package (with its
# pinned dependencies) in editable mode. Safe to run repeatedly.
set -euo pipefail

cd "$(dirname "$0")/.."

# The system Python ships without ensurepip on Debian/Ubuntu, which is required
# to bootstrap a virtualenv. Install the venv package once if it's missing.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

# Create the virtualenv if it doesn't already exist.
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

# Install/refresh dependencies from the pinned pyproject.toml into the venv.
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .

echo "haiku-judge-evals install complete. Activate with: source .venv/bin/activate"
