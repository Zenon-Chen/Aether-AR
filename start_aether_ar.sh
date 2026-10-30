#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "[Setup] Initializing virtual environment for Aether-AR..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.11
        uv pip install -e ".[edge]"
    else
        python3 -m venv .venv
        .venv/bin/pip install --upgrade pip
        .venv/bin/pip install -e ".[edge]"
    fi
fi

exec .venv/bin/python launcher.py "$@"
