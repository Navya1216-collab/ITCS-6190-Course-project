#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Load env vars if .env exists
if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs) || true
fi

python src/01_ingest_eda.py
