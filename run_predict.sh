#!/usr/bin/env bash
set -euo pipefail

python src/03_predictive_model.py \
  --task classify \
  --algo lr \
  --curated_dir "./outputs/curated" \
  --raw_glob "./data/raw/**/*.csv" \
  --models_dir "./outputs/models"
