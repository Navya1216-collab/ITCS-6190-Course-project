#!/usr/bin/env bash
set -e

echo "=== 1) Training ML PipelineModel ==="
python src/03_predictive_model.py \
  --task classify \
  --algo lr \
  --curated_dir ./outputs/curated

echo "=== 2) Creating stream batches ==="
python src/04a_make_stream_batches.py \
  --curated_dir ./outputs/curated \
  --stream_dir ./data/stream

echo "=== 3) Starting Structured Streaming with Predictions ==="
python src/04_stream_predict.py
