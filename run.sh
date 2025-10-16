#!/usr/bin/env bash
set -euo pipefail

log(){ printf "\n[%s] %s\n" "$(date +'%H:%M:%S')" "$*"; }

# venv (if present)
[ -d ".venv" ] && source .venv/bin/activate

# env (optional)
[ -f ".env" ] && export $(grep -v '^#' .env | xargs) || true

# defaults
: "${RAW_DATA_GLOB:=./data/raw/*.csv}"
: "${CURATED_DIR:=./outputs/curated}"
: "${TABLE_DIR:=./outputs/tables}"
: "${PLOTS_DIR:=./outputs/plots}"

mkdir -p "$(dirname "$CURATED_DIR")" "$TABLE_DIR" "$PLOTS_DIR"

# Java for Spark (Codespaces)
if ! command -v java >/dev/null 2>&1; then
  log "Installing OpenJDK 17…"
  sudo apt-get update -y && sudo apt-get install -y openjdk-17-jdk
fi
export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}
export PATH="$JAVA_HOME/bin:$PATH"

# Python deps
pip install --upgrade pip
pip install -r requirements.txt || pip install pyspark python-dotenv pandas matplotlib

log "Running 01_ingest_eda.py (tables)…"
python src/01_ingest_eda.py

log "Running 02_extended_eda.py (plots)…"
python src/02_extended_eda.py \
  --curated_dir "$CURATED_DIR" \
  --plots_dir "$PLOTS_DIR" \
  --sample_for_plots 250000

log "Done ✅
- Tables: $TABLE_DIR
- Plots:  $PLOTS_DIR
- Curated parquet: $CURATED_DIR"