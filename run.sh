#!/usr/bin/env bash
set -euo pipefail

############################################################
# ITCS-6190 COURSE PROJECT — MASTER PIPELINE (run.sh)
# Runs the entire offline pipeline with ONE command:
#
#   1. Ingest raw → curated
#   2. Extended EDA (tables + plots)
#   3. Spark SQL analysis
#   4. ML model training (PipelineModel)
#   5. Create streaming micro-batches
#
# Streaming + prediction demo is run separately using:
#   ./run_stream_predict.sh
############################################################

log(){ printf "\n[%s] %s\n" "$(date +'%H:%M:%S')" "$*"; }

###############################
# ENVIRONMENT
###############################
[ -d ".venv" ] && source .venv/bin/activate || true
[ -f ".env" ] && export $(grep -v '^#' .env | xargs) || true

export RAW_DATA_GLOB="./data/raw/**/*.csv"
export CURATED_DIR="./outputs/curated"
export TABLE_DIR="./outputs/tables"
export PLOTS_DIR="./outputs/plots"
export STREAM_DIR="./data/stream"

mkdir -p "$CURATED_DIR" "$TABLE_DIR" "$PLOTS_DIR" "$STREAM_DIR"

###############################
# JAVA for Spark
###############################
if ! command -v java >/dev/null 2>&1; then
  log "Installing OpenJDK 17…"
  sudo apt-get update -y
  sudo apt-get install -y openjdk-17-jdk
fi
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

###############################
# PYTHON DEPENDENCIES
###############################
log "Installing Python dependencies…"
pip install --upgrade pip
pip install -r requirements.txt || pip install pyspark pandas matplotlib pyarrow python-dotenv

################################
# STEP 1 — INGEST (ONLY IF RAW CSVs EXIST)
################################
echo "[1/5] Checking for raw CSVs in data/raw/ ..."

if ls data/raw/*.csv data/raw/**/*.csv 1>/dev/null 2>&1; then
    echo "[1/5] Raw CSVs found — running ingestion."
    python src/01_ingest_eda.py
else
    echo "[1/5] No raw CSV files found in data/raw/ — skipping ingestion and using existing curated data."
fi



###############################
# STEP 2 — EXTENDED EDA
###############################
if [ -f "src/02_extended_eda.py" ]; then
  log "STEP 2 — Extended EDA (tables + plots)"
  python src/02_extended_eda.py \
    --curated_dir "$CURATED_DIR" \
    --plots_dir "$PLOTS_DIR" \
    --sample_for_plots 250000
  log "✓ EDA tables → outputs/tables/ | Plots → outputs/plots/"
else
  log "⚠ SKIP: src/02_extended_eda.py not found"
fi

###############################
# STEP 3 — SPARK SQL ANALYSIS
###############################
if [ -f "src/02_sql_analysis.py" ]; then
  log "STEP 3 — Spark SQL analysis"
  python src/02_sql_analysis.py
  log "✓ SQL output tables → outputs/tables/"
else
  log "⚠ WARNING: src/02_sql_analysis.py missing — SQL requirement not satisfied"
fi

###############################
# STEP 4 — ML MODEL TRAINING
###############################
log "STEP 4 — Train ML model"
python src/03_predictive_model.py \
  --task classify \
  --algo lr \
  --curated_dir "$CURATED_DIR" \
  --models_dir "./outputs/models"
log "✓ Model saved → outputs/models/"

###############################
# STEP 5 — CREATE STREAM BATCHES
###############################
log "STEP 5 — Create streaming micro-batches"
python src/04a_make_stream_batches.py \
  --curated_dir "$CURATED_DIR" \
  --stream_dir "$STREAM_DIR"
log "✓ Streaming batches → data/stream/"

###############################
# DONE
###############################
log "🎉 Pipeline COMPLETED successfully!"
echo "
=========================================================
 EVERYTHING IS READY:

  • Curated data:        outputs/curated/
  • EDA tables:          outputs/tables/
  • EDA plots:           outputs/plots/
  • SQL analysis:        outputs/tables/
  • Trained model:       outputs/models/
  • Streaming batches:   data/stream/

To run online streaming predictions + ML scoring:
    ./run_stream_predict.sh

To view streaming predictions PNG dashboard:
    python src/viz_stream_live.py
=========================================================
"