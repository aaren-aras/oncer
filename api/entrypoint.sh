#!/usr/bin/env bash
set -e # exit if anything fails

MODEL_PATH=${MODEL_PATH:-/workspace/Oncer/api/src/models/oncer_model.keras}

echo "Checking for model at: $MODEL_PATH"

if [ ! -f "$MODEL_PATH" ]; then
    echo 'Model not found. Generating...'
    # Prepare BraTS 2021 data and generate model files
    python -m src.scripts.data
    python -m src.scripts.model
    echo 'Model generation complete!'
else
    echo 'Model already exists... Skipping.'
fi

echo "Starting backend... on port $API_PORT"
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --reload
