#!/usr/bin/env bash
set -e # exit if anything fails

MODEL_PATH=${MODEL_PATH:-/workspace/Oncer/api/models/oncer_model.keras}
PROCESSED_DATA_DIR=${PROCESSED_DATA_DIR:-/workspace/Oncer/api/data/BraTS2021_Processed_Data}

echo "Checking for processed data at: $PROCESSED_DATA_DIR"

# Checks if folder is populated with at least one .npy file (so manually delete folder to re-prep data)
if [ ! -d "$PROCESSED_DATA_DIR" ] || [ -z "$(find "$PROCESSED_DATA_DIR" -name '*.npy' -print -quit)" ]; then
    echo 'Processed data not found. Running data preparation...'
    python -m src.scripts.data
    echo 'Data preparation complete!'
else
    echo 'Processed data already exists... Skipping data.py.'
fi

echo "Checking for model at: $MODEL_PATH"

if [ ! -f "$MODEL_PATH" ]; then
    echo 'Model not found. Generating...'
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