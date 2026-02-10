#!/bin/bash
# Quick start script for deploying nvidia/NV-Embed-v2 with FastAPI

set -e

MODEL_NAME="${1:-nvidia/NV-Embed-v2}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8001}"
DEVICE="${4:-cuda}"
CUDA_VISIBLE_DEVICES="${5:-3}"
TORCH_DTYPE="${6:-bfloat16}"   # common choice; you can set float16/float32 as needed
MAX_LENGTH="${7:-8192}"
BATCH_SIZE="${8:-16}"

echo "=========================================="
echo "NV-Embed-v2 API Server"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Address: $HOST:$PORT"
echo "Device: $DEVICE"
echo "torch_dtype: $TORCH_DTYPE"
echo "max_length: $MAX_LENGTH"
echo "batch_size: $BATCH_SIZE"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
echo "=========================================="

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "Detected GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | nl -v 0
    echo ""
fi

echo "Checking dependencies..."
if ! python -c "import transformers" 2>/dev/null; then
    echo "transformers not installed, installing..."
    pip install transformers
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "torch not installed, installing..."
    pip install torch
fi

if ! python -c "import fastapi" 2>/dev/null; then
    echo "fastapi not installed, installing..."
    pip install fastapi uvicorn pydantic
fi

# Determine device if auto
if [ "$DEVICE" = "auto" ]; then
    if python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" | grep -q "cuda"; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

echo "Using device: $DEVICE"
echo ""

echo "Starting NV-Embed-v2 API server..."
echo ""

python nv_embed_v2_server.py \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE"
