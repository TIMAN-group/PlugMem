#!/usr/bin/env python3
"""
Simple FastAPI server for nvidia/NV-Embed-v2 model
Uses transformers library for compatibility
OpenAI-compatible endpoints:
  - GET  /v1/models
  - POST /v1/embeddings
  - GET  /health
"""

import argparse
from typing import List, Union, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI(title="NV-Embed-v2 Embedding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model: Optional[torch.nn.Module] = None
tokenizer: Optional[AutoTokenizer] = None
device: Optional[torch.device] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    # optional knobs
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


def load_model(model_name: str, device_name: Optional[str] = None, torch_dtype: Optional[str] = None):
    """
    Load NV-Embed-v2 model & tokenizer.
    """
    global model, tokenizer, device

    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    print(f"Loading model {model_name} on {device}...")

    dtype = None
    if torch_dtype:
        # allow: float16 / bfloat16 / float32
        dtype = getattr(torch, torch_dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype: {torch_dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully on {device}")
        print(f"✓ Hidden size: {getattr(model.config, 'hidden_size', 'unknown')}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise


def _masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling with attention mask.
    last_hidden_state: [B, T, H]
    attention_mask:   [B, T]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B, 1]
    return summed / counts




def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

@torch.no_grad()
def get_embeddings(texts: List[str], max_length:int = 8192, batch_size:int = 16, normalize: bool = True):
    global model, tokenizer, device

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # -----------------------------
        # Path A: preferred for NV-Embed-v2 (remote code)
        # -----------------------------
        if hasattr(model, "encode"):
            # NV-Embed-v2 remote code typically supports encode(prompts=..., ...)
            emb = model.encode(
                prompts=texts,
                instruction="",
                max_length=max_length,   # 先用较小值稳定跑通，再考虑加大
                batch_size=batch_size,
                num_workers=0
            )

            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            elif isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)

            if normalize:
                emb = _l2_normalize(emb)
            return emb

        # -----------------------------
        # Path B: generic HF fallback
        # -----------------------------
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not loaded")

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs might be ModelOutput / dict-like / tuple
        last_hidden = None

        # 1) Standard ModelOutput
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            last_hidden = outputs.last_hidden_state
        # 2) Some models return a dict
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            last_hidden = outputs["last_hidden_state"]
        # 3) Many HF models return tuple where first element is last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            last_hidden = outputs[0]
        # 4) If hidden_states is available, use last layer
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
        elif isinstance(outputs, dict) and "hidden_states" in outputs and outputs["hidden_states"] is not None:
            last_hidden = outputs["hidden_states"][-1]

        if last_hidden is None:
            raise HTTPException(
                status_code=500,
                detail="Model outputs do not contain last_hidden_state/hidden_states and model has no encode(); cannot pool embeddings."
            )

        # Mean pooling with attention mask
        attn = inputs.get("attention_mask", None)
        if attn is None:
            pooled = last_hidden.mean(dim=1)
        else:
            mask = attn.unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        if normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.detach().cpu().numpy()

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
async def root():
    return {
        "message": "NV-Embed-v2 Embedding API Server",
        "status": "running",
        "model": tokenizer.name_or_path if tokenizer else "not loaded",
    }


@app.get("/v1/models")
async def list_models():
    model_name = tokenizer.name_or_path if tokenizer else "unknown"
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "nvidia",
            }
        ],
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    try:
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")

        embeddings = get_embeddings(texts, normalize=request.normalize)

        data = []
        for i, vec in enumerate(embeddings):
            data.append({"object": "embedding", "embedding": vec.tolist(), "index": i})

        # token计数这里仍按 split() 粗略估算，和你 Qwen3 版本保持一致风格
        token_est = sum(len(t.split()) for t in texts)

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={"prompt_tokens": token_est, "total_tokens": token_est},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
    }


def main():
    parser = argparse.ArgumentParser(description="NV-Embed-v2 Embedding API Server")
    parser.add_argument("--model", type=str, default="nvidia/NV-Embed-v2", help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu/cuda:0 etc. Auto-detect if None.")
    parser.add_argument("--torch_dtype", type=str, default=None, help="float16/bfloat16/float32 (optional)")
    parser.add_argument("--max_length", type=int, default=8192, help="Tokenizer max_length (default 8192)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()

    print("=" * 60)
    print("NV-Embed-v2 Embedding API Server")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Device: {args.device or 'auto'}")
    print(f"torch_dtype: {args.torch_dtype or 'default'}")
    print(f"max_length: {args.max_length}")
    print(f"batch_size: {args.batch_size}")
    print("=" * 60)

    try:
        load_model(args.model, args.device, args.torch_dtype)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # stash pooling params into app state (optional)
    app.state.max_length = args.max_length
    app.state.batch_size = args.batch_size

    # monkey-patch get_embeddings default values with CLI args
    # (keeps code simple while allowing runtime flags)
    global get_embeddings
    original_get_embeddings = get_embeddings

    @torch.no_grad()
    def get_embeddings(texts: List[str], normalize: bool = True):
        return original_get_embeddings(
            texts=texts,
            normalize=normalize,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

    print(f"\nStarting server on {args.host}:{args.port}...")
    print("API endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  POST http://{args.host}:{args.port}/v1/embeddings")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
