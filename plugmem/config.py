from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlugMemConfig:
    """Central configuration for the PlugMem service."""

    # LLM
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Embeddings
    embedding_base_url: str = ""
    embedding_model: str = "nvidia/NV-Embed-v2"

    # ChromaDB
    chroma_mode: str = "persistent"  # "persistent" | "http" | "ephemeral"
    chroma_path: str = "./data/chroma"  # for persistent mode
    chroma_host: str = "localhost"  # for http mode
    chroma_port: int = 8000  # for http mode

    # Service
    api_key: Optional[str] = None  # for service auth

    # LLM call defaults
    max_retries: int = 5
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_tokens: int = 4096

    # Embedding limits
    embedding_max_text_len: int = 8192
