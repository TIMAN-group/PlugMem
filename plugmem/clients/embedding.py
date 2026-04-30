"""Embedding client abstractions.

Replaces the standalone get_embedding / get_similarity functions from utils.py
and provides a ChromaDB EmbeddingFunction adapter.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    """Abstract interface for text embedding."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        ...


class HTTPEmbeddingClient(EmbeddingClient):
    """Embedding client that calls an HTTP endpoint (e.g. NV-Embed-v2 server)."""

    def __init__(
        self,
        base_url: str,
        model: str = "nvidia/NV-Embed-v2",
        max_text_len: int = 8192,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        timeout: int = 60,
    ):
        self.base_url = base_url
        self.model = model
        self.max_text_len = max_text_len
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        text = text[: self.max_text_len]
        headers = {"Content-Type": "application/json"}
        data = {"model": self.model, "input": text}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url, json=data, headers=headers, timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                return result["data"][0]["embedding"]
            except Exception as e:
                logger.warning("[Attempt %d/%d] Embedding error: %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to get embedding after {self.max_retries} attempts")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class PlugMemEmbeddingFunction:
    """ChromaDB EmbeddingFunction adapter.

    Wraps an EmbeddingClient so ChromaDB can auto-embed documents on insert.
    Implements the chromadb.EmbeddingFunction protocol.
    """

    def __init__(self, client: EmbeddingClient):
        self._client = client

    def name(self) -> str:
        return "plugmem"

    def is_legacy(self) -> bool:
        return False

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._client.embed_batch(input)


def create_embedding_client_from_env() -> EmbeddingClient:
    """Create an embedding client from environment variables (backward compat)."""
    return HTTPEmbeddingClient(
        base_url=os.environ["EMBEDDING_BASE_URL"],
        model="nvidia/NV-Embed-v2",
    )


def get_similarity(x: List[float], y: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(x, dtype=np.float32)
    b = np.asarray(y, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
