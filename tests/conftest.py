"""Shared test fixtures — mock LLM/embedding clients, ephemeral ChromaDB."""
from __future__ import annotations

import random
from typing import Dict, List
from unittest.mock import MagicMock

import chromadb
import pytest
from fastapi.testclient import TestClient

from plugmem.api import dependencies as deps
from plugmem.api.app import create_app
from plugmem.clients.embedding import EmbeddingClient, PlugMemEmbeddingFunction
from plugmem.clients.llm import LLMClient
from plugmem.graph_manager import GraphManager
from plugmem.storage.chroma import ChromaStorage


# ------------------------------------------------------------------ #
# Fake clients
# ------------------------------------------------------------------ #

class FakeLLM(LLMClient):
    """Returns canned responses for testing."""

    def __init__(self):
        self.calls: list = []

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        self.calls.append(messages)
        # Return a plausible mode for get_mode calls
        last_msg = messages[-1]["content"] if messages else ""
        if "which type of memory" in last_msg.lower() or "mode" in last_msg.lower():
            return "semantic_memory"
        # Return plausible plan/tags
        if "tag" in last_msg.lower() or "plan" in last_msg.lower():
            return '{"next_subgoal": "test goal", "query_tags": ["tag1"]}'
        return "test response"


class FakeEmbedder(EmbeddingClient):
    """Returns deterministic random embeddings for testing."""

    DIM = 64

    def embed(self, text: str) -> List[float]:
        rng = random.Random(hash(text) % (2**31))
        vec = [rng.gauss(0, 1) for _ in range(self.DIM)]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture()
def fake_llm():
    return FakeLLM()


@pytest.fixture()
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture()
def chroma_client():
    return chromadb.EphemeralClient()


@pytest.fixture()
def storage(chroma_client, fake_embedder):
    embed_fn = PlugMemEmbeddingFunction(fake_embedder)
    return ChromaStorage(
        client=chroma_client,
        embedding_function=embed_fn,
        embedding_client=fake_embedder,
    )


@pytest.fixture()
def graph_manager(storage, fake_llm, fake_embedder):
    return GraphManager(storage=storage, llm=fake_llm, embedder=fake_embedder)


@pytest.fixture()
def client(graph_manager, fake_llm, fake_embedder):
    """TestClient wired to fake backends."""
    deps.reset_singletons()

    # Override the singletons
    deps._llm_client = fake_llm
    deps._embedding_client = fake_embedder
    deps._graph_manager = graph_manager

    app = create_app()
    with TestClient(app) as tc:
        yield tc

    deps.reset_singletons()
