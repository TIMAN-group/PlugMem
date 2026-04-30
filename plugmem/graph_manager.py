"""Multi-graph lifecycle manager.

Manages multiple MemoryGraph instances, each identified by a graph_id.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from plugmem.clients.embedding import EmbeddingClient
from plugmem.clients.llm import LLMClient
from plugmem.clients.llm_router import LLMRouter
from plugmem.core.memory_graph import MemoryGraph
from plugmem.storage.chroma import ChromaStorage


class GraphManager:
    """Manages creation, retrieval, and deletion of MemoryGraph instances."""

    def __init__(
        self,
        storage: ChromaStorage,
        llm: LLMClient,
        embedder: EmbeddingClient,
    ):
        self._storage = storage
        self._llm = llm
        self._embedder = embedder
        self._graphs: Dict[str, MemoryGraph] = {}

    def create_graph(self, graph_id: Optional[str] = None) -> str:
        """Create a new memory graph. Returns the graph_id."""
        if graph_id is None:
            graph_id = uuid.uuid4().hex[:12]

        self._storage.create_graph(graph_id)
        graph = MemoryGraph(
            graph_id=graph_id,
            storage=self._storage,
            llm=self._llm,
            embedder=self._embedder,
        )
        self._graphs[graph_id] = graph
        return graph_id

    def get_graph(self, graph_id: str) -> MemoryGraph:
        """Get or load a MemoryGraph by ID."""
        if graph_id in self._graphs:
            return self._graphs[graph_id]

        if not self._storage.graph_exists(graph_id):
            raise KeyError(f"Graph '{graph_id}' does not exist")

        graph = MemoryGraph(
            graph_id=graph_id,
            storage=self._storage,
            llm=self._llm,
            embedder=self._embedder,
        )
        graph.load()
        self._graphs[graph_id] = graph
        return graph

    def delete_graph(self, graph_id: str) -> None:
        """Delete a graph and all its data."""
        self._storage.delete_graph(graph_id)
        self._graphs.pop(graph_id, None)

    def list_graphs(self) -> List[str]:
        """List all graph IDs."""
        return self._storage.list_graphs()

    def get_stats(self, graph_id: str) -> Dict[str, int]:
        """Get node counts for a graph."""
        return self._storage.get_graph_stats(graph_id)
