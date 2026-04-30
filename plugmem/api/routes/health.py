"""Health check endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from plugmem import __version__
from plugmem.api.dependencies import get_config, get_embedder, get_graph_manager, get_llm
from plugmem.api.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    cfg = get_config()

    llm_ok = False
    try:
        llm = get_llm(cfg)
        if llm is not None:
            llm_ok = True
    except Exception:
        logger.debug("LLM health check failed", exc_info=True)

    embedding_ok = False
    try:
        embedder = get_embedder(cfg)
        if embedder is not None:
            embedding_ok = True
    except Exception:
        logger.debug("Embedding health check failed", exc_info=True)

    chroma_ok = False
    try:
        gm = get_graph_manager(cfg)
        # Verify ChromaDB connection by listing graphs
        gm.list_graphs()
        chroma_ok = True
    except Exception:
        logger.debug("ChromaDB health check failed", exc_info=True)

    return HealthResponse(
        status="ok" if (llm_ok and embedding_ok and chroma_ok) else "degraded",
        version=__version__,
        llm_available=llm_ok,
        embedding_available=embedding_ok,
        chroma_available=chroma_ok,
    )
