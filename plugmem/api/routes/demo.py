"""Demo data seeding endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from plugmem.api.auth import require_api_key
from plugmem.api.demo import DEMO_GRAPH_ID, seed_demo_graph
from plugmem.api.dependencies import get_graph_manager
from plugmem.api.schemas import GraphResponse

router = APIRouter(prefix="/demo", tags=["demo"], dependencies=[Depends(require_api_key)])


@router.post("/seed", response_model=GraphResponse)
async def seed(graph_id: str = DEMO_GRAPH_ID, reset: bool = False) -> GraphResponse:
    """Seed a realistic demo memory graph.

    Idempotent: if ``graph_id`` already exists and ``reset`` is False, returns
    the existing stats without rewriting.
    """
    gm = get_graph_manager()
    try:
        stats = seed_demo_graph(
            gm.storage,
            graph_id=graph_id,
            reset=reset,
            embedder=gm.embedder,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"seed failed: {exc}") from exc

    # Drop any cached MemoryGraph so the next read sees the fresh chroma state.
    gm.invalidate_cache(graph_id)
    return GraphResponse(graph_id=graph_id, stats=stats)
