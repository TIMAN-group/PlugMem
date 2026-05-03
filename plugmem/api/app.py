"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from plugmem import __version__
from plugmem.api.routes import extract, graphs, health, memories, retrieval

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and return the PlugMem FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("PlugMem service v%s starting", __version__)
        yield

    app = FastAPI(
        title="PlugMem",
        description="Pluggable memory system for LLM agents",
        version=__version__,
        lifespan=lifespan,
    )

    # Mount route modules under /api/v1
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(graphs.router, prefix="/api/v1")
    app.include_router(memories.router, prefix="/api/v1")
    app.include_router(retrieval.router, prefix="/api/v1")
    app.include_router(extract.router, prefix="/api/v1")

    return app


app = create_app()
