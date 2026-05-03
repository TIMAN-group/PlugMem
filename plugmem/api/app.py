"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from plugmem import __version__
from plugmem.api.routes import demo, inspector, extract, graphs, health, memories, retrieval

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


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
    app.include_router(inspector.router, prefix="/api/v1")
    app.include_router(demo.router, prefix="/api/v1")

    # Memory Inspector — static SPA mounted at /inspector/
    inspector_dir = _STATIC_DIR / "inspector"
    if inspector_dir.is_dir():
        app.mount(
            "/inspector",
            StaticFiles(directory=str(inspector_dir), html=True),
            name="inspector",
        )

        @app.get("/", include_in_schema=False)
        async def _root() -> RedirectResponse:
            return RedirectResponse(url="/inspector/")

    return app


app = create_app()
