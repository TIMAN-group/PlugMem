"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from plugmem import __version__
from plugmem.api.routes import demo, inspector, extract, graphs, health, memories, retrieval

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class RequestLoggingMiddleware:
    """ASGI middleware to log requests and resource metrics with contextvar propagation support."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/v1"):
            await self.app(scope, receive, send)
            return

        import uuid
        import time
        import json
        from datetime import datetime, timezone
        from plugmem.api.logging_ctx import RequestContextLog, current_log_ctx, get_process_memory_mb, get_directory_size_mb
        from plugmem.api.dependencies import get_config

        # 1. Extract/generate Task ID
        headers = dict(scope.get("headers", []))
        task_id_bytes = headers.get(b"x-task-id") or headers.get(b"x-request-id")
        task_id = task_id_bytes.decode("utf-8") if task_id_bytes else f"task_{uuid.uuid4()}"

        log_ctx = RequestContextLog(task_id=task_id)
        token = current_log_ctx.set(log_ctx)

        start_time = time.perf_counter()
        status_code = [500]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            current_log_ctx.reset(token)
            latency = time.perf_counter() - start_time

            # Calculate tokens
            total_prompt_tokens = sum(c.get("prompt_tokens", 0) for c in log_ctx.llm_calls)
            total_completion_tokens = sum(c.get("completion_tokens", 0) for c in log_ctx.llm_calls)

            # Get resources
            ram_usage = get_process_memory_mb()
            
            try:
                cfg = get_config()
                db_dir = cfg.chroma_path
            except Exception:
                db_dir = "./data/chroma"
            db_size = get_directory_size_mb(db_dir)

            # Build record
            log_record = {
                "task_id": log_ctx.task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "endpoint": scope["path"],
                "method": scope["method"],
                "status_code": status_code[0],
                "latency_sec": latency,
                "memory_retrieved": log_ctx.memory_retrieved,
                "agent_output": log_ctx.agent_output,
                "llm_calls": log_ctx.llm_calls,
                "retrieval_calls": log_ctx.retrieval_calls,
                "consolidation_calls": log_ctx.consolidation_calls,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "ram_usage_mb": ram_usage,
                "db_disk_usage_mb": db_size,
            }

            try:
                req_logger = logging.getLogger("plugmem.request_logger")
                if not req_logger.handlers:
                    req_logger.setLevel(logging.INFO)
                    req_logger.propagate = False
                    log_file = Path("logs/request_logs.jsonl")
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    handler = logging.FileHandler(str(log_file), encoding="utf-8")
                    handler.setFormatter(logging.Formatter("%(message)s"))
                    req_logger.addHandler(handler)
                
                req_logger.info(json.dumps(log_record))
            except Exception:
                pass


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

    app.add_middleware(RequestLoggingMiddleware)

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
