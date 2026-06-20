"""Request-scoped logging context and resource monitoring helpers."""
from __future__ import annotations

import contextvars
import os
import time
from typing import Any, Dict, List, Optional


class RequestContextLog:
    """Aggregates all execution traces and resource usages for a single API request."""

    def __init__(self, task_id: str):
        self.task_id: str = task_id
        self.start_time: float = time.perf_counter()
        self.llm_calls: List[Dict[str, Any]] = []
        self.retrieval_calls: List[Dict[str, Any]] = []
        self.consolidation_calls: List[Dict[str, Any]] = []
        self.memory_retrieved: Optional[str] = None
        self.agent_output: Optional[str] = None

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_sec: float,
    ) -> None:
        """Add LLM usage statistics to the request context."""
        self.llm_calls.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_sec": latency_sec,
        })

    def record_retrieval(self, mode: str, latency_sec: float, retrieved_mem: str) -> None:
        """Add memory retrieval stats to the request context."""
        self.retrieval_calls.append({
            "mode": mode,
            "latency_sec": latency_sec,
        })
        self.memory_retrieved = retrieved_mem

    def record_consolidation(self, latency_sec: float, stats: Dict[str, int]) -> None:
        """Add consolidation stats to the request context."""
        self.consolidation_calls.append({
            "latency_sec": latency_sec,
            "stats": stats,
        })


current_log_ctx: contextvars.ContextVar[Optional[RequestContextLog]] = contextvars.ContextVar(
    "current_log_ctx", default=None
)


def get_process_memory_mb() -> float:
    """Estimate memory (RAM RSS) footprint of the current python process in MB.

    Uses `psutil` if installed, with a standard-library fallback on Unix systems.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss) / (1024 * 1024)
    except ImportError:
        try:
            import resource
            # On Linux maxrss is in kilobytes, on macOS it is in bytes.
            # Convert to MB.
            import sys
            divisor = 1024.0 if sys.platform != "darwin" else (1024.0 * 1024.0)
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / divisor
        except (ImportError, AttributeError):
            return 0.0


def get_directory_size_mb(path: str) -> float:
    """Calculate the total size in MB occupied by files in a directory."""
    if not os.path.isdir(path):
        return 0.0
    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Skip symlinks to avoid infinite loops/double counting
                if os.path.exists(fp) and not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except Exception:
        pass
    return float(total_size) / (1024 * 1024)
