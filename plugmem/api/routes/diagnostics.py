"""Diagnostics endpoints — logs, token metrics, and storage footprint."""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from plugmem.api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"], dependencies=[Depends(require_api_key)])

LOG_FILE_PATH = Path("logs/request_logs.jsonl")

@router.get("/logs")
async def get_logs(limit: int = 200, graph_id: str | None = None) -> List[Dict[str, Any]]:
    """Return the last `limit` request logs from `request_logs.jsonl`, optionally filtered by graph_id."""
    if not LOG_FILE_PATH.exists():
        return []

    read_limit = 2000 if graph_id else limit
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            lines = deque(f, maxlen=read_limit)
    except Exception as e:
        logger.error(f"Failed to read log file: {e}")
        return []

    logs = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            endpoint = record.get("endpoint", "")
            
            # Filter out old diagnostics endpoints from rendering in visualizer
            if "/diagnostics" in endpoint:
                continue
                
            # Filter by graph_id if provided
            if graph_id:
                pattern = f"/graphs/{graph_id}"
                if pattern not in endpoint:
                    continue
                    
            logs.append(record)
            if len(logs) >= limit:
                break
        except json.JSONDecodeError:
            continue
    return logs


@router.get("/stats")
async def get_stats(graph_id: str | None = None) -> Dict[str, Any]:
    """Calculate aggregate stats over the entire log file history, optionally filtered by graph_id."""
    if not LOG_FILE_PATH.exists():
        return {
            "total_llm_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "avg_latency_sec": 0.0,
            "latest_ram_usage_mb": 0.0,
            "latest_db_disk_usage_mb": 0.0,
        }

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_calls = 0
    latest_ram = 0.0
    latest_db = 0.0

    # For average latency, we average over the last 100 LLM calls
    # to measure model speed rather than database-only API requests.
    llm_latencies = deque(maxlen=100)

    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    endpoint = record.get("endpoint", "")
                    
                    # Filter by graph_id if provided
                    if graph_id:
                        pattern = f"/graphs/{graph_id}"
                        if pattern not in endpoint:
                            continue

                    p_tok = record.get("total_prompt_tokens", 0) or 0
                    c_tok = record.get("total_completion_tokens", 0) or 0
                    total_prompt_tokens += p_tok
                    total_completion_tokens += c_tok
                    
                    llm_calls = record.get("llm_calls") or []
                    total_llm_calls += len(llm_calls)

                    for call in llm_calls:
                        lat = call.get("latency_sec")
                        if lat is not None:
                            llm_latencies.append(lat)

                    latest_ram = record.get("ram_usage_mb", 0.0) or 0.0
                    latest_db = record.get("db_disk_usage_mb", 0.0) or 0.0
                except (json.JSONDecodeError, TypeError):
                    continue
    except Exception as e:
        logger.error(f"Failed to read log file for stats: {e}")
        return {}

    avg_latency = (sum(llm_latencies) / len(llm_latencies)) if llm_latencies else 0.0

    return {
        "total_llm_calls": total_llm_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "avg_latency_sec": avg_latency,
        "latest_ram_usage_mb": latest_ram,
        "latest_db_disk_usage_mb": latest_db,
    }
