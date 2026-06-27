"""Integration tests for the request logging middleware."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

LOG_FILE = Path("logs/request_logs.jsonl")


def _seed_graph(client, graph_id="mw_test"):
    """Create a graph and insert some semantic memories."""
    client.post("/api/v1/graphs", json={"graph_id": graph_id})
    client.post(f"/api/v1/graphs/{graph_id}/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "Madrid is the capital of Spain.", "tags": ["Spain", "Madrid"]},
        ],
    })


def test_middleware_logging_reason(client):
    _seed_graph(client, "mw_reason")

    # Lower the TagRelevant/SemanticRelevant thresholds on the graph instance
    # so we get matches in mock tests.
    from plugmem.api.dependencies import get_graph_manager
    gm = get_graph_manager()
    graph = gm.get_graph("mw_reason")
    graph.tag_relevant.value_threshold = -1.0
    graph.semantic_relevant.value_threshold = -1.0

    # Override the LLM client to record LLM calls
    from tests.conftest import FakeLLM
    class LoggingFakeLLM(FakeLLM):
        def complete(self, messages, **kwargs):
            from plugmem.api.logging_ctx import current_log_ctx
            ctx = current_log_ctx.get()
            if ctx is not None:
                ctx.record_llm_call(
                    model="logging-fake-llm",
                    prompt_tokens=100,
                    completion_tokens=50,
                    latency_sec=0.05,
                )
            return super().complete(messages, **kwargs)

    graph.llm = LoggingFakeLLM()

    task_id = "test-task-12345"
    response = client.post(
        "/api/v1/graphs/mw_reason/reason",
        json={
            "observation": "What is the capital of Spain?",
            "mode": "semantic_memory",
        },
        headers={"X-Task-ID": task_id}
    )
    assert response.status_code == 200

    # Ensure log file exists
    assert LOG_FILE.exists(), "Log file was not created"

    # Read the log entries
    lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    log_data = None
    for line in reversed(lines):
        if not line.strip():
            continue
        data = json.loads(line)
        if data.get("task_id") == task_id:
            log_data = data
            break

    assert log_data is not None, f"Log entry with task_id {task_id} not found"

    # Validate schema and fields
    assert log_data["endpoint"] == "/api/v1/graphs/mw_reason/reason"
    assert log_data["method"] == "POST"
    assert log_data["status_code"] == 200
    assert isinstance(log_data["latency_sec"], float)
    assert log_data["latency_sec"] >= 0.0

    # Check RAM/Disk usage
    assert isinstance(log_data["ram_usage_mb"], float)
    assert isinstance(log_data["db_disk_usage_mb"], float)

    # Check retrieval logs
    assert len(log_data["retrieval_calls"]) == 1
    assert log_data["retrieval_calls"][0]["mode"] == "semantic_memory"

    # Check LLM complete call was logged
    assert len(log_data["llm_calls"]) >= 1
    # Check total tokens are aggregated
    assert log_data["total_prompt_tokens"] >= 0
    assert log_data["total_completion_tokens"] >= 0

    # Verify agent output was captured
    assert log_data["agent_output"] == "test response"


def test_middleware_logging_retrieve_auto_task_id(client):
    _seed_graph(client, "mw_retrieve")

    # Lower the TagRelevant/SemanticRelevant thresholds on the graph instance
    from plugmem.api.dependencies import get_graph_manager
    gm = get_graph_manager()
    graph = gm.get_graph("mw_retrieve")
    graph.tag_relevant.value_threshold = -1.0
    graph.semantic_relevant.value_threshold = -1.0

    response = client.post(
        "/api/v1/graphs/mw_retrieve/retrieve",
        json={
            "observation": "What is the capital of Spain?",
            "mode": "semantic_memory",
        }
    )
    assert response.status_code == 200

    # Read the log entry (the last line is the retrieve call we just made)
    lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    log_data = json.loads(non_empty_lines[-1])

    # Verify auto-generated Task ID (UUID prefix "task_")
    assert log_data["task_id"].startswith("task_")
    assert log_data["endpoint"] == "/api/v1/graphs/mw_retrieve/retrieve"
    assert log_data["agent_output"] is None  # retrieve endpoint has no agent output
    assert log_data["memory_retrieved"] is not None
