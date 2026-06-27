"""Unit tests for MemoryGraph timing and context logging."""
from __future__ import annotations

import pytest
from plugmem.api.logging_ctx import RequestContextLog, current_log_ctx
from plugmem.core.memory import Memory


def test_graph_retrieval_logging(graph_manager, fake_llm, fake_embedder):
    # Create and fetch a graph instance
    graph_manager.create_graph("test_retrieval_log_graph")
    mg = graph_manager.get_graph("test_retrieval_log_graph")

    # Insert a dummy structured memory
    mem = Memory(
        goal="Answer query",
        observation="Madrid is capital of Spain",
        llm=fake_llm,
        embedder=fake_embedder,
    )
    mem.memory["semantic"].append({
        "semantic_memory": "Madrid is capital of Spain.",
        "tags": ["Spain", "Madrid"],
        "trajectory_num": 0,
        "turn_num": 0,
        "time": "2026-06-04",
    })
    # Use fake_embedder to generate a matching embedding to prevent flaky random projection failures
    mem.memory_embedding["semantic"].append({
        "semantic_memory": fake_embedder.embed("Madrid is capital of Spain."),
        "tags": [[0.1] * 64, [0.1] * 64],
    })
    mg.insert(mem)

    # Initialize context
    log = RequestContextLog(task_id="retrieval-test-task")
    token = current_log_ctx.set(log)

    try:
        messages, variables, mode = mg.retrieve_memory(
            goal="Locate capital",
            observation="Madrid is capital of Spain.",
            task_type="qa",
        )

        # Check retrieval call is recorded
        assert len(log.retrieval_calls) == 1
        ret_call = log.retrieval_calls[0]
        assert ret_call["mode"] == "semantic_memory"
        assert isinstance(ret_call["latency_sec"], float)
        assert ret_call["latency_sec"] >= 0.0

        # Check retrieved memory was recorded
        assert log.memory_retrieved is not None
        assert "Madrid" in log.memory_retrieved

    finally:
        current_log_ctx.reset(token)


def test_graph_consolidation_logging(graph_manager):
    graph_manager.create_graph("test_consolidation_log_graph")
    mg = graph_manager.get_graph("test_consolidation_log_graph")

    # Initialize context
    log = RequestContextLog(task_id="consolidation-test-task")
    token = current_log_ctx.set(log)

    try:
        stats = mg.update_semantic_subgraph(
            merge_threshold=0.5
        )

        # Check consolidation stats are recorded
        assert len(log.consolidation_calls) == 1
        cons_call = log.consolidation_calls[0]
        assert isinstance(cons_call["latency_sec"], float)
        assert cons_call["latency_sec"] >= 0.0
        assert cons_call["stats"] == stats

    finally:
        current_log_ctx.reset(token)
