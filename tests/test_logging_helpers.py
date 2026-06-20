"""Unit tests for request logging context and resource monitoring helpers."""
from __future__ import annotations

import os
import shutil
import tempfile
from plugmem.api.logging_ctx import (
    RequestContextLog,
    current_log_ctx,
    get_process_memory_mb,
    get_directory_size_mb,
)


def test_request_context_log():
    log = RequestContextLog(task_id="test-task-id")
    assert log.task_id == "test-task-id"
    assert log.start_time > 0
    assert len(log.llm_calls) == 0
    assert len(log.retrieval_calls) == 0
    assert log.memory_retrieved is None
    assert log.agent_output is None

    # Record LLM call
    log.record_llm_call(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        latency_sec=1.5,
    )
    assert len(log.llm_calls) == 1
    assert log.llm_calls[0]["model"] == "gpt-4o"
    assert log.llm_calls[0]["prompt_tokens"] == 100
    assert log.llm_calls[0]["completion_tokens"] == 50
    assert log.llm_calls[0]["latency_sec"] == 1.5

    # Record Retrieval call
    log.record_retrieval(mode="semantic_memory", latency_sec=0.2, retrieved_mem="fact 1")
    assert len(log.retrieval_calls) == 1
    assert log.retrieval_calls[0]["mode"] == "semantic_memory"
    assert log.retrieval_calls[0]["latency_sec"] == 0.2
    assert log.memory_retrieved == "fact 1"


def test_current_log_ctx_var():
    assert current_log_ctx.get() is None

    log = RequestContextLog(task_id="another-task-id")
    token = current_log_ctx.set(log)
    try:
        assert current_log_ctx.get() is log
        assert current_log_ctx.get().task_id == "another-task-id"
    finally:
        current_log_ctx.reset(token)

    assert current_log_ctx.get() is None


def test_get_process_memory_mb():
    mem = get_process_memory_mb()
    assert isinstance(mem, float)
    assert mem >= 0.0


def test_get_directory_size_mb():
    # Size of non-existent directory
    assert get_directory_size_mb("non_existent_directory_abc_123") == 0.0

    # Size of a directory with files
    temp_dir = tempfile.mkdtemp()
    try:
        test_file = os.path.join(temp_dir, "test.txt")
        # Write 1024 bytes (1 KB)
        with open(test_file, "wb") as f:
            f.write(b"x" * 1024)

        size_mb = get_directory_size_mb(temp_dir)
        # 1 KB = 1 / 1024 MB = 0.0009765625 MB
        assert size_mb > 0.0
        assert abs(size_mb - (1.0 / 1024.0)) < 1e-5
    finally:
        shutil.rmtree(temp_dir)
