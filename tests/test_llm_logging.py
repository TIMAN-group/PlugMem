"""Unit tests for LLM client logging hooks."""
from __future__ import annotations

from unittest.mock import MagicMock
from plugmem.api.logging_ctx import RequestContextLog, current_log_ctx
from plugmem.clients.llm import OpenAICompatibleLLMClient


def test_llm_logging_hook():
    # Setup mock response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "  test response content  "
    mock_response.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 120
    mock_usage.completion_tokens = 60
    # Mock model_dump to avoid issues inside _log_usage helper
    mock_usage.model_dump.return_value = {
        "prompt_tokens": 120,
        "completion_tokens": 60,
    }
    mock_response.usage = mock_usage

    # Create client and override internal openai completions client
    client = OpenAICompatibleLLMClient(
        base_url="http://fake-endpoint",
        api_key="fake-key",
        model="test-coder-model",
    )
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = mock_response

    # Initialize request-scoped log context
    log = RequestContextLog(task_id="llm-trace-task")
    token = current_log_ctx.set(log)

    try:
        content = client.complete(messages=[{"role": "user", "content": "hello"}])
        assert content == "test response content"

        # Check that metrics were correctly logged in context
        assert len(log.llm_calls) == 1
        llm_record = log.llm_calls[0]
        assert llm_record["model"] == "test-coder-model"
        assert llm_record["prompt_tokens"] == 120
        assert llm_record["completion_tokens"] == 60
        assert isinstance(llm_record["latency_sec"], float)
        assert llm_record["latency_sec"] >= 0.0

    finally:
        # Cleanup context
        current_log_ctx.reset(token)
