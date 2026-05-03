"""Tests for phase-tagged token usage logging."""
from __future__ import annotations

from plugmem.clients.llm import current_phase, with_phase


def test_default_phase():
    assert current_phase() == "default"


def test_with_phase_sets_phase():
    with with_phase("extract"):
        assert current_phase() == "extract"
    assert current_phase() == "default"


def test_nested_phases_restore_outer():
    with with_phase("retrieve"):
        assert current_phase() == "retrieve"
        with with_phase("reason"):
            assert current_phase() == "reason"
        assert current_phase() == "retrieve"
    assert current_phase() == "default"


def test_phase_appears_in_log_entry(tmp_path):
    """The LLM client logs the active phase from `with_phase` into each
    JSONL token-usage record."""
    import json

    from plugmem.clients.llm import OpenAICompatibleLLMClient

    log_path = tmp_path / "tokens.jsonl"

    class FakeUsage:
        def model_dump(self):
            return {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

    class FakeResponse:
        def __init__(self):
            self.usage = FakeUsage()
            self.choices = [type("Choice", (), {
                "message": type("M", (), {"content": "ok"})(),
            })()]

    # Build a real OpenAICompatibleLLMClient with the openai SDK call
    # stubbed out so we exercise _log_usage without a network round trip.
    fake_client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    fake_client.model = "fake"
    fake_client.max_retries = 1
    fake_client.retry_delay = 0.01
    fake_client.token_usage_file = str(log_path)
    fake_client._client = type("C", (), {
        "chat": type("Chat", (), {
            "completions": type("Comp", (), {
                "create": staticmethod(lambda **kw: FakeResponse()),
            })(),
        })(),
    })()

    messages = [{"role": "user", "content": "hi"}]

    # Default phase.
    fake_client.complete(messages=messages)
    # Tagged extract.
    with with_phase("extract"):
        fake_client.complete(messages=messages)
    # Tagged reason.
    with with_phase("reason"):
        fake_client.complete(messages=messages)

    assert log_path.exists()
    phases = [json.loads(line)["phase"] for line in log_path.read_text().strip().splitlines()]
    assert phases == ["default", "extract", "reason"]
