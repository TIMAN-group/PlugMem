"""Tests for LLMRouter — multi-LLM configuration."""
import os
import tempfile

import pytest

from plugmem.clients.llm_router import LLMRouter


def _write_yaml(content: str) -> str:
    """Write YAML to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


class FakeClient:
    """Tracks which model was used."""

    def __init__(self, model: str):
        self.model = model

    def complete(self, messages, **kw):
        return f"response from {self.model}"


# ------------------------------------------------------------------ #
# Unit tests
# ------------------------------------------------------------------ #

def test_from_single_client():
    client = FakeClient("gpt-4o")
    router = LLMRouter.from_single_client(client)

    assert router.for_role("default") is client
    assert router.structuring is client
    assert router.retrieval is client
    assert router.reasoning is client
    assert router.consolidation is client
    assert router.complete([]) == "response from gpt-4o"


def test_explicit_clients():
    default = FakeClient("default-model")
    structuring = FakeClient("struct-model")
    reasoning = FakeClient("reason-model")

    router = LLMRouter({
        "default": default,
        "structuring": structuring,
        "reasoning": reasoning,
    })

    assert router.structuring is structuring
    assert router.reasoning is reasoning
    # retrieval and consolidation fall back to default
    assert router.retrieval is default
    assert router.consolidation is default
    # complete() delegates to default
    assert router.complete([]) == "response from default-model"


def test_requires_default():
    with pytest.raises(ValueError, match="default"):
        LLMRouter({"structuring": FakeClient("x")})


# ------------------------------------------------------------------ #
# YAML loading
# ------------------------------------------------------------------ #

def test_from_yaml_all_roles():
    path = _write_yaml("""
default:
  base_url: "http://localhost:8000/v1"
  api_key: "key-default"
  model: "default-model"

structuring:
  model: "struct-model"

retrieval:
  model: "retrieval-model"

reasoning:
  base_url: "http://other:9000/v1"
  api_key: "key-reason"
  model: "reason-model"

consolidation:
  model: "consol-model"
""")
    try:
        router = LLMRouter.from_yaml(path)

        assert router.for_role("default").model == "default-model"
        assert router.structuring.model == "struct-model"
        assert router.retrieval.model == "retrieval-model"
        assert router.reasoning.model == "reason-model"
        assert router.consolidation.model == "consol-model"

        # structuring inherits base_url from default (OpenAI SDK normalises with trailing /)
        assert "localhost:8000" in str(router.structuring._client.base_url)
        # reasoning overrides base_url
        assert "other:9000" in str(router.reasoning._client.base_url)
    finally:
        os.unlink(path)


def test_from_yaml_default_only():
    path = _write_yaml("""
default:
  base_url: "http://localhost:8000/v1"
  api_key: "key"
  model: "the-model"
""")
    try:
        router = LLMRouter.from_yaml(path)
        # All roles fall back to default
        assert router.structuring.model == "the-model"
        assert router.retrieval.model == "the-model"
        assert router.reasoning.model == "the-model"
        assert router.consolidation.model == "the-model"
    finally:
        os.unlink(path)


def test_from_yaml_env_expansion(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret-key-123")
    monkeypatch.setenv("TEST_LLM_MODEL", "my-model")

    path = _write_yaml("""
default:
  base_url: "http://localhost/v1"
  api_key: "${TEST_LLM_KEY}"
  model: "${TEST_LLM_MODEL}"
""")
    try:
        router = LLMRouter.from_yaml(path)
        # Env vars should be expanded
        assert router.for_role("default").model == "my-model"
    finally:
        os.unlink(path)


def test_from_yaml_empty_raises():
    path = _write_yaml("")
    try:
        with pytest.raises(ValueError, match="Empty"):
            LLMRouter.from_yaml(path)
    finally:
        os.unlink(path)


def test_from_yaml_no_default_raises():
    path = _write_yaml("""
structuring:
  model: "x"
""")
    try:
        with pytest.raises(ValueError, match="default"):
            LLMRouter.from_yaml(path)
    finally:
        os.unlink(path)
