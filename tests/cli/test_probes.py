"""Tests for plugmem.cli.wizard.probes — Ollama detect + LLM/embed probes."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from plugmem.cli.wizard.probes import (
    OllamaInfo,
    detect_ollama,
    probe_embedding,
    probe_llm,
)


# ── detect_ollama ───────────────────────────────────────────────────


def _ollama_response(models: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"models": models}
    return resp


def test_detect_ollama_success() -> None:
    with patch("requests.get", return_value=_ollama_response([
        {"name": "qwen2.5:14b"},
        {"name": "nomic-embed-text"},
    ])):
        info = detect_ollama()
    assert info is not None
    assert info.base_url == "http://127.0.0.1:11434/v1"
    assert info.models == ["nomic-embed-text", "qwen2.5:14b"]  # sorted


def test_detect_ollama_returns_none_on_connection_error() -> None:
    with patch("requests.get", side_effect=requests.ConnectionError()):
        assert detect_ollama() is None


def test_detect_ollama_returns_none_on_non_200() -> None:
    bad = MagicMock(status_code=500)
    with patch("requests.get", return_value=bad):
        assert detect_ollama() is None


def test_detect_ollama_returns_none_on_unparseable_json() -> None:
    bad = MagicMock(status_code=200)
    bad.json.side_effect = ValueError("not json")
    with patch("requests.get", return_value=bad):
        assert detect_ollama() is None


def test_detect_ollama_handles_empty_models_list() -> None:
    with patch("requests.get", return_value=_ollama_response([])):
        info = detect_ollama()
    assert info is not None
    assert info.models == []


# ── probe_llm ───────────────────────────────────────────────────────


def _llm_ok_response() -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    return resp


def test_probe_llm_success() -> None:
    with patch("requests.post", return_value=_llm_ok_response()):
        ok, msg = probe_llm("http://x", "k", "m")
    assert ok
    assert msg == "OK"


def test_probe_llm_failure_returns_status_code() -> None:
    bad = MagicMock(status_code=401, text="invalid api key")
    with patch("requests.post", return_value=bad):
        ok, msg = probe_llm("http://x", "k", "m")
    assert not ok
    assert "401" in msg
    assert "invalid api key" in msg


def test_probe_llm_handles_timeout() -> None:
    with patch("requests.post", side_effect=requests.Timeout()):
        ok, msg = probe_llm("http://x", "k", "m", timeout=1)
    assert not ok
    assert "Timed out" in msg


def test_probe_llm_handles_connection_error() -> None:
    with patch("requests.post", side_effect=requests.ConnectionError("refused")):
        ok, msg = probe_llm("http://x", "k", "m")
    assert not ok
    assert "Connection error" in msg


def test_probe_llm_handles_200_with_bad_shape() -> None:
    bad = MagicMock(status_code=200)
    bad.json.return_value = {"unexpected": "shape"}
    with patch("requests.post", return_value=bad):
        ok, msg = probe_llm("http://x", "k", "m")
    assert not ok
    assert "unexpected" in msg.lower()


def test_probe_llm_strips_trailing_slash_from_base_url() -> None:
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        return _llm_ok_response()

    with patch("requests.post", side_effect=fake_post):
        probe_llm("http://x/", "k", "m")
    assert captured["url"] == "http://x/chat/completions"


def test_probe_llm_omits_auth_header_when_no_key() -> None:
    captured = {}

    def fake_post(url, **kw):
        captured["headers"] = kw.get("headers", {})
        return _llm_ok_response()

    with patch("requests.post", side_effect=fake_post):
        probe_llm("http://ollama/v1", "", "qwen2.5:14b")
    assert "Authorization" not in captured["headers"]


# ── probe_embedding ─────────────────────────────────────────────────


def _embed_ok_response() -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    return resp


def test_probe_embedding_success() -> None:
    with patch("requests.post", return_value=_embed_ok_response()):
        ok, msg = probe_embedding("http://x", "", "m")
    assert ok


def test_probe_embedding_failure_on_empty_data() -> None:
    bad = MagicMock(status_code=200)
    bad.json.return_value = {"data": []}
    with patch("requests.post", return_value=bad):
        ok, msg = probe_embedding("http://x", "", "m")
    assert not ok


def test_probe_embedding_handles_500() -> None:
    bad = MagicMock(status_code=500, text="internal error")
    with patch("requests.post", return_value=bad):
        ok, msg = probe_embedding("http://x", "", "m")
    assert not ok
    assert "500" in msg
