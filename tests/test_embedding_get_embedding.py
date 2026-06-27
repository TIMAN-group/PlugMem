"""I1 — get_embedding must honor EMBEDDING_BASE_URL (not hardcoded ports)."""
from __future__ import annotations

import os
import sys

import pytest

# src/utils.py is eval-side, not part of the plugmem package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
utils = pytest.importorskip("utils")


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_get_embedding_posts_to_env_base_url(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://my-emb:9000/v1/embeddings")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        return _FakeResp({"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr(utils.requests, "post", fake_post)

    out = utils.get_embedding("hello")
    assert out == [0.1, 0.2, 0.3]
    assert captured["url"] == "http://my-emb:9000/v1/embeddings"


def test_resolve_base_urls_splits_comma(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://a/emb, http://b/emb ")
    assert utils._resolve_embedding_base_urls() == ["http://a/emb", "http://b/emb"]


def test_resolve_base_urls_dev_fallback_when_unset(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    urls = utils._resolve_embedding_base_urls()
    assert urls == [
        "http://localhost:8555/v1/embeddings",
        "http://localhost:8556/v1/embeddings",
    ]
