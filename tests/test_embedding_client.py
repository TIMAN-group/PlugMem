"""I3 — HTTPEmbeddingClient.embed_batch must target a single split URL,
not the raw comma-joined base_url string."""
from __future__ import annotations

from plugmem.clients.embedding import HTTPEmbeddingClient


class _FakeResp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}


def test_embed_batch_uses_split_url_not_comma(monkeypatch):
    client = HTTPEmbeddingClient(base_url="http://a/emb,http://b/emb")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        return _FakeResp()

    monkeypatch.setattr("plugmem.clients.embedding.requests.post", fake_post)

    out = client.embed_batch(["x"])
    assert out == [[0.1, 0.2]]
    assert captured["url"] in ("http://a/emb", "http://b/emb")
    assert "," not in captured["url"]
