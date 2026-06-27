"""I4 — add_*_batch must not pad missing embeddings with a fixed-dim zero
vector; missing embeddings are computed from the document text instead."""
from __future__ import annotations


def test_add_semantic_batch_computes_missing_embeddings(storage, fake_embedder):
    storage.create_graph("batch_mix")
    nodes = [
        {"semantic_id": 0, "text": "has embedding",
         "embedding": fake_embedder.embed("has embedding")},
        {"semantic_id": 1, "text": "missing embedding"},  # None -> computed, not zeros
    ]
    storage.add_semantic_batch("batch_mix", nodes)

    data = storage.get_all_semantic("batch_mix")
    assert len(data["ids"]) == 2

    embs = data["embeddings"]
    # Dimension must match the model (fake_embedder.DIM == 64), never the old
    # hardcoded 4096; and the computed vector must not be all-zeros.
    for vec in embs:
        assert len(vec) == fake_embedder.DIM
    computed = embs[data["ids"].index("1")]
    assert any(abs(x) > 0 for x in computed)
