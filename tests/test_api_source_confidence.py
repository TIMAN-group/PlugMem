"""Tests for the source/confidence metadata added in Stage 2.

Covers insertion (round-trip through chroma), retrieval filtering, and
backwards compatibility for callers that don't supply the new fields.
"""
from __future__ import annotations


def _make_graph(client, graph_id):
    client.post("/api/v1/graphs", json={"graph_id": graph_id})


def test_insert_semantic_with_source_and_confidence(client):
    _make_graph(client, "src_sem")
    resp = client.post("/api/v1/graphs/src_sem/memories", json={
        "mode": "structured",
        "semantic": [
            {
                "semantic_memory": "Tests live in tests/",
                "tags": ["repo"],
                "source": "correction",
                "confidence": 0.9,
            },
        ],
    })
    assert resp.status_code == 200
    assert resp.json()["stats"]["semantic"] == 1


def test_insert_procedural_with_source_and_confidence(client):
    _make_graph(client, "src_proc")
    resp = client.post("/api/v1/graphs/src_proc/memories", json={
        "mode": "structured",
        "procedural": [
            {
                "subgoal": "run tests",
                "procedural_memory": "pytest -k auth",
                "source": "failure_delta",
                "confidence": 0.8,
            },
        ],
    })
    assert resp.status_code == 200
    assert resp.json()["stats"]["procedural"] == 1


def test_insert_rejects_invalid_source(client):
    _make_graph(client, "src_bad")
    resp = client.post("/api/v1/graphs/src_bad/memories", json={
        "mode": "structured",
        "semantic": [
            {
                "semantic_memory": "x",
                "source": "definitely-not-a-real-source",
            },
        ],
    })
    assert resp.status_code == 422


def test_insert_rejects_out_of_range_confidence(client):
    _make_graph(client, "conf_bad")
    resp = client.post("/api/v1/graphs/conf_bad/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "x", "confidence": 1.5},
        ],
    })
    assert resp.status_code == 422


def test_legacy_insert_without_metadata_still_works(client):
    """Pre-Stage-2 callers (chat plugin, OpenClaw) keep working unchanged."""
    _make_graph(client, "legacy")
    resp = client.post("/api/v1/graphs/legacy/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "old-style memory", "tags": []},
        ],
        "procedural": [
            {"subgoal": "do thing", "procedural_memory": "step 1"},
        ],
    })
    assert resp.status_code == 200
    stats = resp.json()["stats"]
    assert stats["semantic"] == 1
    assert stats["procedural"] == 1


def test_retrieve_filters_by_min_confidence(client):
    """A memory below the threshold should not surface."""
    _make_graph(client, "filt_conf")
    client.post("/api/v1/graphs/filt_conf/memories", json={
        "mode": "structured",
        "semantic": [
            {
                "semantic_memory": "high-confidence fact",
                "tags": ["x"],
                "confidence": 0.9,
            },
            {
                "semantic_memory": "low-confidence guess",
                "tags": ["x"],
                "confidence": 0.2,
            },
        ],
    })

    # Without filter: both rank (assuming the value-function admits them).
    resp_unfiltered = client.post(
        "/api/v1/graphs/filt_conf/retrieve",
        json={"observation": "x", "mode": "semantic_memory"},
    )
    assert resp_unfiltered.status_code == 200

    # With min_confidence=0.5: low-confidence should be excluded.
    resp_filtered = client.post(
        "/api/v1/graphs/filt_conf/retrieve",
        json={
            "observation": "x",
            "mode": "semantic_memory",
            "min_confidence": 0.5,
        },
    )
    assert resp_filtered.status_code == 200
    semantic_block = resp_filtered.json()["variables"]["semantic_memory"]
    assert "low-confidence guess" not in semantic_block


def test_retrieve_accepts_source_in_filter(client):
    """Smoke: the API accepts source_in and returns 200.

    Functional correctness of the filter is verified by the unit tests
    below — those don't depend on the random-embedding value function."""
    _make_graph(client, "filt_src")
    client.post("/api/v1/graphs/filt_src/memories", json={
        "mode": "structured",
        "semantic": [
            {
                "semantic_memory": "from a correction",
                "tags": ["x"],
                "source": "correction",
                "confidence": 0.9,
            },
        ],
    })
    resp = client.post(
        "/api/v1/graphs/filt_src/retrieve",
        json={
            "observation": "x",
            "mode": "semantic_memory",
            "source_in": ["correction"],
        },
    )
    assert resp.status_code == 200


def test_reason_accepts_filter_params(client):
    """reason() shares the request shape; smoke-test that it doesn't 422."""
    _make_graph(client, "reason_filt")
    client.post("/api/v1/graphs/reason_filt/memories", json={
        "mode": "structured",
        "semantic": [
            {
                "semantic_memory": "high-conf",
                "tags": ["x"],
                "source": "correction",
                "confidence": 0.9,
            },
        ],
    })
    resp = client.post("/api/v1/graphs/reason_filt/reason", json={
        "observation": "x",
        "mode": "semantic_memory",
        "min_confidence": 0.5,
        "source_in": ["correction"],
    })
    assert resp.status_code == 200


# ------------------------------------------------------------------ #
# Unit tests for _passes_metadata_filter — independent of value func
# ------------------------------------------------------------------ #

class _FakeNode:
    def __init__(self, source=None, confidence=0.5):
        self.source = source
        self.confidence = confidence


def test_filter_no_constraints_passes_everything():
    from plugmem.core.memory_graph import _passes_metadata_filter
    assert _passes_metadata_filter(_FakeNode(source=None, confidence=0.0), None, None)
    assert _passes_metadata_filter(_FakeNode(source="explicit", confidence=1.0), None, None)


def test_filter_min_confidence_excludes_below():
    from plugmem.core.memory_graph import _passes_metadata_filter
    low = _FakeNode(confidence=0.2)
    high = _FakeNode(confidence=0.9)
    assert not _passes_metadata_filter(low, 0.5, None)
    assert _passes_metadata_filter(high, 0.5, None)
    # Edge: equal-to-threshold passes.
    assert _passes_metadata_filter(_FakeNode(confidence=0.5), 0.5, None)


def test_filter_source_in_excludes_other_sources():
    from plugmem.core.memory_graph import _passes_metadata_filter
    correction = _FakeNode(source="correction")
    explicit = _FakeNode(source="explicit")
    legacy = _FakeNode(source=None)
    assert _passes_metadata_filter(correction, None, ["correction"])
    assert not _passes_metadata_filter(explicit, None, ["correction"])
    assert not _passes_metadata_filter(legacy, None, ["correction"])


def test_filter_combined_constraints_must_both_pass():
    from plugmem.core.memory_graph import _passes_metadata_filter
    # Right source, low confidence.
    low_conf = _FakeNode(source="correction", confidence=0.2)
    # Wrong source, high confidence.
    wrong_src = _FakeNode(source="explicit", confidence=0.9)
    # Both right.
    both = _FakeNode(source="correction", confidence=0.9)
    assert not _passes_metadata_filter(low_conf, 0.5, ["correction"])
    assert not _passes_metadata_filter(wrong_src, 0.5, ["correction"])
    assert _passes_metadata_filter(both, 0.5, ["correction"])


def test_filter_default_confidence_when_attr_missing():
    """A node missing the confidence attribute is treated as 0.5."""
    from plugmem.core.memory_graph import _passes_metadata_filter

    class Bare:
        pass

    n = Bare()
    # Defaults to 0.5 — passes a 0.5 threshold but not 0.6.
    assert _passes_metadata_filter(n, 0.5, None)
    assert not _passes_metadata_filter(n, 0.6, None)
