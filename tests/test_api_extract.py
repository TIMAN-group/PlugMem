"""Tests for the /api/v1/extract promotion-gate endpoint."""
from __future__ import annotations

import json

import pytest

from plugmem.api import dependencies as deps
from plugmem.inference import promotion as promo


class CannedLLM:
    """Returns a fixed completion regardless of input. Used to drive the
    extractor with a known JSON output."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list = []

    def complete(self, messages, temperature=0, top_p=1.0, max_tokens=4096):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.response


def _swap_llm(canned: CannedLLM):
    deps._llm_client = canned  # type: ignore[assignment]


def test_extract_no_candidates_returns_empty(client):
    resp = client.post("/api/v1/extract", json={"candidates": []})
    assert resp.status_code == 200
    assert resp.json() == {"memories": []}


def test_extract_returns_parsed_memories(client):
    canned = CannedLLM(json.dumps([
        {
            "type": "semantic",
            "semantic_memory": "Use httpx, not requests",
            "tags": ["python", "convention"],
            "source": "correction",
            "confidence": 0.9,
        },
        {
            "type": "procedural",
            "subgoal": "fix import error in tests",
            "procedural_memory": "pip install -e . then pytest",
            "source": "failure_delta",
            "confidence": 0.7,
        },
    ]))
    _swap_llm(canned)
    resp = client.post("/api/v1/extract", json={
        "candidates": [
            {"kind": "correction", "window": "user said: actually use httpx"},
            {"kind": "failure_delta", "window": "pytest failed; pip install fixed it"},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 2
    assert data["memories"][0]["type"] == "semantic"
    assert data["memories"][0]["source"] == "correction"
    assert data["memories"][1]["type"] == "procedural"


def test_extract_drops_invalid_memories(client):
    """Bad memories from the LLM are silently dropped, not propagated."""
    canned = CannedLLM(json.dumps([
        {  # missing source
            "type": "semantic",
            "semantic_memory": "x",
            "confidence": 0.5,
        },
        {  # confidence out of range
            "type": "semantic",
            "semantic_memory": "y",
            "source": "correction",
            "confidence": 99,
        },
        {  # valid
            "type": "semantic",
            "semantic_memory": "z",
            "source": "correction",
            "confidence": 0.6,
        },
    ]))
    _swap_llm(canned)
    resp = client.post("/api/v1/extract", json={
        "candidates": [{"kind": "correction", "window": "..."}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 1
    assert data["memories"][0]["semantic_memory"] == "z"


def test_extract_handles_unparseable_llm_output(client):
    canned = CannedLLM("this is not JSON at all")
    _swap_llm(canned)
    resp = client.post("/api/v1/extract", json={
        "candidates": [{"kind": "correction", "window": "..."}],
    })
    assert resp.status_code == 200
    assert resp.json() == {"memories": []}


def test_extract_strips_code_fences(client):
    canned = CannedLLM(
        "```json\n"
        + json.dumps([{
            "type": "semantic",
            "semantic_memory": "fenced",
            "source": "correction",
            "confidence": 0.7,
        }])
        + "\n```"
    )
    _swap_llm(canned)
    resp = client.post("/api/v1/extract", json={
        "candidates": [{"kind": "correction", "window": "..."}],
    })
    assert resp.status_code == 200
    assert len(resp.json()["memories"]) == 1


def test_extract_rejects_invalid_kind(client):
    resp = client.post("/api/v1/extract", json={
        "candidates": [{"kind": "made_up_kind", "window": "x"}],
    })
    assert resp.status_code == 422


# ------------------------------------------------------------------ #
# Direct unit tests on the extractor (no LLM round-trip)
# ------------------------------------------------------------------ #

def test_extractor_returns_empty_for_no_candidates():
    canned = CannedLLM("")
    assert promo.extract_coding_memories(canned, []) == []
    # Should not call the LLM when no candidates.
    assert canned.calls == []


def test_extractor_filters_invalid_source():
    canned = CannedLLM(json.dumps([
        {
            "type": "semantic",
            "semantic_memory": "x",
            "source": "not_a_real_source",
            "confidence": 0.7,
        },
    ]))
    out = promo.extract_coding_memories(canned, [{"kind": "correction", "window": "."}])
    assert out == []


def test_extractor_filters_missing_semantic_text():
    canned = CannedLLM(json.dumps([
        {
            "type": "semantic",
            "semantic_memory": "",
            "source": "correction",
            "confidence": 0.7,
        },
    ]))
    out = promo.extract_coding_memories(canned, [{"kind": "correction", "window": "."}])
    assert out == []


def test_extractor_filters_partial_procedural():
    canned = CannedLLM(json.dumps([
        {
            "type": "procedural",
            "subgoal": "x",
            # missing procedural_memory
            "source": "failure_delta",
            "confidence": 0.7,
        },
    ]))
    out = promo.extract_coding_memories(canned, [{"kind": "failure_delta", "window": "."}])
    assert out == []
