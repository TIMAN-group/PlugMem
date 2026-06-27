"""Memory-type (mode) selection: PlugMem decides by default, requests may override.

Covers the contract realignment:
  * `_normalize_mode` strips planner markdown and validates against known types.
  * Omitting mode runs the server's planner (get_mode) — PlugMem decides.
  * An explicit mode skips type-selection but still runs the planner (decoupled).
"""
from __future__ import annotations

import pytest

from plugmem.core.memory import Memory
from plugmem.core.memory_graph import _VALID_MODES, _normalize_mode


# ── _normalize_mode ──────────────────────────────────────────────────

@pytest.mark.parametrize("value", _VALID_MODES)
def test_normalize_mode_passes_known_values(value):
    assert _normalize_mode(value) == value


def test_normalize_mode_strips_markdown():
    assert _normalize_mode("**semantic_memory**") == "semantic_memory"
    assert _normalize_mode("### procedural_memory") == "procedural_memory"
    assert _normalize_mode("  episodic_memory  ") == "episodic_memory"


def test_normalize_mode_falls_back_on_unknown():
    assert _normalize_mode("mixed_memory") == "semantic_memory"
    assert _normalize_mode("") == "semantic_memory"
    assert _normalize_mode(None) == "semantic_memory"


def test_normalize_mode_custom_default():
    assert _normalize_mode("nonsense", default="procedural_memory") == "procedural_memory"


# ── type-selection vs. planning ──────────────────────────────────────

def _seed(graph_manager, fake_llm, fake_embedder, graph_id):
    graph_manager.create_graph(graph_id)
    mg = graph_manager.get_graph(graph_id)
    mem = Memory(
        goal="Answer query",
        observation="Madrid is the capital of Spain",
        llm=fake_llm,
        embedder=fake_embedder,
    )
    mem.memory["semantic"].append({
        "semantic_memory": "Madrid is the capital of Spain.",
        "tags": ["Spain", "Madrid"],
        "trajectory_num": 0,
        "turn_num": 0,
        "time": "",
    })
    mem.memory_embedding["semantic"].append({
        "semantic_memory": fake_embedder.embed("Madrid is the capital of Spain."),
        "tags": [fake_embedder.embed("Spain"), fake_embedder.embed("Madrid")],
    })
    mg.insert(mem)
    return mg


def test_omitting_mode_lets_plugmem_decide(graph_manager, fake_llm, fake_embedder):
    """mode=None → server runs get_mode (decide) AND get_plan (plan): 2 LLM calls."""
    mg = _seed(graph_manager, fake_llm, fake_embedder, "decide_default")
    fake_llm.calls.clear()

    _, _, mode = mg.retrieve_memory(observation="capital of Spain", task_type="qa")

    assert mode in _VALID_MODES
    assert len(fake_llm.calls) == 2  # get_mode + get_plan


def test_explicit_mode_skips_decision_but_still_plans(graph_manager, fake_llm, fake_embedder):
    """Explicit mode → get_mode skipped, but get_plan still runs (decoupled): 1 LLM call."""
    mg = _seed(graph_manager, fake_llm, fake_embedder, "explicit_plan")
    fake_llm.calls.clear()

    _, _, mode = mg.retrieve_memory(observation="capital of Spain", mode="semantic_memory")

    assert mode == "semantic_memory"
    assert len(fake_llm.calls) == 1  # only get_plan; type selection was overridden
