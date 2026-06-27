"""I6 — create_graph must be a cheap no-op when the graph is already loaded
in-process (no full graph.load() on every PlugMemClient re-ensure)."""
from __future__ import annotations


def test_create_graph_skips_reload_when_cached(graph_manager, monkeypatch):
    gid = graph_manager.create_graph("cached_create")
    mg = graph_manager.get_graph(gid)

    calls = {"n": 0}
    monkeypatch.setattr(mg, "load", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    # Re-creating an already-loaded graph must not trigger a reload...
    graph_manager.create_graph("cached_create")
    assert calls["n"] == 0
    # ...and must hand back the same cached instance.
    assert graph_manager.get_graph("cached_create") is mg
