"""Tests for retrieve, reason, and consolidate endpoints."""


def _seed_graph(client, graph_id="ret_test"):
    """Create a graph and insert some semantic memories."""
    client.post("/api/v1/graphs", json={"graph_id": graph_id})
    client.post(f"/api/v1/graphs/{graph_id}/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "Water boils at 100 degrees Celsius", "tags": ["physics", "water"]},
            {"semantic_memory": "The Earth orbits the Sun", "tags": ["astronomy"]},
        ],
    })


def test_retrieve(client):
    _seed_graph(client)
    resp = client.post("/api/v1/graphs/ret_test/retrieve", json={
        "observation": "What temperature does water boil?",
        "mode": "semantic_memory",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "semantic_memory"
    assert isinstance(data["reasoning_prompt"], list)


def test_reason(client):
    _seed_graph(client, "reason_test")
    resp = client.post("/api/v1/graphs/reason_test/reason", json={
        "observation": "What temperature does water boil?",
        "mode": "semantic_memory",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "semantic_memory"
    assert isinstance(data["reasoning"], str)
    assert len(data["reasoning"]) > 0


def test_retrieve_not_found(client):
    resp = client.post("/api/v1/graphs/nonexistent/retrieve", json={
        "observation": "test",
    })
    assert resp.status_code == 404


def test_consolidate(client):
    _seed_graph(client, "consol_test")
    resp = client.post("/api/v1/graphs/consol_test/consolidate", json={
        "merge_threshold": 0.5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "merged_pairs" in data["stats"]


# ── Recall audit log ──


def test_retrieve_writes_audit_row(client):
    _seed_graph(client, "audit_test")
    resp = client.post("/api/v1/graphs/audit_test/retrieve", json={
        "observation": "boiling point",
        "mode": "semantic_memory",
        "session_id": "run-1",
    })
    assert resp.status_code == 200

    resp = client.get("/api/v1/graphs/audit_test/recalls")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    row = data["recalls"][0]
    assert row["endpoint"] == "retrieve"
    assert row["session_id"] == "run-1"
    assert row["observation"] == "boiling point"
    assert row["mode"] == "semantic_memory"
    assert row["n_messages"] > 0


def test_recalls_filter_by_session_id(client):
    _seed_graph(client, "audit_filter")
    # Two recalls: one with session, one without
    client.post("/api/v1/graphs/audit_filter/retrieve", json={
        "observation": "q1", "mode": "semantic_memory", "session_id": "run-A",
    })
    client.post("/api/v1/graphs/audit_filter/retrieve", json={
        "observation": "q2", "mode": "semantic_memory", "session_id": "run-B",
    })
    client.post("/api/v1/graphs/audit_filter/retrieve", json={
        "observation": "q3", "mode": "semantic_memory",
    })

    resp = client.get("/api/v1/graphs/audit_filter/recalls?session_id=run-A")
    assert resp.status_code == 200
    rows = resp.json()["recalls"]
    assert len(rows) == 1
    assert rows[0]["session_id"] == "run-A"
    assert rows[0]["observation"] == "q1"

    # No filter — all three
    resp = client.get("/api/v1/graphs/audit_filter/recalls")
    assert resp.json()["count"] == 3


def test_sessions_endpoint_lists_distinct_session_ids(client):
    _seed_graph(client, "sess_list")
    # Insert with session_id
    client.post("/api/v1/graphs/sess_list/memories", json={
        "mode": "structured",
        "session_id": "run-X",
        "semantic": [{"semantic_memory": "fact for X", "tags": []}],
    })
    # Recall under a different session_id
    client.post("/api/v1/graphs/sess_list/retrieve", json={
        "observation": "anything", "mode": "semantic_memory", "session_id": "run-Y",
    })

    resp = client.get("/api/v1/graphs/sess_list/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert "run-X" in sessions
    assert "run-Y" in sessions


def test_session_timeline_merges_inserts_and_recalls(client):
    """Timeline returns inserts + recalls for a session, sorted by time."""
    client.post("/api/v1/graphs", json={"graph_id": "tl_test"})
    # Insert 2 semantics under run-A
    client.post("/api/v1/graphs/tl_test/memories", json={
        "mode": "structured",
        "session_id": "run-A",
        "semantic": [
            {"semantic_memory": "first fact", "tags": ["t1"]},
            {"semantic_memory": "second fact", "tags": ["t1"]},
        ],
    })
    # Run a recall under the same session
    client.post("/api/v1/graphs/tl_test/retrieve", json={
        "observation": "tell me about facts",
        "mode": "semantic_memory",
        "session_id": "run-A",
    })
    # Insert one node under a different session — must NOT appear in run-A timeline
    client.post("/api/v1/graphs/tl_test/memories", json={
        "mode": "structured",
        "session_id": "run-B",
        "semantic": [{"semantic_memory": "other session", "tags": []}],
    })

    resp = client.get("/api/v1/graphs/tl_test/sessions/run-A")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "run-A"
    assert data["count"] == 3, f"expected 2 inserts + 1 recall, got {data['count']}"

    kinds = [e["kind"] for e in data["events"]]
    assert kinds.count("insert") == 2
    assert kinds.count("recall") == 1

    # Recall must reference the inserts via selected_semantic_ids
    recall = next(e for e in data["events"] if e["kind"] == "recall")
    assert recall["endpoint"] == "retrieve"
    assert recall["mode"] == "semantic_memory"

    # No leakage from run-B
    texts = [e.get("text", "") for e in data["events"] if e["kind"] == "insert"]
    assert not any("other session" in t for t in texts)


def test_session_timeline_sorted_inserts_before_recalls_at_same_time(client):
    """An insert at time t and a recall at the same graph_time should
    sort insert-first so the timeline reads chronologically."""
    client.post("/api/v1/graphs", json={"graph_id": "tl_order"})
    client.post("/api/v1/graphs/tl_order/memories", json={
        "mode": "structured",
        "session_id": "run-Z",
        "semantic": [{"semantic_memory": "fact at t0", "tags": []}],
    })
    client.post("/api/v1/graphs/tl_order/retrieve", json={
        "observation": "anything",
        "mode": "semantic_memory",
        "session_id": "run-Z",
    })
    events = client.get("/api/v1/graphs/tl_order/sessions/run-Z").json()["events"]
    # First event must be the insert (graph_time=0), second the recall.
    assert events[0]["kind"] == "insert"
    assert events[-1]["kind"] == "recall"
