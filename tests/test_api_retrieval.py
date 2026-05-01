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
