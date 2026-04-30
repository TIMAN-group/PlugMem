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
