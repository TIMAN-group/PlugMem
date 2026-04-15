"""Tests for graph CRUD endpoints."""


def test_create_graph(client):
    resp = client.post("/api/v1/graphs", json={})
    assert resp.status_code == 201
    data = resp.json()
    assert "graph_id" in data
    assert isinstance(data["stats"], dict)


def test_create_graph_with_custom_id(client):
    resp = client.post("/api/v1/graphs", json={"graph_id": "my_graph"})
    assert resp.status_code == 201
    assert resp.json()["graph_id"] == "my_graph"


def test_list_graphs(client):
    client.post("/api/v1/graphs", json={"graph_id": "g1"})
    client.post("/api/v1/graphs", json={"graph_id": "g2"})
    resp = client.get("/api/v1/graphs")
    assert resp.status_code == 200
    assert set(resp.json()["graphs"]) >= {"g1", "g2"}


def test_get_graph(client):
    client.post("/api/v1/graphs", json={"graph_id": "test"})
    resp = client.get("/api/v1/graphs/test")
    assert resp.status_code == 200
    assert resp.json()["graph_id"] == "test"


def test_get_graph_not_found(client):
    resp = client.get("/api/v1/graphs/nonexistent")
    assert resp.status_code == 404


def test_delete_graph(client):
    client.post("/api/v1/graphs", json={"graph_id": "deleteme"})
    resp = client.delete("/api/v1/graphs/deleteme")
    assert resp.status_code == 204

    resp = client.get("/api/v1/graphs/deleteme")
    assert resp.status_code == 404


def test_get_stats(client):
    client.post("/api/v1/graphs", json={"graph_id": "stats_test"})
    resp = client.get("/api/v1/graphs/stats_test/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["graph_id"] == "stats_test"
    assert "semantic" in data["stats"]


def test_browse_nodes_empty(client):
    client.post("/api/v1/graphs", json={"graph_id": "browse"})
    resp = client.get("/api/v1/graphs/browse/nodes?node_type=semantic")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["nodes"] == []


def test_browse_nodes_invalid_type(client):
    client.post("/api/v1/graphs", json={"graph_id": "browse2"})
    resp = client.get("/api/v1/graphs/browse2/nodes?node_type=invalid")
    assert resp.status_code == 400
