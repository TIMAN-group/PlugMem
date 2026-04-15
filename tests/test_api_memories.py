"""Tests for memory insertion endpoints."""


def test_insert_structured_semantic(client):
    client.post("/api/v1/graphs", json={"graph_id": "mem_test"})
    resp = client.post("/api/v1/graphs/mem_test/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "The capital of France is Paris", "tags": ["geography", "france"]},
            {"semantic_memory": "Python was created by Guido van Rossum", "tags": ["programming"]},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["stats"]["semantic"] == 2

    # Verify via stats endpoint
    resp = client.get("/api/v1/graphs/mem_test/stats")
    assert resp.json()["stats"]["semantic"] == 2
    assert resp.json()["stats"]["tag"] >= 2  # at least "geography", "france", "programming"


def test_insert_structured_procedural(client):
    client.post("/api/v1/graphs", json={"graph_id": "proc_test"})
    resp = client.post("/api/v1/graphs/proc_test/memories", json={
        "mode": "structured",
        "procedural": [
            {"subgoal": "deploy app", "procedural_memory": "run docker-compose up"},
        ],
    })
    assert resp.status_code == 200
    assert resp.json()["stats"]["procedural"] == 1


def test_insert_structured_episodic(client):
    client.post("/api/v1/graphs", json={"graph_id": "epi_test"})
    resp = client.post("/api/v1/graphs/epi_test/memories", json={
        "mode": "structured",
        "episodic": [[
            {"observation": "saw login page", "action": "clicked login"},
            {"observation": "saw dashboard", "action": "clicked settings"},
        ]],
    })
    assert resp.status_code == 200
    assert resp.json()["stats"]["episodic"] == 2


def test_insert_graph_not_found(client):
    resp = client.post("/api/v1/graphs/nope/memories", json={
        "mode": "structured",
        "semantic": [{"semantic_memory": "test", "tags": []}],
    })
    assert resp.status_code == 404


def test_insert_trajectory_missing_goal(client):
    client.post("/api/v1/graphs", json={"graph_id": "traj_err"})
    resp = client.post("/api/v1/graphs/traj_err/memories", json={
        "mode": "trajectory",
        "steps": [{"observation": "a", "action": "b"}],
    })
    assert resp.status_code == 422
