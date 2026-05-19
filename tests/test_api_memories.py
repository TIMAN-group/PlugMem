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


def test_insert_trajectory_with_initial_observation(client):
    client.post("/api/v1/graphs", json={"graph_id": "traj_initial"})
    resp = client.post("/api/v1/graphs/traj_initial/memories", json={
        "mode": "trajectory",
        "goal": "finish task",
        "initial_observation": "initial scene",
        "steps": [
            {"action": "click first", "observation": "after first click"},
            {"action": "click second", "observation": "after second click"},
        ],
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "ok"
    assert resp.json()["stats"]["episodic"] == 2

    r = client.get("/api/v1/graphs/traj_initial/search?node_type=episodic&limit=10")
    nodes = sorted(r.json()["nodes"], key=lambda node: node["episodic_id"])
    assert nodes[0]["observation"] == "initial scene"
    assert nodes[0]["action"] == "click first"
    assert nodes[1]["observation"] == "after first click"
    assert nodes[1]["action"] == "click second"


def test_insert_trajectory_without_initial_observation_uses_first_step(client):
    client.post("/api/v1/graphs", json={"graph_id": "traj_legacy"})
    resp = client.post("/api/v1/graphs/traj_legacy/memories", json={
        "mode": "trajectory",
        "goal": "finish task",
        "steps": [
            {"action": "click first", "observation": "legacy first observation"},
            {"action": "click second", "observation": "legacy second observation"},
        ],
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "ok"
    assert resp.json()["stats"]["episodic"] == 2

    r = client.get("/api/v1/graphs/traj_legacy/search?node_type=episodic&limit=10")
    nodes = sorted(r.json()["nodes"], key=lambda node: node["episodic_id"])
    assert nodes[0]["observation"] == "legacy first observation"
    assert nodes[0]["action"] == "click first"


def test_insert_trajectory_empty_steps_still_fails_with_initial_observation(client):
    client.post("/api/v1/graphs", json={"graph_id": "traj_empty_steps"})
    resp = client.post("/api/v1/graphs/traj_empty_steps/memories", json={
        "mode": "trajectory",
        "goal": "finish task",
        "initial_observation": "initial scene",
        "steps": [],
    })
    assert resp.status_code == 422


def test_insert_stamps_session_id_on_all_node_types(client):
    """An insert with session_id should stamp it on every newly-created node."""
    client.post("/api/v1/graphs", json={"graph_id": "sess_test"})
    resp = client.post("/api/v1/graphs/sess_test/memories", json={
        "mode": "structured",
        "session_id": "run-2026-05-01",
        "episodic": [[{"observation": "obs", "action": "act"}]],
        "semantic": [{"semantic_memory": "fact", "tags": ["t"]}],
        "procedural": [{"subgoal": "do", "procedural_memory": "how"}],
    })
    assert resp.status_code == 200

    # Read each node type back via the inspector and check session_id field
    for node_type, expected_count in [("episodic", 1), ("semantic", 1), ("procedural", 1)]:
        r = client.get(
            f"/api/v1/graphs/sess_test/search?node_type={node_type}&limit=10"
        )
        assert r.status_code == 200, r.text
        nodes = r.json()["nodes"]
        assert len(nodes) == expected_count
        assert nodes[0]["session_id"] == "run-2026-05-01", (
            f"{node_type} node missing session_id: {nodes[0]}"
        )


def test_insert_without_session_id_leaves_field_null(client):
    client.post("/api/v1/graphs", json={"graph_id": "no_sess_test"})
    resp = client.post("/api/v1/graphs/no_sess_test/memories", json={
        "mode": "structured",
        "semantic": [{"semantic_memory": "fact", "tags": []}],
    })
    assert resp.status_code == 200

    r = client.get("/api/v1/graphs/no_sess_test/search?node_type=semantic")
    nodes = r.json()["nodes"]
    assert nodes[0]["session_id"] is None
