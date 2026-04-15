"""Tests for health check endpoint."""


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == "0.1.0"
    assert data["llm_available"] is True
    assert data["embedding_available"] is True
    assert data["chroma_available"] is True
    assert data["status"] == "ok"
