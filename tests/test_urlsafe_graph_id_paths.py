"""Graph IDs must survive the request path.

Clients percent-encode the ID into one path segment (`encodeURIComponent`), but
ASGI servers hand over an already-decoded `scope["path"]`, so `%2F` arrives as a
real `/`, the path gains segments, and every graph-scoped route 404s.

These tests pin both halves of the fix: the middleware keeps encoded slashes
encoded long enough for routing to match, and the route class decodes the param
before the handler runs.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plugmem.api.urlsafe import (
    EncodedSlashPathMiddleware,
    UnquotedPathParamsRoute,
    decode_path_preserving_slashes,
)

# The IDs the coding adapters actually mint.
GRAPH_IDS = [
    "repo://claude-code/github.com/owner/repo",
    "repo://claude-code/localM:/Projects",
    "repo://opencode/gitlab.com/group/subgroup/repo",
    "user://claude-code/dave",
    "plain-id",
]


@pytest.fixture
def client():
    app = FastAPI()
    app.add_middleware(EncodedSlashPathMiddleware)

    from fastapi import APIRouter

    router = APIRouter(prefix="/graphs", route_class=UnquotedPathParamsRoute)

    @router.get("/{graph_id}")
    def get_graph(graph_id: str):
        return {"graph_id": graph_id}

    @router.get("/{graph_id}/stats")
    def get_stats(graph_id: str):
        return {"graph_id": graph_id, "where": "stats"}

    @router.get("/{graph_id}/node/{node_type}/{node_id}")
    def get_node(graph_id: str, node_type: str, node_id: str):
        return {"graph_id": graph_id, "node_type": node_type, "node_id": node_id}

    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def _encoded(graph_id: str) -> str:
    from urllib.parse import quote

    return quote(graph_id, safe="")


@pytest.mark.parametrize("graph_id", GRAPH_IDS)
def test_bare_graph_route_matches_and_decodes(client, graph_id):
    r = client.get(f"/api/v1/graphs/{_encoded(graph_id)}")
    assert r.status_code == 200
    assert r.json()["graph_id"] == graph_id


@pytest.mark.parametrize("graph_id", GRAPH_IDS)
def test_suffixed_route_is_not_shadowed(client, graph_id):
    """The bug that would break `{graph_id:path}`: /stats must still resolve."""
    r = client.get(f"/api/v1/graphs/{_encoded(graph_id)}/stats")
    assert r.status_code == 200
    assert r.json() == {"graph_id": graph_id, "where": "stats"}


@pytest.mark.parametrize("graph_id", GRAPH_IDS)
def test_multi_param_route_splits_correctly(client, graph_id):
    """Trailing params must not be absorbed into the graph ID."""
    r = client.get(f"/api/v1/graphs/{_encoded(graph_id)}/node/semantic/node-7")
    assert r.status_code == 200
    assert r.json() == {
        "graph_id": graph_id,
        "node_type": "semantic",
        "node_id": "node-7",
    }


def test_unencoded_slashes_still_do_not_match(client):
    """We fix encoded slashes only. A literal slash is genuinely another segment."""
    r = client.get("/api/v1/graphs/repo://claude-code/github.com/owner/repo")
    assert r.status_code == 404


def test_other_escapes_are_decoded_normally(client):
    """Spaces and the like must not survive as escapes."""
    r = client.get(f"/api/v1/graphs/{_encoded('graph with spaces')}")
    assert r.status_code == 200
    assert r.json()["graph_id"] == "graph with spaces"


class TestDecodePathPreservingSlashes:
    def test_leaves_encoded_slash_encoded(self):
        assert decode_path_preserving_slashes("/g/a%2Fb") == "/g/a%2Fb"

    def test_lowercase_escape_normalizes(self):
        assert decode_path_preserving_slashes("/g/a%2fb") == "/g/a%2Fb"

    def test_decodes_everything_else(self):
        assert decode_path_preserving_slashes("/g/a%3Ab%20c") == "/g/a:b c"

    def test_real_slashes_untouched(self):
        assert decode_path_preserving_slashes("/api/v1/graphs") == "/api/v1/graphs"

    def test_percent_sign_itself_round_trips(self):
        assert decode_path_preserving_slashes("/g/100%25") == "/g/100%"
