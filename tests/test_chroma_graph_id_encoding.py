"""Graph IDs must survive the trip into ChromaDB collection names and back.

ChromaDB restricts collection names to [a-zA-Z0-9._-] with alphanumeric ends.
The coding adapters mint IDs containing "/" and ":", which Chroma rejects, so
storage encodes them. list_graphs() reverses that encoding, which is why the
mapping has to be exactly invertible rather than merely legal.
"""
import re

import pytest

from plugmem.storage.chroma import (
    NODE_TYPES,
    _collection_name,
    _decode_graph_id,
    _encode_graph_id,
)

# Chroma's own constraint, restated here so the test fails if we drift from it.
CHROMA_NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]")

UNSAFE_IDS = [
    "repo://claude-code/github.com/owner/repo",
    "repo://claude-code/localM:/Projects",
    "repo://claude-code/localC:/Users/Dave/.claude",
    "repo://opencode/gitlab.com/group/subgroup/repo",
    "user://claude-code/dave",
    "repo://claude-code/github.com/owner/repo.git",
    "graph with spaces",
    "graph/with/slashes",
]

SAFE_IDS = [
    "test-safe-id",
    "my_graph",  # underscore is legal for Chroma even though we split on it
    "a.b.c",
    "abc",
]


@pytest.mark.parametrize("graph_id", UNSAFE_IDS + SAFE_IDS)
def test_round_trips_exactly(graph_id):
    assert _decode_graph_id(_encode_graph_id(graph_id)) == graph_id


@pytest.mark.parametrize("graph_id", UNSAFE_IDS + SAFE_IDS)
@pytest.mark.parametrize("node_type", NODE_TYPES + ("recall_audit",))
def test_collection_name_is_chroma_legal(graph_id, node_type):
    assert CHROMA_NAME_RE.fullmatch(_collection_name(graph_id, node_type))


@pytest.mark.parametrize("graph_id", SAFE_IDS)
def test_already_legal_ids_pass_through_untouched(graph_id):
    """Graphs created before this encoding existed must keep resolving."""
    assert _encode_graph_id(graph_id) == graph_id
    assert _collection_name(graph_id, "semantic") == f"{graph_id}_semantic"


def test_distinct_ids_never_collide():
    """A lossy slug would fold these together; the encoding must not."""
    colliding_shapes = [
        "repo://claude-code/github.com/owner/repo",
        "repo-//claude-code/github.com/owner/repo",
        "repo://claude-code/github.com/owner:repo",
        "repo:/claude-code/github.com/owner/repo",
    ]
    encoded = [_encode_graph_id(g) for g in colliding_shapes]
    assert len(set(encoded)) == len(colliding_shapes)


def test_long_ids_stay_within_chroma_name_limit():
    """A path-derived ID at the limit must still produce a legal name."""
    from plugmem.storage.chroma import _MAX_ENCODED_GRAPH_ID

    at_limit = "repo://claude-code/local" + "M:/very-long-path" * 15
    at_limit = at_limit[:_MAX_ENCODED_GRAPH_ID]
    for node_type in NODE_TYPES + ("recall_audit",):
        name = _collection_name(at_limit, node_type)
        assert len(name) <= 512
        assert CHROMA_NAME_RE.fullmatch(name)
    assert _decode_graph_id(_encode_graph_id(at_limit)) == at_limit


def test_over_limit_id_fails_with_a_named_error():
    from plugmem.storage.chroma import _MAX_ENCODED_GRAPH_ID

    too_long = "repo://" + "x" * _MAX_ENCODED_GRAPH_ID
    with pytest.raises(ValueError, match="too long to encode"):
        _encode_graph_id(too_long)


def test_realistic_ids_are_nowhere_near_the_limit():
    """Guards against the limit being tightened into everyday IDs."""
    from plugmem.storage.chroma import _MAX_ENCODED_GRAPH_ID

    for graph_id in UNSAFE_IDS:
        assert len(graph_id) < _MAX_ENCODED_GRAPH_ID / 2


def test_marker_prefixed_literal_is_not_mistaken_for_encoded():
    """An ID that itself starts with the marker gets encoded, not passed through."""
    graph_id = "b32-not-actually-encoded"
    encoded = _encode_graph_id(graph_id)
    assert encoded != graph_id
    assert _decode_graph_id(encoded) == graph_id


def test_list_graphs_recovers_original_ids(tmp_path):
    """The end-to-end constraint: what goes into Chroma comes back out intact."""
    chromadb = pytest.importorskip("chromadb")
    from plugmem.clients.embedding import (
        LocalDeterministicEmbeddingClient,
        PlugMemEmbeddingFunction,
    )
    from plugmem.storage.chroma import ChromaStorage

    storage = ChromaStorage(
        client=chromadb.PersistentClient(path=str(tmp_path)),
        embedding_function=PlugMemEmbeddingFunction(
            LocalDeterministicEmbeddingClient()
        ),
    )

    wanted = [
        "repo://claude-code/github.com/owner/repo",
        "repo://claude-code/localM:/Projects",
        "test-safe-id",
    ]
    for graph_id in wanted:
        storage.create_graph(graph_id)
        assert storage.graph_exists(graph_id)

    assert sorted(storage.list_graphs()) == sorted(wanted)

    storage.delete_graph(wanted[0])
    assert not storage.graph_exists(wanted[0])
    assert sorted(storage.list_graphs()) == sorted(wanted[1:])
