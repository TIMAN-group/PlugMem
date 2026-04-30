"""Memory Inspector endpoints — read/search/inspect/deactivate."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager
from plugmem.api.schemas import (
    NodeDetailResponse,
    SearchResponse,
    SemanticUpdateRequest,
)
from plugmem.graph_manager import GraphManager

router = APIRouter(prefix="/graphs", tags=["inspector"], dependencies=[Depends(require_api_key)])

NODE_TYPES = ("semantic", "procedural", "tag", "subgoal", "episodic")


def _manager() -> GraphManager:
    return get_graph_manager()


def _get_graph(graph_id: str):
    gm = _manager()
    try:
        return gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")


def _check_type(node_type: str) -> None:
    if node_type not in NODE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid node_type '{node_type}'. Must be one of: {list(NODE_TYPES)}",
        )


# ------------------------------------------------------------------ #
# Serializers — single source of truth shared by /search and /node
# ------------------------------------------------------------------ #


def _serialize_episodic(n) -> Dict[str, Any]:
    return {
        "id": n.episodic_id,
        "episodic_id": n.episodic_id,
        "observation": n.observation,
        "action": n.action,
        "subgoal": n.subgoal,
        "state": n.state,
        "reward": n.reward,
        "session_id": n.session_id,
        "time": n.time,
    }


def _serialize_semantic(n) -> Dict[str, Any]:
    return {
        "id": n.semantic_id,
        "semantic_id": n.semantic_id,
        "text": n.get_semantic_memory(),
        "tags": [t.tag for t in n.tag_nodes] or list(n.tags),
        "is_active": n.is_active,
        "credibility": getattr(n, "Credibility", 10),
        "session_id": getattr(n, "session_id", None),
        "date": getattr(n, "date", ""),
        "time": n.time,
        "n_tags": len(n.tag_nodes),
        "n_episodics": len(n.episodic_nodes),
        "n_bro": len(n.bro_semantic_nodes),
    }


def _serialize_tag(n) -> Dict[str, Any]:
    return {
        "id": n.tag_id,
        "tag_id": n.tag_id,
        "tag": n.tag,
        "importance": n.importance,
        "time": n.time,
        "n_semantics": len(n.semantic_nodes),
    }


def _serialize_subgoal(n) -> Dict[str, Any]:
    return {
        "id": n.subgoal_id,
        "subgoal_id": n.subgoal_id,
        "subgoal": n.subgoal,
        "time": n.time,
        "n_procedurals": len(n.procedural_nodes),
        "activated": n.activate,
    }


def _serialize_procedural(n) -> Dict[str, Any]:
    return {
        "id": n.procedural_id,
        "procedural_id": n.procedural_id,
        "text": n.get_procedural_memory(),
        "subgoals": [s.subgoal for s in n.subgoal_nodes] or list(n.subgoals),
        "return": n.Return,
        "time": n.time,
        "n_episodics": len(n.episodic_nodes),
    }


SERIALIZERS = {
    "episodic": _serialize_episodic,
    "semantic": _serialize_semantic,
    "tag": _serialize_tag,
    "subgoal": _serialize_subgoal,
    "procedural": _serialize_procedural,
}


def _node_text(node_type: str, node) -> str:
    """Return the searchable text field for a node."""
    if node_type == "semantic":
        return node.get_semantic_memory() or ""
    if node_type == "procedural":
        return node.get_procedural_memory() or ""
    if node_type == "tag":
        return node.tag or ""
    if node_type == "subgoal":
        return node.subgoal or ""
    if node_type == "episodic":
        return f"{node.observation}\n{node.action}"
    return ""


def _list_for_type(graph, node_type: str) -> List:
    return {
        "episodic": graph.episodic_nodes,
        "semantic": graph.semantic_nodes,
        "tag": graph.tag_nodes,
        "subgoal": graph.subgoal_nodes,
        "procedural": graph.procedural_nodes,
    }[node_type]


def _lookup_for_type(graph, node_type: str):
    return {
        "episodic": graph.episodic_id2node,
        "semantic": graph.semantic_id2node,
        "tag": graph.tag_id2node,
        "subgoal": graph.subgoal_id2node,
        "procedural": graph.procedural_id2node,
    }[node_type]


# ------------------------------------------------------------------ #
# /search — substring filter on the node text field
# ------------------------------------------------------------------ #


@router.get("/{graph_id}/search", response_model=SearchResponse)
async def search_nodes(
    graph_id: str,
    q: str = "",
    node_type: str = "semantic",
    limit: int = 50,
    only_active: bool = False,
) -> SearchResponse:
    _check_type(node_type)
    graph = _get_graph(graph_id)

    nodes = _list_for_type(graph, node_type)
    serializer = SERIALIZERS[node_type]
    needle = q.casefold().strip()

    matches: List[Dict[str, Any]] = []
    for node in nodes:
        if only_active and node_type == "semantic" and not node.is_active:
            continue
        if needle and needle not in _node_text(node_type, node).casefold():
            continue
        matches.append(serializer(node))

    # sort newest first
    matches.sort(key=lambda d: d.get("time") if isinstance(d.get("time"), (int, float)) else 0, reverse=True)
    truncated = matches[: max(0, limit)]

    return SearchResponse(
        graph_id=graph_id,
        node_type=node_type,
        query=q,
        count=len(matches),
        nodes=truncated,
    )


# ------------------------------------------------------------------ #
# /node/{type}/{id} — single node + one-hop edges
# ------------------------------------------------------------------ #


@router.get("/{graph_id}/node/{node_type}/{node_id}", response_model=NodeDetailResponse)
async def get_node_detail(graph_id: str, node_type: str, node_id: int) -> NodeDetailResponse:
    _check_type(node_type)
    graph = _get_graph(graph_id)

    lookup = _lookup_for_type(graph, node_type)
    node = lookup.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"{node_type} node {node_id} not found")

    serializer = SERIALIZERS[node_type]
    edges: Dict[str, List[Dict[str, Any]]] = {}

    if node_type == "semantic":
        edges["tags"] = [_serialize_tag(t) for t in node.tag_nodes]
        edges["episodics"] = [_serialize_episodic(e) for e in node.episodic_nodes]
        edges["bro_semantics"] = [_serialize_semantic(s) for s in node.bro_semantic_nodes]
        edges["son_semantics"] = [_serialize_semantic(s) for s in getattr(node, "son_semantic", [])]
    elif node_type == "tag":
        edges["semantics"] = [_serialize_semantic(s) for s in node.semantic_nodes]
    elif node_type == "subgoal":
        edges["procedurals"] = [_serialize_procedural(p) for p in node.procedural_nodes]
    elif node_type == "procedural":
        edges["subgoals"] = [_serialize_subgoal(s) for s in node.subgoal_nodes]
        edges["episodics"] = [_serialize_episodic(e) for e in node.episodic_nodes]
    elif node_type == "episodic":
        # reverse lookup: which semantics linked back?
        linked = [s for s in graph.semantic_nodes if any(e.episodic_id == node.episodic_id for e in s.episodic_nodes)]
        edges["semantics"] = [_serialize_semantic(s) for s in linked]

    return NodeDetailResponse(
        graph_id=graph_id,
        node_type=node_type,
        node=serializer(node),
        edges=edges,
    )


# ------------------------------------------------------------------ #
# PATCH semantic node — currently only is_active is mutable
# ------------------------------------------------------------------ #


@router.patch("/{graph_id}/semantic/{semantic_id}", response_model=NodeDetailResponse)
async def update_semantic(
    graph_id: str,
    semantic_id: int,
    body: SemanticUpdateRequest,
) -> NodeDetailResponse:
    graph = _get_graph(graph_id)
    node = graph.semantic_id2node.get(semantic_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"semantic node {semantic_id} not found")

    updates: Dict[str, Any] = {}
    if body.is_active is not None:
        node.is_active = bool(body.is_active)
        updates["is_active"] = node.is_active

    if not updates:
        raise HTTPException(status_code=400, detail="no mutable fields supplied")

    graph.storage.update_semantic(graph_id, semantic_id, metadata_updates=updates)

    return NodeDetailResponse(
        graph_id=graph_id,
        node_type="semantic",
        node=_serialize_semantic(node),
        edges={
            "tags": [_serialize_tag(t) for t in node.tag_nodes],
        },
    )
