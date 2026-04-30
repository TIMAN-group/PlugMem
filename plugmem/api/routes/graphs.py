"""Graph CRUD endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager
from plugmem.api.schemas import (
    GraphCreateRequest,
    GraphListResponse,
    GraphResponse,
    StatsResponse,
    NodeListResponse,
)
from plugmem.graph_manager import GraphManager

router = APIRouter(prefix="/graphs", tags=["graphs"], dependencies=[Depends(require_api_key)])


def _manager() -> GraphManager:
    return get_graph_manager()


@router.post("", response_model=GraphResponse, status_code=status.HTTP_201_CREATED)
async def create_graph(body: GraphCreateRequest) -> GraphResponse:
    gm = _manager()
    try:
        graph_id = gm.create_graph(graph_id=body.graph_id)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    stats = gm.get_stats(graph_id)
    return GraphResponse(graph_id=graph_id, stats=stats)


@router.get("", response_model=GraphListResponse)
async def list_graphs() -> GraphListResponse:
    gm = _manager()
    return GraphListResponse(graphs=gm.list_graphs())


@router.get("/{graph_id}", response_model=GraphResponse)
async def get_graph(graph_id: str) -> GraphResponse:
    gm = _manager()
    try:
        gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
    stats = gm.get_stats(graph_id)
    return GraphResponse(graph_id=graph_id, stats=stats)


@router.delete("/{graph_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_graph(graph_id: str) -> None:
    gm = _manager()
    try:
        gm.delete_graph(graph_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{graph_id}/stats", response_model=StatsResponse)
async def get_stats(graph_id: str) -> StatsResponse:
    gm = _manager()
    try:
        stats = gm.get_stats(graph_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
    return StatsResponse(graph_id=graph_id, stats=stats)


@router.get("/{graph_id}/nodes", response_model=NodeListResponse)
async def browse_nodes(
    graph_id: str,
    node_type: str = "semantic",
    limit: int = 50,
    offset: int = 0,
) -> NodeListResponse:
    gm = _manager()
    try:
        graph = gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    type_map = {
        "episodic": graph.episodic_nodes,
        "semantic": graph.semantic_nodes,
        "tag": graph.tag_nodes,
        "subgoal": graph.subgoal_nodes,
        "procedural": graph.procedural_nodes,
    }
    if node_type not in type_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid node_type '{node_type}'. Must be one of: {list(type_map)}",
        )

    nodes = type_map[node_type]
    page = nodes[offset : offset + limit]

    serialized = []
    for n in page:
        d: dict = {}
        if node_type == "episodic":
            d = {"episodic_id": n.episodic_id, "observation": n.observation, "action": n.action,
                 "subgoal": n.subgoal, "state": n.state, "reward": n.reward, "time": n.time}
        elif node_type == "semantic":
            d = {"semantic_id": n.semantic_id, "semantic_memory": n.get_semantic_memory(),
                 "tags": [t.tag for t in n.tag_nodes], "is_active": n.is_active,
                 "credibility": n.credibility, "time": n.time}
        elif node_type == "tag":
            d = {"tag_id": n.tag_id, "tag": n.tag, "importance": n.importance, "time": n.time}
        elif node_type == "subgoal":
            d = {"subgoal_id": n.subgoal_id, "subgoal": n.subgoal, "time": n.time}
        elif node_type == "procedural":
            d = {"procedural_id": n.procedural_id,
                 "procedural_memory": n.get_procedural_memory(),
                 "subgoal": n.subgoal, "time": n.time}
        serialized.append(d)

    return NodeListResponse(
        graph_id=graph_id,
        node_type=node_type,
        count=len(nodes),
        nodes=serialized,
    )
