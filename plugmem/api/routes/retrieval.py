"""Retrieve and Reason endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager
from plugmem.api.schemas import (
    ConsolidateRequest,
    ConsolidateResponse,
    ReasonRequest,
    ReasonResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from plugmem.graph_manager import GraphManager

router = APIRouter(prefix="/graphs", tags=["retrieval"], dependencies=[Depends(require_api_key)])


def _manager() -> GraphManager:
    return get_graph_manager()


def _get_graph(graph_id: str):
    gm = _manager()
    try:
        return gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")


@router.post("/{graph_id}/retrieve", response_model=RetrieveResponse)
async def retrieve(graph_id: str, body: RetrieveRequest) -> RetrieveResponse:
    graph = _get_graph(graph_id)

    messages, variables, mode = graph.retrieve_memory(
        goal=body.goal,
        subgoal=body.subgoal,
        state=body.state,
        observation=body.observation,
        time=body.time,
        task_type=body.task_type,
        mode=body.mode,
    )

    return RetrieveResponse(
        mode=mode,
        reasoning_prompt=messages,
        variables=variables,
    )


@router.post("/{graph_id}/reason", response_model=ReasonResponse)
async def reason(graph_id: str, body: ReasonRequest) -> ReasonResponse:
    graph = _get_graph(graph_id)

    messages, variables, mode = graph.retrieve_memory(
        goal=body.goal,
        subgoal=body.subgoal,
        state=body.state,
        observation=body.observation,
        time=body.time,
        task_type=body.task_type,
        mode=body.mode,
    )

    reasoning = graph.llm.complete(messages=messages)

    return ReasonResponse(
        mode=mode,
        reasoning=reasoning,
        reasoning_prompt=messages,
    )


@router.post("/{graph_id}/consolidate", response_model=ConsolidateResponse)
async def consolidate(graph_id: str, body: ConsolidateRequest) -> ConsolidateResponse:
    graph = _get_graph(graph_id)

    stats = graph.update_semantic_subgraph(
        merge_threshold=body.merge_threshold,
        max_merges_per_node=body.max_merges_per_node,
        max_candidates_per_tag=body.max_candidates_per_tag,
        max_total_candidates=body.max_total_candidates,
        min_credibility_to_keep_active=body.min_credibility_to_keep_active,
        credibility_decay=body.credibility_decay,
        only_update_recent_window=body.only_update_recent_window,
        allow_merge_with_common_episodic_nodes=body.allow_merge_with_common_episodic_nodes,
    )

    return ConsolidateResponse(status="ok", stats=stats)
