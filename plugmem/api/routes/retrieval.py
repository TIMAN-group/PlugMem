"""Retrieve and Reason endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

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
from plugmem.clients.llm import with_phase
from plugmem.graph_manager import GraphManager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_audit(
    graph,
    *,
    endpoint: str,
    body,
    audit: Dict[str, Any],
    mode: str,
    n_messages: int,
) -> None:
    """Best-effort audit write — never breaks the recall path."""
    try:
        graph.storage.add_recall(
            graph.graph_id,
            endpoint=endpoint,
            ts=_now_iso(),
            graph_time=graph.semantic_time,
            session_id=getattr(body, "session_id", None),
            observation=body.observation or "",
            goal=body.goal or "",
            subgoal=body.subgoal or "",
            state=body.state or "",
            task_type=body.task_type or "",
            mode=mode,
            next_subgoal=audit.get("next_subgoal", ""),
            query_tags=audit.get("query_tags", []),
            selected_semantic_ids=audit.get("selected_semantic_ids", []),
            selected_procedural_ids=audit.get("selected_procedural_ids", []),
            n_messages=n_messages,
        )
    except Exception:
        # Don't let an audit-log failure break a working recall.
        pass

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

    audit: Dict[str, Any] = {}
    with with_phase("retrieve"):
        messages, variables, mode = graph.retrieve_memory(
            goal=body.goal,
            subgoal=body.subgoal,
            state=body.state,
            observation=body.observation,
            time=body.time,
            task_type=body.task_type,
            mode=body.mode,
            min_confidence=body.min_confidence,
            source_in=body.source_in,
            _audit=audit,
        )
    _write_audit(graph, endpoint="retrieve", body=body, audit=audit, mode=mode, n_messages=len(messages))

    return RetrieveResponse(
        mode=mode,
        reasoning_prompt=messages,
        variables=variables,
    )


@router.post("/{graph_id}/reason", response_model=ReasonResponse)
async def reason(graph_id: str, body: ReasonRequest) -> ReasonResponse:
    graph = _get_graph(graph_id)

    audit: Dict[str, Any] = {}
    with with_phase("retrieve"):
        messages, variables, mode = graph.retrieve_memory(
            goal=body.goal,
            subgoal=body.subgoal,
            state=body.state,
            observation=body.observation,
            time=body.time,
            task_type=body.task_type,
            mode=body.mode,
            min_confidence=body.min_confidence,
            source_in=body.source_in,
            _audit=audit,
        )

    with with_phase("reason"):
        reasoning = graph.llm.complete(messages=messages)
    
    _write_audit(graph, endpoint="reason", body=body, audit=audit, mode=mode, n_messages=len(messages))

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
