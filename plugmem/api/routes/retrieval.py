"""Retrieve and Reason endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager, api_lock
from plugmem.api.schemas import (
    ConsolidateRequest,
    ConsolidateResponse,
    ReasonRequest,
    ReasonResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from plugmem.api.urlsafe import UnquotedPathParamsRoute
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
        with api_lock:
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


def _audit_from_trace(result: Dict[str, Any]) -> Dict[str, Any]:
    """Lift the recall-audit fields out of a retrieve_with_trace() result."""
    plan = result.get("plan", {}) or {}
    selected = result.get("selected", {}) or {}
    return {
        "next_subgoal": plan.get("next_subgoal", ""),
        "query_tags": plan.get("query_tags", []),
        "selected_semantic_ids": selected.get("semantic_ids", []),
        "selected_procedural_ids": selected.get("procedural_ids", []),
    }

router = APIRouter(
    prefix="/graphs",
    tags=["retrieval"],
    dependencies=[Depends(require_api_key)],
    route_class=UnquotedPathParamsRoute,
)


def _manager() -> GraphManager:
    return get_graph_manager()


def _get_graph(graph_id: str):
    gm = _manager()
    try:
        return gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")


def _parse_value_funcs(body: Union[RetrieveRequest, ReasonRequest]):
    tag_relevant = None
    if body.tag_k is not None or body.tag_threshold is not None:
        from plugmem.core.value_functions import TagRelevant
        tk = body.tag_k if body.tag_k is not None else 1
        tthr = body.tag_threshold if body.tag_threshold is not None else 0.8
        tag_relevant = TagRelevant(k=tk, value_threshold=tthr)

    semantic_relevant = None
    if body.semantic_k is not None or body.semantic_threshold is not None:
        from plugmem.core.value_functions import SemanticRelevant
        sk = body.semantic_k if body.semantic_k is not None else 5
        sthr = body.semantic_threshold if body.semantic_threshold is not None else 0.0
        semantic_relevant = SemanticRelevant(k=sk, value_threshold=sthr)

    procedural_relevant = None
    if body.procedural_k is not None or body.procedural_threshold is not None:
        from plugmem.core.value_functions import ProceduralRelevant
        pk = body.procedural_k if body.procedural_k is not None else 5
        pthr = body.procedural_threshold if body.procedural_threshold is not None else 0.0
        procedural_relevant = ProceduralRelevant(k=pk, value_threshold=pthr)

    subgoal_relevant = None
    if body.subgoal_k is not None or body.subgoal_threshold is not None:
        from plugmem.core.value_functions import SubgoalRelevant
        sgk = body.subgoal_k if body.subgoal_k is not None else 5
        sgthr = body.subgoal_threshold if body.subgoal_threshold is not None else 0.0
        subgoal_relevant = SubgoalRelevant(k=sgk, value_threshold=sgthr)
        
    semantic_relevant4episodic = None
    if body.episodic_k is not None or body.episodic_threshold is not None:
        from plugmem.core.value_functions import SemanticRelevant
        ek = body.episodic_k if body.episodic_k is not None else 5
        ethr = body.episodic_threshold if body.episodic_threshold is not None else 0.0
        semantic_relevant4episodic = SemanticRelevant(k=ek, value_threshold=ethr)

    return tag_relevant, semantic_relevant, procedural_relevant, subgoal_relevant, semantic_relevant4episodic


@router.post("/{graph_id}/retrieve", response_model=RetrieveResponse)
def retrieve(graph_id: str, body: RetrieveRequest) -> RetrieveResponse:
    graph = _get_graph(graph_id)

    tag_rel, sem_rel, proc_rel, sub_rel, ep_rel = _parse_value_funcs(body)

    with with_phase("retrieve"):
        result = graph.retrieve_with_trace(
            goal=body.goal,
            subgoal=body.subgoal,
            state=body.state,
            observation=body.observation,
            time=body.time,
            task_type=body.task_type,
            mode=body.mode,
            min_confidence=body.min_confidence,
            source_in=body.source_in,
            tag_relevant=tag_rel,
            semantic_relevant=sem_rel,
            procedural_relevant=proc_rel,
            subgoal_relevant=sub_rel,
            semantic_relevant4episodic=ep_rel,
            auto_plan=True,
        )

    messages = result.get("rendered_prompt", [])
    variables = result.get("variables", {})
    mode = result.get("mode", "semantic_memory")
    audit = _audit_from_trace(result)

    _write_audit(graph, endpoint="retrieve", body=body, audit=audit, mode=mode, n_messages=len(messages))

    return RetrieveResponse(
        mode=mode,
        reasoning_prompt=messages,
        variables=variables,
    )


@router.post("/{graph_id}/reason", response_model=ReasonResponse)
def reason(graph_id: str, body: ReasonRequest) -> ReasonResponse:
    graph = _get_graph(graph_id)

    tag_rel, sem_rel, proc_rel, sub_rel, ep_rel = _parse_value_funcs(body)

    with with_phase("retrieve"):
        result = graph.retrieve_with_trace(
            goal=body.goal,
            subgoal=body.subgoal,
            state=body.state,
            observation=body.observation,
            time=body.time,
            task_type=body.task_type,
            mode=body.mode,
            min_confidence=body.min_confidence,
            source_in=body.source_in,
            tag_relevant=tag_rel,
            semantic_relevant=sem_rel,
            procedural_relevant=proc_rel,
            subgoal_relevant=sub_rel,
            semantic_relevant4episodic=ep_rel,
            auto_plan=True,
        )

    messages = result.get("rendered_prompt", [])
    mode = result.get("mode", "semantic_memory")
    audit = _audit_from_trace(result)

    with with_phase("reason"):
        reasoning = graph.llm.complete(messages=messages)

    _write_audit(graph, endpoint="reason", body=body, audit=audit, mode=mode, n_messages=len(messages))

    from plugmem.api.logging_ctx import current_log_ctx
    ctx = current_log_ctx.get()
    if ctx is not None:
        ctx.agent_output = reasoning

    return ReasonResponse(
        mode=mode,
        reasoning=reasoning,
        reasoning_prompt=messages,
    )


@router.post("/{graph_id}/consolidate", response_model=ConsolidateResponse)
def consolidate(graph_id: str, body: ConsolidateRequest) -> ConsolidateResponse:
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
