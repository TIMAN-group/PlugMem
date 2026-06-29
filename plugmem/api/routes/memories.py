"""Memory insertion endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_embedder, get_graph_manager, get_llm
from plugmem.api.schemas import MemoryInsertRequest, MemoryInsertResponse
from plugmem.core.memory import Memory
from plugmem.graph_manager import GraphManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graphs", tags=["memories"], dependencies=[Depends(require_api_key)])


def _manager() -> GraphManager:
    return get_graph_manager()


@router.post("/{graph_id}/memories", response_model=MemoryInsertResponse)
def insert_memories(graph_id: str, body: MemoryInsertRequest) -> MemoryInsertResponse:
    gm = _manager()

    try:
        graph = gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    if body.mode == "trajectory":
        return _insert_trajectory(graph, body)
    else:
        return _insert_structured(graph, body)


def _insert_trajectory(graph, body: MemoryInsertRequest) -> MemoryInsertResponse:
    if not body.goal:
        raise HTTPException(status_code=422, detail="'goal' is required for trajectory mode")
    if not body.steps:
        raise HTTPException(status_code=422, detail="'steps' is required for trajectory mode")

    llm = get_llm()
    embedder = get_embedder()

    mem = Memory(
        goal=body.goal,
        observation=body.steps[0].observation,
        llm=llm,
        embedder=embedder,
        time=graph.semantic_time,
        session_id=body.session_id,
    )
    for step in body.steps:
        mem.append(action_t0=step.action, observation_t1=step.observation)
    mem.close()
    graph.insert(mem)

    stats = graph.storage.get_graph_stats(graph.graph_id)
    return MemoryInsertResponse(status="ok", stats=stats)


def _insert_structured(graph, body: MemoryInsertRequest) -> MemoryInsertResponse:
    embedder = get_embedder()

    # Build a Memory-like object with pre-structured data
    mem = Memory.__new__(Memory)
    mem.time = graph.semantic_time
    mem.session_id = body.session_id
    mem.llm = get_llm()
    mem.embedder = embedder
    mem.memory = {
        "goal": "",
        "episodic": [],
        "semantic": [],
        "procedural": [],
    }
    mem.memory_embedding = {
        "semantic": [],
        "procedural": [],
    }

    # Episodic: list of trajectories (list of step dicts)
    if body.episodic:
        for trajectory in body.episodic:
            mem.memory["episodic"].append([
                {
                    "observation": step.observation,
                    "action": step.action,
                    "subgoal": step.subgoal,
                    "state": step.state,
                    "reward": step.reward,
                    "time": step.time or graph.semantic_time,
                }
                for step in trajectory
            ])

    # Semantic: embed text + tags
    if body.semantic:
        for sem in body.semantic:
            sem_dict = {
                "semantic_memory": sem.semantic_memory,
                "tags": sem.tags,
                "source": sem.source,
                "confidence": sem.confidence,
            }
            # Only pass grounding indices when supplied — the graph defaults to
            # whole-trajectory grounding otherwise (None breaks the `< len`
            # comparison in graph.insert).
            if sem.trajectory_num is not None:
                sem_dict["trajectory_num"] = sem.trajectory_num
            if sem.turn_num is not None:
                sem_dict["turn_num"] = sem.turn_num
            mem.memory["semantic"].append(sem_dict)
            # Use pre-computed embeddings if provided, otherwise fall back to embedder
            sem_emb = sem.embedding if sem.embedding is not None else embedder.embed(sem.semantic_memory)
            if sem.tag_embeddings is not None and len(sem.tag_embeddings) == len(sem.tags):
                tag_embs = sem.tag_embeddings
            else:
                tag_embs = [embedder.embed(tag) for tag in sem.tags]
            mem.memory_embedding["semantic"].append({
                "semantic_memory": sem_emb,
                "tags": tag_embs,
            })

    # Procedural: embed subgoal
    if body.procedural:
        for proc in body.procedural:
            proc_dict = {
                "subgoal": proc.subgoal,
                "procedural_memory": proc.procedural_memory,
                "time": graph.semantic_time,
                "return": proc.return_value,
                "source": proc.source,
                "confidence": proc.confidence,
            }
            # Ground this experience on its own episodic segment (the split
            # sub-sequence) instead of trajectory 0 / the whole trajectory.
            if proc.trajectory_num is not None:
                proc_dict["trajectory_num"] = proc.trajectory_num
            mem.memory["procedural"].append(proc_dict)
            # Use pre-computed embedding if provided, otherwise fall back to embedder
            subgoal_emb = proc.subgoal_embedding if proc.subgoal_embedding is not None else embedder.embed(proc.subgoal)
            mem.memory_embedding["procedural"].append({
                "subgoal": subgoal_emb,
            })

    # Apply insertion duplicate-detection thresholds if provided
    if body.tag_equal_threshold is not None:
        graph.tag_equal.value_threshold = body.tag_equal_threshold
    if body.semantic_equal_threshold is not None:
        graph.semantic_equal.value_threshold = body.semantic_equal_threshold
    if body.procedural_equal_threshold is not None:
        graph.procedural_equal.value_threshold = body.procedural_equal_threshold
    if body.subgoal_equal_threshold is not None:
        graph.subgoal_equal.value_threshold = body.subgoal_equal_threshold

    graph.insert(mem)

    stats = graph.storage.get_graph_stats(graph.graph_id)
    return MemoryInsertResponse(status="ok", stats=stats)
