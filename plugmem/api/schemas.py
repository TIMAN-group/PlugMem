"""Pydantic request/response models for the PlugMem API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Why a memory was promoted into the graph. None = legacy / trajectory-derived
# (no promotion gate ran). Used by the coding-agent adapter to filter what
# surfaces on recall.
MemorySource = Literal[
    "failure_delta",
    "correction",
    "merged",
    "repeated_lookup",
    "explicit",
]


# ------------------------------------------------------------------ #
# Graphs
# ------------------------------------------------------------------ #

class GraphCreateRequest(BaseModel):
    graph_id: Optional[str] = Field(
        None,
        description="Optional custom graph ID. Auto-generated if omitted.",
    )


class GraphResponse(BaseModel):
    graph_id: str
    stats: Dict[str, int] = Field(default_factory=dict)


class GraphListResponse(BaseModel):
    graphs: List[str]


# ------------------------------------------------------------------ #
# Memory insertion
# ------------------------------------------------------------------ #

class TrajectoryStep(BaseModel):
    observation: str
    action: str


class SemanticMemoryInput(BaseModel):
    semantic_memory: str
    tags: List[str] = Field(default_factory=list)
    source: Optional[MemorySource] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class ProceduralMemoryInput(BaseModel):
    subgoal: str
    procedural_memory: str
    return_value: float = Field(0.0, alias="return")
    source: Optional[MemorySource] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class EpisodicStep(BaseModel):
    observation: str = ""
    action: str = ""
    subgoal: str = ""
    state: str = ""
    reward: str = ""
    time: Any = ""


class MemoryInsertRequest(BaseModel):
    mode: str = Field(
        ...,
        description='"trajectory" or "structured"',
        pattern="^(trajectory|structured)$",
    )

    # trajectory mode
    goal: Optional[str] = None
    steps: Optional[List[TrajectoryStep]] = None

    # structured mode
    episodic: Optional[List[List[EpisodicStep]]] = None
    semantic: Optional[List[SemanticMemoryInput]] = None
    procedural: Optional[List[ProceduralMemoryInput]] = None


class MemoryInsertResponse(BaseModel):
    status: str = "ok"
    stats: Dict[str, int] = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# Retrieval
# ------------------------------------------------------------------ #

class RetrieveRequest(BaseModel):
    observation: str
    goal: Optional[str] = None
    subgoal: Optional[str] = None
    state: Optional[str] = None
    task_type: str = ""
    time: str = ""
    mode: Optional[str] = Field(
        None,
        description=(
            'null (auto-detect), "semantic_memory", '
            '"episodic_memory", or "procedural_memory"'
        ),
    )
    min_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Exclude memories with confidence below this threshold.",
    )
    source_in: Optional[List[MemorySource]] = Field(
        None,
        description="Restrict recall to memories whose source is in this list.",
    )


class RetrieveResponse(BaseModel):
    mode: str
    reasoning_prompt: List[Dict[str, str]]
    variables: Dict[str, Any] = Field(default_factory=dict)


class ReasonRequest(BaseModel):
    observation: str
    goal: Optional[str] = None
    subgoal: Optional[str] = None
    state: Optional[str] = None
    task_type: str = ""
    time: str = ""
    mode: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_in: Optional[List[MemorySource]] = None


class ReasonResponse(BaseModel):
    mode: str
    reasoning: str
    reasoning_prompt: List[Dict[str, str]]


# ------------------------------------------------------------------ #
# Consolidation
# ------------------------------------------------------------------ #

class ConsolidateRequest(BaseModel):
    merge_threshold: float = 0.5
    max_merges_per_node: int = 1
    max_candidates_per_tag: int = 200
    max_total_candidates: int = 800
    min_credibility_to_keep_active: int = -10
    credibility_decay: int = 0
    only_update_recent_window: Optional[int] = None
    allow_merge_with_common_episodic_nodes: bool = False


class ConsolidateResponse(BaseModel):
    status: str = "ok"
    stats: Dict[str, int] = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# Promotion-gate extraction
# ------------------------------------------------------------------ #

CandidateKind = Literal["failure_delta", "correction"]


class CandidateInput(BaseModel):
    kind: CandidateKind
    window: str = Field(..., description="Text context for the candidate.")


class ExtractRequest(BaseModel):
    candidates: List[CandidateInput] = Field(default_factory=list)


class ExtractedMemory(BaseModel):
    type: Literal["semantic", "procedural"]
    semantic_memory: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    subgoal: Optional[str] = None
    procedural_memory: Optional[str] = None
    source: MemorySource
    confidence: float = Field(..., ge=0.0, le=1.0)


class ExtractResponse(BaseModel):
    memories: List[ExtractedMemory] = Field(default_factory=list)


# ------------------------------------------------------------------ #
# Stats / Nodes
# ------------------------------------------------------------------ #

class StatsResponse(BaseModel):
    graph_id: str
    stats: Dict[str, int]


class NodeListResponse(BaseModel):
    graph_id: str
    node_type: str
    count: int
    nodes: List[Dict[str, Any]]


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool
    embedding_available: bool
    chroma_available: bool
