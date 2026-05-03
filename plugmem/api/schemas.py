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
    session_id: Optional[str] = Field(
        None,
        description=(
            "Stamps every node created by this insert with the given session id. "
            "Used by the Sessions view + recall audit log to group nodes by run."
        ),
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
    session_id: Optional[str] = Field(
        None,
        description="If set, the recall is logged against this session id.",
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
    session_id: Optional[str] = Field(
        None,
        description="If set, the reasoning recall is logged against this session id.",
    )


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
# Inspector
# ------------------------------------------------------------------ #

class SearchResponse(BaseModel):
    graph_id: str
    node_type: str
    query: str
    count: int
    nodes: List[Dict[str, Any]]


class NodeDetailResponse(BaseModel):
    graph_id: str
    node_type: str
    node: Dict[str, Any]
    edges: Dict[str, List[Dict[str, Any]]]


class SemanticUpdateRequest(BaseModel):
    is_active: Optional[bool] = None


class RecallTraceRequest(BaseModel):
    observation: str
    goal: Optional[str] = None
    subgoal: Optional[str] = None
    state: Optional[str] = None
    task_type: str = ""
    time: str = ""
    mode: Optional[str] = Field(
        None,
        description=(
            'null/omit (default = semantic_memory unless auto_plan), '
            '"semantic_memory", "episodic_memory", or "procedural_memory"'
        ),
    )
    query_tags: Optional[List[str]] = Field(
        None,
        description="Manual tags to skip the LLM planner. Empty list disables tag voting.",
    )
    next_subgoal: Optional[str] = None
    auto_plan: bool = Field(
        False,
        description="If True, fill missing mode/tags/subgoal via the LLM planner (paid).",
    )
    session_id: Optional[str] = Field(
        None,
        description="If set, the trace is logged to the recall audit under this session id.",
    )


class RecallTraceResponse(BaseModel):
    mode: str
    plan: Dict[str, Any]
    trace: Dict[str, Any]
    selected: Dict[str, List[int]]
    rendered_prompt: List[Dict[str, str]]


class TopologyResponse(BaseModel):
    graph_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    counts: Dict[str, int]
    truncated: bool
    node_limit: int


class RecallAuditEntry(BaseModel):
    recall_id: int
    endpoint: str
    ts: str
    graph_time: int = 0
    session_id: Optional[str] = None
    observation: str = ""
    goal: str = ""
    subgoal: str = ""
    state: str = ""
    task_type: str = ""
    mode: str = ""
    next_subgoal: str = ""
    query_tags: List[str] = Field(default_factory=list)
    selected_semantic_ids: List[int] = Field(default_factory=list)
    selected_procedural_ids: List[int] = Field(default_factory=list)
    n_messages: int = 0


class RecallListResponse(BaseModel):
    graph_id: str
    count: int
    session_id: Optional[str] = None
    recalls: List[RecallAuditEntry]


class SessionListResponse(BaseModel):
    graph_id: str
    sessions: List[str]


class SessionEvent(BaseModel):
    """One row in the chronological session view.

    Two flavours: ``kind="insert"`` (a node was created) and
    ``kind="recall"`` (a /retrieve, /reason, or /recall_trace fired).
    Both carry ``time`` so the frontend sorts on a unified axis; nodes
    use the graph's monotonic time counter, recalls use ``graph_time``
    captured when the recall was logged.
    """
    kind: str
    time: int
    # insert fields
    node_type: Optional[str] = None
    node_id: Optional[int] = None
    label: Optional[str] = None
    text: Optional[str] = None
    is_active: Optional[bool] = None
    credibility: Optional[int] = None
    return_value: Optional[float] = None
    subgoal: Optional[str] = None
    # recall fields
    endpoint: Optional[str] = None
    recall_id: Optional[int] = None
    ts: Optional[str] = None
    observation: Optional[str] = None
    mode: Optional[str] = None
    next_subgoal: Optional[str] = None
    query_tags: List[str] = Field(default_factory=list)
    selected_semantic_ids: List[int] = Field(default_factory=list)
    selected_procedural_ids: List[int] = Field(default_factory=list)
    n_messages: Optional[int] = None


class SessionTimelineResponse(BaseModel):
    graph_id: str
    session_id: str
    count: int
    events: List[SessionEvent]


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool
    embedding_available: bool
    chroma_available: bool
