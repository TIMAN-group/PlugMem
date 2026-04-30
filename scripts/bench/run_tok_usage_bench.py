"""Token-cost benchmark: PlugMem vs OpenClaw-default memory replay.

Emits JSONL to --out with one record per LLM/embedding call:

    {ts, trajectory_id, query_id, system, scope, phase,
     model, input_tokens, output_tokens, wall_ms, extra}

Scopes (see memory/feedback_bench_token_split.md):
    EXPOSED         tokens the agent's main LLM sees (hot path, user-billed)
    INTERNAL        tokens burned inside the memory service
                    (structuring, classify, reasoning-synthesis)
    INTERNAL_EMBED  embedding calls (different unit cost; logged separately)

Systems:
    A   OpenClaw-default replay: full transcript + query -> answer
    B   PlugMem trajectory-mode: structured on insert, recall via /reason
    C   PlugMem semantic-mode:   pre-distilled facts on insert, recall via /reason

Usage:
    uv run python scripts/bench/run_tok_usage_bench.py \
        --trajectories data/bench/trajectories.jsonl \
        --out           data/bench/results.jsonl \
        --systems       A,B,C \
        --retrievals    1,3,5,10,20
"""
from __future__ import annotations

import argparse
import contextvars
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from plugmem.clients.llm import LLMClient, OpenAICompatibleLLMClient
from plugmem.clients.embedding import EmbeddingClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scope-aware call logging
# ---------------------------------------------------------------------------

# Set at the boundary of each logical operation so nested client calls inherit
# the right tag without plumbing parameters everywhere.
_current_scope: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bench_scope", default="INTERNAL"
)
_current_phase: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bench_phase", default="unspecified"
)
_current_trajectory: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bench_trajectory", default=""
)
_current_query: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bench_query", default=""
)
_current_system: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bench_system", default=""
)


class BenchLogger:
    """Append-only JSONL writer. One record per LLM or embedding call."""

    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_path, "a", buffering=1)  # line-buffered

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        wall_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "ts": time.time(),
            "trajectory_id": _current_trajectory.get(),
            "query_id": _current_query.get(),
            "system": _current_system.get(),
            "scope": _current_scope.get(),
            "phase": _current_phase.get(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "wall_ms": wall_ms,
            "extra": extra or {},
        }
        self._fh.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        self._fh.close()


class scope:
    """Context manager: set scope + phase for all calls inside the block.

    Example:
        with scope("EXPOSED", "agent_answer"):
            answer = exposed_llm.complete(messages)
    """

    def __init__(self, scope_name: str, phase: str):
        self.scope_name = scope_name
        self.phase = phase
        self._tok_scope = None
        self._tok_phase = None

    def __enter__(self) -> "scope":
        self._tok_scope = _current_scope.set(self.scope_name)
        self._tok_phase = _current_phase.set(self.phase)
        return self

    def __exit__(self, *exc) -> None:
        _current_scope.reset(self._tok_scope)
        _current_phase.reset(self._tok_phase)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Estimate tokens. Prefer tiktoken if installed; fall back to len/4."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def _count_messages(messages: List[Dict[str, str]]) -> int:
    return sum(_count_tokens(m.get("content", "")) for m in messages)


# ---------------------------------------------------------------------------
# Instrumented LLM client
# ---------------------------------------------------------------------------

class InstrumentedLLMClient(LLMClient):
    """Decorates any LLMClient with per-call scope-tagged logging.

    All calls are logged against the *current* contextvars (scope/phase/etc).
    Use `with scope("EXPOSED", "agent_answer"):` to tag a region.
    """

    def __init__(self, inner: LLMClient, bench_logger: BenchLogger, model: str):
        self._inner = inner
        self._logger = bench_logger
        self._model = model

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        t0 = time.perf_counter()
        out = self._inner.complete(messages, temperature, top_p, max_tokens)
        wall_ms = (time.perf_counter() - t0) * 1000
        self._logger.record(
            model=self._model,
            input_tokens=_count_messages(messages),
            output_tokens=_count_tokens(out),
            wall_ms=wall_ms,
        )
        return out


class InstrumentedEmbeddingClient(EmbeddingClient):
    """Same idea for embeddings. Always tagged INTERNAL_EMBED regardless of outer scope."""

    def __init__(self, inner: EmbeddingClient, bench_logger: BenchLogger, model: str):
        self._inner = inner
        self._logger = bench_logger
        self._model = model

    def embed(self, text: str) -> List[float]:
        t0 = time.perf_counter()
        # Force scope to INTERNAL_EMBED without disturbing the outer phase label.
        tok = _current_scope.set("INTERNAL_EMBED")
        try:
            out = self._inner.embed(text)
        finally:
            _current_scope.reset(tok)
        wall_ms = (time.perf_counter() - t0) * 1000
        self._logger.record(
            model=self._model,
            input_tokens=_count_tokens(text),
            output_tokens=0,
            wall_ms=wall_ms,
            extra={"dim": len(out) if out else 0},
        )
        return out


# ---------------------------------------------------------------------------
# Trajectory / query schema
# ---------------------------------------------------------------------------

@dataclass
class Step:
    observation: str
    action: str


@dataclass
class Query:
    query_id: str
    question: str
    gold: str


@dataclass
class Trajectory:
    trajectory_id: str
    goal: str
    steps: List[Step]
    queries: List[Query] = field(default_factory=list)
    # Pre-distilled one-line facts for system C (semantic-mode). One per step.
    semantic_facts: List[str] = field(default_factory=list)


def load_trajectories(path: Path) -> List[Trajectory]:
    """Load trajectories from a JSONL file produced by generate_trajectories.py."""
    out: List[Trajectory] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(
                Trajectory(
                    trajectory_id=d["trajectory_id"],
                    goal=d.get("goal", ""),
                    steps=[Step(**s) for s in d.get("steps", [])],
                    queries=[Query(**q) for q in d.get("queries", [])],
                    semantic_facts=d.get("semantic_facts", []),
                )
            )
    return out


# ---------------------------------------------------------------------------
# System runners
# ---------------------------------------------------------------------------

def _render_transcript(traj: Trajectory) -> str:
    lines = [f"Goal: {traj.goal}"]
    for i, s in enumerate(traj.steps):
        lines.append(f"[{i}] observation: {s.observation}")
        lines.append(f"[{i}] action: {s.action}")
    return "\n".join(lines)


def run_system_a(
    traj: Trajectory,
    queries: List[Query],
    exposed_llm: InstrumentedLLMClient,
) -> List[Tuple[str, str]]:
    """OpenClaw-default replay. Full transcript + query -> answer on every call.

    Every call is EXPOSED because memory == prompt.
    Returns [(query_id, answer), ...] for downstream judging.
    """
    transcript = _render_transcript(traj)
    answers: List[Tuple[str, str]] = []
    for q in queries:
        _current_query.set(q.query_id)
        with scope("EXPOSED", "baseline_replay"):
            ans = exposed_llm.complete(
                messages=[
                    {"role": "system", "content": "Answer using the session transcript."},
                    {"role": "user", "content": f"{transcript}\n\nQuestion: {q.question}"},
                ]
            )
        answers.append((q.query_id, ans))
    return answers


def run_system_b(
    traj: Trajectory,
    queries: List[Query],
    graph,  # plugmem.core.memory_graph.MemoryGraph
    exposed_llm: InstrumentedLLMClient,
) -> List[Tuple[str, str]]:
    """PlugMem trajectory-mode. Pay structuring once, then /reason per query.

    Insert path is INTERNAL (structuring). Retrieval path runs retrieve_memory
    (INTERNAL: classify + reasoning-prompt build), then a compact message goes
    to the agent's main LLM -> that answer is EXPOSED.
    Returns [(query_id, answer), ...] for downstream judging.
    """
    # --- INSERT (structuring) ------------------------------------------------
    from plugmem.core.memory import Memory

    with scope("INTERNAL", "structuring"):
        mem = Memory(
            goal=traj.goal,
            observation=traj.steps[0].observation,
            llm=graph.llm,           # instrumented
            embedder=graph.embedder, # instrumented
            time=graph.semantic_time,
        )
        for s in traj.steps:
            mem.append(action_t0=s.action, observation_t1=s.observation)
        mem.close()
        graph.insert(mem)

    # --- RETRIEVE + REASON ---------------------------------------------------
    answers: List[Tuple[str, str]] = []
    for q in queries:
        _current_query.set(q.query_id)

        # retrieve_memory internally runs get_plan / get_mode via graph.llm.
        with scope("INTERNAL", "classify"):
            messages, _vars, _mode = graph.retrieve_memory(
                goal=traj.goal,
                observation=q.question,
                task_type="",
                time="",
                mode=None,
            )

        # The reasoning-synthesis call happens on graph.llm — INTERNAL.
        with scope("INTERNAL", "reason_synthesis"):
            recall = graph.llm.complete(messages=messages)

        # The compact recall result is what the agent's main LLM sees.
        with scope("EXPOSED", "agent_answer"):
            ans = exposed_llm.complete(
                messages=[
                    {"role": "system", "content": "Answer using the retrieved memory."},
                    {"role": "user", "content": f"Memory:\n{recall}\n\nQuestion: {q.question}"},
                ]
            )
        answers.append((q.query_id, ans))
    return answers


def run_system_c(
    traj: Trajectory,
    queries: List[Query],
    graph,
    exposed_llm: InstrumentedLLMClient,
) -> List[Tuple[str, str]]:
    """PlugMem semantic-mode. Pre-distilled facts inserted directly — no structuring.

    Embeddings are still INTERNAL_EMBED. No structuring LLM calls on insert.
    Returns [(query_id, answer), ...] for downstream judging.
    """
    from plugmem.core.memory import Memory

    if not traj.semantic_facts:
        raise ValueError(f"Trajectory {traj.trajectory_id} has no semantic_facts for system C")

    # --- INSERT (pre-structured) --------------------------------------------
    with scope("INTERNAL", "semantic_insert"):
        mem = Memory.__new__(Memory)
        mem.time = graph.semantic_time
        mem.llm = graph.llm
        mem.embedder = graph.embedder
        mem.memory = {"goal": traj.goal, "episodic": [], "semantic": [], "procedural": []}
        mem.memory_embedding = {"semantic": [], "procedural": []}
        for fact in traj.semantic_facts:
            mem.memory["semantic"].append({"semantic_memory": fact, "tags": []})
            mem.memory_embedding["semantic"].append({
                "semantic_memory": graph.embedder.embed(fact),
                "tags": [],
            })
        graph.insert(mem)

    # --- RETRIEVE + REASON (same as B) --------------------------------------
    answers: List[Tuple[str, str]] = []
    for q in queries:
        _current_query.set(q.query_id)
        with scope("INTERNAL", "classify"):
            messages, _vars, _mode = graph.retrieve_memory(
                goal=traj.goal,
                observation=q.question,
                task_type="",
                time="",
                mode=None,
            )
        with scope("INTERNAL", "reason_synthesis"):
            recall = graph.llm.complete(messages=messages)
        with scope("EXPOSED", "agent_answer"):
            ans = exposed_llm.complete(
                messages=[
                    {"role": "system", "content": "Answer using the retrieved memory."},
                    {"role": "user", "content": f"Memory:\n{recall}\n\nQuestion: {q.question}"},
                ]
            )
        answers.append((q.query_id, ans))
    return answers


# ---------------------------------------------------------------------------
# Answer-quality judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict grader. Given a question, a gold answer, and a candidate "
    "answer, output a single integer 0-5:\n"
    "  5 = fully correct & complete\n"
    "  3 = partially correct (key fact present but incomplete or imprecise)\n"
    "  1 = mostly wrong but on-topic\n"
    "  0 = wrong, off-topic, or refusal\n"
    "Output ONLY the integer. No prose."
)


def _parse_judge_score(raw: str) -> int:
    """Extract 0-5 integer from judge output. Returns -1 if unparseable."""
    s = (raw or "").strip()
    for tok in s.replace(",", " ").split():
        try:
            n = int(tok)
        except ValueError:
            continue
        if 0 <= n <= 5:
            return n
    return -1


def judge_answers(
    traj: Trajectory,
    answers: List[Tuple[str, str]],
    judge_llm: InstrumentedLLMClient,
    bench_logger: BenchLogger,
) -> None:
    """Score each (query_id, answer) against the gold and emit EVAL records.

    Two records per query:
      1. The judge LLM call itself, tagged scope=EVAL phase=judge_call (token cost).
      2. A zero-token EVAL record carrying the parsed score, the answer, and
         the gold — phase=judge_score. Cost analysis filters EVAL out; quality
         analysis reads the judge_score records.
    """
    gold_by_qid = {q.query_id: (q.question, q.gold) for q in traj.queries}
    for qid, ans in answers:
        if qid not in gold_by_qid:
            continue
        question, gold = gold_by_qid[qid]
        _current_query.set(qid)

        with scope("EVAL", "judge_call"):
            raw = judge_llm.complete(
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Gold: {gold}\n"
                            f"Candidate: {ans}\n\n"
                            "Score:"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=4,
            )
        score = _parse_judge_score(raw)

        # Zero-cost record carrying the score + raw strings for downstream eval.
        with scope("EVAL", "judge_score"):
            bench_logger.record(
                model="judge",
                input_tokens=0,
                output_tokens=0,
                wall_ms=0.0,
                extra={
                    "judge_score": score,
                    "judge_raw": raw,
                    "answer": ans,
                    "gold": gold,
                    "question": question,
                },
            )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def build_clients(bench_logger: BenchLogger):
    """Build exposed and internal LLM clients, plus an embedding client.

    Separating exposed from internal lets the two scopes run on different
    models. Configure via env:
        EXPOSED_LLM_MODEL / EXPOSED_LLM_BASE_URL / EXPOSED_LLM_API_KEY
        LLM_MODEL         / LLM_BASE_URL         / LLM_API_KEY          (internal)
    """
    internal = OpenAICompatibleLLMClient(
        base_url=os.environ.get("LLM_BASE_URL", ""),
        api_key=os.environ.get("LLM_API_KEY", ""),
        model=os.environ.get("LLM_MODEL", ""),
    )
    exposed = OpenAICompatibleLLMClient(
        base_url=os.environ.get("EXPOSED_LLM_BASE_URL", os.environ.get("LLM_BASE_URL", "")),
        api_key=os.environ.get("EXPOSED_LLM_API_KEY", os.environ.get("LLM_API_KEY", "")),
        model=os.environ.get("EXPOSED_LLM_MODEL", os.environ.get("LLM_MODEL", "")),
    )

    internal_instr = InstrumentedLLMClient(internal, bench_logger, model=internal.model)
    exposed_instr = InstrumentedLLMClient(exposed, bench_logger, model=exposed.model)

    from plugmem.clients.embedding import HTTPEmbeddingClient
    embedder = HTTPEmbeddingClient(
        base_url=os.environ.get("EMBEDDING_BASE_URL", ""),
        model=os.environ.get("EMBEDDING_MODEL", "nvidia/NV-Embed-v2"),
    )
    embed_instr = InstrumentedEmbeddingClient(
        embedder, bench_logger, model=embedder.model if hasattr(embedder, "model") else "embed"
    )

    return internal_instr, exposed_instr, embed_instr


def fresh_graph(internal_llm, embedder):
    """Build a cold ephemeral MemoryGraph. One per (trajectory, system) run."""
    import chromadb
    from plugmem.clients.embedding import PlugMemEmbeddingFunction
    from plugmem.storage.chroma import ChromaStorage
    from plugmem.graph_manager import GraphManager

    client = chromadb.EphemeralClient()
    storage = ChromaStorage(
        client=client,
        embedding_function=PlugMemEmbeddingFunction(embedder),
        embedding_client=embedder,
    )
    gm = GraphManager(storage=storage, llm=internal_llm, embedder=embedder)
    graph_id = gm.create_graph(graph_id=f"bench_{uuid.uuid4().hex[:8]}")
    return gm.get_graph(graph_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--systems", default="A,B,C",
                    help="Comma-separated subset of {A,B,C}")
    ap.add_argument("--retrievals", default="1,3,5,10,20",
                    help="Comma-separated R values (subsample queries per trajectory)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Run only the first N trajectories (0 = all)")
    ap.add_argument("--judge", action="store_true",
                    help="Score answers with an LLM judge (adds EVAL-scoped records)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    r_values = sorted({int(x) for x in args.retrievals.split(",")})
    trajectories = load_trajectories(args.trajectories)
    if args.limit:
        trajectories = trajectories[: args.limit]
    logger.info("loaded %d trajectories; systems=%s; R=%s",
                len(trajectories), systems, r_values)

    bench_logger = BenchLogger(args.out)
    try:
        internal_llm, exposed_llm, embedder = build_clients(bench_logger)
        # Judge reuses the internal client by default — instrumented, so its
        # tokens get logged under scope=EVAL and excluded from cost curves.
        judge_llm = internal_llm

        for traj in trajectories:
            _current_trajectory.set(traj.trajectory_id)
            # For each R, subsample queries deterministically (prefix).
            for r in r_values:
                queries = traj.queries[:r]
                if not queries:
                    continue
                for system in systems:
                    _current_system.set(f"{system}@R={r}")
                    logger.info("traj=%s R=%d system=%s", traj.trajectory_id, r, system)
                    if system == "A":
                        answers = run_system_a(traj, queries, exposed_llm)
                    elif system == "B":
                        graph = fresh_graph(internal_llm, embedder)
                        answers = run_system_b(traj, queries, graph, exposed_llm)
                    elif system == "C":
                        graph = fresh_graph(internal_llm, embedder)
                        answers = run_system_c(traj, queries, graph, exposed_llm)
                    else:
                        raise ValueError(f"Unknown system: {system}")

                    if args.judge:
                        judge_answers(traj, answers, judge_llm, bench_logger)
    finally:
        bench_logger.close()


if __name__ == "__main__":
    main()
