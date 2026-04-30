"""Generate synthetic OpenClaw-style trajectories for the token-cost bench.

Each output line is one trajectory in the schema consumed by
``run_tok_usage_bench.py``::

    {
      "trajectory_id": "...",
      "goal": "...",
      "steps":           [{"observation": "...", "action": "..."}, ...],
      "queries":         [{"query_id": "...", "question": "...", "gold": "..."}, ...],
      "semantic_facts":  ["one-line fact per step", ...]   # for system C
    }

Variables (cells of the bench grid):
    --length    L   trajectory step count   (e.g. 5,20,50,100)
    --diversity D   "single" or "mixed"     (mixed interleaves 3 topics)
    --queries   K   queries per trajectory  (must be >= max R you plan to use)
    --count     N   trajectories per (L,D) cell
    --seed      S   deterministic seed for the generator LLM

Usage:
    uv run python scripts/bench/generate_trajectories.py \
        --out data/bench/trajectories.jsonl \
        --length 20,50,100 --diversity single,mixed \
        --queries 20 --count 40 --seed 42
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from plugmem.clients.llm import OpenAICompatibleLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema (mirrors run_tok_usage_bench.Trajectory)
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
    semantic_facts: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Topic seeds — change these to broaden domain coverage
# ---------------------------------------------------------------------------

TOPICS_SINGLE = [
    "buying a desk lamp on an e-commerce site",
    "filing a tax extension on a government portal",
    "comparing cloud GPU pricing across three vendors",
    "booking a multi-leg train ticket through Europe",
    "debugging a CI pipeline failure on GitHub Actions",
    "researching scholarship deadlines for a CS PhD application",
]

TOPICS_MIXED_BUNDLES = [
    # Each bundle = 3 interleaved topics within one session
    [
        "checking flight status",
        "rescheduling a hotel booking",
        "messaging a colleague about a project deadline",
    ],
    [
        "comparing two open-source databases",
        "drafting a budget approval email",
        "looking up a coworker's PTO calendar",
    ],
]


# ---------------------------------------------------------------------------
# Prompt (one LLM call per trajectory)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are generating a synthetic OpenClaw-style web-agent session for a research benchmark.

Output STRICT JSON only — no prose, no code fences. Schema:

{
  "goal": "<one-line user goal>",
  "steps": [
    {"observation": "<page state, ~1-3 sentences>", "action": "<agent action like CLICK[#submit] or TYPE[input#q,'foo']>"},
    ...
  ],
  "semantic_facts": ["<one-line factual atom per step, third person, self-contained>", ...],
  "queries": [
    {"question": "<question whose answer requires a SPECIFIC step>", "gold": "<short ground-truth answer>"},
    ...
  ]
}

Hard requirements:
- len(steps) == {length}
- len(semantic_facts) == len(steps)
- len(queries) == {queries}
- Every gold answer must be derivable from one or more steps. Do not invent facts outside the trajectory.
- {diversity_clause}
- Mix fact-recall queries (single-step) and procedure-recall queries (multi-step).
- Keep each observation under 200 characters; keep each action under 80 characters.
"""

_DIVERSITY_SINGLE = "All steps stay on a single topic / site / coherent goal."
_DIVERSITY_MIXED = (
    "Interleave 3 distinct sub-tasks within the session so context-switching is required; "
    "label each step's topic implicitly via the observation."
)


def _build_prompt(length: int, queries: int, diversity: str, topic: str) -> List[Dict[str, str]]:
    diversity_clause = _DIVERSITY_SINGLE if diversity == "single" else _DIVERSITY_MIXED
    sys = _SYSTEM_PROMPT.format(
        length=length, queries=queries, diversity_clause=diversity_clause
    )
    user = f"Generate one trajectory. Topic seed: {topic}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _make_llm() -> OpenAICompatibleLLMClient:
    """Build the generator LLM. Reuse the project's standard env vars."""
    return OpenAICompatibleLLMClient(
        base_url=os.environ.get("LLM_BASE_URL", ""),
        api_key=os.environ.get("LLM_API_KEY", ""),
        model=os.environ.get("LLM_MODEL", ""),
    )


def _parse_trajectory_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parse. Returns None on failure."""
    s = raw.strip()
    if s.startswith("```"):
        # strip markdown fences if the model added them
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to recover: locate the first `{` and the last `}`.
        l, r = s.find("{"), s.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(s[l : r + 1])
            except json.JSONDecodeError:
                return None
    return None


def _validate(d: Dict[str, Any], length: int, queries: int) -> Optional[str]:
    """Return error message or None if OK."""
    if "steps" not in d or len(d["steps"]) != length:
        return f"steps len {len(d.get('steps', []))} != {length}"
    if "semantic_facts" not in d or len(d["semantic_facts"]) != length:
        return f"semantic_facts len {len(d.get('semantic_facts', []))} != {length}"
    if "queries" not in d or len(d["queries"]) != queries:
        return f"queries len {len(d.get('queries', []))} != {queries}"
    for s in d["steps"]:
        if "observation" not in s or "action" not in s:
            return "step missing observation/action"
    for q in d["queries"]:
        if "question" not in q or "gold" not in q:
            return "query missing question/gold"
    return None


def generate_one(
    llm,
    length: int,
    queries: int,
    diversity: str,
    topic: str,
    max_attempts: int = 3,
) -> Optional[Trajectory]:
    messages = _build_prompt(length, queries, diversity, topic)
    for attempt in range(max_attempts):
        raw = llm.complete(messages, temperature=0.7, max_tokens=4096)
        parsed = _parse_trajectory_json(raw)
        if parsed is None:
            logger.warning("attempt %d: JSON parse failed", attempt + 1)
            continue
        err = _validate(parsed, length, queries)
        if err:
            logger.warning("attempt %d: validation failed (%s)", attempt + 1, err)
            continue

        traj = Trajectory(
            trajectory_id=f"L{length}-{diversity}-{uuid.uuid4().hex[:8]}",
            goal=parsed.get("goal", ""),
            steps=[Step(**s) for s in parsed["steps"]],
            semantic_facts=list(parsed["semantic_facts"]),
            queries=[
                Query(query_id=f"q{i}", question=q["question"], gold=q["gold"])
                for i, q in enumerate(parsed["queries"])
            ],
            meta={"diversity": diversity, "topic": topic, "length": length},
        )
        return traj
    logger.error("giving up after %d attempts (L=%d D=%s topic=%r)",
                 max_attempts, length, diversity, topic)
    return None


def _write(traj: Trajectory, fh) -> None:
    d = asdict(traj)
    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--length", default="20,50,100",
                    help="Comma-separated step counts L")
    ap.add_argument("--diversity", default="single,mixed",
                    help="Comma-separated diversity values: single|mixed")
    ap.add_argument("--queries", type=int, default=20,
                    help="Queries per trajectory (>= max R)")
    ap.add_argument("--count", type=int, default=10,
                    help="Trajectories per (L,D) cell")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)

    lengths = [int(x) for x in args.length.split(",")]
    diversities = [x.strip() for x in args.diversity.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    llm = _make_llm()
    n_total = 0
    n_failed = 0
    with open(args.out, "a") as fh:
        for L, D in itertools.product(lengths, diversities):
            topics = TOPICS_SINGLE if D == "single" else [
                " | ".join(b) for b in TOPICS_MIXED_BUNDLES
            ]
            for i in range(args.count):
                topic = random.choice(topics)
                logger.info("generating L=%d D=%s i=%d/%d topic=%s",
                            L, D, i + 1, args.count, topic)
                traj = generate_one(llm, length=L, queries=args.queries,
                                    diversity=D, topic=topic)
                if traj is None:
                    n_failed += 1
                    continue
                _write(traj, fh)
                n_total += 1

    logger.info("wrote %d trajectories (failed %d) -> %s", n_total, n_failed, args.out)


if __name__ == "__main__":
    main()
