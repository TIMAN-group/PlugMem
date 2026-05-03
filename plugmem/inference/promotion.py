"""Coding-agent promotion-gate extractor.

Given a list of detected promotion candidates (failure→success deltas,
user corrections), prompt an LLM to emit 0–N structured memory nodes
with `source` / `confidence` metadata.

The extractor is intentionally conservative: when in doubt, return
fewer memories. Bad memories actively harm coding agents (a stale
"always use X" rule leads to wrong code), so the prompt biases toward
*omitting* uncertain extractions.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from plugmem.clients.llm import LLMClient, with_phase

logger = logging.getLogger(__name__)


CandidateKind = Literal["failure_delta", "correction"]


_SYSTEM_PROMPT = """You extract durable, reusable memory nodes from coding-agent session signals.

You receive a list of CANDIDATES, each a short context window from a session that may indicate something worth remembering across future sessions. Possible kinds:

- failure_delta: a tool call failed, then a follow-up succeeded. The fix may be a debugging recipe (procedural) or a configuration fact (semantic).
- correction: the user explicitly corrected the agent ("don't", "stop", "actually"). The rule is a semantic fact about preferences/conventions.

For each candidate, decide whether to emit 0, 1, or rarely 2 memory nodes. Be conservative — bad memories actively harm. Emit nothing if:
- The fix was trivial (typo, syntax error)
- The correction was ambiguous or session-specific
- You're not sure the rule generalizes beyond this session

Output JSON only, no prose. Schema:

[
  {
    "type": "semantic",
    "semantic_memory": "short factual statement",
    "tags": ["tag1", "tag2"],
    "source": "correction" | "failure_delta",
    "confidence": 0.0-1.0
  },
  {
    "type": "procedural",
    "subgoal": "what was being attempted",
    "procedural_memory": "the steps that worked",
    "source": "failure_delta",
    "confidence": 0.0-1.0
  }
]

Confidence guide: 0.9+ for explicit user rules, 0.7-0.8 for clear failure→success patterns, 0.5-0.6 for plausible but ambiguous. Below 0.5: don't emit at all.

Return [] if no candidate warrants a memory."""


def extract_coding_memories(
    llm: LLMClient,
    candidates: List[Dict[str, str]],
    *,
    max_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    """Run the promotion-gate LLM on a candidate list.

    `candidates` is a list of `{"kind": ..., "window": ...}` dicts. Returns
    a list of structured memory dicts shaped for downstream insert. On LLM
    failure or unparseable output, returns an empty list rather than
    raising — the caller treats "no candidates worth promoting" the same
    as "extractor failed."
    """
    if not candidates:
        return []

    user_msg = _format_candidates(candidates)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    with with_phase("extract"):
        raw = llm.complete(messages=messages, temperature=0, max_tokens=max_tokens)
    if not raw:
        logger.info("promotion extractor: empty LLM response")
        return []

    parsed = _parse_json_array(raw)
    if parsed is None:
        logger.info("promotion extractor: unparseable LLM output: %r", raw[:200])
        return []

    return [m for m in parsed if _is_valid_memory(m)]


def _format_candidates(candidates: List[Dict[str, str]]) -> str:
    out = ["CANDIDATES:"]
    for i, c in enumerate(candidates, 1):
        kind = c.get("kind", "unknown")
        window = c.get("window", "")
        out.append(f"\n--- Candidate {i} (kind: {kind}) ---\n{window}")
    return "\n".join(out)


def _parse_json_array(raw: str) -> Optional[List[Any]]:
    # Strip code fences if the model wrapped output in ```json ... ```
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


_VALID_SOURCES = {"failure_delta", "correction", "merged", "repeated_lookup", "explicit"}


def _is_valid_memory(m: Any) -> bool:
    if not isinstance(m, dict):
        return False
    mtype = m.get("type")
    source = m.get("source")
    confidence = m.get("confidence")
    if source not in _VALID_SOURCES:
        return False
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        return False
    if mtype == "semantic":
        return isinstance(m.get("semantic_memory"), str) and m["semantic_memory"].strip() != ""
    if mtype == "procedural":
        return (
            isinstance(m.get("subgoal"), str)
            and m["subgoal"].strip() != ""
            and isinstance(m.get("procedural_memory"), str)
            and m["procedural_memory"].strip() != ""
        )
    return False
