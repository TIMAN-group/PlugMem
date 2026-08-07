"""
PlugMem Memory Provider for Hermes Agent.

Persistent cross-session memory via a PlugMem knowledge graph service.
Registers four tools: plugmem_remember, plugmem_recall, plugmem_learn, plugmem_procedure.

Feature parity with PlugMem OpenClaw connector, plus Hermes-native lifecycle hooks
(prefetch, sync_turn, on_session_end, on_pre_compress, on_session_switch).

Installation:
    1. Start PlugMem service:  uvicorn plugmem.api.app:app --port 8080
    2. Symlink: ln -sf <repo>/hermes-plugmem-plugin/memory_plugmem
                ~/.hermes/hermes-agent/plugins/memory/plugmem
    3. Configure:  hermes config set memory.provider plugmem

Config (in $HERMES_HOME/.env):
    PLUGMEM_BASE_URL=http://localhost:8080
    PLUGMEM_DEFAULT_GRAPH_ID=hermes-default
    PLUGMEM_SHARED_GRAPH_IDS=team-graph,org-graph
    PLUGMEM_AUTO_REMEMBER_ENABLED=true
    PLUGMEM_AUTO_REMEMBER_MIN_STEPS=2
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_GRAPH_ID = "hermes-default"
CONSOLIDATE_EVERY_N_TURNS = 50
PREFETCH_MIN_QUERY_LEN = 8
PREFETCH_MAX_RETRIES = 2
PREFETCH_RETRY_DELAY = 0.5


def _env_bool(key: str, default: bool = True) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Lightweight REST client (stdlib only — no extra deps)
# ---------------------------------------------------------------------------


class PlugMemError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class PlugMemClient:
    """Minimal HTTP client for the PlugMem REST API."""

    _API_PREFIX = "/api/v1"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = "", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self, method: str, path: str, body: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 204:
                    return {}
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body_text).get("detail", body_text)
            except json.JSONDecodeError:
                detail = body_text
            raise PlugMemError(str(detail), e.code) from e
        except Exception as e:
            raise PlugMemError(str(e)) from e

    # -- Health / graphs --

    def health(self) -> Dict[str, Any]:
        return self._request("GET", f"{self._API_PREFIX}/health")

    def ensure_graph(self, graph_id: str) -> Dict[str, Any]:
        try:
            return self._request("GET", f"{self._API_PREFIX}/graphs/{graph_id}/stats")
        except PlugMemError:
            return self._request("POST", f"{self._API_PREFIX}/graphs", {"graph_id": graph_id})

    def get_stats(self, graph_id: str) -> Dict[str, Any]:
        return self._request("GET", f"{self._API_PREFIX}/graphs/{graph_id}/stats")

    # -- Memory --

    def insert_structured(
        self, graph_id: str, semantic: List[Dict], **extra
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"mode": "structured", "semantic": semantic}
        body.update(extra)
        return self._request(
            "POST", f"{self._API_PREFIX}/graphs/{graph_id}/memories", body
        )

    def insert_trajectory(
        self,
        graph_id: str,
        goal: str,
        steps: List[Dict[str, str]],
        session_id: str = "",
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"mode": "trajectory", "goal": goal, "steps": steps}
        if session_id:
            body["session_id"] = session_id
        return self._request(
            "POST", f"{self._API_PREFIX}/graphs/{graph_id}/memories", body
        )

    # -- Retrieval --

    def reason(
        self, graph_id: str, observation: str, mode: str = "", **kwargs
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"observation": observation}
        if mode:
            body["mode"] = mode
        body.update(kwargs)
        return self._request(
            "POST", f"{self._API_PREFIX}/graphs/{graph_id}/reason", body
        )

    def retrieve(
        self, graph_id: str, observation: str, mode: str = "", **kwargs
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"observation": observation}
        if mode:
            body["mode"] = mode
        body.update(kwargs)
        return self._request(
            "POST", f"{self._API_PREFIX}/graphs/{graph_id}/retrieve", body
        )

    # -- Consolidation --

    def consolidate(self, graph_id: str) -> Dict[str, Any]:
        return self._request(
            "POST", f"{self._API_PREFIX}/graphs/{graph_id}/consolidate", {}
        )


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_GRAPH_ID_PARAM = {
    "type": "string",
    "description": (
        "Memory graph to target (defaults to the configured default graph). "
        "Use to read/write a specific project or team graph."
    ),
}

_SESSION_ID_PARAM = {
    "type": "string",
    "description": "Session ID for grouping (auto-attached if omitted).",
}

REMEMBER_SCHEMA = {
    "name": "plugmem_remember",
    "description": (
        "Store information in long-term PlugMem knowledge graph memory. "
        "Use for durable facts, user preferences, decisions, or observation/action "
        "trajectories. PlugMem structures input into semantic facts, procedural "
        "knowledge, and episodic traces that persist across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Free-text fact, preference, or knowledge to remember.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional category tags (e.g. ['project', 'preference']).",
            },
            "goal": {
                "type": "string",
                "description": "Task goal — required for trajectory mode.",
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "observation": {"type": "string"},
                        "action": {"type": "string"},
                    },
                    "required": ["observation", "action"],
                },
                "description": "Observation/action pairs (with 'goal').",
            },
            "graph_id": _GRAPH_ID_PARAM,
            "session_id": _SESSION_ID_PARAM,
        },
    },
}

RECALL_SCHEMA = {
    "name": "plugmem_recall",
    "description": (
        "Recall relevant memories from PlugMem's knowledge graph. Returns "
        "LLM-synthesized reasoning over the most relevant semantic, procedural, "
        "and episodic memories. Queries all configured graphs (personal + shared). "
        "Use when you need past context, facts, decisions, or procedures."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "observation": {
                "type": "string",
                "description": "What do you need to recall?",
            },
            "mode": {
                "type": "string",
                "enum": ["semantic_memory", "episodic_memory", "procedural_memory"],
                "description": "Memory type. Omit to auto-detect.",
            },
            "raw": {
                "type": "boolean",
                "description": (
                    "If true, return the raw retrieval prompt instead of LLM reasoning. "
                    "Useful for debugging or when the base agent wants to reason directly."
                ),
                "default": False,
            },
            "graph_id": _GRAPH_ID_PARAM,
        },
        "required": ["observation"],
    },
}

LEARN_SCHEMA = {
    "name": "plugmem_learn",
    "description": (
        "Store a reusable procedure / how-to in PlugMem. "
        "Use when you discover a workflow that should be repeatable — deployment "
        "steps, debugging sequences, setup instructions, build commands. "
        "PlugMem structures these into procedural memory for future recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Procedure name (e.g. 'Deploy to VPS').",
            },
            "description": {
                "type": "string",
                "description": "What it accomplishes and when to use it.",
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "observation": {"type": "string"},
                        "action": {"type": "string"},
                    },
                    "required": ["observation", "action"],
                },
                "description": "Ordered observation/action pairs.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags (e.g. ['deployment', 'testing']).",
            },
            "graph_id": _GRAPH_ID_PARAM,
        },
        "required": ["title", "steps"],
    },
}

PROCEDURE_SCHEMA = {
    "name": "plugmem_procedure",
    "description": (
        "Find a stored procedure/how-to from PlugMem. "
        "Returns the full procedure steps if found. "
        "Use before performing a task — check for a stored workflow."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What procedure? (e.g. 'deploy', 'run tests')",
            },
            "graph_id": _GRAPH_ID_PARAM,
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_stats(stats: Dict[str, Any]) -> str:
    parts = []
    for k, v in stats.items():
        if v and isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
    return ", ".join(parts) if parts else "empty"


def _extract_text_content(content: Any) -> str:
    """Extract readable text from a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    parts.append("[image]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", "")))
    return str(content) if content else ""


def _messages_to_steps(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Convert agent messages to observation/action steps."""
    steps: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "")
        text = _extract_text_content(msg.get("content", ""))
        if not text:
            # Check for tool calls in assistant messages
            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                    text = f"[tool calls: {', '.join(names)}]"
            if not text:
                continue
        if role == "user":
            steps.append({"observation": text, "action": ""})
        elif role == "assistant":
            if steps and not steps[-1].get("action"):
                steps[-1]["action"] = text
            else:
                steps.append({"observation": "", "action": text})
        elif role == "tool":
            # Tool results become observations for the next assistant action
            tool_name = msg.get("name", msg.get("tool_call_id", "tool"))
            result_text = _extract_text_content(msg.get("content", ""))
            label = f"[{tool_name} result]"
            obs = f"{label} {_truncate(result_text, 300)}" if result_text else label
            steps.append({"observation": obs, "action": ""})
    return steps



def _get_client() -> PlugMemClient:
    return PlugMemClient(
        base_url=os.environ.get("PLUGMEM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("PLUGMEM_API_KEY", ""),
        timeout=int(os.environ.get("PLUGMEM_TIMEOUT", "30")),
    )


def _is_empty_reasoning(result: Dict[str, Any]) -> bool:
    reasoning = result.get("reasoning", "")
    if not reasoning or not reasoning.strip():
        return True
    if "### Information\nnull" in reasoning:
        return True
    if "No relevant" in reasoning and "### Information" in reasoning:
        info_start = reasoning.rfind("### Information")
        if "null" in reasoning[info_start:][:200]:
            return True
    return False


def _merge_reasoning(results: List[Dict[str, Any]]) -> str:
    parts = []
    for r in results:
        if _is_empty_reasoning(r):
            continue
        graph = r.get("graph", "")
        prefix = f"[{graph}] " if graph else ""
        parts.append(f"{prefix}{r.get('reasoning', '')}")
    return "\n\n".join(parts) if parts else ""


def _retry_call(fn, max_retries: int = PREFETCH_MAX_RETRIES, delay: float = PREFETCH_RETRY_DELAY):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except PlugMemError as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------


class PlugMemMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by a PlugMem knowledge graph service."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._client: PlugMemClient | None = None
        self._graph_id = ""
        self._session_id = ""
        self._shared_graph_ids: List[str] = []
        self._turn_count = 0
        self._consolidated = False
        self._auto_remember_enabled = True
        self._auto_remember_min_steps = 2

    # -- Abstract methods -------------------------------------------------

    @property
    def name(self) -> str:
        return "plugmem"

    def is_available(self) -> bool:
        return bool(os.environ.get("PLUGMEM_BASE_URL", DEFAULT_BASE_URL))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._client = _get_client()
        self._graph_id = os.environ.get("PLUGMEM_DEFAULT_GRAPH_ID", DEFAULT_GRAPH_ID)
        self._session_id = session_id
        self._turn_count = 0
        self._consolidated = False
        self._auto_remember_enabled = _env_bool("PLUGMEM_AUTO_REMEMBER_ENABLED", True)
        self._auto_remember_min_steps = _env_int("PLUGMEM_AUTO_REMEMBER_MIN_STEPS", 2)

        shared_raw = os.environ.get("PLUGMEM_SHARED_GRAPH_IDS", "")
        self._shared_graph_ids = (
            [g.strip() for g in shared_raw.split(",") if g.strip()]
            if shared_raw
            else []
        )

        try:
            self._client.ensure_graph(self._graph_id)
            for gid in self._shared_graph_ids:
                self._client.ensure_graph(gid)
        except PlugMemError as e:
            logger.warning("PlugMem graph init failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [REMEMBER_SCHEMA, RECALL_SCHEMA, LEARN_SCHEMA, PROCEDURE_SCHEMA]

    # -- Core lifecycle ----------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        try:
            stats = self._client.get_stats(self._graph_id)
        except Exception:
            return (
                "# PlugMem Memory\n"
                "Service unreachable. Configure PLUGMEM_BASE_URL and start "
                "the PlugMem service.\n"
            )

        total = sum(v for v in stats.values() if isinstance(v, (int, float)))
        shared_note = f" (+{len(self._shared_graph_ids)} shared)" if self._shared_graph_ids else ""

        if total == 0:
            return (
                "# PlugMem Memory\n"
                f"Active. Empty knowledge graph{shared_note} — use "
                "plugmem_remember to store facts, plugmem_learn for procedures, "
                "plugmem_recall to search.\n"
            )
        return (
            f"# PlugMem Memory\n"
            f"Active. {total} units ({_format_stats(stats)}){shared_note}.\n"
            f"Use plugmem_remember for facts, plugmem_learn for procedures, "
            f"plugmem_recall to retrieve LLM-reasoned context.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._client or not query:
            return ""
        if len(query.strip()) < PREFETCH_MIN_QUERY_LEN:
            return ""

        sets = [("", self._graph_id)] + [
            ("shared", gid) for gid in self._shared_graph_ids
        ]
        all_results: List[Dict[str, Any]] = []

        for label, gid in sets:
            try:

                def _do():
                    result = self._client.reason(
                        gid,
                        observation=query,
                        session_id=session_id or self._session_id,
                    )
                    result["graph"] = label or gid
                    return result

                result = _retry_call(_do)
                all_results.append(result)
            except PlugMemError:
                logger.debug("PlugMem prefetch failed for graph %s", gid, exc_info=True)
                continue

        merged = _merge_reasoning(all_results)
        if not merged:
            return ""
        return f"## PlugMem Recall\n{merged}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self._client or not messages:
            return

        self._turn_count += 1

        try:
            # Store full turn as episodic trajectory
            steps = _messages_to_steps(messages)
            if steps and len(steps) >= self._auto_remember_min_steps:
                goal = user_content[:200] if user_content else "Agent turn"
                self._client.insert_trajectory(
                    self._graph_id,
                    goal=goal,
                    steps=steps,
                    session_id=session_id or self._session_id,
                )

            # Periodic consolidation
            if self._turn_count % CONSOLIDATE_EVERY_N_TURNS == 0:
                try:
                    self._client.consolidate(self._graph_id)
                except PlugMemError:
                    pass
        except PlugMemError as e:
            logger.debug("PlugMem sync_turn failed: %s", e)

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        if tool_name == "plugmem_remember":
            return self._handle_remember(args)
        elif tool_name == "plugmem_recall":
            return self._handle_recall(args)
        elif tool_name == "plugmem_learn":
            return self._handle_learn(args)
        elif tool_name == "plugmem_procedure":
            return self._handle_procedure(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._client and self._turn_count > 0 and not self._consolidated:
            try:
                self._client.consolidate(self._graph_id)
            except PlugMemError:
                pass
        self._consolidated = True
        self._client = None

    # -- Optional hooks ----------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def on_session_switch(
        self, new_session_id: str, *, reset: bool = False, **kwargs
    ) -> None:
        """Reset per-session state on /reset or /new."""
        self._session_id = new_session_id
        if reset:
            self._turn_count = 0
            self._consolidated = False

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._client or not messages:
            return
        try:
            steps = _messages_to_steps(messages)
            if steps:
                self._client.insert_trajectory(
                    self._graph_id,
                    goal="Full agent session",
                    steps=steps,
                    session_id=self._session_id,
                )
            self._client.consolidate(self._graph_id)
            self._consolidated = True
        except PlugMemError:
            pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract durable facts before context compression — store + return."""
        if not self._client or not messages:
            return ""
        try:
            text_blob = "\n".join(
                _extract_text_content(m.get("content", ""))
                for m in messages
                if m.get("content")
            )
            if len(text_blob) < 50:
                return ""

            observation = (
                "Extract durable facts, decisions, and preferences from this "
                f"conversation. Return as bullet list: {text_blob[:2000]}"
            )
            result = self._client.reason(self._graph_id, observation=observation)
            reasoning = result.get("reasoning", "")

            if reasoning and not _is_empty_reasoning(result):
                try:
                    self._client.insert_structured(
                        self._graph_id,
                        [{
                            "semantic_memory": reasoning,
                            "tags": ["compressed", "auto-extracted"],
                        }],
                        session_id=self._session_id,
                    )
                except PlugMemError:
                    pass

            return reasoning
        except PlugMemError:
            return ""

    def on_memory_write(
        self, action: str, target: str, content: str, metadata=None
    ) -> None:
        if action == "add" and self._client and content:
            try:
                tags = ["memory" if target == "memory" else "user_profile"]
                self._client.insert_structured(
                    self._graph_id,
                    [{"semantic_memory": content, "tags": tags}],
                    session_id=self._session_id,
                )
            except PlugMemError as e:
                logger.debug("PlugMem mirror failed: %s", e)

    # -- Config ------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "base_url",
                "description": "PlugMem service URL",
                "default": DEFAULT_BASE_URL,
                "env_var": "PLUGMEM_BASE_URL",
            },
            {
                "key": "api_key",
                "description": "API key for PlugMem service authentication",
                "secret": True,
                "env_var": "PLUGMEM_API_KEY",
            },
            {
                "key": "default_graph_id",
                "description": "Default memory graph ID",
                "default": DEFAULT_GRAPH_ID,
                "env_var": "PLUGMEM_DEFAULT_GRAPH_ID",
            },
            {
                "key": "shared_graph_ids",
                "description": "Comma-separated read-only shared graph IDs",
                "env_var": "PLUGMEM_SHARED_GRAPH_IDS",
            },
            {
                "key": "auto_remember_enabled",
                "description": "Auto-store turns and correction patterns",
                "type": "boolean",
                "default": True,
                "env_var": "PLUGMEM_AUTO_REMEMBER_ENABLED",
            },
            {
                "key": "auto_remember_min_steps",
                "description": "Minimum steps before auto-storing a turn",
                "type": "integer",
                "default": 2,
                "env_var": "PLUGMEM_AUTO_REMEMBER_MIN_STEPS",
            },
        ]

    # -- Tool handlers ------------------------------------------------------

    def _resolve_graph(self, args: dict) -> str:
        return args.get("graph_id") or self._graph_id

    def _resolve_session(self, args: dict) -> str:
        return args.get("session_id") or self._session_id

    def _handle_remember(self, args: dict) -> str:
        if not self._client:
            return json.dumps({"error": "PlugMem service not connected"})
        try:
            gid = self._resolve_graph(args)
            sid = self._resolve_session(args)

            if args.get("steps") and args.get("goal"):
                result = self._client.insert_trajectory(
                    gid, goal=args["goal"], steps=args["steps"], session_id=sid,
                )
                return json.dumps({
                    "status": "stored",
                    "steps": len(args["steps"]),
                    "graph": gid,
                    "graph_stats": _format_stats(result.get("stats", {})),
                })

            if args.get("text"):
                result = self._client.insert_structured(
                    gid,
                    [{"semantic_memory": args["text"], "tags": args.get("tags", [])}],
                    session_id=sid,
                )
                return json.dumps({
                    "status": "remembered",
                    "preview": _truncate(args["text"]),
                    "graph": gid,
                    "graph_stats": _format_stats(result.get("stats", {})),
                })

            return json.dumps({
                "error": "Provide 'text' (fact) or 'goal' + 'steps' (trajectory)."
            })
        except PlugMemError as e:
            logger.error("plugmem_remember failed: %s", e)
            return json.dumps({"error": str(e), "status_code": e.status_code})

    def _handle_recall(self, args: dict) -> str:
        if not self._client:
            return json.dumps({"error": "PlugMem service not connected"})
        try:
            primary = self._resolve_graph(args)
            sets = [("", primary)] + [
                ("shared", gid) for gid in self._shared_graph_ids if gid != primary
            ]
            all_results: List[Dict[str, Any]] = []
            raw_mode = args.get("raw", False)

            for label, gid in sets:
                try:
                    if raw_mode:
                        result = self._client.retrieve(
                            gid,
                            observation=args["observation"],
                            mode=args.get("mode", ""),
                        )
                    else:
                        result = self._client.reason(
                            gid,
                            observation=args["observation"],
                            mode=args.get("mode", ""),
                            session_id=self._session_id,
                        )
                    result["graph"] = label or gid
                    all_results.append(result)
                except PlugMemError:
                    continue

            if raw_mode:
                # Return raw retrieval prompts for debugging
                parts = []
                for r in all_results:
                    prompt = r.get("reasoning_prompt", [])
                    graph = r.get("graph", "")
                    prefix = f"[{graph}] " if graph else ""
                    prompt_text = "\n".join(
                        f"**{m.get('role', '?')}**: {m.get('content', '')}"
                        for m in prompt
                    )
                    parts.append(f"{prefix}{r.get('mode', '')}\n{prompt_text}")
                merged = "\n\n---\n\n".join(parts) if parts else "No results."
            else:
                merged = _merge_reasoning(all_results)
                if not merged:
                    merged = "No relevant memories found in any graph."

            return json.dumps({
                "result": merged,
                "graphs_queried": len(sets),
                "mode": args.get("mode", ""),
                "raw": raw_mode,
            })
        except PlugMemError as e:
            logger.error("plugmem_recall failed: %s", e)
            return json.dumps({"error": str(e), "status_code": e.status_code})

    def _handle_learn(self, args: dict) -> str:
        if not self._client:
            return json.dumps({"error": "PlugMem service not connected"})
        try:
            gid = self._resolve_graph(args)
            title = args["title"]
            description = args.get("description", "")
            steps = args["steps"]

            goal_text = f"Procedure: {title}"
            if description:
                goal_text += f" — {description}"

            result = self._client.insert_trajectory(
                gid, goal=goal_text, steps=steps, session_id=self._session_id,
            )
            return json.dumps({
                "status": "learned",
                "procedure": title,
                "steps": len(steps),
                "graph": gid,
                "graph_stats": _format_stats(result.get("stats", {})),
            })
        except PlugMemError as e:
            logger.error("plugmem_learn failed: %s", e)
            return json.dumps({"error": str(e), "status_code": e.status_code})

    def _handle_procedure(self, args: dict) -> str:
        if not self._client:
            return json.dumps({"error": "PlugMem service not connected"})
        try:
            gid = self._resolve_graph(args)
            sets = [("", gid)] + [
                ("shared", sgid) for sgid in self._shared_graph_ids if sgid != gid
            ]
            all_results: List[str] = []

            for label, sgid in sets:
                try:
                    result = self._client.reason(
                        sgid,
                        observation=f"Find the procedure for: {args['query']}",
                        mode="procedural_memory",
                        session_id=self._session_id,
                    )
                    reasoning = result.get("reasoning", "")
                    if reasoning and not _is_empty_reasoning(result):
                        all_results.append(reasoning)
                except PlugMemError:
                    continue

            if not all_results:
                return json.dumps({
                    "result": f"No procedure found for '{args['query']}'.",
                    "graphs_queried": len(sets),
                })
            return json.dumps({
                "result": "\n\n".join(all_results),
                "graphs_queried": len(sets),
            })
        except PlugMemError as e:
            logger.error("plugmem_procedure failed: %s", e)
            return json.dumps({"error": str(e), "status_code": e.status_code})
