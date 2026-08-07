"""
PlugMem Memory Provider for Hermes Agent.

Provides persistent cross-session memory via a PlugMem knowledge graph service.
Registers four tools: plugmem_remember, plugmem_recall, plugmem_learn, plugmem_procedure.
Matches all capabilities of the PlugMem OpenClaw / Claude Code connectors.

Installation:
    1. Start PlugMem service:  uvicorn plugmem.api.app:app --port 8080
    2. Symlink this directory into Hermes plugins:
       ln -sf <repo>/hermes-plugmem-plugin/memory_plugmem
              ~/.hermes/hermes-agent/plugins/memory/plugmem
    3. Configure:  hermes config set memory.provider plugmem

Config (in $HERMES_HOME/.env):
    PLUGMEM_BASE_URL=http://localhost:8080
    PLUGMEM_DEFAULT_GRAPH_ID=hermes-default
    PLUGMEM_SHARED_GRAPH_IDS=team-graph,org-graph
"""

from __future__ import annotations

import json
import logging
import os
import re
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


def _env(key: str, default: str = "") -> str:
    """Read an env var set by Hermes (from profile .env). No dotenv fallback."""
    return os.environ.get(key, default)


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

    def _request(self, method: str, path: str, body: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        headers: Dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
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

    def insert_structured(self, graph_id: str, semantic: List[Dict], **extra) -> Dict[str, Any]:
        body: Dict[str, Any] = {"mode": "structured", "semantic": semantic}
        body.update(extra)
        return self._request("POST", f"{self._API_PREFIX}/graphs/{graph_id}/memories", body)

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
        return self._request("POST", f"{self._API_PREFIX}/graphs/{graph_id}/memories", body)

    # -- Retrieval --

    def reason(
        self,
        graph_id: str,
        observation: str,
        mode: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"observation": observation}
        if mode:
            body["mode"] = mode
        body.update(kwargs)
        return self._request("POST", f"{self._API_PREFIX}/graphs/{graph_id}/reason", body)

    def retrieve(
        self, graph_id: str, observation: str, mode: str = "", **kwargs
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"observation": observation}
        if mode:
            body["mode"] = mode
        body.update(kwargs)
        return self._request("POST", f"{self._API_PREFIX}/graphs/{graph_id}/retrieve", body)

    # -- Consolidation --

    def consolidate(self, graph_id: str) -> Dict[str, Any]:
        return self._request("POST", f"{self._API_PREFIX}/graphs/{graph_id}/consolidate")


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

REMEMBER_SCHEMA = {
    "name": "plugmem_remember",
    "description": (
        "Store information in long-term PlugMem knowledge graph memory. "
        "Use for durable facts, user preferences, decisions, or observation/action "
        "trajectories. PlugMem's structuring pipeline extracts semantic facts, "
        "procedural knowledge, and episodic traces that persist across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Free-text fact, preference, or knowledge to remember permanently.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional category tags (e.g. ['project', 'decision', 'preference']).",
            },
            "goal": {
                "type": "string",
                "description": "Task goal — required when using steps/trajectory mode.",
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
                "description": "Observation/action pairs to structure and store (with 'goal').",
            },
        },
    },
}

RECALL_SCHEMA = {
    "name": "plugmem_recall",
    "description": (
        "Recall relevant memories from PlugMem's knowledge graph. Returns LLM-synthesized "
        "reasoning over the most relevant semantic, procedural, and episodic memories. "
        "Queries all configured graphs (personal + shared). "
        "Use when you need past context, facts, decisions, or procedures to inform your response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "observation": {
                "type": "string",
                "description": "Current query or observation — what do you need to recall?",
            },
            "mode": {
                "type": "string",
                "enum": ["semantic_memory", "episodic_memory", "procedural_memory"],
                "description": "Memory type to search. Omit to let PlugMem decide automatically.",
            },
        },
        "required": ["observation"],
    },
}

LEARN_SCHEMA = {
    "name": "plugmem_learn",
    "description": (
        "Store a reusable procedure / how-to in PlugMem. "
        "Use when you discover a workflow that should be repeatable — deployment steps, "
        "debugging sequences, setup instructions, build commands. "
        "PlugMem structures these into procedural memory for future recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Name of the procedure (e.g. 'Deploy to VPS', 'Run integration tests').",
            },
            "description": {
                "type": "string",
                "description": "What this procedure accomplishes and when to use it.",
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
                "description": "Ordered observation/action pairs defining the procedure.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional category tags (e.g. ['deployment', 'testing']).",
            },
        },
        "required": ["title", "steps"],
    },
}

PROCEDURE_SCHEMA = {
    "name": "plugmem_procedure",
    "description": (
        "Find a stored procedure/how-to from PlugMem. "
        "Returns the full procedure steps if found. "
        "Use before performing a task you've done before — check if there's a stored workflow."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What procedure are you looking for? (e.g. 'deploy', 'run tests')",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 120) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _format_stats(stats: Dict[str, int]) -> str:
    parts = [f"{k}: {v}" for k, v in stats.items() if v]
    return ", ".join(parts) if parts else "empty"


def _get_client() -> PlugMemClient:
    return PlugMemClient(
        base_url=_env("PLUGMEM_BASE_URL", DEFAULT_BASE_URL),
        api_key=_env("PLUGMEM_API_KEY"),
        timeout=int(_env("PLUGMEM_TIMEOUT", "30")),
    )


def _merge_reasoning(results: List[Dict[str, Any]]) -> str:
    """Merge recall results from multiple graphs into one reasoning block."""
    parts = []
    for r in results:
        graph = r.get("graph", "")
        reasoning = r.get("reasoning", "")
        if reasoning and "No relevant" not in reasoning and "null" not in reasoning[:50]:
            prefix = f"[{graph}] " if graph else ""
            parts.append(f"{prefix}{reasoning}")
    if not parts:
        return ""
    return "\n\n".join(parts)


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

    # -- Abstract methods -------------------------------------------------

    @property
    def name(self) -> str:
        return "plugmem"

    def is_available(self) -> bool:
        """Check if config is set — lightweight, no network call."""
        base = os.environ.get("PLUGMEM_BASE_URL", DEFAULT_BASE_URL)
        return bool(base)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._client = _get_client()
        self._graph_id = _env("PLUGMEM_DEFAULT_GRAPH_ID", DEFAULT_GRAPH_ID)
        self._session_id = session_id
        self._turn_count = 0
        shared_raw = _env("PLUGMEM_SHARED_GRAPH_IDS", "")
        self._shared_graph_ids = (
            [g.strip() for g in shared_raw.split(",") if g.strip()] if shared_raw else []
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
                "Service unreachable. Configure PLUGMEM_BASE_URL and start the PlugMem service.\n"
            )

        total = sum(stats.values())
        shared_note = ""
        if self._shared_graph_ids:
            shared_note = f" (+{len(self._shared_graph_ids)} shared graphs)"

        if total == 0:
            return (
                "# PlugMem Memory\n"
                f"Active. Empty knowledge graph{shared_note} — use plugmem_remember to store facts, "
                "plugmem_learn to store procedures, and plugmem_recall to search.\n"
            )
        return (
            f"# PlugMem Memory\n"
            f"Active. {total} knowledge units stored ({_format_stats(stats)}){shared_note}.\n"
            f"Use plugmem_remember for facts, plugmem_learn for procedures, "
            f"plugmem_recall to retrieve LLM-reasoned context.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context before each turn — queries personal + shared graphs."""
        if not self._client or not query or len(query.strip()) < 10:
            return ""

        try:
            # Query all graphs: personal first, then shared
            all_results = []
            sets = [("", self._graph_id)] + [("shared", gid) for gid in self._shared_graph_ids]

            for label, gid in sets:
                try:
                    result = self._client.reason(
                        gid,
                        observation=query,
                        session_id=session_id or self._session_id,
                    )
                    result["graph"] = label or gid
                    all_results.append(result)
                except PlugMemError:
                    continue

            merged = _merge_reasoning(all_results)
            if not merged:
                return ""
            return f"## PlugMem Recall\n{merged}"
        except PlugMemError as e:
            logger.debug("PlugMem prefetch failed: %s", e)
            return ""

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
            steps = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and content:
                    steps.append({"observation": str(content), "action": ""})
                elif role == "assistant" and content:
                    if steps and not steps[-1].get("action"):
                        steps[-1]["action"] = str(content)
                    else:
                        steps.append({"observation": "", "action": str(content)})

            if steps:
                goal = user_content[:200] if user_content else "Agent turn"
                self._client.insert_trajectory(
                    self._graph_id,
                    goal=goal,
                    steps=steps,
                    session_id=session_id or self._session_id,
                )

            # Periodically consolidate (every 10 turns)
            if self._turn_count % 10 == 0:
                try:
                    self._client.consolidate(self._graph_id)
                except PlugMemError:
                    pass
        except PlugMemError as e:
            logger.debug("PlugMem sync_turn failed: %s", e)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
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
        # Run final consolidation on shutdown
        if self._client and self._turn_count > 0:
            try:
                self._client.consolidate(self._graph_id)
            except PlugMemError:
                pass
        self._client = None

    # -- Optional hooks ----------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._client or not messages:
            return
        try:
            steps = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and content:
                    steps.append({"observation": str(content), "action": ""})
                elif role == "assistant" and content:
                    if steps and not steps[-1].get("action"):
                        steps[-1]["action"] = str(content)
                    else:
                        steps.append({"observation": "", "action": str(content)})

            if steps:
                goal = "Full agent session"
                self._client.insert_trajectory(
                    self._graph_id,
                    goal=goal,
                    steps=steps,
                    session_id=self._session_id,
                )
            self._client.consolidate(self._graph_id)
        except PlugMemError:
            pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._client or not messages:
            return ""
        try:
            text_blob = "\n".join(
                str(m.get("content", "")) for m in messages if m.get("content")
            )
            if len(text_blob) < 50:
                return ""

            result = self._client.reason(
                self._graph_id,
                observation=f"Extract durable facts from this conversation: {text_blob[:2000]}",
            )
            return result.get("reasoning", "")
        except PlugMemError:
            return ""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
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
        ]

    # -- Tool handlers ------------------------------------------------------

    def _handle_remember(self, args: dict) -> str:
        if not self._client:
            return json.dumps({"error": "PlugMem service not connected"})

        try:
            # Trajectory mode
            if args.get("steps") and args.get("goal"):
                result = self._client.insert_trajectory(
                    self._graph_id,
                    goal=args["goal"],
                    steps=args["steps"],
                    session_id=self._session_id,
                )
                stats = _format_stats(result.get("stats", {}))
                return json.dumps(
                    {
                        "status": "stored",
                        "steps": len(args["steps"]),
                        "graph_stats": stats,
                    }
                )

            # Semantic mode
            if args.get("text"):
                semantic = [
                    {
                        "semantic_memory": args["text"],
                        "tags": args.get("tags", []),
                    }
                ]
                result = self._client.insert_structured(
                    self._graph_id,
                    semantic,
                    session_id=self._session_id,
                )
                stats = _format_stats(result.get("stats", {}))
                return json.dumps(
                    {
                        "status": "remembered",
                        "preview": _truncate(args["text"]),
                        "graph_stats": stats,
                    }
                )

            return json.dumps(
                {"error": "Provide 'text' (fact) or 'goal' + 'steps' (trajectory)."}
            )

        except PlugMemError as e:
            logger.error("plugmem_remember failed: %s", e)
            return json.dumps({"error": str(e), "status_code": e.status_code})

    def _handle_recall(self, args: dict) -> str:
        if not self._client:
            return json.dumps({"error": "PlugMem service not connected"})

        try:
            # Query all graphs: personal + shared
            all_results = []
            sets = [("", self._graph_id)] + [("shared", gid) for gid in self._shared_graph_ids]

            for label, gid in sets:
                try:
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

            merged = _merge_reasoning(all_results)
            if not merged:
                return json.dumps(
                    {
                        "result": "No relevant memories found in any graph.",
                        "graphs_queried": len(sets),
                    }
                )
            return json.dumps(
                {
                    "result": merged,
                    "graphs_queried": len(sets),
                    "mode": args.get("mode", ""),
                }
            )
        except PlugMemError as e:
            logger.error("plugmem_recall failed: %s", e)
            return json.dumps({"error": str(e), "status_code": e.status_code})

    def _handle_learn(self, args: dict) -> str:
        """Store a procedural skill — structured as a trajectory with procedural tags."""
        if not self._client:
            return json.dumps({"error": "PlugMem service not connected"})

        try:
            title = args["title"]
            description = args.get("description", "")
            steps = args["steps"]
            tags = args.get("tags", [])

            # Build a rich goal description so PlugMem classifies it as procedural
            goal_text = f"Procedure: {title}"
            if description:
                goal_text += f" — {description}"

            result = self._client.insert_trajectory(
                self._graph_id,
                goal=goal_text,
                steps=steps,
                session_id=self._session_id,
            )

            # Also store as semantic memory with procedural tag for cross-reference
            semantic_text = f"Procedure '{title}': {description or title}. Has {len(steps)} steps."
            self._client.insert_structured(
                self._graph_id,
                [{"semantic_memory": semantic_text, "tags": tags + ["procedure"]}],
                session_id=self._session_id,
            )

            stats = _format_stats(result.get("stats", {}))
            return json.dumps(
                {
                    "status": "learned",
                    "procedure": title,
                    "steps": len(steps),
                    "graph_stats": stats,
                }
            )
        except PlugMemError as e:
            logger.error("plugmem_learn failed: %s", e)
            return json.dumps({"error": str(e), "status_code": e.status_code})

    def _handle_procedure(self, args: dict) -> str:
        """Search for a stored procedure."""
        if not self._client:
            return json.dumps({"error": "PlugMem service not connected"})

        try:
            # Search all graphs for procedural knowledge matching the query
            all_results = []
            sets = [("", self._graph_id)] + [("shared", gid) for gid in self._shared_graph_ids]

            for label, gid in sets:
                try:
                    result = self._client.reason(
                        gid,
                        observation=f"Find the procedure for: {args['query']}",
                        mode="procedural_memory",
                        session_id=self._session_id,
                    )
                    reasoning = result.get("reasoning", "")
                    if reasoning and "No relevant" not in reasoning:
                        all_results.append(reasoning)
                except PlugMemError:
                    continue

            if not all_results:
                return json.dumps(
                    {
                        "result": f"No procedure found for '{args['query']}'.",
                        "graphs_queried": len(sets),
                    }
                )
            return json.dumps(
                {
                    "result": "\n\n".join(all_results),
                    "graphs_queried": len(sets),
                }
            )
        except PlugMemError as e:
            logger.error("plugmem_procedure failed: %s", e)
            return json.dumps({"error": str(e), "status_code": e.status_code})
