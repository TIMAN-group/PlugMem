"""
PlugMemClient — HTTP client that replaces the old in-process MemoryGraph.

Drop-in replacement for MemoryGraph in plugmem_agent.py and LME eval scripts.
Talks to the FastAPI/ChromaDB server running at PLUGMEM_API_URL (default: http://localhost:8765).

Usage:
    from plugmem_client import PlugMemClient
    mg = PlugMemClient(graph_id="webarena")   # replaces MemoryGraph()
"""
from __future__ import annotations

import os
import logging
import requests
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BASE_URL = os.environ.get("PLUGMEM_API_URL", "http://localhost:8765")
_API_KEY  = os.environ.get("PLUGMEM_API_KEY", "")


def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if _API_KEY:
        h["X-API-Key"] = _API_KEY
    return h


def _post(path: str, body: Dict) -> Dict:
    url = f"{_BASE_URL}/api/v1{path}"
    try:
        r = requests.post(url, json=body, headers=_headers(), timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"PlugMemClient POST {path} failed: {e}") from e


def _get(path: str) -> Dict:
    url = f"{_BASE_URL}/api/v1{path}"
    try:
        r = requests.get(url, headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"PlugMemClient GET {path} failed: {e}") from e


class DummyRelevant:
    def __init__(self, k: int = 5):
        self.k = k


class DummySemanticNode:
    def __init__(self, semantic_id: int, text: str):
        self.semantic_id = semantic_id
        self.text = text

    def get_semantic_memory(self) -> str:
        return self.text


class PlugMemClient:
    """
    HTTP wrapper around the PlugMem FastAPI server.
    Mirrors the MemoryGraph interface used in plugmem_agent.py:
        - insert(mem)
        - retrieve_memory(...)
        - update_semantic_subgraph(...)
        - get_stats()
    """

    def __init__(self, graph_id: str = "default", auto_create: bool = True, **kwargs):
        self.graph_id = graph_id
        self.tag_relevant = DummyRelevant(kwargs.get("tag_relevant_k", 5))
        self.semantic_relevant = DummyRelevant(kwargs.get("semantic_relevant_k", 5))
        if auto_create:
            self._ensure_graph()

    # ------------------------------------------------------------------
    # Graph lifecycle
    # ------------------------------------------------------------------

    def _ensure_graph(self) -> None:
        """Create the graph if it doesn't exist yet."""
        try:
            _post("/graphs", {"graph_id": self.graph_id})
            logger.info("Created PlugMem graph '%s'", self.graph_id)
        except RuntimeError as e:
            # 409 = already exists, that's fine
            if "409" in str(e):
                logger.debug("Graph '%s' already exists", self.graph_id)
            else:
                logger.warning("Could not create graph '%s': %s", self.graph_id, e)

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, mem) -> None:
        """
        Insert a Memory object (from memory_structuring.memory) into the graph.
        Sends structured memory or trajectory steps to the /memories endpoint.
        """
        try:
            goal = getattr(mem, "goal", "") or ""
            session_id = getattr(mem, "session_id", None)
            if session_id is not None:
                session_id = str(session_id)

            episodic = getattr(mem, "memory", {}).get("episodic", [])
            semantic = getattr(mem, "memory", {}).get("semantic", [])
            procedural = getattr(mem, "memory", {}).get("procedural", [])

            # Normalize episodic to 2D list if it is a 1D list of dicts (HotpotQA format)
            if episodic and isinstance(episodic, (list, tuple)) and isinstance(episodic[0], dict):
                episodic = [episodic]

            # If we already have structured semantic or procedural memories, use structured mode.
            if semantic or procedural:
                # Build episodic steps in structured format
                episodic_payload = []
                for trajectory in episodic:
                    traj_steps = []
                    for step in trajectory:
                        traj_steps.append({
                            "observation": str(step.get("observation", "") or ""),
                            "action": str(step.get("action", "") or ""),
                            "subgoal": str(step.get("subgoal", "") or ""),
                            "state": str(step.get("state", "") or ""),
                            "reward": str(step.get("reward", "") if step.get("reward") is not None else ""),
                            "time": step.get("time"),
                        })
                    episodic_payload.append(traj_steps)

                # Build semantic nodes in structured format
                semantic_payload = []
                for s in semantic:
                    semantic_payload.append({
                        "semantic_memory": str(s.get("semantic_memory", "") or ""),
                        "tags": [str(t) for t in s.get("tags", [])],
                    })

                # Build procedural nodes in structured format
                procedural_payload = []
                for p in procedural:
                    procedural_payload.append({
                        "subgoal": str(p.get("subgoal", "") or ""),
                        "procedural_memory": str(p.get("procedural_memory", "") or ""),
                        "return": float(p.get("return", p.get("return_value", 0.0)) or 0.0),
                    })

                body = {
                    "mode": "structured",
                    "session_id": session_id,
                    "episodic": episodic_payload,
                    "semantic": semantic_payload,
                    "procedural": procedural_payload,
                }
            else:
                # Raw trajectory mode
                steps = []
                for trajectory in episodic:
                    for step in trajectory:
                        steps.append({
                            "observation": str(step.get("observation", "") or ""),
                            "action": str(step.get("action", "") or ""),
                        })

                body = {
                    "mode": "trajectory",
                    "goal": goal,
                    "steps": steps,
                    "session_id": session_id,
                }

            _post(f"/graphs/{self.graph_id}/memories", body)
            logger.info("Inserted memory into graph '%s'", self.graph_id)

        except Exception as e:
            raise RuntimeError(f"insert failed: {e}") from e

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve_memory(
        self,
        goal: str = "",
        subgoal: str = "",
        state: str = "",
        observation: str = "",
        time: Optional[int] = None,
        task_type: str = "",
        mode: Optional[str] = None,
        min_confidence: float = 0.0,
        source_in: Optional[List[str]] = None,
        task_id: Optional[Any] = None,   # accepted but ignored (old interface compat)
        **kwargs,
    ) -> Tuple[List, Dict, str]:
        """
        Returns (messages, variables, mode) matching the old MemoryGraph interface.
        """
        body = {
            "goal": goal or "",
            "subgoal": subgoal or "",
            "state": state or "",
            "observation": observation or "none",
            "time": str(time) if time is not None else "",
            "task_type": task_type or "",
            "mode": mode,
            "min_confidence": min_confidence,
            "source_in": source_in,
        }
        result = _post(f"/graphs/{self.graph_id}/retrieve", body)
        messages  = result.get("reasoning_prompt", [])
        variables = result.get("variables", {})
        ret_mode  = result.get("mode", "none")
        return messages, variables, ret_mode

    # ------------------------------------------------------------------
    # Consolidate / update
    # ------------------------------------------------------------------

    def update_semantic_subgraph(self, **kwargs) -> Dict:
        """Trigger consolidation on the server."""
        body = {
            "merge_threshold":                  kwargs.get("merge_threshold", 0.85),
            "max_merges_per_node":              kwargs.get("max_merges_per_node", 3),
            "max_candidates_per_tag":           kwargs.get("max_candidates_per_tag", 10),
            "max_total_candidates":             kwargs.get("max_total_candidates", 50),
            "min_credibility_to_keep_active":   kwargs.get("min_credibility_to_keep_active", -10),
            "credibility_decay":                kwargs.get("credibility_decay", 0),
            "only_update_recent_window":        kwargs.get("only_update_recent_window", None),
            "allow_merge_with_common_episodic_nodes": kwargs.get("allow_merge_with_common_episodic_nodes", False),
        }
        return _post(f"/graphs/{self.graph_id}/consolidate", body)

    # ------------------------------------------------------------------
    # Stats (used by print_memory_graph_stats)
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        try:
            return _get(f"/graphs/{self.graph_id}/stats")
        except Exception as e:
            logger.warning("Could not fetch stats: %s", e)
            return {}

    # Mimic attribute access used in print_memory_graph_stats and evaluations
    @property
    def semantic_nodes(self) -> List[DummySemanticNode]:
        try:
            result = _get(f"/graphs/{self.graph_id}/nodes?node_type=semantic&limit=10000")
            nodes = result.get("nodes", [])
            return [
                DummySemanticNode(
                    semantic_id=n.get("semantic_id", 0),
                    text=n.get("semantic_memory", "")
                )
                for n in nodes
            ]
        except Exception as e:
            logger.warning("Could not fetch semantic nodes from server: %s", e)
            return []

    @property
    def episodic_nodes(self) -> List:
        try:
            result = _get(f"/graphs/{self.graph_id}/stats")
            count = result.get("episodic", 0)
            return [None] * count
        except Exception:
            return []

    @property
    def procedural_nodes(self) -> List:
        try:
            result = _get(f"/graphs/{self.graph_id}/stats")
            count = result.get("procedural", 0)
            return [None] * count
        except Exception:
            return []

    @property
    def tag_nodes(self) -> List:
        try:
            result = _get(f"/graphs/{self.graph_id}/stats")
            count = result.get("tag", 0)
            return [None] * count
        except Exception:
            return []

    @property
    def subgoal_nodes(self) -> List:
        try:
            result = _get(f"/graphs/{self.graph_id}/stats")
            count = result.get("subgoal", 0)
            return [None] * count
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Compatibility interface methods
    # ------------------------------------------------------------------

    def insert_hpqa_ver(self, mem) -> None:
        """Alias for insert, mapping to HotpotQA evaluation interface."""
        self.insert(mem)

    def build_mem_from_disk_hpqa_ver(self, dir_path: str) -> None:
        """No-op on the server since data is already loaded and persistent."""
        logger.info("Server-based run: build_mem_from_disk_hpqa_ver is a no-op")
        pass

    def build_mem_from_disk_lme_ver(self, file_path: str) -> None:
        """No-op on the server since data is already loaded and persistent."""
        logger.info("Server-based run: build_mem_from_disk_lme_ver is a no-op")
        pass

    def build_mem_from_disk_webarena_ver(self, dir_path: str, **kwargs) -> None:
        """No-op on the server since data is already loaded and persistent."""
        logger.info("Server-based run: build_mem_from_disk_webarena_ver is a no-op")
        pass

    def return_logger(self) -> logging.Logger:
        """Returns standard logger matching MemoryGraph interface."""
        return logging.getLogger("plugmem_client")
