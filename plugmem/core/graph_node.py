"""Refactored graph node types.

All nodes hold data in-memory — no DIR_PATH coupling.
Embeddings are stored directly, not lazy-loaded from disk.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np


class EpisodicNode:
    """Stores a single episodic memory (observation + action)."""

    __slots__ = (
        "episodic_id", "observation", "action", "time",
        "session_id", "subgoal", "state", "reward",
        "semantic_nodes",
    )

    def __init__(
        self,
        episodic_id: int,
        observation: str = "",
        action: str = "",
        time: Any = "",
        session_id: Optional[str] = None,
        subgoal: str = "",
        state: str = "",
        reward: str = "",
    ):
        self.episodic_id = episodic_id
        self.observation = observation
        self.action = action
        self.time = time
        self.session_id = session_id
        self.subgoal = subgoal
        self.state = state
        self.reward = reward
        self.semantic_nodes: List[SemanticNode] = []

    def get_episodic_memory(self, date: bool = True) -> str:
        parts = []
        if self.observation:
            parts.append(self.observation)
        if self.action:
            parts.append(self.action)
        if date and self.time:
            parts.append(str(self.time))
        return "\n".join(parts) if parts else ""

    def get_date(self) -> str:
        return str(self.time) if self.time else ""


class SemanticNode:
    """Stores a semantic/factual memory with embedding."""

    def __init__(
        self,
        semantic_id: int,
        semantic_memory_str: str = "",
        embedding: Any = None,
        time: int = 0,
        is_active: bool = True,
        son: Optional[List[SemanticNode]] = None,
        session_id: Optional[str] = None,
        date: str = "",
        credibility: int = 10,
        source: Optional[str] = None,
        confidence: float = 0.5,
    ):
        self.semantic_id = semantic_id
        self.semantic_memory_str = semantic_memory_str
        self._embedding = embedding
        self.tags: List[str] = []
        self.tag_nodes: List[TagNode] = []
        self.time = time
        self.episodic_nodes: List[EpisodicNode] = []
        self.bro_semantic_nodes: List[SemanticNode] = []
        self.Credibility = credibility
        self.updated = False
        self.is_active = is_active
        self.son_semantic: List[SemanticNode] = son or []
        self.session_id = session_id
        self.date = date
        self.source = source
        self.confidence = confidence

    @property
    def embedding(self) -> Optional[np.ndarray]:
        if self._embedding is None:
            return None
        if isinstance(self._embedding, list):
            self._embedding = np.asarray(self._embedding, dtype=np.float32)
        return self._embedding

    @embedding.setter
    def embedding(self, value: Any) -> None:
        self._embedding = value

    def get_semantic_memory(self) -> str:
        return self.semantic_memory_str


class TagNode:
    """Stores a tag with embedding for categorizing semantic memories."""

    def __init__(
        self,
        tag: str,
        tag_id: int,
        embedding: Any = None,
        time: int = 0,
        importance: int = 1,
    ):
        self.tag = tag
        self.tag_id = tag_id
        self._embedding = embedding
        self.semantic_nodes: List[SemanticNode] = []
        self.importance = importance
        self.time = time

    @property
    def embedding(self) -> Optional[np.ndarray]:
        if self._embedding is None:
            return None
        if isinstance(self._embedding, list):
            self._embedding = np.asarray(self._embedding, dtype=np.float32)
        return self._embedding

    @embedding.setter
    def embedding(self, value: Any) -> None:
        self._embedding = value


class ProceduralNode:
    """Stores procedural/experiential memory with embedding."""

    def __init__(
        self,
        procedural_id: int,
        procedural_memory_str: str = "",
        embedding: Any = None,
        time: int = 0,
        return_value: float = 0.0,
        source: Optional[str] = None,
        confidence: float = 0.5,
        session_id: Optional[str] = None,
    ):
        self.procedural_id = procedural_id
        self.procedural_memory_str = procedural_memory_str
        self._embedding = embedding
        self.subgoals: List[str] = []
        self.subgoal_nodes: List[SubgoalNode] = []
        self.time = time
        self.episodic_nodes: List[EpisodicNode] = []
        self.Return = return_value
        self.source = source
        self.confidence = confidence
        self.session_id = session_id

    @property
    def embedding(self) -> Optional[np.ndarray]:
        if self._embedding is None:
            return None
        if isinstance(self._embedding, list):
            self._embedding = np.asarray(self._embedding, dtype=np.float32)
        return self._embedding

    @embedding.setter
    def embedding(self, value: Any) -> None:
        self._embedding = value

    def get_procedural_memory(self) -> str:
        return self.procedural_memory_str


class SubgoalNode:
    """Stores a subgoal abstraction for grouping procedural memories."""

    def __init__(
        self,
        subgoal: str,
        subgoal_id: int,
        embedding: Any = None,
        time: int = 0,
    ):
        self.subgoal = subgoal
        self.subgoal_id = subgoal_id
        self._embedding = embedding
        self.child_subgoal: List[SubgoalNode] = []
        self.procedural_nodes: List[ProceduralNode] = []
        self.activate = False
        self.importance = 1
        self.edge: List[Any] = []
        self.time = time

    @property
    def embedding(self) -> Optional[np.ndarray]:
        if self._embedding is None:
            return None
        if isinstance(self._embedding, list):
            self._embedding = np.asarray(self._embedding, dtype=np.float32)
        return self._embedding

    @embedding.setter
    def embedding(self, value: Any) -> None:
        self._embedding = value

    def activation(self, procedural_nodes: Optional[List[ProceduralNode]] = None) -> None:
        if procedural_nodes:
            self.procedural_nodes.extend(procedural_nodes)
        self.activate = True

    def get_subgoal(self) -> str:
        return self.subgoal
