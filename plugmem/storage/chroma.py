"""ChromaDB storage wrapper for PlugMem memory graphs.

Replaces all file-based save_*/update_* functions with ChromaDB operations.
Each memory graph gets 5 collections: semantic, procedural, tag, subgoal, episodic.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

logger = logging.getLogger(__name__)

NODE_TYPES = ("semantic", "procedural", "tag", "subgoal", "episodic")


def _collection_name(graph_id: str, node_type: str) -> str:
    return f"{graph_id}_{node_type}"


def _to_list(v: Any) -> Optional[List[float]]:
    """Convert numpy arrays or lists to plain float lists for ChromaDB."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tolist()
    if isinstance(v, list):
        return v
    return list(v)


def _serialize_list(v: Any) -> str:
    """Serialize a Python list to JSON string for ChromaDB metadata."""
    if v is None:
        return "[]"
    return json.dumps(v)


def _deserialize_list(s: str) -> list:
    """Deserialize a JSON string back to a Python list."""
    if not s:
        return []
    return json.loads(s)


class ChromaStorage:
    """Manages ChromaDB collections for PlugMem memory graphs."""

    def __init__(
        self,
        client: chromadb.ClientAPI,
        embedding_function=None,
        embedding_client=None,
    ):
        self._client = client
        # Build embedding function: prefer explicit, then wrap client, else None
        if embedding_function is not None:
            self._embedding_fn = embedding_function
        elif embedding_client is not None:
            from plugmem.clients.embedding import PlugMemEmbeddingFunction
            self._embedding_fn = PlugMemEmbeddingFunction(embedding_client)
        else:
            self._embedding_fn = None

    # ------------------------------------------------------------------ #
    # Graph lifecycle
    # ------------------------------------------------------------------ #

    def create_graph(self, graph_id: str) -> None:
        """Create 5 collections for a new memory graph."""
        for node_type in NODE_TYPES:
            name = _collection_name(graph_id, node_type)
            self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_fn,
            )
        logger.info("Created graph %s with 5 collections", graph_id)

    def delete_graph(self, graph_id: str) -> None:
        """Delete all collections for a memory graph (incl. recall audit)."""
        for node_type in NODE_TYPES:
            try:
                self._client.delete_collection(_collection_name(graph_id, node_type))
            except Exception:
                pass
        try:
            self._client.delete_collection(f"{graph_id}_recall_audit")
        except Exception:
            pass

    def list_graphs(self) -> List[str]:
        """List all graph IDs by inspecting collection names."""
        collections = self._client.list_collections()
        graph_ids: set[str] = set()
        for col in collections:
            # col may be a Collection object or a string depending on chromadb version.
            # In chromadb >= 0.6 list_collections returns names (strings); accessing
            # `.name` on the proxy raises NotImplementedError despite hasattr being True.
            if isinstance(col, str):
                col_name = col
            else:
                try:
                    col_name = col.name
                except (AttributeError, NotImplementedError):
                    col_name = str(col)
            for nt in NODE_TYPES:
                suffix = f"_{nt}"
                if col_name.endswith(suffix):
                    graph_ids.add(col_name[: -len(suffix)])
                    break
        return sorted(graph_ids)

    def graph_exists(self, graph_id: str) -> bool:
        """Check if a graph exists."""
        try:
            self._client.get_collection(_collection_name(graph_id, "semantic"))
            return True
        except Exception:
            return False

    def get_graph_stats(self, graph_id: str) -> Dict[str, int]:
        """Return node counts per type."""
        stats = {}
        for node_type in NODE_TYPES:
            col = self._client.get_collection(
                _collection_name(graph_id, node_type),
                embedding_function=self._embedding_fn,
            )
            stats[node_type] = col.count()
        return stats

    # ------------------------------------------------------------------ #
    # Collection accessors
    # ------------------------------------------------------------------ #

    def _col(self, graph_id: str, node_type: str):
        return self._client.get_collection(
            _collection_name(graph_id, node_type),
            embedding_function=self._embedding_fn,
        )

    # ------------------------------------------------------------------ #
    # Episodic nodes
    # ------------------------------------------------------------------ #

    def add_episodic(
        self,
        graph_id: str,
        episodic_id: int,
        observation: str = "",
        action: str = "",
        time: Any = "",
        session_id: Optional[str] = None,
        subgoal: str = "",
        state: str = "",
        reward: str = "",
        embedding: Optional[List[float]] = None,
    ) -> None:
        doc = f"{observation}\n{action}" if observation or action else ""
        metadata: Dict[str, Any] = {
            "episodic_id": episodic_id,
            "observation": observation,
            "action": action,
            "time": str(time),
            "subgoal": subgoal,
            "state": state,
            "reward": reward,
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        col = self._col(graph_id, "episodic")
        kwargs: Dict[str, Any] = {
            "ids": [str(episodic_id)],
            "documents": [doc],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def get_episodic(self, graph_id: str, episodic_id: int) -> Optional[Dict]:
        col = self._col(graph_id, "episodic")
        result = col.get(ids=[str(episodic_id)], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return result["metadatas"][0]

    def get_all_episodic(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "episodic")
        return col.get(include=["documents", "metadatas"])

    # ------------------------------------------------------------------ #
    # Semantic nodes
    # ------------------------------------------------------------------ #

    def add_semantic(
        self,
        graph_id: str,
        semantic_id: int,
        text: str,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        tag_ids: Optional[List[int]] = None,
        time: int = 0,
        is_active: bool = True,
        episodic_ids: Optional[List[int]] = None,
        bro_semantic_ids: Optional[List[int]] = None,
        son_semantic_ids: Optional[List[int]] = None,
        session_id: Optional[str] = None,
        credibility: int = 10,
        date: str = "",
    ) -> None:
        metadata: Dict[str, Any] = {
            "semantic_id": semantic_id,
            "tags": _serialize_list(tags or []),
            "tag_ids": _serialize_list(tag_ids or []),
            "time": time,
            "is_active": is_active,
            "episodic_ids": _serialize_list(episodic_ids or []),
            "bro_semantic_ids": _serialize_list(bro_semantic_ids or []),
            "son_semantic_ids": _serialize_list(son_semantic_ids or []),
            "credibility": credibility,
            "date": date,
        }
        if session_id is not None:
            metadata["session_id"] = session_id

        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {
            "ids": [str(semantic_id)],
            "documents": [text],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_semantic(
        self,
        graph_id: str,
        semantic_id: int,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {"ids": [str(semantic_id)]}
        if text is not None:
            kwargs["documents"] = [text]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            # Serialize list values
            processed = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def query_semantic(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 20,
        where: Optional[Dict] = None,
    ) -> Dict:
        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {
            "query_embeddings": [_to_list(query_embedding)],
            "n_results": min(n_results, max(col.count(), 1)),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def get_all_semantic(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "semantic")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Tag nodes
    # ------------------------------------------------------------------ #

    def add_tag(
        self,
        graph_id: str,
        tag_id: int,
        tag: str,
        embedding: Optional[List[float]] = None,
        semantic_ids: Optional[List[int]] = None,
        time: int = 0,
        importance: int = 1,
    ) -> None:
        metadata: Dict[str, Any] = {
            "tag_id": tag_id,
            "semantic_ids": _serialize_list(semantic_ids or []),
            "time": time,
            "importance": importance,
        }
        col = self._col(graph_id, "tag")
        kwargs: Dict[str, Any] = {
            "ids": [str(tag_id)],
            "documents": [tag],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_tag(
        self,
        graph_id: str,
        tag_id: int,
        tag: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "tag")
        kwargs: Dict[str, Any] = {"ids": [str(tag_id)]}
        if tag is not None:
            kwargs["documents"] = [tag]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            processed = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def query_tag(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> Dict:
        col = self._col(graph_id, "tag")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_tags(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "tag")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Subgoal nodes
    # ------------------------------------------------------------------ #

    def add_subgoal(
        self,
        graph_id: str,
        subgoal_id: int,
        subgoal: str,
        embedding: Optional[List[float]] = None,
        procedural_ids: Optional[List[int]] = None,
        time: int = 0,
    ) -> None:
        metadata: Dict[str, Any] = {
            "subgoal_id": subgoal_id,
            "procedural_ids": _serialize_list(procedural_ids or []),
            "time": time,
        }
        col = self._col(graph_id, "subgoal")
        kwargs: Dict[str, Any] = {
            "ids": [str(subgoal_id)],
            "documents": [subgoal],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_subgoal(
        self,
        graph_id: str,
        subgoal_id: int,
        subgoal: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "subgoal")
        kwargs: Dict[str, Any] = {"ids": [str(subgoal_id)]}
        if subgoal is not None:
            kwargs["documents"] = [subgoal]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            processed = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def query_subgoal(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict:
        col = self._col(graph_id, "subgoal")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_subgoals(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "subgoal")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Procedural nodes
    # ------------------------------------------------------------------ #

    def add_procedural(
        self,
        graph_id: str,
        procedural_id: int,
        text: str,
        embedding: Optional[List[float]] = None,
        subgoal: str = "",
        subgoal_id: Optional[int] = None,
        episodic_ids: Optional[List[int]] = None,
        time: int = 0,
        return_value: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        metadata: Dict[str, Any] = {
            "procedural_id": procedural_id,
            "subgoal": subgoal,
            "time": time,
            "return": return_value,
            "episodic_ids": _serialize_list(episodic_ids or []),
        }
        if subgoal_id is not None:
            metadata["subgoal_id"] = subgoal_id
        if session_id is not None:
            metadata["session_id"] = session_id
        col = self._col(graph_id, "procedural")
        kwargs: Dict[str, Any] = {
            "ids": [str(procedural_id)],
            "documents": [text],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def update_procedural(
        self,
        graph_id: str,
        procedural_id: int,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "procedural")
        kwargs: Dict[str, Any] = {"ids": [str(procedural_id)]}
        if text is not None:
            kwargs["documents"] = [text]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            processed = {}
            for k, v in metadata_updates.items():
                if isinstance(v, list):
                    processed[k] = _serialize_list(v)
                else:
                    processed[k] = v
            kwargs["metadatas"] = [processed]
        col.update(**kwargs)

    def query_procedural(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> Dict:
        col = self._col(graph_id, "procedural")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_procedural(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "procedural")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Recall audit log
    # ------------------------------------------------------------------ #
    #
    # A separate collection per graph (`{graph_id}_recall_audit`) records
    # every /retrieve, /reason, and /recall_trace call. Lazily created on
    # first append; not enumerated by `list_graphs` because the suffix is
    # outside NODE_TYPES.

    def _recall_col(self, graph_id: str):
        return self._client.get_or_create_collection(
            name=f"{graph_id}_recall_audit",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

    def add_recall(
        self,
        graph_id: str,
        *,
        endpoint: str,
        observation: str,
        ts: str,
        graph_time: int = 0,
        session_id: Optional[str] = None,
        goal: str = "",
        subgoal: str = "",
        state: str = "",
        task_type: str = "",
        mode: str = "",
        next_subgoal: str = "",
        query_tags: Optional[List[str]] = None,
        selected_semantic_ids: Optional[List[int]] = None,
        selected_procedural_ids: Optional[List[int]] = None,
        n_messages: int = 0,
    ) -> int:
        """Append one recall to the audit log. Returns the assigned recall_id."""
        col = self._recall_col(graph_id)
        recall_id = col.count()
        metadata: Dict[str, Any] = {
            "recall_id": recall_id,
            "endpoint": endpoint,
            "ts": ts,
            "graph_time": graph_time,
            "observation": observation,
            "goal": goal,
            "subgoal": subgoal,
            "state": state,
            "task_type": task_type,
            "mode": mode,
            "next_subgoal": next_subgoal,
            "query_tags": _serialize_list(query_tags or []),
            "selected_semantic_ids": _serialize_list(selected_semantic_ids or []),
            "selected_procedural_ids": _serialize_list(selected_procedural_ids or []),
            "n_messages": n_messages,
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        col.add(
            ids=[str(recall_id)],
            documents=[observation or ""],
            metadatas=[metadata],
        )
        return recall_id

    def list_recalls(
        self,
        graph_id: str,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return audit rows newest-first, optionally filtered by session_id."""
        col = self._recall_col(graph_id)
        kwargs: Dict[str, Any] = {"include": ["metadatas"]}
        if session_id is not None:
            kwargs["where"] = {"session_id": session_id}
        data = col.get(**kwargs)
        rows = list(data.get("metadatas") or [])
        for row in rows:
            row["query_tags"] = _deserialize_list(row.get("query_tags", "[]"))
            row["selected_semantic_ids"] = _deserialize_list(row.get("selected_semantic_ids", "[]"))
            row["selected_procedural_ids"] = _deserialize_list(row.get("selected_procedural_ids", "[]"))
        rows.sort(key=lambda r: r.get("recall_id", 0), reverse=True)
        return rows[: max(0, limit)]

    def list_sessions(self, graph_id: str) -> List[str]:
        """Distinct session_ids that appear anywhere in the graph (nodes or recalls)."""
        seen: set = set()
        for node_type in ("episodic", "semantic", "procedural"):
            col = self._col(graph_id, node_type)
            data = col.get(include=["metadatas"])
            for meta in data.get("metadatas", []) or []:
                sid = meta.get("session_id")
                if sid:
                    seen.add(sid)
        try:
            audit_col = self._recall_col(graph_id)
            data = audit_col.get(include=["metadatas"])
            for meta in data.get("metadatas", []) or []:
                sid = meta.get("session_id")
                if sid:
                    seen.add(sid)
        except Exception:
            pass
        return sorted(seen)
