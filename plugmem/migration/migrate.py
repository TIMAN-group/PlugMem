"""Migration utility: read existing JSON files -> insert into ChromaDB.

Supports migrating from all three storage formats:
- HPQA: directory with subdirectories (episodic_memory/, semantic_memory/, etc.)
- LongMemEval: single JSON file with all nodes
- WebArena: directory with numbered JSON files
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from plugmem.clients.embedding import EmbeddingClient
from plugmem.storage.chroma import ChromaStorage

logger = logging.getLogger(__name__)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float_list(v: Any) -> Optional[List[float]]:
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tolist()
    if isinstance(v, list):
        return v
    return None


def migrate_longmemeval(
    file_path: str,
    graph_id: str,
    storage: ChromaStorage,
    embedder: EmbeddingClient,
) -> Dict[str, int]:
    """Migrate a LongMemEval-format JSON file into ChromaDB.

    The JSON file contains all nodes in a single file:
    {
        "session_ids": [...],
        "episodic_nodes": [...],
        "semantic_nodes": [...],
        "tag_nodes": [...],
        "subgoal_nodes": [...],
        "procedural_nodes": [...]
    }
    """
    storage.create_graph(graph_id)
    data = _load_json(file_path)
    stats = {"episodic": 0, "semantic": 0, "tag": 0, "subgoal": 0, "procedural": 0}

    # Episodic
    for item in data.get("episodic_nodes", []):
        storage.add_episodic(
            graph_id,
            episodic_id=item["episodic_id"],
            observation=item.get("observation", ""),
            action=item.get("action", ""),
            time=str(item.get("time", "")),
            session_id=item.get("session_id"),
        )
        stats["episodic"] += 1

    # Semantic
    for item in data.get("semantic_nodes", []):
        text = item.get("semantic_memory", "")
        embedding = _to_float_list(item.get("semantic_embedding"))
        if embedding is None and text:
            embedding = embedder.embed(text)
        storage.add_semantic(
            graph_id,
            semantic_id=item["semantic_id"],
            text=text,
            embedding=embedding,
            tags=item.get("tags", []),
            time=item.get("time", 0),
            episodic_ids=item.get("episodic_nodes", []),
            session_id=item.get("session_id"),
            date=item.get("date", ""),
        )
        stats["semantic"] += 1

    # Tags
    for item in data.get("tag_nodes", []):
        tag_text = item.get("tag", "")
        embedding = _to_float_list(item.get("tag_embedding"))
        if embedding is None and tag_text:
            embedding = embedder.embed(tag_text)
        storage.add_tag(
            graph_id,
            tag_id=item["tag_id"],
            tag=tag_text,
            embedding=embedding,
            semantic_ids=item.get("semantic_nodes", []),
            time=item.get("time", 0),
            importance=item.get("importance", 1),
        )
        stats["tag"] += 1

    # Subgoals
    for i, item in enumerate(data.get("subgoal_nodes", [])):
        subgoal_text = item.get("subgoal", "")
        embedding = _to_float_list(item.get("subgoal_embedding"))
        if embedding is None and subgoal_text:
            embedding = embedder.embed(subgoal_text)
        storage.add_subgoal(
            graph_id,
            subgoal_id=i,
            subgoal=subgoal_text,
            embedding=embedding,
            time=item.get("time", 0),
        )
        stats["subgoal"] += 1

    # Procedural
    for item in data.get("procedural_nodes", []):
        text = item.get("procedural_memory", "")
        embedding = _to_float_list(item.get("procedural_embedding"))
        if embedding is None and text:
            embedding = embedder.embed(text)
        storage.add_procedural(
            graph_id,
            procedural_id=item.get("procedural_id", 0),
            text=text,
            embedding=embedding,
            subgoal=item.get("subgoal", ""),
            time=item.get("time", 0),
            return_value=float(item.get("return", 0.0)),
        )
        stats["procedural"] += 1

    logger.info("Migrated LongMemEval %s -> graph %s: %s", file_path, graph_id, stats)
    return stats


def migrate_hpqa_dir(
    dir_path: str,
    graph_id: str,
    storage: ChromaStorage,
    embedder: EmbeddingClient,
) -> Dict[str, int]:
    """Migrate an HPQA-format directory into ChromaDB.

    Directory structure:
    dir_path/
        episodic_memory/episodic_memory_*.json
        semantic_memory/semantic_memory_*.json
        tag/tag_*.json
        subgoal/subgoal_*.json
        procedural_memory/procedural_memory_*.json
    """
    storage.create_graph(graph_id)
    stats = {"episodic": 0, "semantic": 0, "tag": 0, "subgoal": 0, "procedural": 0}

    # Episodic
    epis_dir = os.path.join(dir_path, "episodic_memory")
    if os.path.isdir(epis_dir):
        for fpath in sorted(glob.glob(os.path.join(epis_dir, "episodic_memory_*.json"))):
            item = _load_json(fpath)
            epis_id = item.get("episodic_id", stats["episodic"])
            observation = item.get("observation", item.get("episodic_memory", ""))
            storage.add_episodic(
                graph_id,
                episodic_id=epis_id,
                observation=observation,
                action=item.get("action", ""),
                time=str(item.get("time", "")),
            )
            stats["episodic"] += 1

    # Semantic
    sem_dir = os.path.join(dir_path, "semantic_memory")
    if os.path.isdir(sem_dir):
        for fpath in sorted(glob.glob(os.path.join(sem_dir, "semantic_memory_*.json"))):
            item = _load_json(fpath)
            sem_id = item.get("semantic_id", stats["semantic"])
            text = item.get("semantic_memory", "")
            embedding = _to_float_list(item.get("semantic_embedding"))
            if embedding is None and text:
                embedding = embedder.embed(text)
            storage.add_semantic(
                graph_id,
                semantic_id=sem_id,
                text=text,
                embedding=embedding,
                tags=item.get("tags", []),
                tag_ids=item.get("tag_ids", []),
                time=item.get("time", 0),
                is_active=item.get("is_active", True),
                episodic_ids=item.get("episodic_ids", []),
                bro_semantic_ids=item.get("bro_semantic_ids", []),
                son_semantic_ids=item.get("son_semantic_ids", []),
            )
            stats["semantic"] += 1

    # Tags
    tag_dir = os.path.join(dir_path, "tag")
    if os.path.isdir(tag_dir):
        for fpath in sorted(glob.glob(os.path.join(tag_dir, "tag_*.json"))):
            item = _load_json(fpath)
            tag_id = item.get("tag_id", stats["tag"])
            tag_text = item.get("tag", "")
            embedding = _to_float_list(item.get("tag_embedding"))
            if embedding is None and tag_text:
                embedding = embedder.embed(tag_text)
            storage.add_tag(
                graph_id,
                tag_id=tag_id,
                tag=tag_text,
                embedding=embedding,
                semantic_ids=item.get("semantic_ids", []),
                time=item.get("time", 0),
                importance=item.get("importance", 1),
            )
            stats["tag"] += 1

    # Subgoals
    sg_dir = os.path.join(dir_path, "subgoal")
    if os.path.isdir(sg_dir):
        for fpath in sorted(glob.glob(os.path.join(sg_dir, "subgoal_*.json"))):
            item = _load_json(fpath)
            sg_id = item.get("subgoal_id", stats["subgoal"])
            subgoal_text = item.get("subgoal", "")
            embedding = _to_float_list(item.get("subgoal_embedding"))
            if embedding is None and subgoal_text:
                embedding = embedder.embed(subgoal_text)
            storage.add_subgoal(
                graph_id,
                subgoal_id=sg_id,
                subgoal=subgoal_text,
                embedding=embedding,
                procedural_ids=item.get("procedural_ids", []),
                time=item.get("time", 0),
            )
            stats["subgoal"] += 1

    # Procedural
    proc_dir = os.path.join(dir_path, "procedural_memory")
    if os.path.isdir(proc_dir):
        for fpath in sorted(glob.glob(os.path.join(proc_dir, "procedural_memory_*.json"))):
            item = _load_json(fpath)
            proc_id = item.get("procedural_id", stats["procedural"])
            text = item.get("procedural_memory", "")
            embedding = _to_float_list(item.get("procedural_embedding"))
            if embedding is None and text:
                embedding = embedder.embed(text)
            storage.add_procedural(
                graph_id,
                procedural_id=proc_id,
                text=text,
                embedding=embedding,
                subgoal=item.get("subgoal", ""),
                subgoal_id=item.get("subgoal_id"),
                episodic_ids=item.get("episodic_ids", []),
                time=item.get("time", 0),
                return_value=float(item.get("return", 0.0)),
            )
            stats["procedural"] += 1

    logger.info("Migrated HPQA dir %s -> graph %s: %s", dir_path, graph_id, stats)
    return stats


def migrate_webarena_dir(
    dir_path: str,
    graph_id: str,
    storage: ChromaStorage,
    embedder: EmbeddingClient,
) -> Dict[str, int]:
    """Migrate a WebArena-format directory into ChromaDB.

    Uses numbered JSON files: semantic_memory_0.json, etc.
    """
    storage.create_graph(graph_id)
    stats = {"episodic": 0, "semantic": 0, "tag": 0, "subgoal": 0, "procedural": 0}

    def _sorted_json_files(directory: str, prefix: str) -> List[str]:
        pattern = os.path.join(directory, f"{prefix}_*.json")
        files = glob.glob(pattern)
        files.sort(key=lambda f: int(os.path.basename(f).split("_")[-1].replace(".json", "")))
        return files

    # Episodic
    epis_dir = os.path.join(dir_path, "episodic_memory")
    if os.path.isdir(epis_dir):
        for fpath in _sorted_json_files(epis_dir, "episodic_memory"):
            item = _load_json(fpath)
            observation = item.get("observation", item.get("episodic_memory", ""))
            storage.add_episodic(
                graph_id,
                episodic_id=stats["episodic"],
                observation=observation,
                action=item.get("action", ""),
                time=str(item.get("time", "")),
                subgoal=item.get("subgoal", ""),
                state=item.get("state", ""),
                reward=item.get("reward", ""),
            )
            stats["episodic"] += 1

    # Semantic
    sem_dir = os.path.join(dir_path, "semantic_memory")
    if os.path.isdir(sem_dir):
        for fpath in _sorted_json_files(sem_dir, "semantic_memory"):
            item = _load_json(fpath)
            text = item.get("semantic_memory", "")
            embedding = _to_float_list(item.get("semantic_embedding"))
            if embedding is None and text:
                embedding = embedder.embed(text)
            tags = item.get("tags", [])
            storage.add_semantic(
                graph_id,
                semantic_id=stats["semantic"],
                text=text,
                embedding=embedding,
                tags=tags,
                time=item.get("time", 0),
                episodic_ids=item.get("episodic_ids", []),
                bro_semantic_ids=item.get("bro_semantic_ids", []),
            )
            stats["semantic"] += 1

    # Subgoals
    sg_dir = os.path.join(dir_path, "subgoal")
    if os.path.isdir(sg_dir):
        for fpath in _sorted_json_files(sg_dir, "subgoal"):
            item = _load_json(fpath)
            subgoal_text = item.get("subgoal", "")
            embedding = _to_float_list(item.get("subgoal_embedding"))
            if embedding is None and subgoal_text:
                embedding = embedder.embed(subgoal_text)
            storage.add_subgoal(
                graph_id,
                subgoal_id=stats["subgoal"],
                subgoal=subgoal_text,
                embedding=embedding,
                procedural_ids=item.get("procedural_ids", []),
                time=item.get("time", 0),
            )
            stats["subgoal"] += 1

    # Procedural
    proc_dir = os.path.join(dir_path, "procedural_memory")
    if os.path.isdir(proc_dir):
        for fpath in _sorted_json_files(proc_dir, "procedural_memory"):
            item = _load_json(fpath)
            text = item.get("procedural_memory", "")
            embedding = _to_float_list(item.get("subgoal_embedding"))
            if embedding is None and text:
                embedding = embedder.embed(text)
            storage.add_procedural(
                graph_id,
                procedural_id=stats["procedural"],
                text=text,
                embedding=embedding,
                subgoal=item.get("subgoal", ""),
                subgoal_id=item.get("subgoal_id"),
                episodic_ids=item.get("episodic_ids", []),
                time=item.get("time", 0),
                return_value=float(item.get("return", 0.0)),
            )
            stats["procedural"] += 1

    logger.info("Migrated WebArena dir %s -> graph %s: %s", dir_path, graph_id, stats)
    return stats
