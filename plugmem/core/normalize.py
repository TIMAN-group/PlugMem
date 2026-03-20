"""Normalize Memory objects from different benchmark formats into the unified shape.

The unified insert() expects:
    memory.memory["episodic"]  = [[step, step, ...], [step, ...], ...]  (list of trajectories)
    memory.memory["semantic"]  = [{"semantic_memory": str, "tags": [...], "trajectory_num": int, "turn_num": int, ...}, ...]
    memory.memory["procedural"]= [{"subgoal": str, "procedural_memory": str, "trajectory_num": int, ...}, ...]
    memory.memory_embedding["semantic"]  = [{"semantic_memory": [...], "tags": [[...], ...]}, ...]
    memory.memory_embedding["procedural"]= [{"subgoal": [...], ...}, ...]

HPQA's build_mem.py produces a flat episodic list:
    memory.memory["episodic"]  = [step, step, ...]  (flat list of dicts or strings)

This module detects the flat shape and wraps it into a single trajectory.
"""
from __future__ import annotations

from typing import Any, Dict


def normalize_memory(memory: Any) -> Any:
    """Normalize a Memory object in-place so its episodic list is always nested.

    Detection logic:
        - If episodic is empty → leave as-is
        - If episodic[0] is a dict or str → flat format → wrap in one trajectory
        - If episodic[0] is a list → already nested → no-op

    Also patches trajectory_num references in semantic/procedural items when
    wrapping a flat list (all point to trajectory 0).

    Returns the same Memory object (mutated in-place) for convenience.
    """
    episodic = memory.memory.get("episodic", [])

    if not episodic:
        return memory

    first = episodic[0]

    # Already nested (list of trajectories)
    if isinstance(first, list):
        return memory

    # Flat list of dicts or strings → wrap as single trajectory
    memory.memory["episodic"] = [episodic]

    # Fix trajectory_num references: all should point to trajectory 0
    for sem in memory.memory.get("semantic", []):
        sem.setdefault("trajectory_num", 0)
        sem["trajectory_num"] = 0

    for proc in memory.memory.get("procedural", []):
        proc.setdefault("trajectory_num", 0)
        proc["trajectory_num"] = 0

    return memory
