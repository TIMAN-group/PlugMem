"""Refactored Memory class — accepts injected LLMClient + EmbeddingClient."""
from __future__ import annotations

from typing import List, Optional

from plugmem.clients.embedding import EmbeddingClient, get_similarity
from plugmem.clients.llm import LLMClient
from plugmem.clients.llm_router import LLMRouter
from plugmem.inference.structuring import (
    get_procedural,
    get_reward,
    get_semantic,
    get_state,
    get_subgoal,
)


class Memory:
    """Trajectory structuring pipeline.

    Converts raw (observation, action) steps into structured episodic,
    semantic, and procedural memories.
    """

    def __init__(
        self,
        goal: str,
        observation: str,
        llm: LLMClient,
        embedder: EmbeddingClient,
        time: int = 0,
        session_id: Optional[str] = None,
    ):
        self.time = time
        self.session_id = session_id
        # If an LLMRouter is passed, use its structuring role
        if isinstance(llm, LLMRouter):
            self.llm = llm.structuring
        else:
            self.llm = llm
        self.embedder = embedder

        self.memory: dict = {
            "goal": goal,
            "episodic": [],
            "procedural": [],
            "semantic": [],
        }
        self.memory_embedding: dict = {
            "procedural": [],
            "semantic": [],
        }

        self.observation_t0 = observation
        self.goal = goal
        self.trajectory: List[dict] = []
        self.state_t0 = ""

    def append(self, action_t0: str, observation_t1: str) -> None:
        subgoal = get_subgoal(
            self.llm,
            goal=self.goal,
            state_t0=self.state_t0,
            observation_t0=self.observation_t0,
            action_t0=action_t0,
        )

        reward = get_reward(
            self.llm,
            goal=subgoal,
            state_t0=self.state_t0,
            action_t0=action_t0,
            observation_t1=observation_t1,
        )

        similarity_subgoal = -1.0
        if self.trajectory:
            emb_prev = self.embedder.embed(self.trajectory[-1]["subgoal"])
            emb_curr = self.embedder.embed(subgoal)
            similarity_subgoal = get_similarity(emb_prev, emb_curr)
            if similarity_subgoal < 0.75:
                self.memory["episodic"].append(self.trajectory)
                self.trajectory = []

        self.trajectory.append({
            "subgoal": subgoal,
            "state": self.state_t0,
            "observation": self.observation_t0,
            "action": action_t0,
            "reward": reward,
            "similarity_subgoal": similarity_subgoal,
            "time": self.time,
        })

        self.state_t0 = get_state(
            self.llm,
            goal=self.goal,
            state_t0=self.state_t0,
            action_t0=action_t0,
            observation_t1=observation_t1,
        )
        self.observation_t0 = observation_t1

    def close(self) -> None:
        self.memory["episodic"].append(self.trajectory)
        self.trajectory = []

        for j, trajectory in enumerate(self.memory["episodic"]):
            trajectory_str = ""
            for i, step in enumerate(trajectory):
                trajectory_str += (
                    f"Step {i}:\n-State: {step['state']}\n"
                    f"-Action: {step['action']}\n-Reward: {step['reward']}\n"
                )
                new_semantic = get_semantic(self.llm, step, j, i, self.time)
                self.memory["semantic"] += new_semantic
                for semantic_memory in new_semantic:
                    self.memory_embedding["semantic"].append({
                        "semantic_memory": self.embedder.embed(semantic_memory["semantic_memory"]),
                        "tags": [self.embedder.embed(tag) for tag in semantic_memory["tags"]],
                    })

            procedural_memory, goal, _return = get_procedural(self.llm, trajectory=trajectory_str)

            self.memory["procedural"].append({
                "subgoal": goal,
                "procedural_memory": procedural_memory,
                "trajectory_num": j,
                "time": self.time,
                "return": _return,
            })
            self.memory_embedding["procedural"].append({
                "subgoal": self.embedder.embed(goal),
            })
