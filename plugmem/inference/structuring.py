"""Structuring inference — accepts injected LLMClient and optional PromptRegistry."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional

from plugmem.clients.llm import LLMClient
from plugmem.prompts.structuring import (
    GetProceduralPrompt,
    GetReturnPrompt,
    GetRewardPrompt,
    GetSemanticPrompt,
    GetStatePrompt,
    GetSubgoalPrompt,
)

if TYPE_CHECKING:
    from plugmem.prompts.registry import PromptRegistry


def _render_messages(prompt_obj, variables: dict) -> List[Dict[str, str]]:
    messages = prompt_obj.render(variables)
    return [{"role": m.role, "content": m.content} for m in messages]


def _resolve(name: str, fallback_cls: type, prompts: Optional[PromptRegistry], graph_id: Optional[str]):
    if prompts is not None:
        return prompts.get(name, graph_id=graph_id)
    return fallback_cls()


def get_subgoal(
    llm: LLMClient, goal: str, state_t0: str, observation_t0: str, action_t0: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> str:
    prompt_obj = _resolve("get_subgoal", GetSubgoalPrompt, prompts, graph_id)
    variables = {"goal": goal, "state": state_t0, "observation": observation_t0, "action": action_t0}
    response = llm.complete(messages=_render_messages(prompt_obj, variables))
    pattern = r"### Subgoal\n(.*)"
    match = re.search(pattern, response, re.S)
    return match.group(1).strip() if match else "<a subgoal>"


def get_reward(
    llm: LLMClient, goal: str, state_t0: str, action_t0: str, observation_t1: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> str:
    prompt_obj = _resolve("get_reward", GetRewardPrompt, prompts, graph_id)
    variables = {"goal": goal, "state": state_t0, "action": action_t0, "observation": observation_t1}
    response = llm.complete(messages=_render_messages(prompt_obj, variables))
    pattern = r"### Reward\n(.*)"
    match = re.search(pattern, response, re.S)
    return match.group(1).strip() if match else "<a reward>"


def get_state(
    llm: LLMClient, goal: str, state_t0: str, action_t0: str, observation_t1: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> str:
    prompt_obj = _resolve("get_state", GetStatePrompt, prompts, graph_id)
    variables = {"goal": goal, "state": state_t0, "action": action_t0, "observation": observation_t1}
    response = llm.complete(messages=_render_messages(prompt_obj, variables))
    pattern = r"### State\n(.*)"
    match = re.search(pattern, response, re.S)
    return match.group(1).strip() if match else "<a state>"


def get_semantic(
    llm: LLMClient, step: dict, trajectory_num: int = 0, turn_num: int = 0, time: int = 0,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> List[dict]:
    prompt_obj = _resolve("get_semantic", GetSemanticPrompt, prompts, graph_id)
    variables = {"observation": step["observation"]}
    response = llm.complete(messages=_render_messages(prompt_obj, variables))
    pattern = r"### Facts\n(.*)"
    match = re.search(pattern, response, re.S)
    facts = match.group(1).strip() if match else None
    semantic_memory: List[dict] = []
    if facts is not None:
        pattern = r'\*\*Statement:\*\*\s*(.*?)\s*\n\s*\*\*Tags:\*\*\s*(.*?)\s*(?:\n|$)'
        matches = re.findall(pattern, facts)
        for statement, tags_str in matches:
            tags = [tag.strip().strip("[]\"'`,:;") for tag in tags_str.split(',')]
            tags = list(set(tags))
            semantic_memory.append({
                "semantic_memory": statement,
                "tags": tags,
                "trajectory_num": trajectory_num,
                "turn_num": turn_num,
                "time": time,
                "st_ed": "mid",
            })
    return semantic_memory


def get_return(
    llm: LLMClient, subgoal: str, procedural_memory: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> float:
    prompt_obj = _resolve("get_return", GetReturnPrompt, prompts, graph_id)
    variables = {"subgoal": subgoal, "procedural_memory": procedural_memory}
    response = llm.complete(messages=_render_messages(prompt_obj, variables))
    pattern = r"### Score\n(.*)"
    match = re.search(pattern, response, re.S)
    try:
        return float(match.group(1).strip()) if match else 0.0
    except (ValueError, TypeError):
        return 0.0


def get_procedural(
    llm: LLMClient, trajectory: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> tuple:
    prompt_obj = _resolve("get_procedural", GetProceduralPrompt, prompts, graph_id)
    variables = {"trajectory": trajectory}
    response = llm.complete(messages=_render_messages(prompt_obj, variables))
    pattern = r"### Goal\n(.*)\n### Experiential Insight"
    goal_match = re.search(pattern, response, re.S)
    goal = goal_match.group(1).strip() if goal_match else "<a goal>"
    pattern = r"### Experiential Insight\n(.*)"
    experience_match = re.search(pattern, response, re.S)
    experience = experience_match.group(1).strip() if experience_match else None
    _return = 0.0
    return experience, goal, _return
