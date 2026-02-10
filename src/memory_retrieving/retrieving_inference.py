import re
from utils import call_gpt, call_qwen, wrapper_call_model
from memory_retrieving.prompt_retrieving import (
    GetPlanPrompt,
    GetNewSemanticPrompt,
    GetNewSubgoalPrompt,
    GetModePrompt,
)
import json
from typing import Any, Dict, Tuple


def get_plan(goal, subgoal, state, observation):
    prompt_obj = GetPlanPrompt()
    variables = {
        "goal": goal,
        "subgoal": subgoal,
        "state": state,
        "observation": observation,
    }
    messages = prompt_obj.render(variables)
    response = wrapper_call_model(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_gpt(messages=[{"role": m.role, "content": m.content} for m in messages])
    tags_pattern = r"\*\*Tags:\*\*\s*(.*)\n"
    tags_match = re.search(tags_pattern, response)
    
    import ast
    import json
    tags: list[str] = []
    if tags_match:
        raw = tags_match.group(1).strip()
        try:
            tags = json.loads(raw)
        except json.JSONDecodeError:
            try:
                tags = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                tags = []
    
    # tags = tags_match.group(1).split(",") if tags_match else [] 
    # tags = [tag.strip() for tag in tags]
    
    subgoal_pattern = r"### Next Subgoal\n(.*)"
    subgoal_match = re.search(subgoal_pattern, response, re.S)
    subgoal = subgoal_match.group(1).strip() if subgoal_match else "<the next subgoal>"
    return subgoal, tags

def get_new_semantic(old_semantic_memory, new_semantic_memory):
    # ================= newly added for semantic merge =================
    _ALLOWED_REL = {
        "UPDATE_SAME_FACT",
        "SAME_TOPIC_MERGE_WELL",
        "WEAK_RELATED_STITCH_RISK",
    }
    _REQUIRED_KEYS = {
        "merged_statement",
        "relationship",
        "deactivate_earlier",
        "deactivate_later",
        "simple_reasoning",
    }

    def _extract_json_object(text: str) -> Dict[str, Any]:
        text = text.strip()
        # Optional fenced block handling
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return json.loads(m.group(1))
        l = text.find("{")
        r = text.rfind("}")
        if l == -1 or r == -1 or r <= l:
            raise ValueError("No JSON object found in model output.")
        return json.loads(text[l:r+1])

    def _to_bool(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, str) and x.strip().lower() in {"true", "false"}:
            return x.strip().lower() == "true"
        raise TypeError(f"Expected boolean, got {x!r} ({type(x)})")

    def parse_merge_decision(model_text: str) -> Dict[str, Any]:
        obj = _extract_json_object(model_text)
        if not isinstance(obj, dict):
            raise TypeError("Parsed JSON is not an object/dict.")
        missing = _REQUIRED_KEYS - set(obj.keys())
        if missing:
            raise KeyError(f"Missing required keys: {sorted(missing)}")
        rel = obj.get("relationship")
        if rel not in _ALLOWED_REL:
            raise ValueError(f"Invalid relationship: {rel}. Allowed: {sorted(_ALLOWED_REL)}")
        merged = obj.get("merged_statement")
        if not isinstance(merged, str) or not merged.strip():
            raise ValueError("merged_statement must be a non-empty string.")
        # Normalize booleans then enforce consistency
        obj["deactivate_earlier"] = _to_bool(obj["deactivate_earlier"])
        obj["deactivate_later"] = _to_bool(obj["deactivate_later"])

        return {
            "merged_statement": merged.strip(),
            "relationship": rel,
            "deactivate_earlier": obj["deactivate_earlier"],
            "deactivate_later": obj["deactivate_later"],
            "simple_reasoning": obj["simple_reasoning"],
        }
    # ================= newly added for semantic merge =================
    
    prompt_obj = GetNewSemanticPrompt()
    variables = {
        "memory_earlier": old_semantic_memory,
        "memory_later": new_semantic_memory,
    }
    messages = prompt_obj.render(variables)
    response = wrapper_call_model(messages=[{"role": m.role, "content": m.content} for m in messages])
    merge_decision = parse_merge_decision(response)
    # response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_gpt(messages=[{"role": m.role, "content": m.content} for m in messages])
    return merge_decision

def get_new_subgoal(old_subgoal, new_subgoal):
    prompt_obj = GetNewSubgoalPrompt()
    variables = {
        "goal_1": old_subgoal,
        "goal_2": new_subgoal
    }
    messages = prompt_obj.render(variables)
    response = wrapper_call_model(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_gpt(messages=[{"role": m.role, "content": m.content} for m in messages])
    return response

def get_mode(observation, task_type):
    prompt_obj = GetModePrompt()
    variables = {
        "observation": observation,
        "task_type": task_type
    }
    messages = prompt_obj.render(variables)
    response = wrapper_call_model(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_gpt(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Memory Type\n(.*)"
    match = re.search(pattern, response)
    mode = match.group(1).strip() if match else "semantic_memory"
    return mode