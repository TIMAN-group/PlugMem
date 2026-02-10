import re
from utils import call_gpt, call_qwen, call_dpsk, call_llm_openrouter_api
from memory_structuring.prompt_structuring import (
    GetSubgoalPrompt,
    GetRewardPrompt,
    GetStatePrompt,
    GetSemanticPrompt,
    GetProceduralPrompt,
    GetReturnPrompt,
)

def get_subgoal(goal, state_t0, observation_t0, action_t0):
    prompt_obj = GetSubgoalPrompt()
    variables = {"goal": goal, "state": state_t0, "observation": observation_t0, "action": action_t0}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Subgoal\n(.*)"
    match = re.search(pattern, response, re.S)
    subgoal = match.group(1).strip() if match else "<a subgoal>"
    #print(f"Subgoal: {subgoal}")
    return subgoal

def get_reward(goal, state_t0, action_t0, observation_t1):
    prompt_obj = GetRewardPrompt()
    variables = {"goal": goal, "state": state_t0, "action": action_t0, "observation": observation_t1}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Reward\n(.*)"
    match = re.search(pattern, response, re.S)
    reward = match.group(1).strip() if match else "<a reward>"
    #print(f"Reward: {reward}")
    return reward

def get_state(goal, state_t0, action_t0, observation_t1):
    prompt_obj = GetStatePrompt()
    variables = {"goal": goal, "state": state_t0, "action": action_t0, "observation": observation_t1}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### State\n(.*)"
    match = re.search(pattern, response, re.S)
    state = match.group(1).strip() if match else "<a state>"
    #print(f"State: {state}")
    return state

def get_semantic(step, trajectory_num=0, turn_num=0, time=0):    
    prompt_obj = GetSemanticPrompt()
    variables = {"observation": step["observation"]}
    messages = prompt_obj.render(variables)
    # response = call_gpt(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_dpsk(messages=[{"role": m.role, "content": m.content} for m in messages])
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_llm_openrouter_api(model_name="openai/gpt-4o-2024-11-20",messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Facts\n(.*)"
    match = re.search(pattern, response, re.S)
    facts = match.group(1).strip() if match else None
    semantic_memory = []
    if not facts == None:
        # pattern = r'\*\*Statement:\*\* (.*?)\n\s*\*\*Tags:\*\* (.*?)\n'
        pattern = r'\*\*Statement:\*\*\s*(.*?)\s*\n\s*\*\*Tags:\*\*\s*(.*?)\s*(?:\n|$)'
        matches = re.findall(pattern, facts)
        for idx, (statement, tags) in enumerate(matches):
            tags=[tag.strip().strip("[]\"'`,:;") for tag in tags.split(',')]
            tags=list(set(tags))
            semantic_memory.append({
                "semantic_memory": statement,
                "tags": tags,
                "trajectory_num" : trajectory_num,
                "turn_num" : turn_num,
                "time": time,
                "st_ed": "mid"
            })
        # if not len(semantic_memory) == 0:
        #     semantic_memory[0]["st_ed"] = "st"
        #     semantic_memory[len(semantic_memory)-1]["st_ed"] = "ed"
    for i in range(len(semantic_memory)):
        print(semantic_memory[i])
    return semantic_memory

def get_return(subgoal: str, procedural_memory: str):
    prompt_obj = GetReturnPrompt()
    variables = {"subgoal": subgoal, "procedural_memory": procedural_memory}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Score\n(.*)"
    match = re.search(pattern, response, re.S)
    _return = match.group(1).strip() if match else 0.0
    return _return

def get_procedural(trajectory: str):
    prompt_obj = GetProceduralPrompt()
    variables = {"trajectory": trajectory}
    messages = prompt_obj.render(variables)
    response = call_qwen(messages=[{"role": m.role, "content": m.content} for m in messages])
    # response = call_llm_openrouter_api(model_name="openai/gpt-4o-2024-11-20",messages=[{"role": m.role, "content": m.content} for m in messages])
    pattern = r"### Goal\n(.*)\n### Experiential Insight"
    goal_match = re.search(pattern, response, re.S)
    goal = goal_match.group(1).strip() if goal_match else "<a goal>"
    pattern = r"### Experiential Insight\n(.*)"
    experience_match = re.search(pattern, response, re.S)
    experience = experience_match.group(1).strip() if experience_match else None
    # _return = get_return(subgoal = goal, procedural_memory = experience)
    _return = 0.0
    res_dict={"procedural_memory": experience, "sub_goal": goal, "return": _return}
    print(res_dict)
    return experience, goal, _return