import json
import re
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算上上级目录路径
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
# 添加到 sys.path
sys.path.append(parent_dir)
from memory_structuring.memory import Memory, Memory_LongMemEval
from memory_retrieving.memory_graph import MemoryGraph
from memory_retrieving.value_longmemeval import TagEqual, TagRelevant, SemanticEqual, SemanticRelevant, SubgoalEqual, SubgoalRelevant, ProceduralEqual, ProceduralRelevant
from utils import wrapper_call_model,load_json,dump_json
from utils import DEFAULT_LLM_NAME, DEFAULT_EMBEDDING_MODEL_NAME

def load_run_prompt() -> str:
    with open("longmemeval_run_prompt.txt", "r") as f:
        return f.read()
run_prompt_template = load_run_prompt()
print("Loading...")
with open("/home/test/test1711/czx/PM/src/LongMemEval/data/longmemeval_s_cleaned.json", "r") as f:
    data = json.load(f)
print("Loading done")

def _build_memory_from_session(session, time, session_id):
    goal = "Answer user's question"
    if session[0]['role'] == 'user':
        memory = Memory_LongMemEval(goal=goal, observation=f"User Say: {session[0]['content']}", time = f"Date: {time}", session_id = session_id)
        st = 1
    else:
        memory = Memory_LongMemEval(goal=goal, observation="User Say: ...", time = f"Date: {time}", session_id = session_id)
        st = 0
    action = "Agent Say: ..."
    for turn in session[st:]:
        if turn["role"] == "assistant":
            action = f"Agent Say: {turn['content']}"
        else:
            memory.append(
                action_t0=action,
                observation_t1=f"User Say: {turn['content']}"
            )
            action = "Agent Say: ..."
    if session[-1]["role"] == "assistant":
        memory.append(
            action_t0=action,
            observation_t1=f"User Say: ..."
        )
    memory.close()
    return memory


worker_count = int(os.getenv("LONGMEMEVAL_SESSION_WORKERS", max(os.cpu_count() or 1, 1)))
cnt = 0

for n in range(500):
    print(n)
    test = data[n]
    question_id = test["question_id"]
    mg = MemoryGraph(
        tag_equal=TagEqual(),
        tag_relevant=TagRelevant(),
        semantic_equal=SemanticEqual(),
        semantic_relevant=SemanticRelevant(), 
        subgoal_equal=SubgoalEqual(),
        subgoal_relevant=SubgoalRelevant(),
        procedural_equal=ProceduralEqual(),
        procedural_relevant=ProceduralRelevant()
    )#1
    question = test["question"]
    sessions = test["haystack_sessions"]
    times = test['haystack_dates']
    task_type = "assistant for user"
    print(f"Loading test {question_id} with {len(sessions)} sessions using {worker_count} workers")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        memories = list(executor.map(_build_memory_from_session, sessions, times))
    print("Memory OK")
    for memory in memories:
        mg.insert(memory)#5
    print("MG OK")
    print("Finish Loading Session")
    goal = "Answer user's question"
    with open("../../../data_longmemeval/retrieve.json", "a",) as input:
        _json = {
            "question_id": question_id,
            "question": question
        }
        input.write(json.dumps(_json) + "\n")
    messages, memory_map, sel_type = mg.retrieve_memory(goal=goal, observation=question, time=f"Date: {test['question_date']}", task_type=task_type)#6
    memory_str = memory_map[sel_type]
    response = wrapper_call_model(model_name="gpt-5.4", messages=messages)#7
    information = response
    with open("../../../data_longmemeval/reasoning.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "messages": messages,
            "response": response
        }
        input.write(json.dumps(_json) + "\n")
    prompt_run = run_prompt_template.format(
        information=information,
        question=question,
        time=test['question_date']
    )
    response = wrapper_call_model(model_name="gpt-5.4", messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt_run}
    ])
    with open("../../../data_longmemeval/hypothesis.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "hypothesis": response
        }
        input.write(json.dumps(_json) + "\n")