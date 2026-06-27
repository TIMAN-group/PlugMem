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
sys.path.append(current_dir)
from memory_structuring.memory import Memory
from plugmem_client import PlugMemClient as MemoryGraph
from memory_retrieving.value_longmemeval import TagEqual, TagRelevant, SemanticEqual, SemanticRelevant, SubgoalEqual, SubgoalRelevant, ProceduralEqual, ProceduralRelevant
from utils import wrapper_call_model,load_json,dump_json
from utils import DEFAULT_LLM_NAME, DEFAULT_EMBEDDING_MODEL_NAME
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--update_merge_first", action="store_true")
parser.add_argument("--sem_merge_threshold", type=float, default=0.5)
parser.add_argument("--max_qa_items", type=int, default=500)
args = parser.parse_args()

def load_run_prompt() -> str:
    with open("longmemeval_run_prompt.txt", "r") as f:
        return f.read()
run_prompt_template = load_run_prompt()
print("Loading...")
with open("../../LongMemEval/data/longmemeval_s_cleaned.json", "r") as f:
    data = json.load(f)
print("Loading done")


def _build_memory_from_session(session, time):
    if not session:
        return None
    goal = "Answer user's question"
    if session[0]['role'] == 'user':
        memory = Memory(goal=goal, observation=session[0]["content"], time = f"Date: {time}")
        st = 1
    else:
        memory = Memory(goal=goal, observation="User: ...")
        st = 0
    action = None
    for turn in session[st:]:
        if turn["role"] == "assistant":
            action = f"Agent Say: {turn['content']}"
        else:
            if action is None:
                raise ValueError("Encountered user turn before any assistant action in session.")
            memory.append(
                action_t0=action,
                observation_t1=f"User Say: {turn['content']}"
            )
            action = None
    memory.close()
    return memory


worker_count = int(os.getenv("LONGMEMEVAL_SESSION_WORKERS", max(os.cpu_count() or 1, 1)))
cnt = 0

for n in range(args.max_qa_items):
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
    memories = [m for m in memories if m is not None]
    print("Memory OK")
    for memory in memories:
        mg.insert(memory)#5
    print("MG OK")
    print("Finish Loading Session")
    if args.update_merge_first:
        mg.update_semantic_subgraph(merge_threshold=args.sem_merge_threshold, write_to_disk=False)
    goal = "Answer user's question"
    with open("../../../data_longmemeval/retrieve.json", "a",) as input:
        _json = {
            "question_id": question_id,
            "question": question
        }
        input.write(json.dumps(_json) + "\n")
    messages, memory_map, sel_type = mg.retrieve_memory(goal=goal, observation=question, time=f"Date: {test['question_date']}", task_type=task_type)#6
    memory_str = memory_map[sel_type]
    response = wrapper_call_model(model_name="gpt-4o", messages=messages)#7
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
    response = wrapper_call_model(model_name="gpt-4o", messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt_run}
    ])
    with open("../../../data_longmemeval/hypothesis.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "hypothesis": response
        }
        input.write(json.dumps(_json) + "\n")
