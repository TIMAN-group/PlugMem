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
from memory_structuring.memory import Memory
from memory_retrieving.memory_graph import MemoryGraph
from memory_retrieving.value_longmemeval import TagEqual, TagRelevant, SemanticEqual, SemanticRelevant, SubgoalEqual, SubgoalRelevant, ProceduralEqual, ProceduralRelevant
from utils import call_qwen, call_gpt, save_episodic_longmem_ver


def load_run_prompt() -> str:
    with open("longmemeval_run_prompt.txt", "r") as f:
        return f.read()
run_prompt_template = load_run_prompt()
print("Loading...")
with open("../../LongMemEval/data/longmemeval_s_cleaned.json", "r") as f:
    data = json.load(f)
print("Loading done")


def _build_memory_from_session(session, time):
    goal = "Answer user's question"
    if session[0]['role'] == 'user':
        memory = Memory(goal=goal, observation=session[0]["content"], time = f"Date: {time}")
        observation_t0 = f"User Say: {session[0]['content']}"
        st = 1
    else:
        memory = Memory(goal=goal, observation="User: ...")
        observation_t0 = f"User Say nothing"
        st = 0
    action = None
    for turn in session[st:]:
        if turn["role"] == "assistant":
            action = f"Agent Say: {turn['content']}"
        else:
            if action is None:
                raise ValueError("Encountered user turn before any assistant action in session.")
            memory.append(
                observation_t0=observation_t0,
                action_t0=action,
                observation_t1=f"User Say: {turn['content']}"
            )
            observation_t0 = f"User Say: {turn['content']}"
            action = None
    memory.close()
    return memory


worker_count = int(os.getenv("LONGMEMEVAL_SESSION_WORKERS", max(os.cpu_count() or 1, 1)))
cnt = 0
#for _ in range(25):
vis_question_id = []
n_prep = [] + list(range(236,243)) + list(range(366,376)) + list(range(444,454))
for n in n_prep:
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
    #sessions = sessions[:1]
    times = test['haystack_dates']
    task_type = "assistant for user"
    print(f"Loading test {question_id} with {len(sessions)} sessions using {worker_count} workers")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        memories = list(executor.map(_build_memory_from_session, sessions, times))
    print("Memory OK")
    for memory in memories:
        with open("result.json", "a") as output:
            output.write(json.dumps(memory.memory)+"\n")
        mg.insert(memory)#5
    print("MG OK")
    print("Finish Loading Session")
    goal = "Answer user's question"
    with open("../../../data_longmemeval/retrieve.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "question": question
        }
        input.write(json.dumps(_json) + "\n")
    messages, memory_str = mg.get_reason_prompt(goal=goal, observation=question, time=f"Date: {test['question_date']}", task_type=task_type)#6
    prompt_run_without_reasoning = run_prompt_template.format(
        information=memory_str,
        question=question,
        time=test['question_date']
    )
    response_without_reasoning = call_gpt(
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt_run_without_reasoning}
        ],
        model_id="gpt-4o-mini"
    )
    with open("../../../data_longmemeval/hh_without_reasoning.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "hypothesis": response_without_reasoning
        }
        input.write(json.dumps(_json) + "\n")