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
from utils import call_qwen, call_gpt, save_episodic_longmem_ver, get_embedding, get_similarity

def load_run_prompt() -> str:
    with open("longmemeval_run_prompt.txt", "r") as f:
        return f.read()
run_prompt_template = load_run_prompt()

def load_reason_prompt() -> str:
    with open("longmemeval_reason_prompt.txt", "r") as f:
        return f.read()
reason_prompt_template = load_reason_prompt()

def load_test_set():
    re = []
    with open("../../../data_longmemeval/question_ids.txt","r",) as input:
        for line in input:
            re.append(line.strip())
    return re

test_set = load_test_set()

print("Loading...")
with open("../../LongMemEval/data/longmemeval_s_cleaned.json", "r") as f:
    data = json.load(f)
print("Loading done")


def _build_memory_from_session(turn):
    memory = {
        "content": turn['content'],
        "time": turn['time'],
        "embedding": get_embedding(turn['content'])
    }
    return memory


worker_count = int(os.getenv("LONGMEMEVAL_SESSION_WORKERS", max(os.cpu_count() or 1, 1)))
cnt = 0
for n in range(500):
    test = data[n]
    question_id = test["question_id"]
    question = test["question"]
    sessions = test["haystack_sessions"]
    times = test['haystack_dates']
    turns = []
    for session, time in zip(sessions, times):
        for turn in session:
            turns.append({
                "content": f"{turn['role']} say: {turn['content']}",
                "time": time
            })
    task_type = "assistant for user"
    print(f"Loading test {question_id} with {len(sessions)} sessions using {worker_count} workers")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        memories = list(executor.map(_build_memory_from_session, turns))
    print("Memory OK")
    print("Finish Loading Session")
    goal = "Answer user's question"
    values = []
    embedding = get_embedding(question)
    for index, memory in enumerate(memories):
        values.append((get_similarity(memory['embedding'], embedding), index))
    values.sort(reverse=True, key=lambda x: x[0])
    values = values[:10]
    memory_str = ""
    for _, index in values:
        memory_str += f"{memories[index]['content']} Date: {memories[index]['time']}"
    prompt_reason = reason_prompt_template.format(
        episodic_memory_semantic = memory_str,
        time = test['question_date'],
        observation = question
    )
    response = call_gpt(prompt=prompt_reason, model_id="gpt-4o-mini")#7
    information = response
    with open("../../../data_longmemeval/reasoning_without_structuring.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "prompt": prompt_reason,
            "response": response
        }
        input.write(json.dumps(_json) + "\n")
    prompt_run = run_prompt_template.format(
        information=information,
        question=question,
        time=test['question_date']
    )
    response = call_gpt(
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt_run}
        ],
        model_id="gpt-4o-mini"
    )
    with open("../../../data_longmemeval/hypothesis_without_structuring.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "hypothesis": response
        }
        input.write(json.dumps(_json) + "\n")