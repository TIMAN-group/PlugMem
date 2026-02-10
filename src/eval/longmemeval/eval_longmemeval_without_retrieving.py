import json
import re
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算上上级目录路径
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)
from memory_retrieving.retrieving_inference import get_mode
from memory_structuring.structuring_inference import get_semantic
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



worker_count = int(os.getenv("LONGMEMEVAL_SESSION_WORKERS", max(os.cpu_count() or 1, 1)))
cnt = 0
#for _ in range(25):
vis_question_id = []
for n in range(500):
    test = data[n]
    question_id = test["question_id"]
    if not question_id in test_set:
        continue
    print(n)
    question = test["question"]
    sessions = test["haystack_sessions"]
    times = test['haystack_dates']
    turns = []
    mode = get_mode(
        observation = question,
        task_type = "assistant for user"
    )
    for session, time in zip(sessions, times):
        for turn in session:
            turns.append({
                "content": f"{turn['role']} say: {turn['content']}",
                "time": time
            })
    nums = random.sample(range(0, len(turns)), 5)
    print(f"Loading test {question_id} with {len(sessions)} sessions using {worker_count} workers")
    memory_str = ""
    if mode == "episodice_memory":
        for num in nums:
            memory_str+= f"{turns[num]['content']}\n"
    else:
        for num in nums:
            semantic_memory = get_semantic(
                step = {
                    "state": '',
                    "observation": turns[num]['content'],
                    "action": '',
                    "reward": ''
                },
                trajectory_num = 0,
                turn_num = 0,
                time = turns[num]['time']
            )
            for i, mem in enumerate(semantic_memory):
                memory_str += f"Relevant Fact{i}: {mem['semantic_memory']}\n"
    task_type = "assistant for user"
    
    print("Memory OK")
    print("Finish Loading Session")
    goal = "Answer user's question"
    prompt_reason = reason_prompt_template.format(
        episodic_memory_semantic = memory_str,
        time = test['question_date'],
        observation = question
    )
    response = call_gpt(prompt=prompt_reason, model_id="gpt-4o-mini")#7
    pattern = r'### Information\n(.*)'
    match = re.search(pattern, response, re.S)#8
    information = match.group(1).strip() if match else "<information>"#9
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
    with open("../../../data_longmemeval/hh_without_retrieving.jsonl", "a",) as input:
        _json = {
            "question_id": question_id,
            "hypothesis": response
        }
        input.write(json.dumps(_json) + "\n")