"""
Build memory graph from HotpotQA corpus.

This script processes HotpotQA corpus data and builds a memory graph structure
by extracting semantic and episodic memories from corpus documents.
"""
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import argparse
import time
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)

from memory_structuring.memory import Memory
from memory_structuring.structuring_inference import get_semantic, get_procedural
from memory_retrieving.memory_graph import MemoryGraph
from memory_retrieving.value_longmemeval import (
    TagEqual, TagRelevant, SemanticEqual, SemanticRelevant,
    SubgoalEqual, SubgoalRelevant, ProceduralEqual, ProceduralRelevant
)
from utils import get_embedding
from funcs_eval import HOTPOTQA_CORPUS_PATH, MUSIQUE_CORPUS_PATH


def _process_single_data(idx: int, data: Dict[str, Any], emb_model: str, max_try: int = 3,) -> Optional[Memory]:
    base_sleep = 0.5
    jitter = 0.2
    for attempt in range(1, max_try + 1):
        try:
            goal = "Answer user's question"
            obs = f"Title: {data['title']}\nText: {data['text']}"

            memory = Memory(goal=goal, observation=obs, time="")

            episodic_memory = [{
                "observation": obs,
                "action": "",
                "state": "",
                "reward": "",
                "subgoal": ""
            }]
            memory.memory["episodic"] = episodic_memory

            semantic_memory = get_semantic(
                step={"observation": obs},
                trajectory_num=0,
                turn_num=0,
                time=""
            )

            memory.memory["semantic"] = semantic_memory
            for sm in semantic_memory:
                memory.memory_embedding["semantic"].append({
                    "semantic_memory": get_embedding(sm["semantic_memory"], emb_model),
                    "tags": [get_embedding(tag, emb_model) for tag in sm["tags"]]
                })

            procedural_memory, goal, _return = get_procedural(trajectory=obs)
            memory.memory["procedural"].append({
                "subgoal": goal,
                "procedural_memory": procedural_memory,
                "trajectory_num": 1,
                "time": memory.time,
                "return": _return,
            })

            memory.memory_embedding["procedural"].append({
                "procedural_memory": get_embedding(procedural_memory, emb_model),
                "subgoal": get_embedding(goal, emb_model)
            })
            return idx,memory

        except Exception as e:
            logger.info(f"[_process_single_data] attempt {attempt}/{max_try} failed: {e}")
            traceback.print_exc()

            if attempt < max_try:
                sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, jitter)
                time.sleep(sleep_s)

    return idx, None


def concurrent_main(mg: MemoryGraph,start_idx: int,end_idx: int,from_disk_only: bool, num_workers: int = 4, chunk_size: int = 50):
    if from_disk_only:
        if DIR_PATH is None:
            raise ValueError("DIR_PATH environment variable is not set.")
        mg.build_mem_from_disk_hpqa_ver(DIR_PATH)
        return
    
    # Load corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)[start_idx:end_idx + 1]
    
    # Load existing memory graph from disk
    mg.build_mem_from_disk_hpqa_ver(DIR_PATH)

    # 建议：mapping 先缓冲，减少频繁 open/close
    map_buffer = []

    def flush_mapping():
        if not map_buffer:
            return
        with open(MAP_PATH, "a", encoding="utf-8") as f:
            for rec in map_buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        map_buffer.clear()

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        pending_set = set()
        next_i = 0  # corpus 内偏移
        corpus_len = len(corpus)

        # 先填满窗口
        while next_i < corpus_len and len(pending_set) < chunk_size:
            idx = start_idx + next_i
            future = ex.submit(_process_single_data, idx, corpus[next_i], EMBEDDING_MODEL)
            pending_set.add(future)
            next_i += 1

        num_inserted = 0
        while pending_set:
            done, pending_set = wait(pending_set, return_when=FIRST_COMPLETED)

            for future in done:
                idx, memory = future.result()
                if memory is not None:
                    sem_num_before = len(mg.semantic_nodes)
                    mg.insert_hpqa_ver(memory)
                    sem_num_after = len(mg.semantic_nodes)

                    if sem_num_after > sem_num_before:
                        map_buffer.append({
                            "corpus_idx": idx,
                            "semantic_start": sem_num_before,
                            "semantic_end": sem_num_after - 1,
                        })

                    num_inserted += 1
                    logger.info(f"num_inserted: {num_inserted}")
                    flush_mapping()

                # 补充提交新的任务，维持窗口大小
                if next_i < corpus_len:
                    new_idx = start_idx + next_i
                    new_fut = ex.submit(_process_single_data, new_idx, corpus[next_i], EMBEDDING_MODEL)
                    pending_set.add(new_fut)
                    next_i += 1

        flush_mapping()




def main( mg: MemoryGraph, start_idx: int, end_idx: int, from_disk_only: bool) -> None:
    if from_disk_only:
        if DIR_PATH is None:
            raise ValueError("DIR_PATH environment variable is not set.")
        mg.build_mem_from_disk_hpqa_ver(DIR_PATH)
        return
    
    # Load corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)[start_idx:end_idx + 1]
    
    # Load existing memory graph from disk
    mg.build_mem_from_disk_hpqa_ver(DIR_PATH)
    
    # Process each corpus item
    logger.info(f"Processing {len(corpus)} items (indices {start_idx} to {end_idx})")
    for i, data in enumerate(corpus):
        idx = i + start_idx
        memory = _process_single_data(data, EMBEDDING_MODEL)
        
        if memory is not None:
            
            sem_num_before = len(mg.semantic_nodes)
            mg.insert_hpqa_ver(memory)
            sem_num_after = len(mg.semantic_nodes)
            logger.info(f"insert new memory for item {idx}")
            
            # Save mapping if new semantic nodes were added
            if sem_num_after > sem_num_before:
                record = {"corpus_idx": idx,"sem_start": sem_num_before,"sem_end": sem_num_after - 1,}
                with open(MAP_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(f"Processed corpus item {idx}: added {sem_num_after - sem_num_before} semantic nodes")


def _setup_memory_graph(dir_path: str, log_file: str) -> Tuple[MemoryGraph, Any]:
    mg = MemoryGraph(
        log_file=log_file,
        tag_equal=TagEqual(),
        tag_relevant=TagRelevant(),
        semantic_equal=SemanticEqual(),
        semantic_relevant=SemanticRelevant(),
        subgoal_equal=SubgoalEqual(),
        subgoal_relevant=SubgoalRelevant(),
        procedural_equal=ProceduralEqual(),
        procedural_relevant=ProceduralRelevant()
    )
    logger = mg.return_logger()
    return mg, logger


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Build memory graph from HotpotQA corpus"
    )
    parser.add_argument("--bench_name", type=str, default="hotpotqa",)
    parser.add_argument("--start_idx", type=int, default=0,)
    parser.add_argument("--end_idx", type=int, default=9,)
    parser.add_argument("--num_workers", type=int, default=8,)
    parser.add_argument("--chunk_size", type=int, default=30,)
    parser.add_argument("--emb_model", type=str, default="NV-Embed-v2",
                       help=f"Embedding model name (default: NV-Embed-v2)")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--from_disk_only", action="store_true",
                       help="Only load memory graph from disk, do not process corpus")
    
    # Constants
    DEFAULT_EMBEDDING_MODEL = "NV-Embed-v2"
    
    DIR_PATH = os.environ.get("DIR_PATH")
    MAP_PATH = os.path.join(DIR_PATH, "corpus_semantic_node_map.jsonl")
    START_TIME = time.time()
    
    os.makedirs(DIR_PATH, exist_ok=True)
    os.makedirs(os.path.join(DIR_PATH, "logs"), exist_ok=True)
    os.makedirs(os.path.join(DIR_PATH, "episodic_memory"), exist_ok=True)
    os.makedirs(os.path.join(DIR_PATH, "semantic_memory"), exist_ok=True)
    os.makedirs(os.path.join(DIR_PATH, "tag"), exist_ok=True)
    os.makedirs(os.path.join(DIR_PATH, "procedural_memory"), exist_ok=True)
    os.makedirs(os.path.join(DIR_PATH, "subgoal"), exist_ok=True)
    
    args = parser.parse_args()
    bench_name=args.bench_name
    start_idx=args.start_idx
    end_idx=args.end_idx
    from_disk_only=args.from_disk_only
    num_workers=args.num_workers
    chunk_size=args.chunk_size
    if bench_name == "hotpotqa":
        corpus_path = HOTPOTQA_CORPUS_PATH
    elif bench_name == "musique":
        corpus_path = MUSIQUE_CORPUS_PATH
    else:
        raise ValueError(f"Unsupported benchmark name: {bench_name}")
    
    # Setup logging
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_file = args.log_file
    if log_file is not None:
        log_file = os.path.join(DIR_PATH, "logs", f"build_mem_{now}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    token_usage_file = os.environ.get("TOKEN_USAGE_FILE", f"usage/build_mem_token_usage_{now}.jsonl")
    
    # Get embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or args.emb_model or DEFAULT_EMBEDDING_MODEL
    
    # Initialize memory graph
    mg, logger = _setup_memory_graph(DIR_PATH, log_file)
    
    # Run main processing
    concurrent_main(
        mg=mg,
        start_idx=start_idx,
        end_idx=end_idx,
        from_disk_only=from_disk_only,
        num_workers=num_workers,
        chunk_size=chunk_size
    )
    
    logger.info(f"Time cost for corpus {start_idx} to {end_idx}: {time.time() - START_TIME} seconds")
    
    
"""
export DIR_PATH=""
mkdir -p $DIR_PATH/logs
nohup env \
  OPENROUTER_BASE_URL="https://openrouter.ai/api/v1" \
  OPENROUTER_API_KEY="" \
  VLLM_QWEN_API_KEY="" \
  TOKEN_USAGE_FILE=None \
  bash -lc 'python build_mem.py --bench_name hotpotqa --start_idx 0 --end_idx 99 --num_workers 2' \
  > $DIR_PATH/logs/build_mem_$(date +%F_%H%M%S).log 2>&1 &
echo $! > run1.pid
disown
"""