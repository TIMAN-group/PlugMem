import os
import json
import re
import time
import argparse
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Set

# -------------------------
# Path bootstrap
# -------------------------
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)

from utils import wrapper_call_model,load_json,dump_json
from utils import DEFAULT_LLM_NAME, DEFAULT_EMBEDDING_MODEL_NAME
from prompt_base import PromptBase, ChatMessage
from memory_retrieving.memory_graph import MemoryGraph
from memory_retrieving.value_longmemeval import TagEqual, TagRelevant, SemanticEqual, SemanticRelevant
from memory_reasoning.prompt_reasoning import DefaultSemanticPrompt
from funcs_eval import (
    build_messages_for_qa,
    extract_gold_context,
    simulate_long_context,
    extract_reasoning_info,
    precache_semantic_facts,
    random_ctx_for_no_retrieving_mode,
    single_f1_score,
    single_exact_match,
)
from funcs_eval import HOTPOTQA_QA_PATH, HOTPOTQA_CORPUS_PATH, MUSIQUE_QA_PATH, MUSIQUE_CORPUS_PATH
START_TIME = time.time()


# -------------------------
# Multi-hop retrieval helpers
# -------------------------
def _extract_semnode_ids(text: str) -> List[int]:
    _SEMNODE_ID_RE = re.compile(r"Sem\s*Node\s*(\d+)", re.I)
    return [int(x) for x in _SEMNODE_ID_RE.findall(text or "")]

def _extract_first_json_obj(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _llm_select_topn(
    question: str,
    semantic_memory_str: str,
    judge_model: str,
    n_facts_new_query: int = 3,
    max_try: int = 3
) -> Tuple[bool, List[int], str]:
    available_ids = sorted(set(_extract_semnode_ids(semantic_memory_str)))
    if not available_ids:
        return False, [], ""

    sys_msg = "You are a retrieval controller for multi-hop question answering."
    user_msg = (
        "Return STRICT JSON only:\n"
        "{\n"
        "  \"enough\": true/false,\n"
        "  \"top_node_ids\": [int, int, ...]\n"
        "}\n\n"
        f"Constraints:\n"
        f"- top_node_ids length <= {n_facts_new_query}\n"
        f"- top_node_ids must be subset of available ids\n"
        f"- if enough=true => top_node_ids=[]\n\n"
        f"Question:\n{question}\n\n"
        f"Available node ids:\n{available_ids}\n\n"
        f"Retrieved facts:\n{semantic_memory_str}\n"
    ).strip()

    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

    last_out = ""
    for _ in range(max_try):
        last_out = wrapper_call_model(model_name=judge_model, messages=messages)
        obj = _extract_first_json_obj(last_out)
        if not isinstance(obj, dict):
            continue

        enough = bool(obj.get("enough", False))
        top_ids = obj.get("top_node_ids", [])
        if not isinstance(top_ids, list):
            continue

        avail = set(available_ids)
        cleaned: List[int] = []
        for x in top_ids:
            try:
                xi = int(x)
            except Exception:
                continue
            if xi in avail and xi not in cleaned:
                cleaned.append(xi)
            if len(cleaned) >= n_facts_new_query:
                break

        if enough:
            return True, [], last_out
        return False, cleaned, last_out

    return False, available_ids[:n_facts_new_query], last_out

def multi_hop_retrieval_sem(
    memgraph: MemoryGraph,
    sem_id2text: Dict[int, str],
    question: str,
    task_type: str,
    judge_model: str,
    max_rounds: int = 3,
    n_facts_new_query: int = 3,
) -> Dict[str, Any]:
    logger = memgraph.return_logger()

    unique_node_ids: Set[int] = set()
    rounds: List[Dict[str, Any]] = []
    query_text = question
    prev_count = 0

    for r in range(1, max_rounds + 1):
        logger.info(f"## retrieval round {r}/{max_rounds}")
        logger.info(f"query:\n{query_text}")

        goal = "Answer the question"
        messages, variables, sel_type = memgraph.retrieve_memory(
            goal=goal,
            observation=query_text,
            time=0,
            task_type=task_type,
            # mode="semantic_memory",
        )
        memory_str = variables.get(sel_type, "")
        
        if sel_type in ["procedural_memory", "episodic_memory"]:
            logger.info("⚠️ memory type is NOT supported for multi-hop, return retrieved memory directly.")
            retrieved_mem = variables.get(sel_type, variables.get(sel_type, ""))
            return {"memory_str_all": retrieved_mem, 
                    "rounds": [{
                        "round": 0,
                        "query_text": query_text,
                        "memory_str": memory_str,
                        "retrieved_node_ids": set(),
                        "unique_node_ids_so_far": set(),
                        "variables": variables,
                        "retrieval_messages": messages,
                    }]}
            
        retrieved_ids = set(_extract_semnode_ids(memory_str))  
        unique_node_ids |= retrieved_ids
        curr_count = len(unique_node_ids)

        round_record = {
            "round": r,
            "query_text": query_text,
            "memory_str": memory_str,
            "retrieved_node_ids": sorted(retrieved_ids),
            "unique_node_ids_so_far": sorted(unique_node_ids),
            "variables": variables,
            "retrieval_messages": messages,
        }
        rounds.append(round_record)

        if curr_count <= prev_count:
            rounds[-1]["stop_reason"] = "no_gain"
            break
        prev_count = curr_count

        enough, top_ids, raw = _llm_select_topn(
            question=question,
            semantic_memory_str=memory_str,
            judge_model=judge_model,
            n_facts_new_query=n_facts_new_query,
        )
        rounds[-1]["llm_enough"] = enough
        rounds[-1]["llm_top_node_ids"] = top_ids
        rounds[-1]["llm_raw"] = raw

        logger.info(f"judge enough={enough}, top_ids={top_ids}")

        if enough:
            rounds[-1]["stop_reason"] = "llm_enough"
            break
        if not top_ids:
            rounds[-1]["stop_reason"] = "no_candidates"
            break

        seed_texts = [sem_id2text.get(i, "").strip() for i in top_ids]
        seed_texts = [t for t in seed_texts if t]
        if not seed_texts:
            rounds[-1]["stop_reason"] = "selected_nodes_empty"
            break

        query_text = "original question: " + question + "\n" + "\n".join(seed_texts)

    aggregated = ""
    for i, nid in enumerate(sorted(unique_node_ids)):
        txt = sem_id2text.get(nid, "").strip()
        if txt:
            aggregated += f"Fact {i} (Sem Node {nid}): {txt}\n"

    return {"memory_str_all": aggregated.strip(), "rounds": rounds}

    
# -------------------------
# Main
# -------------------------
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str, default="hotpotqa",
                        choices=["hotpotqa", "musique"])
    parser.add_argument("--qa_model_name", type=str, default=DEFAULT_LLM_NAME)
    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--max_qa_items", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--write_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--context_mode", type=str, default="retrieval",
                        choices=["retrieval", "oracle", "no_context", "long_context"])
    parser.add_argument("--n_round_retrieval", type=int, default=1)
    parser.add_argument("--sel_mem_type", type=str, default=None,
                        choices=["semantic_memory", "procedural_memory", "episodic_memory"])
    
    parser.add_argument("--tag_top_k", type=int, default=1)
    parser.add_argument("--tag_value_threshold", type=float, default=0.5)
    parser.add_argument("--sem_top_k", type=int, default=10)
    parser.add_argument("--sem_value_threshold", type=float, default=0.0)
    
    parser.add_argument("--no_write", action="store_true")
    parser.add_argument("--no_logging", action="store_true")
    parser.add_argument("--no_retrieving", action="store_true")
    parser.add_argument("--no_reasoning", action="store_true")
    parser.add_argument("--update_merge_first", action="store_true")
    parser.add_argument("--sem_merge_threshold", type=float, default=0.5)


    parser.add_argument("--reasoning_max_tokens", type=int, default=None)
    parser.add_argument("--name_other_suffix", type=str, default="")
    args = parser.parse_args()

    # ---- Extract args ----
    bench_name = args.bench_name
    qa_model_name = args.qa_model_name
    embedding_model_name = args.embedding_model_name
    
    
    max_qa_items = args.max_qa_items
    start_idx = args.start_idx
    write_every = args.write_every
    seed = args.seed
    
    context_mode = args.context_mode
    n_round_retrieval = args.n_round_retrieval
    tag_top_k = args.tag_top_k
    tag_value_threshold = args.tag_value_threshold
    sem_top_k = args.sem_top_k
    sem_value_threshold = args.sem_value_threshold
    
    no_logging_gate = args.no_logging
    no_write_gate = args.no_write
    no_retrieving_gate = args.no_retrieving
    no_reasoning_gate = args.no_reasoning
    
    update_merge_first_gate = args.update_merge_first
    sem_merge_threshold = args.sem_merge_threshold
    
    sel_mem_type = args.sel_mem_type
    reasoning_max_tokens = args.reasoning_max_tokens
    name_other_suffix = args.name_other_suffix
    
    # ---- Env----
    dir_path = os.environ.get("DIR_PATH", None)
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    token_usage_file = os.environ.get("TOKEN_USAGE_FILE", f"usage/eval_token_usage_{qa_model_name.split('/')[-1].replace(':', '_')}_{now}.jsonl")
    if not dir_path:
        raise ValueError("DIR_PATH environment variable is not set.")
    if not token_usage_file:
        raise ValueError("TOKEN_USAGE_FILE environment variable is not set.")
    os.makedirs(os.path.dirname(token_usage_file), exist_ok=True)
    os.environ["EMBEDDING_MODEL_NAME"] = embedding_model_name
    os.environ["LLM_NAME"] = qa_model_name
    
    # ---- QA data path ----
    if bench_name == "hotpotqa":
        QA_DATA_PATH = HOTPOTQA_QA_PATH
    elif bench_name == "musique":
        QA_DATA_PATH = MUSIQUE_QA_PATH
    else:
        raise ValueError(f"Unsupported benchmark name: {bench_name}")

    # ---- output naming ----
    memory_dir = dir_path
    output_dir = dir_path
    pred_name_base = "predictions"
    metric_name_base = "metrics"
    out_suffix=""
    if sel_mem_type != None:
        if sel_mem_type == "semantic_memory":
            out_suffix += f"_sem_only"
        elif sel_mem_type == "procedural_memory":
            out_suffix += f"_proc_only"
        elif sel_mem_type == "episodic_memory":
            out_suffix += f"_epis_only"
            
    if context_mode != "retrieval":
        out_suffix += f"_{context_mode}"
    else:
        pass
    
    if no_reasoning_gate:
        out_suffix += "_no_reasoning"
    
    if no_retrieving_gate:
        out_suffix += "_no_retrieving"
    else:
        if n_round_retrieval >= 2:
            out_suffix += f"_{n_round_retrieval}_round_retrieval"
    
    if update_merge_first_gate:
        if sem_merge_threshold is not None:
            out_suffix += f"_sem_merge_thres_{sem_merge_threshold}"
        
    model_name_in_file = qa_model_name.split("/")[-1].replace(":", "_")
    out_suffix += f"_{model_name_in_file}"
    if start_idx > 0 and max_qa_items > 0:
        out_suffix += f"_{start_idx}_{start_idx + max_qa_items}"
    
    if reasoning_max_tokens is not None:
        out_suffix += f"_max{reasoning_max_tokens}"
    
    if name_other_suffix:
        out_suffix += f"_{name_other_suffix}"
        
    pred_path = os.path.join(output_dir, f"{pred_name_base}{out_suffix}.json")
    metric_path = os.path.join(output_dir, f"{metric_name_base}{out_suffix}.json")

    # ---- logger file ----
    if no_logging_gate:
        log_file=None
        print("no_logging is True, no log file will be written")
        
    else:
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(output_dir, "logs", f"eval_{now_str}{out_suffix}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        print("log_file is set to: ", log_file)
    print("="*100)
    
    # ---- build / load memgraph if needed ----
    mg = MemoryGraph(
        log_file=log_file,
        tag_equal=TagEqual(),
        tag_relevant=TagRelevant(k=tag_top_k, value_threshold=tag_value_threshold),
        semantic_equal=SemanticEqual(),
        semantic_relevant=SemanticRelevant(k=sem_top_k, value_threshold=sem_value_threshold),
    )
    mg.build_mem_from_disk_hpqa_ver(memory_dir)
    if update_merge_first_gate:
        mg.update_semantic_subgraph(merge_threshold=sem_merge_threshold,write_to_disk=True)
    logger = mg.return_logger()
    
    sem_id2text: Dict[int, str] = {}
    if context_mode == "retrieval":
        sem_id2text={node.semantic_id:node.get_semantic_memory() for node in mg.semantic_nodes}

    # ---- log configs ----
    logger.info(
        "configs:\n"
        f"qa_model_name: {qa_model_name}\n"
        f"embedding_model_name: {embedding_model_name}\n"
        f"max_examples: {max_qa_items}\n"
        f"start_idx: {start_idx}\n"
        f"context_mode: {context_mode}\n"
        f"no_retrieving: {no_retrieving_gate}\n"
        f"no_reasoning: {no_reasoning_gate}\n"
        f"write_every: {write_every}\n"
        f"no_write: {no_write_gate}\n"
        f"top_k tag nodes: {mg.tag_relevant.k}\n"
        f"top_k sem nodes: {mg.semantic_relevant.k}\n"
        f"n_round_retrieval: {n_round_retrieval}\n"
        f"predefined_use_mem_type: {sel_mem_type}\n"
        f"pred_path: {pred_path}\n"
        f"metric_path: {metric_path}\n"
        f"log_file: {log_file}\n"
        f"token_usage_file: {token_usage_file}\n"
    )

    # ---- load QA data ----
    data = load_json(QA_DATA_PATH)[start_idx:]
    if max_qa_items > 0:
        data = data[:max_qa_items]
    
    
    # ---- Pre-cache semantic facts for no_retrieving mode ----
    all_semantic_facts: List[str] = []
    if context_mode == "retrieval" and no_retrieving_gate:
        all_semantic_facts = precache_semantic_facts(memory_dir)
        logger.info(f"[Cache] loaded {len(all_semantic_facts)} semantic facts for no_retrieving")

    
    # ---- Resume: load existing predictions (if any) ----
    completed_ids: Set[str] = set()
    total_em, total_f1, n_done = 0.0, 0.0, 0
    if (not no_write_gate) and os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            existing_preds=json.load(f)
        for obj in existing_preds:
            qid = obj.get("id", None)
            if qid is None:
                continue
            completed_ids.add(str(qid))
            n_done += 1
            total_em += obj.get("em", 0.0)
            total_f1 += obj.get("f1", 0.0)

        logger.info(f"[Resume] found existing {n_done} records in {pred_path}. Will skip completed ids.")

    # ---- Main Eval Loop ----
    n = n_done  # start from already done count
    for idx_in_run, qa_item in enumerate(data):
        qid = str(qa_item.get("_id", qa_item.get("id", str(idx_in_run))))
        if qid in completed_ids:
            logger.info(f"Skipping completed id: {qid}")
            continue

        question = qa_item["question"]
        gold_ans = qa_item.get("answer", "")
        _, gold_context_str = extract_gold_context(qa_item=qa_item, bench_name=bench_name)

        task_type = "answer the question based on objective knowledge or information."
        retrieved_mem = ""
        rag_context = ""
        messages_for_reasoning: Optional[List[Dict[str, str]]] = None

        # ---- context mode handling ----
        if context_mode == "retrieval":
            # retrieving gate: no retrieving
            if no_retrieving_gate:
                # deterministic per question (resume-safe): seed + hash(qid)
                rng = random.Random(seed + (hash(qid) % 1_000_000_007))
                k = mg.semantic_relevant.k
                messages_for_reasoning, variables, sel_type, retrieved_mem = \
                random_ctx_for_no_retrieving_mode(
                    question=question,
                    all_semantic_facts=all_semantic_facts,
                    k=k,
                    rng=rng,
                    prompt_obj = DefaultSemanticPrompt()
                )
                
            # retrieving gate: use retrieving
            else:
                # multi-hop retrieval
                if n_round_retrieval <= 1:
                    goal = "Answer the question"
                    messages_for_reasoning, variables, sel_type = mg.retrieve_memory(
                        goal=goal,
                        observation=question,
                        time=0,
                        task_type=task_type,
                        mode = sel_mem_type,
                        # mode="semantic_memory",
                        # mode="episodic_memory",
                        # mode="procedural_memory",
                    )
                    retrieved_mem = variables.get(sel_type, variables.get(sel_type, ""))

                # single-hop retrieval
                else:
                    result = multi_hop_retrieval_sem(
                        memgraph=mg,
                        sem_id2text=sem_id2text,
                        task_type=task_type,
                        question=question,
                        judge_model=qa_model_name,
                        max_rounds=n_round_retrieval,
                        n_facts_new_query=3,
                    )
                    messages_for_reasoning = result["rounds"][-1].get("retrieval_messages")
                    retrieved_mem = result["memory_str_all"]

            # reasoning gate
            if no_reasoning_gate:
                rag_context = retrieved_mem.strip()
            else:
                if messages_for_reasoning is None:
                    rag_context = retrieved_mem.strip()
                else:
                    if reasoning_max_tokens is None:
                        raw_reasoning_out = wrapper_call_model(model_name=qa_model_name, messages=messages_for_reasoning)
                    else:
                        messages_for_reasoning[1] = {
                            "role":"user", 
                            "content":messages_for_reasoning[1]["content"]+ \
                                # f"\n\nIMPORTANT: \nMake sure your output is within the token budget of {reasoning_max_tokens} tokens! Adjust your output accordingly, if necessary, skip the reasoning process with a placeholder '<skipped>' and only output the extracted information part."
                                f"\n\nIMPORTANT: \nMake sure the length of your output is approximately {reasoning_max_tokens} tokens! Adjust your output accordingly, if necessary, skip the reasoning process with a placeholder '<skipped>' and only output the extracted information part."
                        }
                        raw_reasoning_out = wrapper_call_model(model_name=qa_model_name, messages=messages_for_reasoning, max_tokens=reasoning_max_tokens)
                    
                    logger.info(f"raw_reasoning_out:\n{raw_reasoning_out}")
                    reasoning_out = extract_reasoning_info(text=raw_reasoning_out)
                    if not reasoning_out:
                        reasoning_out = reasoning_out.strip()
                    logger.info(f"reasoning_info:\n{reasoning_out}")
                    rag_context = f"Relevant info: {reasoning_out}\n"
                    # rag_context += f"Original retrieved memories:\n{retrieved_mem}".strip()

        elif context_mode == "oracle":
            _, rag_context = gold_context_str

        elif context_mode == "no_context":
            rag_context = "No information is retrieved. Answer the question based on your knowledge."

        elif context_mode == "long_context":
            raise NotImplementedError
            encoder = ""
            rag_context = simulate_long_context(qa_item, encoder)

        # ---- final answer ----
        messages = build_messages_for_qa(info=rag_context,question=question)
        pred = wrapper_call_model(model_name=qa_model_name, messages=messages).strip()

        em = single_exact_match(pred, gold_ans) if gold_ans else 0.0
        f1 = single_f1_score(pred, gold_ans) if gold_ans else 0.0
        total_em += em
        total_f1 += f1
        n += 1
        completed_ids.add(qid)

        logger.info(f"----- gold_context -----:\n{gold_context_str}")
        logger.info(f"----- question -----: {question}")
        logger.info(f"----- gold_answer -----: {gold_ans}")
        logger.info(f"----- model_answer -----: {pred}")
        logger.info(f"EM={em:.3f}, F1={f1:.3f}")

        record: Dict[str, Any] = {
            "id": qid,
            "question": question,
            "gold": gold_ans,
            "pred": pred,
            "em": em,
            "f1": f1,
            "context_mode": context_mode,
            "no_retrieving": no_retrieving_gate,
            "no_reasoning": no_reasoning_gate,
            "retrieved_facts": retrieved_mem,
            "reasoning_info": rag_context,
        }
        
        if os.path.exists(pred_path): 
            with open(pred_path, "r", encoding="utf-8") as fout: 
                existing_records=json.load(fout) 
        else: 
            existing_records=[]     
        existing_records.append(record) 
        
        if no_write_gate: 
            pass 
        else: 
            with open(pred_path, "w", encoding="utf-8") as fout: 
                json.dump(existing_records, fout, indent=4, ensure_ascii=False)

        if n % write_every == 0:
            avg_em = total_em / n
            avg_f1 = total_f1 / n
            logger.info(f"[{n}/{len(data)}] EM={avg_em:.4f} F1={avg_f1:.4f}")

    # ---- final metrics ----
    metrics = {
        "count": n,
        "em": total_em / n if n else 0.0,
        "f1": total_f1 / n if n else 0.0,
        "context_mode": context_mode,
        "no_retrieving": no_retrieving_gate,
        "no_reasoning": no_reasoning_gate,
        "qa_model_name": qa_model_name,
        "embedding_model_name": embedding_model_name,
        "n_round_retrieval": n_round_retrieval,
        "predefined_use_mem_type": sel_mem_type,
        "pred_path": pred_path,
    }

    if not no_write_gate:
        dump_json(metric_path, metrics)

    logger.info(f"[Done] {json.dumps(metrics, ensure_ascii=False)}")
    logger.info(f"[Time Cost] {time.time() - START_TIME:.3f}s")


if __name__ == "__main__":
    main()


