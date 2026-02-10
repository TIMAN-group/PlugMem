# eval_hotpotqa_wo_structuring.py
# HotpotQA ablation: WITHOUT structuring
# - Treat each HotpotQA corpus item as a raw chunk (title + text)
# - Dense retrieval by cosine similarity over get_embedding()
# - (Optional) LLM "reasoning" step to compress retrieved chunks (still NOT structuring)
# - LLM final answer generation
#
# Requirements (from your utils):
#   - call_qwen(messages=messages) -> str
#   - get_embedding(text) -> List[float] or np.ndarray
#
# HotpotQA QA item fields: {"question", "answer", "_id"}
# HotpotQA corpus item fields: {"title", "text"}  (total ~9800 items)

import os
import json
import re
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Adjust sys.path to import your utils if needed
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)

from utils import wrapper_call_model, get_embedding
from utils import DEFAULT_LLM_NAME, DEFAULT_EMBEDDING_MODEL_NAME
from funcs_eval import single_exact_match, single_f1_score, extract_reasoning_info,build_messages_for_qa
from memory_reasoning.prompt_reasoning import DefaultSemanticPrompt



# -----------------------
# LLM prompts
# -----------------------
REASON_SYSTEM = "You are a retrieval reasoning module for multi-hop QA."
REASON_USER_TEMPLATE = """You are given retrieved passages and a question.
Extract ONLY the minimal set of factual statements that are directly useful to answer the question.
If the passages are insufficient, extract what is relevant anyway.

Return in this exact format:

### Information
- ...

Question:
{question}

Retrieved Passages:
{passages}
"""

ANSWER_SYSTEM = "You are a helpful assistant. Answer concisely and exactly."
ANSWER_USER_TEMPLATE = (
            "You are given retrieved facts from an external memory.\n"
            "Answer the question based on the context and your knowledge. \n"
            "Extract substring from the context as the answer. If extracting is hard, generate the answer from your own knowledge. \n"
            "And the answer is always a short answer with few words/phrases.\n"
            "DO NOT include anything else like reasoning or process or explanation before or after your answer!! \n\n"
            "Here is the input: \n"
            "Retrieved:\n{information}\n"
            "Question: {question}\n\n"
            "Output format: \n<short answer to the question>\n\n"
        )


# -----------------------
# Embedding + retrieval
# -----------------------
def to_np(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def cosine_topk(
    corpus_matrix: np.ndarray,
    corpus_norms: np.ndarray,
    query_vec: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    qn = np.linalg.norm(query_vec) + 1e-12
    dots = corpus_matrix @ query_vec
    sims = dots / (corpus_norms * qn + 1e-12)

    if k >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx_part = np.argpartition(-sims, k)[:k]
        idx = idx_part[np.argsort(-sims[idx_part])]
    return idx, sims[idx]



# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str, default="hotpotqa")
    parser.add_argument("--qa_model_name", type=str, default=DEFAULT_LLM_NAME,)
    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME,)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_qa_items", type=int, default=100)
    parser.add_argument("--max_corpus_items", type=int, default=1000)
    parser.add_argument("--qa_start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write_every", type=int, default=10)
    parser.add_argument("--granularity", type=str, default="paragraph", choices=["paragraph", "sentence"])
    parser.add_argument("--use_reasoning", action="store_true",
                        help="If set, run an LLM 'reasoning' compression step before answering.")
    args = parser.parse_args()

    # -------- Extract CLI args first (avoid args.xxx everywhere) --------
    bench_name = args.bench_name
    dir_path = f"/data/xuan/workdir/PlugMem-exp/data_{bench_name}_no_structuring"
    os.environ["DIR_PATH"] = dir_path
    
    qa_model_name = args.qa_model_name
    embedding_model_name = args.embedding_model_name
    os.environ["EMBEDDING_MODEL_NAME"] = embedding_model_name
    os.environ["LLM_NAME"] = qa_model_name
    
    topk = args.topk
    max_qa_items = args.max_qa_items
    max_corpus_items = args.max_corpus_items
    qa_start_idx = args.qa_start_idx
    seed = args.seed
    write_every = args.write_every
    granularity = args.granularity
    use_reasoning_gate = args.use_reasoning
    np.random.seed(seed)
    
    output_dir = dir_path
    os.makedirs(output_dir, exist_ok=True)
    embed_cache = os.path.join(output_dir,f"no_structuring_embed_{granularity}.npz")
    
    name_suffix = ""
    if granularity == "paragraph":
        name_suffix += "_paragraph"
    elif granularity == "sentence":
        name_suffix += "_sentence"
        
    if not use_reasoning_gate:
        name_suffix += "_no_reasoning"
    if max_qa_items > 0:
        name_suffix += f"_{max_qa_items}qa"
    if topk > 0:
        name_suffix += f"_topk={topk}"
    
    pred_path = os.path.join(output_dir, f"predictions{name_suffix}.json")
    metrics_path = os.path.join(output_dir, f"metrics{name_suffix}.json")
    
    if bench_name == "hotpotqa":
        qa_path = "../../hotpotqa_hipporag/hotpotqa.json"
        corpus_path = "../../hotpotqa_hipporag/hotpotqa_corpus.json"
    elif bench_name == "musique":
        qa_path = "../../hotpotqa_hipporag/musique.json"
        corpus_path = "../../hotpotqa_hipporag/musique_corpus.json"
    else:
        raise ValueError(f"Unsupported benchmark name: {bench_name}")
    
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    if qa_start_idx > 0:
        qa_data = qa_data[qa_start_idx:]

    if max_qa_items > 0:
        qa_data = qa_data[:max_qa_items]
    
    if max_corpus_items > 0:
        corpus = corpus[:max_corpus_items]
    print(f"[Info] Using {len(qa_data)} QA items and {len(corpus)} corpus items")
    
    # ---------- Build/load corpus embeddings ----------
    cache_path = ""
    if embed_cache:
        cache_path = embed_cache
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(output_dir, cache_path)

    corpus_texts: List[str] = []
    if granularity == "paragraph":
        for item in corpus:
            title = item.get("title", "").strip()
            text = item.get("text", "").strip()
            chunk = f"Title: {title}\nText: {text}".strip()
            corpus_texts.append(chunk)
    elif granularity == "sentence":
        for item in qa_data:
            for _, sentences in item["context"]:
                corpus_texts.extend(sentences)

    print(f"===== len(corpus_texts): {len(corpus_texts)}")
    
    if cache_path and os.path.exists(cache_path):
        print(f"[Cache] Loading corpus embeddings from {cache_path}")
        npz = np.load(cache_path, allow_pickle=False)
        corpus_matrix = npz["E"].astype(np.float32)
        corpus_norms = npz["N"].astype(np.float32)
    else:
        print(f"[Embed] Computing corpus embeddings for {len(corpus_texts)} items ...")
        emb_list = []
        for i, txt in tqdm(enumerate(corpus_texts)):
            emb = to_np(get_embedding(txt,embedding_model_name))
            emb_list.append(emb)
            if (i + 1) % 50 == 0:
                print(f"  embedded {i+1}/{len(corpus_texts)}")

        corpus_matrix = np.stack(emb_list, axis=0).astype(np.float32)
        corpus_norms = np.linalg.norm(corpus_matrix, axis=1) + 1e-12

        if cache_path:
            print(f"[Cache] Saving corpus embeddings to {cache_path}")
            np.savez(cache_path, E=corpus_matrix, N=corpus_norms)
    
    
    # ---------- Eval loop ----------
    total_em, total_f1 = 0.0, 0.0
    n = 0

    records=[]
    for qa_item in qa_data:
        qid = qa_item.get("_id", str(n))
        question = qa_item["question"]
        gold_ans = qa_item.get("answer", "")

        q_emb = to_np(get_embedding(question,embedding_model_name))
        top_idx, top_sims = cosine_topk(corpus_matrix, corpus_norms, q_emb, topk)

        passages = []
        for rank, (ci, sim) in enumerate(zip(top_idx.tolist(), top_sims.tolist()), start=1):
            passages.append(f"[{rank}] (sim={sim:.4f}) {corpus_texts[ci]}")
        passages_str = "\n\n".join(passages)
        print(f"\n ----- retrieved passages ----- : \n{passages_str}")
        
        # (Optional) reasoning compression step
        if use_reasoning_gate:
            goal = "Answer the question"
            variables = {
                "goal": goal,
                "subgoal": "", 
                "state": "",
                "observation": question,
                "semantic_memory": passages_str,
                "procedural_memory": "",
                "episodic_memory_semantic": "",
                "episodic_memory_procedural": "",
                "time": "",
            }
            prompt_obj = DefaultSemanticPrompt()
            retrieval_messages = prompt_obj.build_messages(variables)
            retrieval_messages = [{"role": m.role, "content": m.content} for m in retrieval_messages]
            raw_reasoning = wrapper_call_model(model_name=qa_model_name,messages=retrieval_messages)
            info = extract_reasoning_info(raw_reasoning)
            if not info:
                info = raw_reasoning.strip()

        else:
            info = passages_str

        # final answer
        messages = build_messages_for_qa(info=info,question=question)
        pred = wrapper_call_model(model_name=qa_model_name,messages=messages).strip()
        print(f"\n ----- question ----- : {question}")
        print(f"\n ----- gold answer ----- : {gold_ans}")
        print(f"\n ----- model answer ----- : {pred}")
        
        em = single_exact_match(pred, gold_ans) if gold_ans else 0.0
        f1 = single_f1_score(pred, gold_ans) if gold_ans else 0.0

        total_em += em
        total_f1 += f1
        n += 1

        record: Dict[str, Any] = {
            "id": qid,
            "question": question,
            "gold": gold_ans,
            "pred": pred,
            "em": em,
            "f1": f1,
            "topk": topk,
            "use_reasoning": use_reasoning_gate,
            "reasoning_info": info,
            "retrieved_indices": top_idx.tolist(),
            "retrieved_sims": [float(x) for x in top_sims.tolist()],
        }
        records.append(record)
        
        with open(pred_path, "w", encoding="utf-8") as fout:
            json.dump(records,fout, indent=4, ensure_ascii=False)

        if n % write_every == 0:
            print(f"[{n}/{len(qa_data)}] EM={total_em/n:.4f} F1={total_f1/n:.4f}")

    metrics = {
        "count": n,
        "em": (total_em / n) if n else 0.0,
        "f1": (total_f1 / n) if n else 0.0,
        "topk": topk,
        "use_reasoning": use_reasoning_gate,
        "qa_path": qa_path,
        "corpus_path": corpus_path,
        "output_dir": output_dir,
        "embed_cache": cache_path,
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[Done]", metrics)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time cost: {end_time - start_time:.2f} seconds")
    

"""
export DIR_PATH=""
mkdir -p $DIR_PATH/logs
nohup env \
  PYTHONUNBUFFERED=1 \
  VLLM_QWEN_API_KEY="" \
  TOKEN_USAGE_FILE=usage/eval_hotpotqa_no_structuring_1000_topk=10_sentence.jsonl \
  bash -lc 'python eval_qa_no_structuring.py --bench_name hotpotqa --topk 10 --granularity sentence --use_reasoning --qa_start_idx 0 --max_qa_items 1000 --max_corpus_items 10000' \
  > $DIR_PATH/logs/eval_$(date +%F_%H-%M-%S).log 2>&1 &
echo $! > run1.pid
disown
"""
