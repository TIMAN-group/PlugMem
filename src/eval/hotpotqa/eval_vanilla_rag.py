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
import time
import json
import re
import argparse
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Adjust sys.path to import your utils if needed
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)



from memory_reasoning.prompt_reasoning import DefaultSemanticPrompt
from utils import wrapper_call_model, get_embedding
from utils import DEFAULT_LLM_NAME, DEFAULT_EMBEDDING_MODEL_NAME
from funcs_eval import single_exact_match, single_f1_score, extract_reasoning_info, build_messages_for_qa
from funcs_eval import HOTPOTQA_QA_PATH, MUSIQUE_QA_PATH, \
                                HOTPOTQA_CORPUS_PATH, MUSIQUE_CORPUS_PATH, \
                                HOTPOTQA_TRACE_PATH, MUSIQUE_TRACE_PATH, \
                                TRACE_FIELDS_ORDER

DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOPK = 5
DEFAULT_CHUNK_STRIDE = DEFAULT_CHUNK_SIZE - DEFAULT_CHUNK_OVERLAP



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


# -----------------------
# Embedding + retrieval
# -----------------------
def _to_np(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _cosine_topk(
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



def _trace_dict_to_text(d: Dict[str, Any]) -> str:
    """
    Convert one trace dict to a text block. Skip None/empty fields.
    Format: "field: value" each on a new line.
    """
    parts = []
    for k in TRACE_FIELDS_ORDER:
        v = d.get(k, None)
        if v is None:
            continue
        if isinstance(v, str):
            vv = v.strip()
            if not vv:
                continue
            parts.append(f"{k}: {vv}")
        else:
            # 非字符串也转成字符串（如 reward 是数字）
            parts.append(f"{k}: {v}")
    return "\n".join(parts).strip()


def _chunk_text_stream(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Chunk a long text stream by characters with fixed overlap.
    """
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += stride
    return chunks


def build_index_from_raw_traces(
    raw_traces_path: str,
    embedding_model_name: str,
    cache_path: Optional[str],
    max_corpus_items: int = -1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Read raw_traces.json -> flatten to 1D text stream -> chunk -> embed -> cache.
    Returns:
      chunks: List[str]
      corpus_matrix: (M, D) float32
      corpus_norms: (M,) float32
    """

    with open(raw_traces_path, "r", encoding="utf-8") as f:
        traces = json.load(f)
    
    if not isinstance(traces, list):
        raise ValueError("raw_traces.json must be a list of dicts")

    if max_corpus_items and max_corpus_items > 0:
        traces = traces[:max_corpus_items]
    
    random.shuffle(traces)

    # flatten to 1D text stream
    blocks: List[str] = []
    for d in traces:
        if not isinstance(d, dict):
            continue
        blk = _trace_dict_to_text(d)
        if blk:
            blocks.append(blk)

    # 这里用分隔符降低不同 trace 粘连带来的噪声
    text_stream = "\n\n---\n\n".join(blocks)

    chunks = _chunk_text_stream(text_stream, chunk_size=chunk_size, overlap=overlap)
    print(f"[Index] Building {len(chunks)} chunks from {len(traces)} raw trace items")

    if cache_path and os.path.exists(cache_path):
        print(f"[Cache] Loading embeddings from {cache_path}")
        npz = np.load(cache_path, allow_pickle=False)
        corpus_matrix = npz["E"].astype(np.float32)
        corpus_norms = npz["N"].astype(np.float32)

        # 简单一致性检查：cache 的 chunk 数最好匹配
        if corpus_matrix.shape[0] != len(chunks):
            print("[Warn] Cache rows != current chunks count. Recomputing embeddings ...")
        else:
            return chunks, corpus_matrix, corpus_norms

    # compute embeddings
    print(f"[Embed] Computing embeddings for {len(chunks)} chunks ...")
    emb_list = []
    for i, txt in tqdm(enumerate(chunks), total=len(chunks)):
        emb = _to_np(get_embedding(txt, embedding_model_name))
        emb_list.append(emb)
        if (i + 1) % 200 == 0:
            print(f"  embedded {i+1}/{len(chunks)}")

    corpus_matrix = np.stack(emb_list, axis=0).astype(np.float32)
    corpus_norms = np.linalg.norm(corpus_matrix, axis=1) + 1e-12

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"[Cache] Saving embeddings to {cache_path}")
        np.savez(cache_path, E=corpus_matrix, N=corpus_norms)

    return chunks, corpus_matrix, corpus_norms



def run_eval_rag(
    qa_data: List[Dict[str, Any]],
    chunks: List[str],
    corpus_matrix: np.ndarray,
    corpus_norms: np.ndarray,
    qa_model_name: str,
    embedding_model_name: str,
    topk: int,
    no_reasoning_gate: bool,
    pred_path: str,
    metrics_path: str,
    write_every: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    total_em, total_f1 = 0.0, 0.0
    n = 0
    records: List[Dict[str, Any]] = []

    for qa_item in qa_data:
        qid = qa_item.get("_id", str(n))
        question = qa_item["question"]
        gold_ans = qa_item.get("answer", "")
        
        q_emb = _to_np(get_embedding(question, embedding_model_name))
        top_idx, top_sims = _cosine_topk(corpus_matrix, corpus_norms, q_emb, topk)

        passages = []
        for rank, (ci, sim) in enumerate(zip(top_idx.tolist(), top_sims.tolist()), start=1):
            passages.append(f"[{rank}] (sim={sim:.4f}) {chunks[ci]}")
        passages_str = "\n\n".join(passages)

        # Optional reasoning compression
        if no_reasoning_gate:
            info = passages_str
        else:
            variables = {
                "goal": "Answer the question",
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

            raw_reasoning = wrapper_call_model(model_name=qa_model_name, messages=retrieval_messages)
            info = extract_reasoning_info(raw_reasoning) or raw_reasoning.strip()
            

        # Final answer
        messages = build_messages_for_qa(info, question)
        pred = wrapper_call_model(
            model_name=qa_model_name,
            messages=messages,
        ).strip()

        em = single_exact_match(pred, gold_ans) if gold_ans else 0.0
        f1 = single_f1_score(pred, gold_ans) if gold_ans else 0.0
        print(f"[eval] ---- retrived raw ----- : {passages_str}")
        print(f"[eval] ---- reasoning info ----- : {info}")
        print(f"[eval] ---- question ----- : {question}")
        print(f"[eval] ---- gold ----- : {gold_ans}")
        print(f"[eval] ---- pred ----- : {pred}")
        print(f"[eval] [{n}-th item of {len(qa_data)}] EM={em:.4f} F1={f1:.4f}")
        
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
            "no_reasoning": no_reasoning_gate,
            "reasoning_info": info,
            "retrieved_indices": top_idx.tolist(),
            "retrieved_sims": [float(x) for x in top_sims.tolist()],
        }
        records.append(record)

        # write predictions every step (保持你原行为)
        with open(pred_path, "w", encoding="utf-8") as fout:
            json.dump(records, fout, indent=4, ensure_ascii=False)

        if n % write_every == 0:
            print(f"[eval] [{n}/{len(qa_data)}] total EM={total_em/n:.4f} total F1={total_f1/n:.4f}")

    metrics = {
        "count": n,
        "em": (total_em / n) if n else 0.0,
        "f1": (total_f1 / n) if n else 0.0,
        "topk": topk,
        "no_reasoning": no_reasoning_gate,
        "pred_path": pred_path,
        "metrics_path": metrics_path,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[Done]", metrics)
    return records, metrics



# -----------------------
# Main
# -----------------------
def main():
    # ---- Extract args ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str, default="hotpotqa")
    parser.add_argument("--qa_model_name", type=str, default=DEFAULT_LLM_NAME)
    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--max_qa_items", type=int, default=100)
    parser.add_argument("--max_corpus_items", type=int, default=1000)
    parser.add_argument("--corpus_shuffle_seed", type=int, default=42)
    parser.add_argument("--write_every", type=int, default=10)
    parser.add_argument("--no_reasoning", action="store_true")
    # new: chunk config
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    
    args = parser.parse_args()

    bench_name = args.bench_name
    qa_model_name = args.qa_model_name
    embedding_model_name = args.embedding_model_name

    topk = args.topk
    max_qa_items = args.max_qa_items
    max_corpus_items = args.max_corpus_items
    corpus_shuffle_seed = args.corpus_shuffle_seed
    write_every = args.write_every
    no_reasoning_gate = args.no_reasoning
    
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    
    
    
    # ---- Env ----
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if DIR_PATH is None:
        raise ValueError("DIR_PATH is not set")
    output_dir = DIR_PATH
    os.makedirs(output_dir, exist_ok=True)
    os.environ["EMBEDDING_MODEL_NAME"] = embedding_model_name
    os.environ["LLM_NAME"] = qa_model_name
    np.random.seed(corpus_shuffle_seed)
    random.seed(corpus_shuffle_seed)
    

    # ---- paths ----
    cache_path = os.path.join(output_dir, "cache", f"raw_traces_embed_cs{chunk_size}_ov{chunk_overlap}_max{max_corpus_items}_corpus_items.npz")
    
    name_suffix=f"{max_qa_items}qa_cs{chunk_size}_ov{chunk_overlap}_top{topk}"    
    if no_reasoning_gate:
        name_suffix += "_no_reasoning"

    pred_path = os.path.join(output_dir, f"predictions_{name_suffix}.json")
    metrics_path = os.path.join(output_dir, f"metrics_{name_suffix}.json")
    
    # ---- QA data path ----
    if bench_name == "hotpotqa":
        qa_path = HOTPOTQA_QA_PATH
        raw_traces_path = HOTPOTQA_TRACE_PATH
    elif bench_name == "musique":
        qa_path = MUSIQUE_QA_PATH
        raw_traces_path = MUSIQUE_TRACE_PATH
    else:
        raise ValueError(f"Unsupported benchmark name: {bench_name}")
    
    # ---- load QA data ----
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    if max_qa_items > 0:
        qa_data = qa_data[:max_qa_items]
    
    # ---- log configs ----
    print("configs: "
        f"qa_model_name: {qa_model_name}\n"
        f"embedding_model_name: {embedding_model_name}\n"
        f"topk: {topk}\n"
        f"max_qa_items: {max_qa_items}\n"
        f"max_corpus_items: {max_corpus_items}\n"
        f"corpus_shuffle_seed: {corpus_shuffle_seed}\n"
        f"write_every: {write_every}\n"
        f"no_reasoning_gate: {no_reasoning_gate}\n"
        f"pred_path: {pred_path}\n"
        f"metrics_path: {metrics_path}\n"
        f"raw_traces_path: {raw_traces_path}\n"
        f"chunk_size: {chunk_size}\n"
        f"chunk_overlap: {chunk_overlap}\n"
        f"token_usage_file: {os.environ.get('TOKEN_USAGE_FILE',None)}\n"
        f"==========================================================\n"
    )
    
    # --- indexing ---
    chunks, corpus_matrix, corpus_norms = build_index_from_raw_traces(
        raw_traces_path=raw_traces_path,
        embedding_model_name=embedding_model_name,
        cache_path=cache_path,
        max_corpus_items=max_corpus_items,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    # --- eval ---
    run_eval_rag(
        qa_data=qa_data,
        chunks=chunks,
        corpus_matrix=corpus_matrix,
        corpus_norms=corpus_norms,
        qa_model_name=qa_model_name,
        embedding_model_name=embedding_model_name,
        topk=topk,
        no_reasoning_gate=no_reasoning_gate,
        pred_path=pred_path,
        metrics_path=metrics_path,
        write_every=write_every,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")





"""
export DIR_PATH=""
mkdir -p $DIR_PATH/logs
nohup env \
  PYTHONUNBUFFERED=1 \
  VLLM_QWEN_API_KEY="" \
  TOKEN_USAGE_FILE=usage_hotpotqa/eval_hotpotqa_vanilla_chunk_1000_cs512_ov128.jsonl \
  bash -lc 'python eval_vanilla_rag.py --bench_name hotpotqa --topk 5 --no_reasoning --chunk_size 512 --chunk_overlap 128 --max_qa_items 1000 --max_corpus_items 10000' \
  > $DIR_PATH/logs/run.$(date +%F_%H%M%S).log 2>&1 &
echo $! > vanilla_rag_run.pid
disown
"""