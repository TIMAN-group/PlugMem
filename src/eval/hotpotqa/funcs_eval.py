import os
import json
import re
import time
from typing import List, Dict, Any, Tuple, Set, Optional
import sys
import argparse
import string
from collections import Counter
from collections import defaultdict
import random
from string import Template

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)
from utils import call_qwen, call_gpt


HOTPOTQA_QA_PATH="../../bench_data/hotpotqa_hipporag/hotpotqa.json"   
HOTPOTQA_CORPUS_PATH="../../bench_data/hotpotqa_hipporag/hotpotqa_corpus.json"
HOTPOTQA_TRACE_PATH="../../bench_data/hotpotqa_hipporag/hotpotqa_oas_traces.json"
MUSIQUE_CORPUS_PATH="../../bench_data/hotpotqa_hipporag/musique_corpus.json"
MUSIQUE_QA_PATH="../../bench_data/hotpotqa_hipporag/musique.json"
MUSIQUE_TRACE_PATH="../../bench_data/hotpotqa_hipporag/musique_oas_traces.json"
TRACE_FIELDS_ORDER = ["observation", "action", "state", "reward", "subgoal"]

HOTPOTQA_PREFIX = (
    "You are given retrieved facts from an external memory.\n"
    "Answer the question based on the retrieved facts and your knowledge.\n"
    "Try your best to extract a substring from the retrieved facts (question not included) as the answer.\n"
    "If extracting is hard or provided info is not enough, generate the answer from your **OWN KNOWLEDGE**.\n"
    "The answer is ALWAYS SHORT! For yes/no questions, answer only 'yes' or 'no'.\n"
    "DO NOT include anything else like reasoning or process or explanation before or after your answer!!\n"
)

MAX_CONTEXT_MAPPING = {
    "gpt-4o":128_000,
    "gpt-4o-mini":128_000,
    "gpt-5-mini":128_000,
    "qwen-2.5-7b-instruct":32768,
    "qwen-2.5-14b-instruct":32768,
    "qwen-2.5-72b-instruct":32768,
}

RESERVED_TOKENS = 4_000


        
# ----------------------------
# (EM / F1) metrics
# ----------------------------
def _normalize_answer(s: str) -> str:

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def single_f1_score(pred: str, gold: str) -> float:
    gold_tokens = _normalize_answer(gold).split()
    predicted_tokens = _normalize_answer(pred).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def single_exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0


# ----------------------------
# Question Rephrasing
# ----------------------------
def rephrase_question(question: str, answer: str, max_try: int = 3, sleep_sec: float = 0.3) -> str:
    
    def _parse_rephrased_question(response: str) -> Optional[str]:
        _REPHRASED_PATTERN = re.compile(
            r"###\s*Rephrased Question\s*\n(.+)",
            flags=re.S
        )
        if not response or not isinstance(response, str):
            return None
        m = _REPHRASED_PATTERN.search(response)
        if not m:
            return None
        text = m.group(1).strip()
        if not text:
            return None
        # Keep first non-empty line to avoid the model adding extra sections
        line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if not line:
            return None
        # Normalize ending punctuation
        if not line.endswith("?"):
            line += "?"
        return line
    
    system_prompt = (
        "You are a careful question rewriter for QA datasets. "
        "Rewrite the user's question to remove ambiguity while preserving meaning. "
        "Do not answer the question."
    )

    user_prompt = (
        "Rewrite the following question to be as clear and unambiguous as possible.\n\n"
        "You are given:\n"
        "- The original question.\n"
        "- Its gold answer (ground-truth answer).\n"
        "Use the gold answer ONLY to resolve ambiguity (e.g., unclear references, missing entity names), "
        "but do NOT reveal the answer or change what is being asked.\n\n"
        "Rules:\n"
        "- Preserve the original intent and target answer.\n"
        "- Use the gold answer only to clarify which entity/event the question refers to when there is ambiguity.\n"
        "- Do NOT state, paraphrase, or leak the answer itself in the rephrased question.\n"
        "- Resolve ambiguous references by explicitly stating what is being asked.\n"
        "- Prefer a single, direct interrogative sentence.\n"
        "- Do NOT add external facts, names, or guesses beyond what is implied by the question and answer.\n"
        "- Output EXACTLY in this format (no extra text):\n\n"
        "### Rephrased Question\n"
        "<one clear question>\n\n"
        "Question:\n"
        f"{question}\n"
        "Gold Answer:\n"
        f"{answer}"
    )


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    for attempt in range(1, max_try + 1):
        try:
            response = call_gpt(messages=messages)
            rephrased = _parse_rephrased_question(response)
            if rephrased is not None:
                return rephrased
        except Exception as e:
            last_err = e

        # small backoff before retry (avoid hammering API / transient failures)
        if attempt < max_try:
            time.sleep(sleep_sec * attempt)

    if last_err is not None:
        print(f"[WARN] Rephrase failed after {max_try} attempts due to error: {last_err}. Returning original question.")
    else:
        print(f"[WARN] Rephrase failed after {max_try} attempts (bad format). Returning original question.")
    return question


# ----------------------------
# QA prompt for HotpotQA
# ----------------------------
def build_messages_for_qa(info: str, question: str) ->str:
    user_prompt=(
            HOTPOTQA_PREFIX +
            f"Retrieved information:\n{info}\n"
            f"Question: {question}\n\n"
            "Output:"
        )
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]
    return messages

    
# ----------------------------
# Gold Context Extraction for oracle setting
# ----------------------------
def extract_gold_context(qa_item, bench_name="hotpotqa", sep="\n",include_title=False):
    if bench_name == "hotpotqa":
        return extract_gold_context_hotpotqa(qa_item,sep,include_title)
    elif bench_name == "musique":
        return extract_gold_context_musique(qa_item,sep,include_title)
    else:
        raise ValueError(f"Unsupported benchmark name: {bench_name}")


def extract_gold_context_hotpotqa(qa_item,sep="\n",include_title=False):
    title2sents = {title: sents for title, sents in qa_item["context"]}
    idxs = defaultdict(set)
    for title, sent_idx in qa_item["supporting_facts"]:
        idxs[title].add(sent_idx)
    out = []
    for title, sents in qa_item["context"]:
        if title not in idxs:
            continue
        for i in sorted(idxs[title]):
            if 0 <= i < len(sents):
                if include_title:
                    out.append(f"{title}\n{sents[i]}")
                else:
                    out.append(sents[i])
    return out, sep.join(out)


def extract_gold_context_musique(qa_item,sep="\n",include_title=False):
    paragraphs = qa_item.get("paragraphs", [])
    if not isinstance(paragraphs, list):
        return ""

    supp_idxs = []
    for p in paragraphs:
        if isinstance(p, dict) and p.get("is_supporting") is True and isinstance(p.get("idx"), int):
            supp_idxs.append(p["idx"])

    # 2) Fallback: use question_decomposition paragraph_support_idx
    if not supp_idxs:
        decomp = qa_item.get("question_decomposition", [])
        if isinstance(decomp, list):
            for step in decomp:
                if isinstance(step, dict) and isinstance(step.get("paragraph_support_idx"), int):
                    supp_idxs.append(step["paragraph_support_idx"])

    supp_set = set(supp_idxs)
    if not supp_set:
        return [], ""

    # Build idx -> paragraph mapping (idx should be unique)
    idx2p = {}
    for p in paragraphs:
        if isinstance(p, dict) and isinstance(p.get("idx"), int):
            idx2p[p["idx"]] = p

    out: List[str] = []
    for idx in sorted(supp_set):
        p = idx2p.get(idx)
        if not p:
            continue
        text = p.get("paragraph_text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        title = p.get("title", "")
        if include_title:
            if isinstance(title, str) and title.strip():
                out.append(f"{title}\n{text}")
            else:
                out.append(text)
        else:
            out.append(text)

    return out, sep.join(out)

# ----------------------------
# Long Context Simulation - randomly select corpus items until ctx window limit
# ----------------------------
def simulate_long_context(
    qa_item, 
    encoder,
    doc_tokens: List[List[int]],
    doc_lens: List[int],
    max_context_tokens: int = 128_000,
    reserved_for_other: int = RESERVED_TOKENS,
) -> Tuple[str, List[int]]:
    question = qa_item['question']
    
    q_tokens = len(encoder.encode(question))

    budget = max_context_tokens - q_tokens - reserved_for_other
    if budget <= 0:
        raise ValueError("too small budget to input question only with no context")

    n = len(doc_tokens)
    indices = list(range(n))
    random.shuffle(indices)

    picked = []
    total = 0

    for idx in indices:
        l = doc_lens[idx]
        if total + l > budget:
            continue
        picked.append(idx)
        total += l
        
        # stop early
        if total > budget * 0.99:
            break

    sep_tokens = encoder.encode("\n\n")
    joined_tokens = []
    for i, idx in enumerate(picked):
        if i > 0:
            joined_tokens.extend(sep_tokens)
        joined_tokens.extend(doc_tokens[idx])

    context_str = encoder.decode(joined_tokens)
    return context_str, picked


# ----------------------------
# No-retrieving Simulation - randomly select top-k items
# ----------------------------
def precache_semantic_facts(memory_dir: str) -> List[str]:
    semantic_dir = os.path.join(memory_dir, "semantic_memory")
    if not os.path.isdir(semantic_dir):
        raise FileNotFoundError(f"semantic_memory dir not found: {semantic_dir}")

    files = sorted([fn for fn in os.listdir(semantic_dir) if fn.endswith(".json")])
    facts: List[str] = []
    for fn in files:
        path = os.path.join(semantic_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            sem = obj.get("semantic_memory", "")
            if isinstance(sem, str) and sem.strip():
                facts.append(sem.strip())
        except Exception:
            continue

    if not facts:
        raise ValueError("No semantic_memory facts loaded from disk.")
    return facts

def random_ctx_for_no_retrieving_mode(
    question: str,
    all_semantic_facts: List[str],
    k: int,
    rng: random.Random,
    prompt_obj,
) -> Tuple[List[Dict[str, str]], Dict[str, str], str, str]:
    """
    Same behavior as your no_retrieving baseline:
      - randomly sample k semantic facts (from cached list)
      - build retrieval_messages using DefaultSemanticPrompt
    """
    sampled = rng.sample(all_semantic_facts, min(k, len(all_semantic_facts)))
    semantic_memory_str = "\n".join(
        [f"Relevant Fact {i}: {sem}" for i, sem in enumerate(sampled)]
    ).strip()

    variables = {
        "goal": "Answer the question",
        "subgoal": "",
        "state": "",
        "observation": question,
        "semantic_memory": semantic_memory_str,
        "procedural_memory": "",
        "episodic_memory_semantic": "",
        "episodic_memory_procedural": "",
        "time": "",
    }
    sel_type = "semantic_memory"
    built = prompt_obj.build_messages(variables)
    messages = [{"role": m.role, "content": m.content} for m in built]

    return messages, variables, sel_type, semantic_memory_str


# ----------------------------
# Reasoning output parsing helpers
# ----------------------------
def extract_reasoning_info(text: str) -> str:
    INFO_PATTERNS = [
        r"###\s*Information\s*\n(.*)",
        r"###\s*Relevant\s*Information\s*\n(.*)",
        r"###\s*Evidence\s*\n(.*)",
    ]
    for pat in INFO_PATTERNS:
        m = re.search(pat, text, re.S | re.I)
        if m:
            return m.group(1).strip()
    return text


