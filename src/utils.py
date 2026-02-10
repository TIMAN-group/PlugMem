import numpy as np
import time
import sys
import json
import requests
import openai
from openai import OpenAI, AzureOpenAI
import re
import os
import logging
import traceback
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Set, Optional

MAX_TRY=5
DEFAULT_EMBEDDING_MODEL_NAME = "NV-Embed-v2"
DEFAULT_LLM_NAME = "qwen-2.5-32b-instruct"
DEFAULT_LLM_NAME_ALIAS = ["qwen-2.5-32b-instruct",
                    "qwen2.5-32b-instruct",
                    "Qwen2.5-7B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct", 
                    "CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic"]


class PrintLogger:
    """A minimal logger-like object whose .info/.debug/... just call print."""
    def info(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        print(msg)

    def debug(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        print(msg)

    def warning(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        print(msg)

    def error(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        print(msg)

    def exception(self, msg, *args, **kwargs):
        # optional: you can also print stack trace from kwargs.get("exc_info")
        if args:
            msg = msg % args
        print(msg)


def set_logger(
    log_file: str | None = None,  
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = "[%(asctime)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler -> stdout
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Only add file handler if log_file is provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger

# ----------------------------
# LLM API
# ----------------------------
def wrapper_call_model(model_name: str=None,messages:List[Dict[str, str]]=None,prompt=None,temperature=0, top_p=1.0, max_tokens=4096, token_usage_file=None,) -> str:
    model_name = model_name or os.environ.get("LLM_NAME",None) or DEFAULT_LLM_NAME
    token_usage_file = os.environ.get("TOKEN_USAGE_FILE", None)  # *.jsonl
    if token_usage_file is not None:
        token_usage_file = token_usage_file.replace(".jsonl", f"_{model_name}.jsonl")
        
    if model_name.lower() in DEFAULT_LLM_NAME_ALIAS:
        return call_qwen(messages=messages, prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, token_usage_file=token_usage_file)  
    elif model_name in ["4o","4o-mini","4.1","o1","5.1","5.2"]:
        return call_gpt(model_id=model_name, messages=messages, prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, token_usage_file=token_usage_file)
    elif model_name.lower() == "deepseek-v3-0324":
        return call_dpsk(model_id="DeepSeek-V3-0324", messages=messages, prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, token_usage_file=token_usage_file)
    return call_llm_openrouter_api(model_name=model_name, messages=messages, prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, token_usage_file=token_usage_file)



def call_llm_openrouter_api(model_name,messages=None,prompt=None,temperature=0, top_p=1.0, max_tokens=4096, token_usage_file=None, ):
    url = f"https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
               }
    
    if messages == None:
        messages = [
            {"role": "user", "content": prompt}
        ]
    
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    for attempt in range(1, MAX_TRY + 1):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            usage = response.json()["usage"]
            response.raise_for_status()  
            usage = response.json()["usage"]
            usage_log = token_usage_file or os.environ.get("TOKEN_USAGE_FILE") or None
            if usage_log is not None:
                with open(usage_log, "a") as f:
                    usage_json={"model": model_name,"prompt_first100": messages[1]["content"][:100]}
                    usage_json.update(usage)
                    f.write(json.dumps(usage_json) + "\n")
            return response.json()["choices"][0]["message"]["content"]

        except ImportError:
            print("Need to install openai library: pip install openai")

        except requests.exceptions.SSLError as e:
            print(f"[Attempt {attempt}/{MAX_TRY}] SSL error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"[Attempt {attempt}/{MAX_TRY}] Request error: {e}")

        time.sleep(5)
    return  ""


def call_qwen(prompt = None, messages = None, temperature=0, top_p=1.0, max_tokens=4096, token_usage_file=None, system_prompt="You are a helpful assistant."):
    VLLM_QWEN_API_KEY = os.environ["VLLM_QWEN_API_KEY"]
    QWEN_BASE_URL = os.environ["QWEN_BASE_URL"]
    client = OpenAI(
        base_url=QWEN_BASE_URL,
        api_key=VLLM_QWEN_API_KEY,
    )
    
    if messages == None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    model_id = "CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic"
    for attempt in range(1, MAX_TRY + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )    
            usage = response.usage
            usage_log = token_usage_file or os.environ.get("TOKEN_USAGE_FILE") or None
            if usage_log is not None:
                with open(usage_log, "a") as f:
                    usage_json={"model": model_id,"prompt_first100": messages[1]["content"][:100]}
                    usage_json.update(usage.model_dump())
                    f.write(json.dumps(usage_json) + "\n")
            return response.choices[0].message.content

        except ImportError:
            print("Need to install openai library: pip install openai")

        except Exception as e:
            print(f"[Attempt {attempt}/{MAX_TRY}]. Error: {e}")

        time.sleep(5)
    return  ""



def call_dpsk(prompt = None, messages = None, model_id="DeepSeek-V3-0324", temperature=0, top_p=1.0, max_tokens=4096, token_usage_file=None, system_prompt="You are a helpful assistant.", ):
    import os
    from openai import AzureOpenAI

    api_version = "2024-05-01-preview"
    AZURE_DPSK_API_KEY = os.environ.get("AZURE_DPSK_API_KEY", None)
    AZURE_DPSK_ENDPOINT = os.environ.get("AZURE_DPSK_ENDPOINT", None)
    
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=AZURE_DPSK_ENDPOINT,
        api_key=AZURE_DPSK_API_KEY,
    )
    num_attempts = 0
    if messages == None:
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
        
    while True:
        if num_attempts >= MAX_TRY:
            raise ValueError("OpenAI request failed.")
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            usage = response.usage  # prompt_tokens, completion_tokens, total_tokens

            usage_log = token_usage_file or os.environ.get("TOKEN_USAGE_FILE") or None
            if usage_log is not None:
                with open(usage_log, "a") as f:
                    usage_json={"model": model_id,"prompt_first100": messages[1]["content"][:100]}
                    usage_json.update(usage.model_dump())
                    f.write(json.dumps(usage_json) + "\n")
            return response.choices[0].message.content.strip()
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1



def call_gpt(prompt = None, messages = None, model_id="gpt-4o", temperature=0, top_p=1.0, max_tokens=4096, token_usage_file=None, system_prompt="You are a helpful assistanct."):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
    client = OpenAI() if not AZURE_ENDPOINT else AzureOpenAI(azure_endpoint = AZURE_ENDPOINT, api_key=OPENAI_API_KEY, api_version="2024-12-01-preview")
    
    if messages == None:
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
        
    num_attempts = 0
    while True:
        if num_attempts >= MAX_TRY:
            raise ValueError("OpenAI request failed.")
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            usage = response.usage  # prompt_tokens, completion_tokens, total_tokens

            usage_log = token_usage_file or os.environ.get("TOKEN_USAGE_FILE") or None
            if usage_log is not None:
                with open(usage_log, "a") as f:
                    usage_json={"model": model_id,"prompt_first100": messages[1]["content"][:100]}
                    usage_json.update(usage.model_dump())
                    f.write(json.dumps(usage_json) + "\n")
            return response.choices[0].message.content.strip()
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1




# ----------------------------
# Embedding Model API
# ----------------------------
def get_embedding(text, embedding_model=None):
    MAX_TRY = 5
    
    model_name = "nvidia/NV-Embed-v2"
    url = os.environ["EMBEDDING_BASE_URL"]
    text = text[:8192]

    # print(f"using {model_name}")
    
    for attempt in range(1, MAX_TRY + 1):
        try:
            headers = {"Content-Type": "application/json",}
            data = {"model": model_name,"input": text,}
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            embedding = result["data"][0]["embedding"]
            # print("√ Embedding generated")
            return embedding

        except Exception as e:
            print(f"[Attempt {attempt}/{MAX_TRY}] error: {repr(e)}", flush=True)

        time.sleep(5.0)

    return None


# ----------------------------
# MemGraph Operations
# ----------------------------
def get_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def array_to_list(x: Any):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).tolist()
    elif isinstance(x, list):
        return x
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

def memory_load(filepath):
    with open(filepath,"r",) as input:
        memory = json.load(input)
    return memory

# ================ Default Version ================
def save_semantic(semantic_memory, tags, semantic_id, time):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    with open(
        DIR_PATH + f"/semantic_memory/semantic_memory_{semantic_id}.json",
        "w",
        encoding="utf-8",
    ) as output:
        _json = {
            "semantic_memory": semantic_memory,
            "tags": tags,
            "time": time,
        }
        json.dump(_json, output, indent=4, ensure_ascii=False)


# ================ Default Version ================
def update_semantic(*, semantic_id=None, semantic_memory=None, tags=None, time=None):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if semantic_id is None:
        print("update_semantic: semantic_id is None; skip updating.")
        return
    path = DIR_PATH + f"/semantic_memory/semantic_memory_{semantic_id}.json"
    update_fields = {
        "semantic_memory": semantic_memory,
        "tags": tags,
        "time": time,
    }
    if all(v is None for v in update_fields.values()):
        print("update_semantic: no fields provided; skip updating.")
        return
    with open(path, "r", encoding="utf-8") as f:
        semantic_data = json.load(f)
    for k, v in update_fields.items():
        if v is not None:
            semantic_data[k] = v
    with open(path, "w", encoding="utf-8") as output:
        json.dump(semantic_data, output, indent=4, ensure_ascii=False)


# ================ Default Version ================
def save_procedural(procedural_memory, procedural_id):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    with open(
        DIR_PATH + f"/procedural_memory/procedural_memory_{procedural_id}.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(procedural_memory, output, indent=4, ensure_ascii=False)


# ================ Default Version ================
def update_procedural(*, procedural_id=None, procedural_memory=None, updates=None):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if procedural_id is None:
        print("update_procedural: procedural_id is None; skip updating.")
        return
    path = DIR_PATH + f"/procedural_memory/procedural_memory_{procedural_id}.json"
    if procedural_memory is None and not updates:
        print("update_procedural: no fields provided; skip updating.")
        return
    with open(path, "r", encoding="utf-8") as f:
        procedural_data = json.load(f)
    if isinstance(procedural_memory, dict):
        procedural_data.update(procedural_memory)
    if isinstance(updates, dict):
        procedural_data.update(updates)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(procedural_data, output, indent=4, ensure_ascii=False)


# ================ Default Version ================
def save_subgoal(subgoal, subgoal_id):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    with open(
        DIR_PATH + f"/subgoal/subgoal_{subgoal_id}.json",
        "w",
        encoding="utf-8",
    ) as output:
        _json = {
            "subgoal": subgoal,
        }
        json.dump(_json, output, indent=4, ensure_ascii=False)


# ================ Default Version ================
def update_subgoal(*, subgoal_id=None, subgoal=None):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if subgoal_id is None:
        print("update_subgoal: subgoal_id is None; skip updating.")
        return
    if subgoal is None:
        print("update_subgoal: no fields provided; skip updating.")
        return
    path = DIR_PATH + f"/subgoal/subgoal_{subgoal_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        subgoal_data = json.load(f)
    subgoal_data["subgoal"] = subgoal
    with open(path, "w", encoding="utf-8") as output:
        json.dump(subgoal_data, output, indent=4, ensure_ascii=False)


# ================ Default Version ================
def save_episodic(episodic_memory, episodic_id):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    with open(
        DIR_PATH + f"/episodic_memory/episodic_memory_{episodic_id}.json",
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(episodic_memory, output, indent=4, ensure_ascii=False)


# ================ Default Version ================
def update_episodic(*, episodic_id=None, episodic_memory=None, updates=None):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if episodic_id is None:
        print("update_episodic: episodic_id is None; skip updating.")
        return
    if episodic_memory is None and not updates:
        print("update_episodic: no fields provided; skip updating.")
        return
    path = DIR_PATH + f"/episodic_memory/episodic_memory_{episodic_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        episodic_data = json.load(f)
    if isinstance(episodic_memory, dict):
        episodic_data.update(episodic_memory)
    if isinstance(updates, dict):
        episodic_data.update(updates)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(episodic_data, output, indent=4, ensure_ascii=False)

# ================ HPQA ================
def save_episodic_hpqa_ver(episodic_memory_str, episodic_id):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    with open(DIR_PATH+f"/episodic_memory/episodic_memory_{episodic_id}.json","w",) as output:
        _json = {
            "episodic_id": episodic_id,
            "episodic_memory": episodic_memory_str,
        }
        json.dump(_json, output, indent=4,ensure_ascii=False)
        
# ================ HPQA ================
def save_semantic_hpqa_ver(
    semantic_memory_str,
    semantic_id,
    semantic_embedding,
    episodic_ids=None,
    episodic_id=None,
    tags=None,
    tag_ids=None,
    time=None,
    bro_semantic_ids=None,
    son_semantic_ids=[],
):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if hasattr(semantic_embedding, "tolist"):
        semantic_embedding = semantic_embedding.tolist()
    semantic_embedding = array_to_list(semantic_embedding)

    if episodic_ids is None:
        episodic_ids = [episodic_id] if episodic_id is not None else []
    
    with open(
        os.path.join(DIR_PATH, f"semantic_memory/semantic_memory_{semantic_id}.json"),
        "w",
        encoding="utf-8",
    ) as output:
        _json = {
            "semantic_id": semantic_id,
            "semantic_memory": semantic_memory_str,
            "episodic_ids": episodic_ids,
            "episodic_id": episodic_id if episodic_id is not None else (episodic_ids[0] if episodic_ids else None),
            "bro_semantic_ids": bro_semantic_ids or [],
            "tags": tags or [],
            "tag_ids": tag_ids or [],            
            "time": time,                  
            "semantic_embedding": semantic_embedding,  
            "son_semantic_ids": son_semantic_ids,
        }
        json.dump(_json, output, indent=4,ensure_ascii=False)

# ================ HPQA ================
def update_semantic_hpqa_ver(
    semantic_id=None,
    semantic_memory_str=None,
    semantic_embedding=None,
    episodic_ids=None,
    episodic_id=None,
    bro_semantic_ids=None,
    tags=None,
    tag_ids=None,
    time=None,
    son_semantic_ids=None,
    is_active=None,
):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    # If semantic_id missing, we can't locate the file; keep behavior simple (no extra handling)
    if semantic_id is None:
        print("update_semantic: semantic_id is None; skip updating.")
        return
    path = DIR_PATH + f"/semantic_memory/semantic_memory_{semantic_id}.json"

    if episodic_ids is None and episodic_id is not None:
        episodic_ids = [episodic_id]

    update_fields = {
        "semantic_memory": semantic_memory_str,
        "semantic_embedding": semantic_embedding,
        "episodic_ids": episodic_ids,
        "episodic_id": episodic_id,
        "bro_semantic_ids": bro_semantic_ids,
        "tags": tags,
        "tag_ids": tag_ids,
        "time": time,
        "son_semantic_ids": son_semantic_ids,
        "is_active": is_active,
    }
    # If all args are None -> nothing needs to update
    if all(v is None for v in update_fields.values()):
        print("update_semantic: no fields provided (all None); skip updating.")
        return

    with open(path, "r", encoding="utf-8") as f:
        semantic_data = json.load(f)

    if semantic_embedding is not None:
        if hasattr(semantic_embedding, "tolist"):
            semantic_embedding = semantic_embedding.tolist()
        semantic_embedding = array_to_list(semantic_embedding)
        update_fields["semantic_embedding"] = semantic_embedding

    for k, v in update_fields.items():
        if v is not None:
            semantic_data[k] = v

    # Ensure semantic_id stays consistent (optional but usually desired)
    semantic_data["semantic_id"] = semantic_id

    with open(path, "w", encoding="utf-8") as output:
        json.dump(semantic_data, output, indent=4, ensure_ascii=False)

# ================ HPQA ================
def save_tag_hpqa_ver(tag, tag_id, semantic_ids, time, tag_embedding, importance):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if hasattr(tag_embedding, "tolist"):
        tag_embedding = tag_embedding.tolist()
    tag_embedding = array_to_list(tag_embedding)

    with open(DIR_PATH+f"/tag/tag_{tag_id}.json","w",encoding="utf-8") as output:
        _json = {
            "tag_id": tag_id,
            "tag": tag,
            "semantic_ids": semantic_ids,
            "time": time, 
            "importance":importance,
            "tag_embedding": tag_embedding,  
        }
        json.dump(_json, output, indent=4, ensure_ascii=False)

# ================ HPQA ================
def update_tag_hpqa_ver(tag, tag_id, semantic_ids, time, tag_embedding, importance):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if hasattr(tag_embedding, "tolist"):
        tag_embedding = tag_embedding.tolist()
    with open(DIR_PATH+f"/tag/tag_{tag_id}.json","r",) as f:
        tag_data=json.load(f)
    
    tag_data["semantic_ids"]=semantic_ids
    tag_data["time"]=time
    tag_data["importance"]=importance
    
    with open(DIR_PATH+f"/tag/tag_{tag_id}.json","w",) as output:
        json.dump(tag_data, output, indent=4)
    
# ================ HPQA ================
def save_procedural_hpqa_ver(
    procedural_memory_str,
    procedural_embedding,
    procedural_id,
    subgoal,
    subgoal_id,
    episodic_ids=None,
    episodic_id=None,
    time=None,
    _return=None,
):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if hasattr(procedural_embedding, "tolist"):
        procedural_embedding = procedural_embedding.tolist()
    procedural_embedding = array_to_list(procedural_embedding)

    if episodic_ids is None:
        episodic_ids = [episodic_id] if episodic_id is not None else []
    
    with open(
        os.path.join(DIR_PATH, f"procedural_memory/procedural_memory_{procedural_id}.json"),
        "w",
        encoding="utf-8",
    ) as output:
        _json = {
            "procedural_id": procedural_id,
            "procedural_memory": procedural_memory_str,
            "episodic_ids": episodic_ids,
            "episodic_id": episodic_id if episodic_id is not None else (episodic_ids[0] if episodic_ids else None),
            "subgoal": subgoal,       
            "subgoal_id": subgoal_id,                          
            "time": time,
            "return": _return,
            "procedural_embedding": procedural_embedding,  
        }
        json.dump(_json, output, indent=4,ensure_ascii=False)

# ================ HPQA ================
def save_subgoal_hpqa_ver(subgoal, subgoal_id, subgoal_embedding,procedural_id,time):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    subgoal_embedding = array_to_list(subgoal_embedding)
    
    with open(DIR_PATH+f"/subgoal/subgoal_{subgoal_id}.json","w",) as output:
        _json = {
            "subgoal_id": subgoal_id,
            "subgoal": subgoal,
            "procedural_id": procedural_id,
            "time": time,
            "subgoal_embedding": subgoal_embedding,        
        }
        json.dump(_json, output, indent=4)
    

# ================ WebArena Version ================
def save_episodic_webarena_ver(episodic_memory, episodic_id):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    with open(DIR_PATH + f"/episodic_memory/episodic_memory_{episodic_id}.json", "w", encoding="utf-8") as output:
        _json = {
            "episodic_id": episodic_id,
            "subgoal": episodic_memory.get("subgoal"),
            "state": episodic_memory.get("state"),
            "observation": episodic_memory.get("observation"),
            "action": episodic_memory.get("action"),
            "reward": episodic_memory.get("reward"),
            "time": episodic_memory.get("time"),
        }
        json.dump(_json, output, indent=4, ensure_ascii=False)


# ================ WebArena Version ================
def save_semantic_webarena_ver(
    semantic_memory,
    tags,
    semantic_id,
    time,
    episodic_ids=None,
    bro_semantic_ids=None,
    semantic_embedding=None,
    tag_embeddings=None,
):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if hasattr(semantic_embedding, "tolist"):
        semantic_embedding = semantic_embedding.tolist()
    semantic_embedding = array_to_list(semantic_embedding)
    if tag_embeddings:
        tag_embeddings = {
            tag: array_to_list(emb) for tag, emb in tag_embeddings.items()
        }
    with open(DIR_PATH + f"/semantic_memory/semantic_memory_{semantic_id}.json", "w", encoding="utf-8") as output:
        _json = {
            "semantic_id": semantic_id,
            "semantic_memory": semantic_memory,
            "tags": tags,
            "time": time,
            "episodic_ids": episodic_ids or [],
            "bro_semantic_ids": bro_semantic_ids or [],
            "embedding": semantic_embedding,
            "tag_embeddings": tag_embeddings or {},
        }
        json.dump(_json, output, indent=4, ensure_ascii=False)


# ================ WebArena Version ================
def update_semantic_webarena_ver(
    *,
    semantic_id=None,
    semantic_memory=None,
    tags=None,
    time=None,
    episodic_ids=None,
    bro_semantic_ids=None,
    semantic_embedding=None,
    tag_embeddings=None,
):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if semantic_id is None:
        print("update_semantic_webarena_ver: semantic_id is None; skip updating.")
        return
    path = DIR_PATH + f"/semantic_memory/semantic_memory_{semantic_id}.json"
    update_fields = {
        "semantic_memory": semantic_memory,
        "tags": tags,
        "time": time,
        "episodic_ids": episodic_ids,
        "bro_semantic_ids": bro_semantic_ids,
        "embedding": semantic_embedding,
        "tag_embeddings": tag_embeddings,
    }
    if all(v is None for v in update_fields.values()):
        print("update_semantic_webarena_ver: no fields provided; skip updating.")
        return
    with open(path, "r", encoding="utf-8") as f:
        semantic_data = json.load(f)
    if semantic_embedding is not None:
        if hasattr(semantic_embedding, "tolist"):
            semantic_embedding = semantic_embedding.tolist()
        update_fields["embedding"] = array_to_list(semantic_embedding)
    if tag_embeddings is not None:
        update_fields["tag_embeddings"] = {
            tag: array_to_list(emb) for tag, emb in tag_embeddings.items()
        }
    for k, v in update_fields.items():
        if v is not None:
            semantic_data[k] = v
    semantic_data["semantic_id"] = semantic_id
    with open(path, "w", encoding="utf-8") as output:
        json.dump(semantic_data, output, indent=4, ensure_ascii=False)


# ================ WebArena Version ================
def save_subgoal_webarena_ver(subgoal, subgoal_id, procedural_ids=None, subgoal_embedding=None):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    subgoal_embedding = array_to_list(subgoal_embedding)
    with open(DIR_PATH + f"/subgoal/subgoal_{subgoal_id}.json", "w", encoding="utf-8") as output:
        _json = {
            "subgoal_id": subgoal_id,
            "subgoal": subgoal,
            "procedural_ids": procedural_ids or [],
            "embedding": subgoal_embedding,
        }
        json.dump(_json, output, indent=4, ensure_ascii=False)


# ================ WebArena Version ================
def update_subgoal_webarena_ver(
    *,
    subgoal_id=None,
    subgoal=None,
    procedural_ids=None,
    subgoal_embedding=None,
):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if subgoal_id is None:
        print("update_subgoal_webarena_ver: subgoal_id is None; skip updating.")
        return
    path = DIR_PATH + f"/subgoal/subgoal_{subgoal_id}.json"
    update_fields = {
        "subgoal": subgoal,
        "procedural_ids": procedural_ids,
        "embedding": subgoal_embedding,
    }
    if all(v is None for v in update_fields.values()):
        print("update_subgoal_webarena_ver: no fields provided; skip updating.")
        return
    with open(path, "r", encoding="utf-8") as f:
        subgoal_data = json.load(f)
    if subgoal_embedding is not None:
        update_fields["embedding"] = array_to_list(subgoal_embedding)
    for k, v in update_fields.items():
        if v is not None:
            subgoal_data[k] = v
    subgoal_data["subgoal_id"] = subgoal_id
    with open(path, "w", encoding="utf-8") as output:
        json.dump(subgoal_data, output, indent=4, ensure_ascii=False)


# ================ WebArena Version ================
def save_procedural_webarena_ver(
    procedural_memory,
    procedural_id,
    subgoal_id=None,
    episodic_ids=None,
    subgoal_embedding=None,
):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    subgoal_embedding = array_to_list(subgoal_embedding)
    with open(DIR_PATH + f"/procedural_memory/procedural_memory_{procedural_id}.json", "w", encoding="utf-8") as output:
        _json = {
            "procedural_id": procedural_id,
            "procedural_memory": procedural_memory.get("procedural_memory"),
            "subgoal": procedural_memory.get("subgoal"),
            "time": procedural_memory.get("time"),
            "return": procedural_memory.get("return"),
            "subgoal_id": subgoal_id,
            "episodic_ids": episodic_ids or [],
            "subgoal_embedding": subgoal_embedding,
        }
        json.dump(_json, output, indent=4, ensure_ascii=False)


# ================ WebArena Version ================
def update_procedural_webarena_ver(
    *,
    procedural_id=None,
    procedural_memory=None,
    subgoal=None,
    time=None,
    _return=None,
    subgoal_id=None,
    episodic_ids=None,
    subgoal_embedding=None,
):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    if procedural_id is None:
        print("update_procedural_webarena_ver: procedural_id is None; skip updating.")
        return
    path = DIR_PATH + f"/procedural_memory/procedural_memory_{procedural_id}.json"
    update_fields = {
        "procedural_memory": procedural_memory,
        "subgoal": subgoal,
        "time": time,
        "return": _return,
        "subgoal_id": subgoal_id,
        "episodic_ids": episodic_ids,
        "subgoal_embedding": subgoal_embedding,
    }
    if all(v is None for v in update_fields.values()):
        print("update_procedural_webarena_ver: no fields provided; skip updating.")
        return
    with open(path, "r", encoding="utf-8") as f:
        procedural_data = json.load(f)
    if subgoal_embedding is not None:
        update_fields["subgoal_embedding"] = array_to_list(subgoal_embedding)
    for k, v in update_fields.items():
        if v is not None:
            procedural_data[k] = v
    procedural_data["procedural_id"] = procedural_id
    with open(path, "w", encoding="utf-8") as output:
        json.dump(procedural_data, output, indent=4, ensure_ascii=False)


# ================ LongMemEval Version ================
def save_semantic_longmem_ver(semantic_memory, tags, semantic_id, time, blk_sz: int=1):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    blk_id = semantic_id // blk_sz
    file_path = DIR_PATH+f"/semantic_memory/semantic_memory_{blk_id}.json"
    if os.path.exists(file_path):
        with open(file_path, "r", ) as input:
            _json = json.load(input)
        existed = False
        for semantic_node in _json:
            if semantic_node['semantic_id'] == semantic_id:
                semantic_node = {
                    "semantic_id": semantic_id,
                    "semantic_memory": semantic_memory,
                    "tags": tags,
                    "time": time
                }
                existed = True
                break
        if existed == False:
            _json.append({
            "semantic_id": semantic_id,
            "semantic_memory": semantic_memory,
            "tags": tags,
            "time": time
        })
    else:
        _json = [{
            "semantic_id": semantic_id,
            "semantic_memory": semantic_memory,
            "tags": tags,
            "time": time
        }]
    with open(file_path,"w",) as output:
        json.dump(_json, output, indent=4)

# ================ LongMemEval Version ================
def save_tag_longmem_ver(tag, tag_id, blk_sz: int=1):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    blk_id = tag_id // blk_sz
    file_path = DIR_PATH+f"/tag/tag_{blk_id}.json"
    if os.path.exists(file_path):
        with open(file_path, "r", ) as input:
            _json = json.load(input)
        existed = False
        for tag_node in _json:
            if tag_node['tag_id'] == tag_id:
                tag_node = {
                    "tag": tag,
                    "tag_id": tag_id
                }
                existed = True
                break
        if existed == False:
            _json.append({
                "tag": tag,
                "tag_id": tag_id
            })
    else:
        _json = [{
            "tag": tag,
            "tag_id": tag_id
        }]
    with open(file_path,"w",) as output:
        json.dump(_json, output, indent=4)

# ================ LongMemEval Version ================
def save_procedural_longmem_ver(procedural_memory, procedural_id, blk_sz: int=1):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    blk_id = procedural_id // blk_sz
    file_path = DIR_PATH+f"/procedural_memory/procedural_memory_{blk_id}.json"
    if os.path.exists(file_path):
        with open(file_path, "r", ) as input:
            _json = json.load(input)
        existed = False
        for procedural_node in _json:
            if procedural_node['procedural_id'] == procedural_id:
                procedural_node = {
                    "procedural_id": procedural_id,
                    "procedural_memory": procedural_memory['procedural_memory'],
                    "subgoal": procedural_memory['subgoal']
                }
                existed = True
                break
        if existed == False:
            _json.append({
            "procedural_id": procedural_id,
            "procedural_memory": procedural_memory['procedural_memory'],
            "subgoal": procedural_memory['subgoal']
        })
    else:
        _json = [{
            "procedural_id": procedural_id,
            "procedural_memory": procedural_memory['procedural_memory'],
            "subgoal": procedural_memory['subgoal']
        }]
    with open(file_path,"w",) as output:
        json.dump(_json, output, indent=4)  

# ================ LongMemEval Version ================
def save_subgoal_longmem_ver(subgoal, subgoal_id, blk_sz: int=1):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    blk_id = subgoal_id // blk_sz
    file_path = DIR_PATH+f"/subgoal/subgoal_{blk_id}.json"
    if os.path.exists(file_path):
        with open(file_path, "r", ) as input:
            _json = json.load(input)
        existed = False
        for subgoal_node in _json:
            if subgoal_node['subgoal_id'] == subgoal_id:
                subgoal_node = {
                    "subgoal_id": subgoal_id,
                    "subgoal": subgoal
                }
                existed = True
                break
        if existed == False:
            _json.append({
            "subgoal_id": subgoal_id,
            "subgoal": subgoal
        })
    else:
        _json = [{
            "subgoal_id": subgoal_id,
            "subgoal": subgoal
        }]
    with open(file_path,"w",) as output:
        json.dump(_json, output, indent=4)

# ================ LongMemEval Version ================
def save_episodic_longmem_ver(episodic_memory, episodic_id, blk_sz: int=1):
    DIR_PATH = os.environ.get("DIR_PATH", None)
    blk_id = episodic_id // blk_sz
    file_path = DIR_PATH+f"/episodic_memory/episodic_memory_{blk_id}.json"
    if os.path.exists(file_path):
        with open(file_path, "r", ) as input:
            _json = json.load(input)
        existed = False
        for episodic_node in _json:
            if episodic_node['episodic_id'] == episodic_id:
                episodic_node = {
                    "episodic_id": episodic_id,
                    "subgoal": episodic_memory["subgoal"],
                    "state": episodic_memory['state'],
                    "observation": episodic_memory['observation'],
                    'action': episodic_memory['action'],
                    'reward': episodic_memory['reward'],
                    "time": episodic_memory['time']
                }
                existed = True
                break
        if existed == False:
            _json.append({
                "episodic_id": episodic_id,
                "subgoal": episodic_memory["subgoal"],
                "state": episodic_memory['state'],
                "observation": episodic_memory['observation'],
                'action': episodic_memory['action'],
                'reward': episodic_memory['reward'],
                "time": episodic_memory['time']
            })
    else:
        _json = [{
            "episodic_id": episodic_id,
            "subgoal": episodic_memory["subgoal"],
            "state": episodic_memory['state'],
            "observation": episodic_memory['observation'],
            'action': episodic_memory['action'],
            'reward': episodic_memory['reward'],
            "time": episodic_memory['time']
        }]
    with open(file_path,"w",) as output:
        json.dump(_json, output, indent=4)
# ================ LongMemEval Version ================





# -------------------------
# Utils: IO
# -------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)