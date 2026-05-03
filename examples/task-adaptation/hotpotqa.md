## Environment variables

The HotpotQA pipeline reads its configuration from env vars. Set the ones you need before running either script.

#### Memory graph location

| Variable | Description |
| --- | --- |
| `DIR_PATH` | Directory the memory graph + per-run outputs (predictions, metrics, logs) live under. Pick a fresh path per dataset/run, e.g. `data/hotpotqa_1000`. |

#### QA LLM — pick **one** of (A) or (B)

The wrapper auto-routes to Azure when `AZURE_ENDPOINT` is set; otherwise it uses the OpenAI-compatible path. The model id is whatever you pass via `--qa_model_name`; nothing is whitelisted.

- **A. OpenAI-compatible API** — vanilla OpenAI, OpenRouter, vLLM, Together, etc.
  - `OPENAI_BASE_URL` — endpoint URL of the provider
  - `OPENAI_API_KEY` — API key for the provider
- **B. Azure OpenAI**
  - `AZURE_ENDPOINT` — Azure endpoint URL (its presence triggers the Azure branch)
  - `OPENAI_API_KEY` — Azure access key

#### Embedding model — pick **at least one** of (A), (B), or (C)

`get_embedding` falls back through the three backends in order until one succeeds.

- **A. Self-hosted server** — e.g. NV-Embed-v2 on vLLM
  - `EMBEDDING_BASE_URL` — URL of the embedding server (`/v1/embeddings`-style endpoint)
- **B. Third-party OpenAI-compatible embeddings API**
  - `EMBEDDING_API_BASE_URL` — endpoint of the embeddings API (falls back to `OPENAI_BASE_URL` if unset)
  - `EMBEDDING_API_KEY` — key for that API (falls back to `OPENAI_API_KEY` if unset)
- **C. Local model** — no env var needed
  - `sentence-transformers` is loaded on first call and embeds in-process; requires `pip install sentence-transformers`

#### Optional

| Variable | Description |
| --- | --- |
| `TOKEN_USAGE_FILE` | JSONL path for per-call token accounting. Auto-named under `usage/` if unset. |

---

## How to run

#### 1. Build the memory graph

```bash
export DIR_PATH="data/hotpotqa_1000"

# === QA LLM: uncomment ONE block ===
# (A) OpenAI-compatible
export OPENAI_BASE_URL="<endpoint>"
export OPENAI_API_KEY="<key>"
# (B) Azure
# export AZURE_ENDPOINT="<https://...azure.com/>"
# export OPENAI_API_KEY="<azure key>"

# === Embedding: uncomment ONE block ===
# (A) Self-hosted
export EMBEDDING_BASE_URL="http://<host>:<port>/v1/embeddings"
# (B) Third-party API
# export EMBEDDING_API_BASE_URL="<endpoint>"
# export EMBEDDING_API_KEY="<key>"
# (C) Local sentence-transformers — no env var; just `pip install sentence-transformers`

python src/eval/hotpotqa/build_mem.py \
  --bench_name hotpotqa \
  --start_idx 0 --end_idx 999 \
  --num_workers 2 --chunk_size 30
```

`--end_idx` defaults to `9` (toy run); bump to cover the full corpus.

#### 2. Run QA evaluation

```bash
export DIR_PATH="data/hotpotqa_1000"

# === QA LLM: uncomment ONE block ===
# (A) OpenAI-compatible
export OPENAI_BASE_URL="<endpoint>"
export OPENAI_API_KEY="<key>"
# (B) Azure
# export AZURE_ENDPOINT="<https://...azure.com/>"
# export OPENAI_API_KEY="<azure key>"

# === Embedding: uncomment ONE block ===
# (A) Self-hosted
export EMBEDDING_BASE_URL="http://<host>:<port>/v1/embeddings"
# (B) Third-party API
# export EMBEDDING_API_BASE_URL="<endpoint>"
# export EMBEDDING_API_KEY="<key>"
# (C) Local sentence-transformers — no env var; just `pip install sentence-transformers`

python src/eval/hotpotqa/eval_qa_all.py \
  --bench_name hotpotqa \
  --qa_model_name <model id passed straight to provider> \
  --n_round_retrieval 2
```

Predictions, metrics, and logs land under `$DIR_PATH`.

---

## Task adaptation

To adapt PlugMem to HotpotQA with deeper multi-hop reasoning and a stronger backbone, only two parameters in `src/eval/hotpotqa/eval_qa_all.py` need to be changed:

- `--n_round_retrieval` from the default `2` to `3` (allows one additional retrieval round)
- `--qa_model_name` to `openai/gpt-5.4`

```
python src/eval/hotpotqa/eval_qa_all.py \
  --bench_name hotpotqa \
  --qa_model_name oepnai/gpt-5.4 \
  --n_round_retrieval 3
```

#### Performance
To further assess the potential of PlugMem as a memory backbone, we increase the maximum multi-hop depth $T_{\max}$ from 2 to 3 and adopt a stronger backbone model, GPT-5.4. Under this setting, PlugMem achieves an F1 score of **79.1**, substantially outperforming the **74.1** F1 score reported in the main experiments. In addition to token-level F1, we further evaluate answer correctness using LLM-as-Judge accuracy, where this configuration achieves **91.1%**.

To estimate the upper bound under ideal evidence retrieval, we report an oracle baseline that directly answers with the gold paragraphs as context, which obtains an F1 score of **83.8** and an LLM-as-Judge accuracy of **95.0%**. After task adaptation with deeper reasoning depth and a stronger backbone model, PlugMem reaches **94.4%** of the oracle F1 score and **95.9%** of the oracle LLM-as-Judge accuracy.
