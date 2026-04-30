# Token-cost benchmark — PlugMem vs OpenClaw default

A controlled experiment to measure where LLM tokens are spent when an agent
uses PlugMem instead of OpenClaw's default replay-the-whole-transcript memory.

The point isn't just *fewer tokens overall* — the point is that PlugMem
**shifts** tokens off the agent's hot path (its main LLM) and onto a cheap,
async-friendly background path inside the memory service. The bench
quantifies that shift.

## Files

| File | Role |
|---|---|
| `generate_trajectories.py` | Synthesises OpenClaw-style sessions + grounded queries (one LLM call per trajectory). |
| `run_tok_usage_bench.py` | Drives systems A/B/C against each trajectory. Emits one JSONL record per LLM/embedding call. |
| `analyze.py` | Aggregates JSONL → tables, curves, crossover R\*, optional plots. |
| `pricing.yaml` | Per-model `$/M` tokens for input/output. Edit to match your vendor pricing. |

## Concepts

### Token scopes

Every recorded call is tagged with one of three **scopes**. They are billed
to different parties and almost always run on different models — never report
a single combined total.

| Scope | What it is | Who pays | Hot path? |
|---|---|---|---|
| `EXPOSED` | Tokens the agent's main LLM sees: system prompt, user query, retrieved memory context, completion. | Agent operator | Yes — blocks the user. |
| `INTERNAL` | Tokens burned inside the memory service: structuring on insert, plan/mode classification, reasoning-synthesis. | Memory-service operator | No — can be async / cheap model. |
| `INTERNAL_EMBED` | Embedding calls. Different unit cost from completion tokens. | Memory-service operator | No. |
| `EVAL` | Judge LLM tokens + zero-cost score sentinels. **Excluded from cost curves**; used only for quality retention. | Researcher | N/A |

### Systems under test

| Code | System | Insert cost | Per-retrieval cost |
|---|---|---|---|
| `A` | **OpenClaw default** (baseline) | 0 | full transcript replay → all `EXPOSED` |
| `B` | **PlugMem trajectory-mode** | structuring (`INTERNAL`) | classify + reason-synthesis (`INTERNAL`) + compact recall (`EXPOSED`) |
| `C` | **PlugMem semantic-mode** | only embeddings (`INTERNAL_EMBED`) | same as B |

The `B` vs `C` split tests whether structuring is worth its upfront LLM cost
when the agent could distil facts itself.

## End-to-end

### 1. Configure environment

The bench reads the same env vars as the rest of PlugMem, plus optional
overrides for the agent-facing model so you can run exposed and internal on
different models (Sonnet exposed, Haiku internal — that's where the dollar
crossover lives).

```bash
# Memory-service-internal LLM (structuring, classify, reason-synthesis).
export LLM_BASE_URL=...
export LLM_API_KEY=...
export LLM_MODEL=claude-haiku-4-5-20251001

# Embedding endpoint.
export EMBEDDING_BASE_URL=...
export EMBEDDING_MODEL=nvidia/NV-Embed-v2

# Agent's main LLM (the one that "sees" memory). Falls back to LLM_* if unset.
export EXPOSED_LLM_BASE_URL=...
export EXPOSED_LLM_API_KEY=...
export EXPOSED_LLM_MODEL=claude-sonnet-4-6
```

### 2. Generate trajectories

```bash
uv run python scripts/bench/generate_trajectories.py \
    --out       data/bench/trajectories.jsonl \
    --length    20,50,100 \
    --diversity single,mixed \
    --queries   20 \
    --count     10 \
    --seed      42
```

Each line is one `Trajectory`:

```json
{
  "trajectory_id": "L20-single-...",
  "goal": "...",
  "steps":          [{"observation": "...", "action": "..."}, ...],
  "queries":        [{"query_id": "q0", "question": "...", "gold": "..."}, ...],
  "semantic_facts": ["one-line atom per step", ...]
}
```

`semantic_facts` is required only for system C. If you skip C, you can drop
that field.

### 3. Run the bench

```bash
uv run python scripts/bench/run_tok_usage_bench.py \
    --trajectories data/bench/trajectories.jsonl \
    --out          data/bench/results.jsonl \
    --systems      A,B,C \
    --retrievals   1,3,5,10,20 \
    --judge
```

Flags:

| Flag | Meaning |
|---|---|
| `--systems` | Comma-separated subset of `A,B,C`. |
| `--retrievals` | R values to sweep — query subsamples per trajectory. The first R queries (deterministic prefix) are used. |
| `--limit N` | Run only the first N trajectories (smoke testing). |
| `--judge` | Score answers with an LLM judge (0–5 rubric); emits `EVAL` records. |

Each bench iteration runs in a **fresh ephemeral ChromaDB graph** (one per
`(trajectory, system)`) so caches never flatter PlugMem's numbers.

### 4. Analyze

```bash
uv run python scripts/bench/analyze.py \
    --in      data/bench/results.jsonl \
    --pricing scripts/bench/pricing.yaml \
    --report  data/bench/report.md \
    --plots   data/bench/plots/
```

The report includes:
- per-`(system, R, scope)` cost cells (mean per trajectory)
- crossover R\* (smallest R where PlugMem cum-cost ≤ baseline) — both
  total-cost and **exposed-only**, in tokens *and* dollars
- exposed-token reduction ratio at max R
- answer-quality retention vs A (if `--judge` was set)

## JSONL record schema

`run_tok_usage_bench.py` writes one line per call:

```json
{
  "ts": 1714000000.0,
  "trajectory_id": "L20-single-abc12345",
  "query_id": "q3",
  "system": "B@R=5",
  "scope": "INTERNAL",
  "phase": "reason_synthesis",
  "model": "claude-haiku-4-5-20251001",
  "input_tokens": 412,
  "output_tokens": 87,
  "wall_ms": 624.1,
  "extra": {}
}
```

Phase values currently emitted:

| Phase | Scope | Source |
|---|---|---|
| `baseline_replay` | `EXPOSED` | system A — full-transcript answer call |
| `structuring` | `INTERNAL` | system B — `Memory.append`/`close` LLM calls |
| `semantic_insert` | `INTERNAL` | system C — semantic-mode insert (embeddings only) |
| `classify` | `INTERNAL` | `retrieve_memory` (`get_plan`, `get_mode`) |
| `reason_synthesis` | `INTERNAL` | reasoning-prompt completion in `/reason` |
| `agent_answer` | `EXPOSED` | the agent's main LLM answering with retrieved memory |
| `judge_call` | `EVAL` | judge LLM scoring an answer |
| `judge_score` | `EVAL` | zero-token sentinel carrying parsed score + answer + gold |

## Pricing

`pricing.yaml` maps `model → {input, output}` in USD per 1M tokens. Self-hosted
models should use *amortized* costs (GPU-hours × utilization ÷ throughput),
not vendor list prices. Embedding entries leave `output: 0.00` since their
output is vectors, not tokens. Unmapped models fall back to `default`.

## Token counting

The instrumentation prefers `tiktoken` (`cl100k_base`) for token counts and
falls back to `len(text) // 4` if tiktoken is not installed. Both are
**estimates**: the only authoritative source is the vendor's `usage` field.
For headline numbers, swap in vendor-reported usage by extending
`InstrumentedLLMClient.complete` to capture the response object before
returning.

## Quality guardrail

The judge uses a 0–5 rubric (5 = fully correct, 0 = wrong/refusal). The bench
reports per-`(system, R)` mean score and retention-vs-A. Hypothesis H4 from
the experiment design says efficiency numbers are meaningless if PlugMem
loses more than 5 points (out of 100, i.e. 0.25 on the 0–5 rubric) on
average. Validate the LLM judge against ~50 human-labeled samples before
trusting the headline.

## What this bench is *not*

- **Not a latency bench.** `wall_ms` is logged but the bench does no
  warm-ups, no concurrency control, and runs against whatever endpoint
  happens to respond. Latency numbers are sanity-only.
- **Not a load test.** Per-query, single-thread, no batching.
- **Not a real-OpenClaw integration test.** The "OpenClaw default" baseline
  is a *simulation* of full-history replay, not the OpenClaw runtime itself.
  For end-to-end validation, plug the same trajectories through OpenClaw
  with and without the `openclaw-plugmem-plugin`.
- **Not a fixed-cost bench.** Vendor pricing changes. Re-run `analyze.py`
  with an updated `pricing.yaml` rather than re-running the whole bench.

## Known caveats / open work

- Synthetic trajectories are LLM-generated and may be cleaner than real
  OpenClaw sessions. Add at least one cell of real session transcripts
  before publishing a headline number.
- `tiktoken` is an approximation. For final numbers, capture vendor `usage`.
- The judge is itself an LLM; its scores are correlated with a human rubric
  but not identical. Spot-check.
- No retries / resume — if `run_tok_usage_bench.py` crashes mid-run, restart
  with `--limit` and stitch JSONL outputs.
- `EXPOSED` cost for systems B/C currently only counts the final
  agent-answer call. If your real agent feeds the recall result through
  multiple turns of its own reasoning loop, instrument those turns too.
