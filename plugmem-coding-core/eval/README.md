# PlugMem coding-agent eval harness

Drives synthetic event sequences through the real `createCore` code path
against a live PlugMem service. Captures per-session promotion counts,
recall block sizes, and (optionally) per-phase token costs read from the
server's `token_usage_file`.

The harness is intentionally **not** a Claude Code integration test — it
fires the abstract events directly. That gives us a deterministic,
reproducible way to exercise the memory loop without spending API credits
on every commit. Real-CC eval (recording transcripts of actual sessions
and replaying their event streams) is a layer to build on top of this.

## Setup

You need a running PlugMem service. From the repo root:

```bash
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini
export EMBEDDING_BASE_URL=http://localhost:8001/v1
export EMBEDDING_MODEL=nvidia/NV-Embed-v2
export PLUGMEM_API_KEY=dev-key-change-me

# To collect internal-token costs split by phase:
export TOKEN_USAGE_FILE=$(pwd)/eval-tokens.jsonl

uv run uvicorn plugmem.api.app:app --host 0.0.0.0 --port 8080
```

In another shell, build the eval CLI:

```bash
cd plugmem-coding-core
npm install
npm run build
```

## Run

```bash
node dist/eval/cli.js run \
  --fixture eval/fixtures/smoke.json \
  --out eval-report.json
```

The runner truncates the configured graph (`graphId` in the fixture)
before each run unless you pass `--no-reset`. Reports go to stdout if
`--out` is omitted.

## What's in the report

```jsonc
{
  "fixture": "smoke",
  "startedAt": "...", "finishedAt": "...",
  "sessions": [
    {
      "id": 1,
      "events": [
        { "type": "session_start", "injection": { "text": "...", "charCount": 0 } },
        // ...
      ],
      "statsAfter": { "semantic": 1, "procedural": 0, ... },
      "promotions": 1,
      "exposedChars": 0
    }
  ],
  "totalPromotions": 1,
  "totalExposedChars": 312,
  "finalStats": { "semantic": 1, ... }
}
```

- **`promotions`** — net new memory nodes inserted during this session
  (semantic + procedural + episodic delta). Read this as "how many
  candidates the LLM extractor turned into memories."
- **`exposedChars`** — total chars of recall blocks injected to the
  agent during this session. Per the bench-token-split rule, this is
  **exposed** cost — what the agent's main LLM sees. Tokens ≈ chars/4.
- **`statsAfter`** — graph composition snapshot. Useful for crossover
  curves ("how many semantic memories does the graph have at session N").

## Internal-token cost (optional)

If you set `TOKEN_USAGE_FILE` on the server, run the summarizer after
the run to get per-phase token totals:

```bash
node dist/eval/cli.js summarize --token-log /path/to/tokens.jsonl
```

Phases the harness cares about:
- `extract` — promotion-gate LLM calls
- `retrieve` — sub-task / session-start retrieval (no LLM in current
  flow, but tagged for forward compat with `/reason`)
- `reason` — final reasoning synthesis (chat-plugin path; not used by
  coding adapter)
- `default` — untagged calls (legacy chat plugin)

These are **internal** tokens — what the memory service spent on the
agent's behalf, billed to the memory service's LLM provider, separate
from the agent's own model usage.

## Writing a fixture

A fixture is JSON with this shape:

```jsonc
{
  "name": "...",
  "baseUrl": "http://localhost:8080",
  "apiKey": "...",
  "graphId": "eval://your-name",
  "harness": "claude-code",
  "cwd": "/tmp/whatever",
  "recall": { /* optional overrides — see RecallConfig types */ },
  "sessions": [
    {
      "id": 1,
      "events": [
        { "type": "session_start", "label": "..." },
        { "type": "user_prompt", "prompt": "..." },
        {
          "type": "pre_tool", "toolName": "Bash",
          "toolInput": {"...": "..."}, "callId": "c1"
        },
        {
          "type": "post_tool", "toolName": "Bash",
          "toolInput": {"...": "..."}, "callId": "c1",
          "toolResult": "...", "outcome": "success" | "failure" | "unknown"
        },
        { "type": "session_end" }
      ]
    }
  ]
}
```

Ordering matters: the runner fires events in array order with the same
session ID. To exercise the failure-delta detector, emit a `pre_tool` →
`post_tool` (failure) → another `pre_tool` → `post_tool` (success) for
the same `toolName` within a single session.

## Limitations / next layer

- **Doesn't test agent quality.** The runner doesn't run an agent; it
  fires hand-authored events. The crossover-vs-CLAUDE.md eval needs
  real Claude Code (or OpenCode) sessions producing event streams. Plan
  is to record those in a separate harness that captures real session
  transcripts and replays the corresponding abstract events here.
- **No baseline-comparison automation.** Run the fixture once per
  baseline (empty / hand-written CLAUDE.md / PlugMem) and diff the
  reports manually. Multi-run aggregation is a future addition.
- **No success-rate measurement.** Task-completion correctness needs
  domain-specific assertions (does the test pass? does the diff
  compile?) — outside the scope of a fixture-driven harness.
