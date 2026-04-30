# PlugMem for OpenClaw — Onboarding

Long-term memory for your OpenClaw agent. Installs as a plugin, auto-saves
session trajectories, and gives the agent two tools: `plugmem.remember` and
`plugmem.recall`. This doc takes you from zero to a working memory loop in
about 10 minutes.

## What you'll have when you're done

- A PlugMem service running at `http://localhost:8080`.
- An OpenClaw agent that:
  - Exposes `plugmem.remember` / `plugmem.recall` tools.
  - Automatically stores the session trajectory on `/reset` and before
    context compaction.
  - Recalls prior sessions on demand into the current context.

## Prerequisites

- **Python 3.10+** with [`uv`](https://docs.astral.sh/uv/).
- **Node.js 18+** and `npm` (for building the plugin).
- **OpenClaw** host running locally — you'll register the plugin against it.
- **An OpenAI-compatible LLM endpoint** (OpenAI, vLLM, Ollama with the
  OpenAI shim, etc.).
- **An embedding endpoint** exposing an OpenAI-compatible `/embeddings`
  route. The server defaults to the `nvidia/NV-Embed-v2` model name — if
  you're pointing at anything else, set `EMBEDDING_MODEL`.

---

## 1. Start the PlugMem service

```bash
cd PlugMem
uv sync
```

Create a `.env` next to `pyproject.toml` (the server reads env vars, not a
dotfile — source this before launching):

```bash
# LLM (required for trajectory structuring + reasoning)
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"

# Embeddings (required for storage + retrieval)
export EMBEDDING_BASE_URL="http://localhost:8001/v1"
export EMBEDDING_MODEL="nvidia/NV-Embed-v2"   # override if you use another

# Storage
export CHROMA_MODE="persistent"               # or "ephemeral" for throwaway
export CHROMA_PATH="./data/chroma"

# Service auth — pick any string; the plugin must send the same value
export PLUGMEM_API_KEY="dev-key-change-me"
```

Launch:

```bash
source .env
uv run uvicorn plugmem.api.app:app --host 0.0.0.0 --port 8080
```

Health check (in a second shell):

```bash
curl -s http://localhost:8080/health | jq
```

You want `status: "ok"` and all three of `llm_available`,
`embedding_available`, `chroma_available` set to `true`. If any is
`false`, fix the corresponding env var before continuing — the plugin will
appear to work but memory insertion and retrieval will fail.

## 2. Create your first graph

A graph is a namespace for memories (typically one per agent or per user).
You **must** create it before inserting memories:

```bash
curl -X POST http://localhost:8080/api/v1/graphs \
  -H "X-API-Key: dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"my-agent"}'
```

## 3. Build and register the plugin

```bash
cd openclaw-plugmem-plugin
npm install
npm run build
```

Register with OpenClaw. The shape depends on how your host loads plugins,
but the call is always the same:

```ts
import { createPlugMemPlugin } from "@plugmem/openclaw-plugin";

const plugin = createPlugMemPlugin({
  baseUrl: "http://localhost:8080",
  apiKey: "dev-key-change-me",      // must match PLUGMEM_API_KEY
  defaultGraphId: "my-agent",       // same ID you created in step 2
});

// Inside your OpenClaw host:
plugin.activate(openclawPluginApi);
```

### Config reference

| Field | Default | Notes |
|---|---|---|
| `baseUrl` | — (required) | No trailing slash needed; stripped automatically. |
| `apiKey` | none | Sent as `X-API-Key`. Required if `PLUGMEM_API_KEY` is set on the server. |
| `defaultGraphId` | none | **Needed for auto-remember.** Without it, `/reset` and compaction hooks silently no-op. |
| `sharedReadGraphIds` | `[]` | Extra graphs fanned in on `recall`. Read-only — writes never target these. See "Shared graphs" below. |
| `timeoutMs` | `30000` | Per-request timeout. |
| `maxRetries` | `3` | Retries on 408/429/502/503/504 with exponential backoff. |
| `autoRemember.onSessionReset` | `true` | Save trajectory when the session resets (`/new`, `/reset`). |
| `autoRemember.onCompaction` | `true` | Save trajectory before context compaction. |
| `autoRemember.minSteps` | `2` | Skip auto-save if the session has fewer (obs, action) pairs. |
| `autoRemember` | — | Set to `false` to disable auto-save entirely. |

## 4. Verify the loop end-to-end

In an OpenClaw session, ask the agent to call the tool directly:

```
Use plugmem.remember with text="The user prefers meetings in the morning" and tags=["user-pref"]
```

Then in the same session (or a new one):

```
Use plugmem.recall with observation="When does the user like to meet?"
```

You should see an LLM-synthesized answer grounded in the stored memory.
Confirm it landed:

```bash
curl -s http://localhost:8080/api/v1/graphs/my-agent/stats \
  -H "X-API-Key: dev-key-change-me" | jq
```

`stats.semantic` should be `>= 1`.

## 5. What the agent sees

### `plugmem.remember`

Two modes, picked by which params you pass:

- **Semantic** — `text` (+ optional `tags`). One-liner facts, user
  preferences, domain knowledge.
- **Trajectory** — `goal` + `steps[{observation, action}]`. A full
  agent run. PlugMem's structuring pipeline will extract semantic,
  procedural, and episodic memories from it.

### `plugmem.recall`

Takes an `observation` (required) and optional `goal`, `mode`, `raw`. The
default return is an LLM-synthesized answer over the top-ranked
memories. Pass `raw: true` to get the retrieval prompt without synthesis —
useful when you want to feed the memories into your own prompt.

### Auto-remember

Fires on two hooks:

- `before_reset` — user runs `/new` or `/reset`.
- `before_compaction` — OpenClaw is about to compact older messages
  out of context. The plugin falls back to reading the session JSONL
  file because current OpenClaw builds don't populate `event.messages`
  on this hook.

Both paths convert the OpenClaw message log to `(observation, action)`
steps and post a trajectory to `defaultGraphId`. Failures are logged to
`stderr` and swallowed — they won't crash the session.

## Shared graphs (cross-agent memory)

Procedural and episodic memories don't generalize across agents — a calendar
agent shouldn't be retrieving a coding agent's tool traces. Semantic
memories (user preferences, facts about the user) *do* generalize. To get
that without polluting either agent's main graph:

1. Create a user-level graph once:
   ```bash
   curl -X POST http://localhost:8080/api/v1/graphs \
     -H "X-API-Key: dev-key-change-me" \
     -H "Content-Type: application/json" \
     -d '{"graph_id":"user-facts"}'
   ```

2. Point every agent's plugin at it as a read-only fan-in:
   ```ts
   createPlugMemPlugin({
     baseUrl: "http://localhost:8080",
     apiKey: "dev-key-change-me",
     defaultGraphId: "calendar-agent",        // writes go here
     sharedReadGraphIds: ["user-facts"],       // also read on recall
   });
   ```

3. To populate `user-facts`, call `plugmem.remember` with an explicit
   `graph_id`:
   ```
   plugmem.remember(graph_id="user-facts", text="User prefers async standups", tags=["user-pref"])
   ```

On `recall`, the plugin fans out to `[defaultGraphId, ...sharedReadGraphIds]`
in parallel and labels each section in the response:

```
[graph:calendar-agent | episodic_memory]
<reasoning over calendar-agent memories>

---

[graph:user-facts | semantic_memory]
<reasoning over user-facts memories>
```

If a shared graph 404s or times out, the call still succeeds — the failing
graph appears as `[graph:X | error]` and the rest of the response is intact.

Writes (`plugmem.remember` without an explicit `graph_id`, auto-remember
hooks) always go to `defaultGraphId` only. The shared list is read-only by
design.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/health` returns `llm_available: false` | `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` not set or endpoint unreachable. |
| `plugmem.remember` returns `404` | Graph doesn't exist. Run the `POST /graphs` call from step 2. |
| `plugmem.remember` returns `401` | `apiKey` in plugin config doesn't match server's `PLUGMEM_API_KEY`. |
| Auto-remember never fires | `defaultGraphId` not set, or session had `< minSteps` turns. Check server stderr for `[plugmem] auto-remember` lines. |
| `recall` returns empty reasoning | Graph exists but has no memories yet, or query embedding misses. Check `/stats`. |
| Embedding errors on insert | `EMBEDDING_MODEL` mismatch with your endpoint. Default is `nvidia/NV-Embed-v2`. |

## Further reading

- API reference: every route lives under `plugmem/api/routes/` — one file
  per resource (`graphs.py`, `memories.py`, `retrieval.py`).
- Env vars: `ENV_REFERENCE.md` at the repo root.
- Architecture: the paper
  (https://arxiv.org/abs/2603.03296) and `CLAUDE.md` for the pipeline
  (structure → store → retrieve → reason).
