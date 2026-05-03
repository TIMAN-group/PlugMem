# PlugMem for Claude Code — Onboarding

Cross-session memory for Claude Code, packaged as a hook-based plugin.
The agent learns project conventions, debugging recipes, and personal
preferences from your sessions and recalls them at the start of future
sessions — like a self-writing CLAUDE.md.

This doc walks you from zero to a verified working memory loop in about
fifteen minutes.

## What you get

- A `SessionStart` hook that injects a compact "learned conventions" block
  into the agent's context, drawn from a per-repo memory graph.
- A `UserPromptSubmit` hook that detects user corrections ("don't use X,
  we use Y") and queues them for promotion.
- `PreToolUse` / `PostToolUse` hooks that pair tool failures with later
  successes to extract debugging recipes.
- A `PreCompact` / `SessionEnd` flush that runs the promotion-gate LLM
  on accumulated candidates and writes the high-quality ones to the graph.

## Prerequisites

- **Node 18+** (the hook dispatcher is a Node CLI)
- **Claude Code 2.x** with plugin support
- A **running PlugMem service** at a URL you control (local Docker /
  vLLM / OpenAI-compatible LLM + embedding endpoints — see the main
  repo README)
- The plugin's npm dependencies installed

## 1. Install dependencies and build

```bash
cd plugmem-coding-claude-code
npm install
npm run build
```

`dist/bin/cc-hook.js` is what the hooks invoke. If you change source,
re-run `npm run build`.

## 2. Start the PlugMem service

You need an accessible PlugMem instance. Quick local-dev recipe (see
the main repo's README for the full version):

```bash
# In the PlugMem repo root:
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini
export EMBEDDING_BASE_URL=http://localhost:8001/v1
export EMBEDDING_MODEL=nvidia/NV-Embed-v2
export PLUGMEM_API_KEY=dev-key-change-me

uv run uvicorn plugmem.api.app:app --host 0.0.0.0 --port 8080
```

Sanity-check:

```bash
curl http://localhost:8080/health
# expect: {"status":"ok",...,"llm_available":true,"embedding_available":true,"chroma_available":true}
```

## 3. Configure environment for the plugin

The dispatcher reads env vars from your shell. Set these before launching
Claude Code:

| Variable | Required? | Purpose |
|---|---|---|
| `PLUGMEM_BASE_URL` | yes | URL of the PlugMem service (e.g., `http://localhost:8080`) |
| `PLUGMEM_API_KEY` | if server has auth | matches the server's `PLUGMEM_API_KEY` |
| `PLUGMEM_USER_ID` | optional | enables a per-user shared-read graph (future use) |
| `PLUGMEM_TIMEOUT_MS` | optional | per-request timeout, default 30000 |
| `PLUGMEM_MAX_RETRIES` | optional | default 3 |
| `PLUGMEM_DEBUG` | optional | `1` to log dispatcher activity to stderr |
| `PLUGMEM_STATE_DIR` | optional | override for `~/.cache/plugmem/sessions` |
| `PLUGMEM_KEEP_STATE` | optional | `1` to skip per-session state cleanup at `SessionEnd` (debug aid) |

If `PLUGMEM_BASE_URL` is unset, the dispatcher silently exits 0 — your
session keeps working, but no memory writes happen.

## 4. Install the plugin into Claude Code

### Local development (recommended for first-time setup)

Run Claude Code with the plugin directory loaded:

```bash
claude --plugin-dir /absolute/path/to/plugmem-coding-claude-code
```

Verify it loaded:

```
/plugin list
```

You should see `plugmem-coding 0.1.0`.

### Validate the manifest

```
/plugin validate
```

Should report no errors for `plugmem-coding`.

## 5. Verify the loop end-to-end

The "fact stated in session 1 surfaces in session 2" test:

**Session 1** — in a real git repo (run `claude` from inside one):

```
You: actually, we use httpx in this project, not requests
[do anything for a turn or two]
[/exit or close the session]
```

When the session ends, the dispatcher fires `SessionEnd`, the core
drains the correction candidate, posts to `/api/v1/extract`, and
inserts a semantic memory tagged `source: "correction"`.

Confirm the write landed:

```bash
# repo graph id is repo://claude-code/<host>/<owner>/<repo> derived from
# `git remote get-url origin`. List graphs to see exact ID:
curl -s http://localhost:8080/api/v1/graphs \
  -H "X-API-Key: dev-key-change-me" | jq

# Check stats:
curl -s "http://localhost:8080/api/v1/graphs/$GRAPH_ID/stats" \
  -H "X-API-Key: dev-key-change-me" | jq
# stats.semantic should be >= 1
```

**Session 2** — in the same repo, start a fresh `claude` session.
The dispatcher's `SessionStart` hook calls `/retrieve` against the
per-repo graph and injects the result. With `PLUGMEM_DEBUG=1` you'll
see lines like:

```
[plugmem-coding-core] (recall block injected)
```

In the agent's context, look for a `<plugmem-recall trigger="session-start" graph="repo://...">` block at the top of its first message — it should mention `httpx` (or whatever fact you taught it).

## What the agent sees

A recall block injected at session start looks like:

```
<plugmem-recall trigger="session-start" graph="repo://claude-code/github.com/owner/repo">
Fact 0: We use httpx in this project, not requests
Fact 1: ...
</plugmem-recall>
```

Sub-task recall (when you submit a substantial prompt mid-session)
uses `trigger="user-prompt"`. Tool-family recall (configured per-call,
default off) uses `trigger="tool:<toolname>"`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| No injection block in session 2 | `PLUGMEM_BASE_URL` not set, server unreachable, or graph empty |
| `404 Graph not found` in stderr | Graph derivation failed; check `git remote get-url origin` works in the repo |
| `401 Unauthorized` in stderr | `PLUGMEM_API_KEY` doesn't match server's `PLUGMEM_API_KEY` env |
| Nothing happens at all | Dispatcher is silently no-op'ing; set `PLUGMEM_DEBUG=1` to see why |
| Promotion gate fires but nothing inserted | Server's LLM rejected the candidate; the extractor is conservative by design — set `PLUGMEM_DEBUG=1` and check stderr for `0 memories` lines |
| `~/.cache/plugmem/sessions/` keeps growing | Cleanup is enabled by default at `SessionEnd`; check that hook is firing. Set `PLUGMEM_KEEP_STATE=1` only when you want to inspect candidate state |

For deeper inspection:

```bash
PLUGMEM_DEBUG=1 PLUGMEM_BASE_URL=http://localhost:8080 \
  claude --plugin-dir /path/to/plugmem-coding-claude-code 2> /tmp/plugmem.log

# In another shell:
tail -f /tmp/plugmem.log
```

## Disabling specific behaviors

The plugin has reasonable defaults. To disable a single trigger, pass
options through `createCore` programmatically — but for end-users the
simplest knobs are env vars:

- Tool-family recall is off by default. Enable by editing
  `bin/cc-hook.ts` to pass a `toolFamilyRecall.mapping` to `createCore`.
  (We can promote this to env var if it gets used.)

To disable the plugin entirely without uninstalling:

```bash
unset PLUGMEM_BASE_URL
```

The dispatcher will silently no-op until the env var comes back.

## What's next

- **Cross-session learning gets better with use.** First-session impact
  is small; the value compounds as the graph accumulates.
- **The plugin only writes to a per-repo graph.** Switching repos
  switches graphs automatically (derived from `git remote get-url
  origin`).
- **Memory does not flow between Claude Code and OpenCode** — by
  design. Graph IDs include a `<harness>` segment so a Claude Code
  graph and an OpenCode graph for the same repo are isolated. See the
  design doc's "no cross-harness sharing" decision.

For deeper architecture, see `design_docs/plugmem_for_coding.md` in
the repo root.
