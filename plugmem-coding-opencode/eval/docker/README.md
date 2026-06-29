# OpenCode end-to-end eval (Docker)

Runs **real OpenCode sessions** against the PlugMem plugin + server in an isolated
container, to validate the cross-session memory loop on genuine OpenCode events.

The container bundles: the OpenCode CLI, the built PlugMem plugin (+ core), and the
PlugMem server (Python, in a venv). The agent backend and the server's
LLM/embeddings are reached over the network (e.g. your ngrok endpoints).

## Build (context = repo root)

```bash
docker build -f plugmem-coding-opencode/eval/docker/Dockerfile -t plugmem-eval .
```

## Run

```bash
docker run -d --name plugmem-eval \
  -e LLM_BASE_URL=https://jizejtestada00.ngrok.app/v1 \
  -e LLM_MODEL=CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic \
  -e LLM_API_KEY=EMPTY \
  -e EMBEDDING_BASE_URL=https://jizejtestada01.ngrok.app/v1/embeddings,https://jizejtestada02.ngrok.app/v1/embeddings \
  -e EMBEDDING_MODEL=nvidia/NV-Embed-v2 \
  plugmem-eval

# memory loop against a PERSISTENT opencode server (recommended — see below)
docker exec plugmem-eval /app/PlugMem/plugmem-coding-opencode/eval/docker/serve-eval.sh
```

`OPENCODE_BASE_URL` / `OPENCODE_MODEL` override the agent backend for `run-eval.sh`.
`serve-eval.sh` takes `OC_MODEL` (e.g. `OC_MODEL=anthropic/claude-sonnet-4-6` with
`-e ANTHROPIC_API_KEY=...`, or `OC_MODEL=opencode/north-mini-code-free` for a free,
keyless, tool-capable model).

## Verified result

With a tool-calling backend (Claude Sonnet, `anthropic/claude-sonnet-4-6`):
- Session 1 — the agent did real work (wrote `http_client.py` using `httpx` via real
  `write`/`read`/`bash` tool calls, which flow through the adapter's outcome parser),
  and the correction was promoted: `inserted 1 semantic + 1 procedural`.
- Session 2 — **session-start recall and user-prompt recall both selected the memory**
  (recall audit `sem_ids:[0]`). Stored fact:
  `"Use httpx instead of requests for HTTP calls in this project."`

Also validated against a self-hosted OpenAI-compatible endpoint (MiniMax-M2.7 on a
llama.cpp server) via `OPENCODE_BASE_URL=<.../v1> OC_MODEL=local-qwen/<model-id>` — same
result (1 semantic + 1 procedural written, recalled in session 2).

Note: the agent backend must serve the OpenAI/Anthropic API directly with tool-calling
support — a self-hosted vLLM needs `--enable-auto-tool-choice --tool-call-parser`, and
endpoints fronted by an ngrok interstitial / 502 (`content_type: text/html`) won't work.

## Findings / requirements (learned from running this)

1. **The agent backend must serve the OpenAI API directly AND have vLLM tool-calling
   enabled.** OpenCode agents use `tool_choice: "auto"`, so the vLLM server must be
   launched with `--enable-auto-tool-choice --tool-call-parser <parser>` (e.g. `hermes`
   for Qwen2.5). Without it, every turn fails with:
   `Error: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`.
   Also ensure the endpoint returns JSON, not an ngrok browser-warning page
   (`content_type: text/html` ⇒ the tunnel/interstitial is in the way).

2. **Use a persistent `opencode serve` for headless eval, not one-shot `opencode run`.**
   OpenCode fires plugin events fire-and-forget; one-shot `run` exits before the
   plugin's async `session.idle` work (extract → insert, network round-trips to the
   PlugMem server) finishes, so the memory is never persisted. `serve-eval.sh` starts a
   persistent server and drives sessions via `opencode run --attach`, so the plugin
   outlives each run and the insert completes. (Interactive TUI use is unaffected — the
   process stays alive across turns.)

`run-eval.sh` is the simpler one-shot variant; it exercises the event path but will
**not** persist memories for the reason in (2). Prefer `serve-eval.sh`.
