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

`OPENCODE_BASE_URL` / `OPENCODE_MODEL` override the agent backend.

## Verified result

Session 1 (a correction prompt) → the plugin promotes a semantic memory; Session 2
(a related prompt) → session-start **and** user-prompt recall both surface it. Confirmed
against the server: `stats.semantic == 1`, the stored fact
`"Use httpx instead of requests for HTTP calls in this project."`, and recall-audit
entries selecting that node.

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
