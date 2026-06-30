# OpenCode adapter — integration harness

`integration.mjs` drives the **built** plugin (`../dist/index.js`) with real
OpenCode-shaped events against a running PlugMem server, then verifies recall.
It exercises the runtime paths unit tests can't reach end-to-end:

- `tool.execute.after` outcome parsing on the real `{title, output, metadata}` payload;
- failure → success pairing into a `failure_delta` memory;
- per-event-family session-id resolution (`getSessionId`);
- the episodic-candidate filter at `/extract`;
- the `session.idle` transcript-replay **dedup** (repeated idles must not duplicate).

This is a **manual** check (it needs a real LLM + embedder), so it is not part of
`npm test`.

## Run

```bash
# 1. build the plugin
npm run build

# 2. start a PlugMem server with a real LLM + embedder (ephemeral keeps it clean)
#    (from the repo root, with LLM_BASE_URL / EMBEDDING_BASE_URL set or in .env)
CHROMA_MODE=ephemeral python -m uvicorn plugmem.api.app:app --port 8077

# 3. run the harness
PLUGMEM_PORT=8077 node eval/integration.mjs
```

Exit code is non-zero if any check fails. Expected: `7 passed, 0 failed`.
