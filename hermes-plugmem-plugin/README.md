# PlugMem for Hermes Agent

Hermes memory provider backed by a [PlugMem](https://github.com/TIMAN-group/PlugMem) knowledge graph service.

PlugMem transforms raw agent interactions into structured knowledge units (semantic facts, procedural skills, episodic traces) organized in a graph. The Hermes provider injects relevant context before each turn and registers `plugmem_remember` + `plugmem_recall` tools.

## Quick Start

```bash
# 1. Start the PlugMem service
pip install -e /path/to/PlugMem
uvicorn plugmem.api.app:app --port 8080

# 2. Install the Hermes plugin
mkdir -p ~/.hermes/hermes-agent/plugins/memory
ln -sf "$(pwd)/hermes-plugmem-plugin/memory_plugmem" \
       ~/.hermes/hermes-agent/plugins/memory/memory_plugmem

# 3. Configure
hermes config set memory.provider plugmem
echo "PLUGMEM_BASE_URL=http://localhost:8080" >> ~/.hermes/.env
echo "PLUGMEM_DEFAULT_GRAPH_ID=hermes-default" >> ~/.hermes/.env

# 4. Verify
hermes memory status
```

## Provider Lifecycle

The provider hooks into Hermes's MemoryProvider lifecycle:

| Hook | What it does |
|------|-------------|
| `system_prompt_block()` | Shows graph stats in the system prompt |
| `prefetch(query)` | Calls PlugMem's `/reason` endpoint — injects relevant context before each turn |
| `sync_turn(user, asst)` | Stores each turn as an episodic trajectory |
| `on_session_end(messages)` | Extracts structured knowledge at session boundaries |
| `on_pre_compress(messages)` | Extracts facts before context compression discards them |
| `on_memory_write(action, target, content)` | Mirrors built-in memory writes to PlugMem |

## Tools

### `plugmem_remember`

Store durable facts, preferences, decisions, or full trajectories.

```
plugmem_remember(text="User prefers dark mode", tags=["preference", "ui"])
plugmem_remember(goal="Fix deploy", steps=[{"observation": "...", "action": "..."}])
```

### `plugmem_recall`

Search memory with LLM-synthesized reasoning.

```
plugmem_recall(observation="What are the user's UI preferences?")
plugmem_recall(observation="How was the deployment issue resolved?", mode="procedural_memory")
```

## Configuration

Set in `$HERMES_HOME/.env`:

| Key | Default | Description |
|-----|---------|-------------|
| `PLUGMEM_BASE_URL` | `http://localhost:8080` | PlugMem service URL |
| `PLUGMEM_API_KEY` | — | API key (if auth enabled) |
| `PLUGMEM_DEFAULT_GRAPH_ID` | `hermes-default` | Primary memory graph |
| `PLUGMEM_SHARED_GRAPH_IDS` | — | Comma-separated shared graph IDs |
| `PLUGMEM_TIMEOUT` | `30` | Request timeout in seconds |

## Architecture

```
Hermes Agent
    │
    ├─ prefetch() ──────────→ POST /api/v1/graphs/{id}/reason
    ├─ sync_turn() ─────────→ POST /api/v1/graphs/{id}/memories (trajectory)
    ├─ on_memory_write() ───→ POST /api/v1/graphs/{id}/memories (structured)
    ├─ on_session_end() ────→ POST /api/v1/graphs/{id}/memories (trajectory)
    ├─ plugmem_remember ────→ POST /api/v1/graphs/{id}/memories
    └─ plugmem_recall ──────→ POST /api/v1/graphs/{id}/reason
    │
    ▼
PlugMem Service (FastAPI :8080)
    │
    ├─ Structuring  (LLM: raw → knowledge units)
    ├─ Retrieval    (vector similarity + value functions)
    ├─ Reasoning    (LLM: nodes → synthesized guidance)
    └─ Storage      (ChromaDB)
```

## License

Apache 2.0 — matches the PlugMem upstream license.
