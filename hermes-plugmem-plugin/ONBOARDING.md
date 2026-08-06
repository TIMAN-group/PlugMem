# Hermes PlugMem — Onboarding

Walk through setting up PlugMem memory for a Hermes agent in ~10 minutes.

## Prerequisites

- Hermes Agent installed (`hermes --version`)
- Python 3.10+
- OpenAI-compatible API key (for PlugMem's LLM + embeddings)

## Step 1: Install PlugMem

```bash
cd /path/to/PlugMem
pip install -e .
```

Or from git:

```bash
pip install "plugmem @ git+https://github.com/TIMAN-group/PlugMem.git"
```

## Step 2: Configure the service

Create a `plugmem.env` in your home directory or export env vars:

```bash
export PLUGMEM_LLM_BASE_URL="https://api.openai.com/v1"
export PLUGMEM_LLM_API_KEY="sk-..."
export PLUGMEM_LLM_MODEL="gpt-4o-mini"
export PLUGMEM_EMBEDDING_BASE_URL="https://api.openai.com/v1"
export PLUGMEM_EMBEDDING_MODEL="text-embedding-3-small"
export PLUGMEM_CHROMA_PATH="./data/chroma"
```

> **Model recommendations:**
> - LLM: `gpt-4o-mini` (fast/cheap for structuring) or `deepseek-chat` (cheaper)
> - Embeddings: `text-embedding-3-small` (OpenAI) or point at any OpenAI-compatible endpoint
> - ChromaDB: persistent mode stores data in `./data/chroma`

## Step 3: Start the service

```bash
plugmem serve
```

Verify:

```bash
curl http://localhost:8080/health
# → {"status": "ok", "version": "0.1.0"}
```

## Step 4: Install the Hermes plugin

```bash
# From the PlugMem repo root
cd /path/to/PlugMem

# Create the Hermes plugins directory if it doesn't exist
mkdir -p ~/.hermes/hermes-agent/plugins/memory

# Symlink the provider
ln -sf "$(pwd)/hermes-plugmem-plugin/memory_plugmem" \
       ~/.hermes/hermes-agent/plugins/memory/memory_plugmem
```

> **Important:** The directory must be named `memory_plugmem` (underscore) — Hermes uses the directory name as the provider key.

## Step 5: Configure Hermes

```bash
hermes config set memory.provider plugmem

# Add to ~/.hermes/.env
echo "PLUGMEM_BASE_URL=http://localhost:8080" >> ~/.hermes/.env
echo "PLUGMEM_DEFAULT_GRAPH_ID=hermes-default" >> ~/.hermes/.env
```

## Step 6: Verify

```bash
hermes memory status
# → External provider: plugmem (active)

# Start a session and test
hermes
```

In your Hermes session, test the tools:

```
> Remember that I prefer concise responses and dark mode.

> What do you know about my preferences?
```

The agent should call `plugmem_remember` then `plugmem_recall` to retrieve.

## Troubleshooting

### "Provider 'plugmem' not found"
- Check the symlink: `ls -la ~/.hermes/hermes-agent/plugins/memory/memory_plugmem/__init__.py`
- The directory MUST be named `memory_plugmem` (underscore, not hyphen)

### "Connection refused" on recall
- Ensure the PlugMem service is running: `curl http://localhost:8080/health`
- Check `PLUGMEM_BASE_URL` is set correctly

### "No such file: plugmem.env"
- Create it or use environment variables directly
- The service looks for `plugmem.env` in the current working directory, then `$HOME/.plugmem.env`

## Next Steps

- **Shared graphs:** Configure `PLUGMEM_SHARED_GRAPH_IDS` to fan out recall across multiple graphs (e.g. a `user-facts` graph shared across profiles)
- **Memory Inspector:** Open `http://localhost:8080/inspector/` in a browser to browse the memory graph
- **Consolidation:** Run `plugmem consolidate <graph_id>` periodically to merge similar facts
