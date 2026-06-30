#!/usr/bin/env bash
# Real OpenCode end-to-end memory loop + diagnostics.
#   Session 1 — a correction-bearing task -> plugin should promote a memory.
#   Session 2 — a related task -> session-start recall should surface it.
# Prints the actual OpenCode event taxonomy so we can see what fires.
set -u

BASE="http://127.0.0.1:8000/api/v1"
export PLUGMEM_URL="http://127.0.0.1:8000"

# Agent backend. ada00 serves the OpenAI API directly (no ngrok header needed);
# blackwell00 returns the ngrok HTML interstitial, so it is unusable here.
OC_BASE="${OPENCODE_BASE_URL:-https://jizejtestada00.ngrok.app/v1}"
OC_MODEL_ID="${OPENCODE_MODEL:-CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic}"
MODEL="local-qwen/${OC_MODEL_ID}"
PLUGIN="/app/PlugMem/plugmem-coding-opencode/dist/index.js"
TIMEOUT="${OC_TIMEOUT:-300}"

PROJ=/work/proj
rm -rf "$PROJ"; mkdir -p "$PROJ"; cd "$PROJ"
git init -q  # no remote -> deterministic repo_opencode_local_* graph id

cat > opencode.json <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-qwen": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Qwen",
      "options": { "baseURL": "${OC_BASE}", "apiKey": "EMPTY" },
      "models": { "${OC_MODEL_ID}": { "name": "Qwen 32B" } }
    }
  },
  "model": "${MODEL}",
  "plugin": ["${PLUGIN}"]
}
JSON

run_session () {
  local title="$1"; shift; local prompt="$1"; shift
  echo "==================== $title ===================="
  echo "PROMPT: $prompt"
  timeout "$TIMEOUT" opencode run --model "$MODEL" "$prompt" 2>&1 | tail -30
  echo "[opencode exit: ${PIPESTATUS[0]}]"
  sleep 4   # allow any end-of-session drain to finish
}

event_summary () {
  echo ">>> event types seen:"
  grep -oE 'EVENT RECEIVED: [a-z.]+' "$PROJ/plugin-debug.log" 2>/dev/null | sort | uniq -c
  echo ">>> adapter markers:"
  grep -E "TOOL BEFORE|TOOL AFTER|Processing user prompt|promotion-gate|recall" \
    "$PROJ/plugin-debug.log" 2>/dev/null | tail -25
}

run_session "SESSION 1 (correction)" \
  "Stop using the requests library. We use httpx for all HTTP calls in this project."

GID="$(curl -s "$BASE/graphs" | jq -r '.graphs[]?' | grep '^repo_opencode' | head -1)"
echo ">>> graph: ${GID:-<none>}"
echo ">>> stats after S1: $(curl -s "$BASE/graphs/$GID/stats")"
event_summary

run_session "SESSION 2 (recall)" \
  "What HTTP client should this project use, and why? Answer in one sentence."

echo ">>> stats after S2: $(curl -s "$BASE/graphs/$GID/stats")"
echo ">>> retrieve(semantic): $(curl -s -X POST "$BASE/graphs/$GID/retrieve" -H 'Content-Type: application/json' -d '{"observation":"which HTTP client should I use","mode":"semantic_memory"}' | jq -r '.variables.semantic_memory')"
event_summary
