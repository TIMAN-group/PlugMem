#!/usr/bin/env bash
# Drives a PERSISTENT `opencode serve` (so the plugin outlives each run and its
# async session-end insert completes), runs a 2-session memory loop, and verifies.
#
# Agent model/backend (overridable):
#   OC_MODEL=anthropic/claude-sonnet-4-6              (needs ANTHROPIC_API_KEY)
#   OC_MODEL=opencode/north-mini-code-free           (free Zen model, no key)
#   OC_MODEL=local-qwen/<id> OPENCODE_BASE_URL=<openai-compatible /v1 url>
set -u
BASE="http://127.0.0.1:8000/api/v1"
GID="repo_opencode_local__work_proj"
PLUGIN="/app/PlugMem/plugmem-coding-opencode/dist/index.js"

OC_MODEL="${OC_MODEL:-local-qwen/CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic}"
OPENCODE_BASE_URL="${OPENCODE_BASE_URL:-https://jizejtestada00.ngrok.app/v1}"
OC_TIMEOUT="${OC_TIMEOUT:-360}"
case "$OC_MODEL" in
  local-qwen/*) LOCAL_ID="${OC_MODEL#local-qwen/}";;
  *)            LOCAL_ID="placeholder";;
esac

mkdir -p /work/proj && cd /work/proj || { echo "no /work/proj"; exit 1; }
rm -f plugin-debug.log http_client.py
[ -d .git ] || git init -q

cat > opencode.json <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-qwen": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local",
      "options": { "baseURL": "${OPENCODE_BASE_URL}", "apiKey": "EMPTY", "headers": { "ngrok-skip-browser-warning": "true" } },
      "models": { "${LOCAL_ID}": { "name": "local" } }
    }
  },
  "model": "${OC_MODEL}",
  "plugin": ["${PLUGIN}"]
}
JSON

echo "model=$OC_MODEL baseURL=$OPENCODE_BASE_URL"
PLUGMEM_URL=http://127.0.0.1:8000 opencode serve --port 4096 --hostname 127.0.0.1 >/tmp/serve.log 2>&1 &
SERVE_PID=$!
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4096/ 2>/dev/null)
  if [ "$code" != "000" ]; then echo "opencode server up http=$code"; break; fi
  sleep 1
done

run_one () {
  echo "==================== $1 ===================="
  timeout "$OC_TIMEOUT" opencode run --attach http://127.0.0.1:4096 --model "$OC_MODEL" "$2" 2>&1 | tail -8
  echo "[run exit ${PIPESTATUS[0]}]"
  echo "waiting 12s for async promotion (extract+insert) in the persistent server..."
  sleep 12
  echo "stats: $(curl -s "$BASE/graphs/$GID/stats")"
}

run_one "SESSION 1 (correction + tool action)" \
  "Stop using the requests library. We use httpx for all HTTP calls in this project. Create http_client.py with a function get_json(url) that fetches and returns JSON using httpx."
echo "http_client.py: $(test -f http_client.py && echo CREATED || echo MISSING)"
run_one "SESSION 2 (recall)" "What HTTP client should this project use, and why?"

echo "=== retrieve(semantic) ==="
curl -s -X POST "$BASE/graphs/$GID/retrieve" -H 'Content-Type: application/json' \
  -d '{"observation":"which HTTP client should I use","mode":"semantic_memory"}' | head -c 400
echo
echo "=== adapter log (tools / promotion / recall) ==="
grep -E 'TOOL BEFORE|TOOL AFTER|promotion-gate|inserted|extract call|Processing user prompt' plugin-debug.log | tail -40

kill "$SERVE_PID" 2>/dev/null
echo done
