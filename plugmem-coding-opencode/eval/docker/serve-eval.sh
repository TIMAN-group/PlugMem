#!/usr/bin/env bash
# Variant of the eval that drives a PERSISTENT `opencode serve` so the plugin
# outlives each run and its async session-end work (extract -> insert) completes.
set -u
BASE="http://127.0.0.1:8000/api/v1"
# Agent model is overridable: e.g. OC_MODEL=anthropic/claude-sonnet-4-6 (needs
# ANTHROPIC_API_KEY) or OC_MODEL=opencode/north-mini-code-free (free, no key).
M="${OC_MODEL:-local-qwen/CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic}"
GID="repo_opencode_local__work_proj"

cd /work/proj || { echo "no /work/proj"; exit 1; }
rm -f plugin-debug.log

PLUGMEM_URL=http://127.0.0.1:8000 opencode serve --port 4096 --hostname 127.0.0.1 >/tmp/serve.log 2>&1 &
SERVE_PID=$!
echo "serve pid=$SERVE_PID"
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4096/ 2>/dev/null)
  if [ "$code" != "000" ]; then echo "server up http=$code"; break; fi
  sleep 1
done

run_one () {
  echo "==================== $1 ===================="
  opencode run --attach http://127.0.0.1:4096 --model "$M" "$2" 2>&1 | tail -6
  echo "[run exit ${PIPESTATUS[0]}]"
  echo "waiting 10s for async promotion (extract+insert) in the persistent server..."
  sleep 10
  echo "stats: $(curl -s "$BASE/graphs/$GID/stats")"
}

rm -f http_client.py
run_one "SESSION 1 (correction + tool action)" \
  "Stop using the requests library. We use httpx for all HTTP calls in this project. Create http_client.py with a function get_json(url) that fetches and returns JSON using httpx."
echo "http_client.py: $(test -f http_client.py && echo CREATED || echo MISSING)"
run_one "SESSION 2 (recall)" "What HTTP client should this project use, and why?"

echo "=== retrieve(semantic) ==="
curl -s -X POST "$BASE/graphs/$GID/retrieve" -H 'Content-Type: application/json' \
  -d '{"observation":"which HTTP client should I use","mode":"semantic_memory"}' | head -c 500
echo
echo "=== adapter log (tools / promotion / recall) ==="
grep -E 'TOOL BEFORE|TOOL AFTER|promotion-gate|inserted|extract call|Processing user prompt' plugin-debug.log | tail -40

kill "$SERVE_PID" 2>/dev/null
echo "done"
