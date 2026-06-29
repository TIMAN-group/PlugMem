#!/usr/bin/env bash
# Validates the failure_delta (failure -> success) procedural-memory path with a
# REAL agent: force a failing `python3 run.py` (ModuleNotFoundError) then a fix
# (pip install) then a successful rerun. The prompt contains no correction/episodic
# trigger, so any promoted memory must come from the failure_delta detector.
set -u
BASE="http://127.0.0.1:8000/api/v1"
GID="repo_opencode_local__work_proj"
PLUGIN="/app/PlugMem/plugmem-coding-opencode/dist/index.js"
OC_MODEL="${OC_MODEL:-local-qwen/unsloth/MiniMax-M2.7-GGUF:UD-Q4_K_XL}"
OPENCODE_BASE_URL="${OPENCODE_BASE_URL:-https://jizejtestblackwell01.ngrok.app/v1}"
OC_TIMEOUT="${OC_TIMEOUT:-420}"
case "$OC_MODEL" in local-qwen/*) LOCAL_ID="${OC_MODEL#local-qwen/}";; *) LOCAL_ID="placeholder";; esac

cd /work/proj || exit 1
rm -f plugin-debug.log run.py
[ -d .git ] || git init -q
cat > opencode.json <<JSON
{ "\$schema":"https://opencode.ai/config.json",
  "provider":{"local-qwen":{"npm":"@ai-sdk/openai-compatible","name":"Local","options":{"baseURL":"${OPENCODE_BASE_URL}","apiKey":"EMPTY","headers":{"ngrok-skip-browser-warning":"true"}},"models":{"${LOCAL_ID}":{"name":"local"}}}},
  "model":"${OC_MODEL}", "plugin":["${PLUGIN}"] }
JSON

PLUGMEM_URL=http://127.0.0.1:8000 opencode serve --port 4096 --hostname 127.0.0.1 >/tmp/serve.log 2>&1 &
SERVE_PID=$!
for i in $(seq 1 30); do c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4096/ 2>/dev/null); [ "$c" != "000" ] && { echo "opencode server up $c"; break; }; sleep 1; done

echo "==================== SESSION 1 (failure -> success) ===================="
timeout "$OC_TIMEOUT" opencode run --attach http://127.0.0.1:4096 --model "$OC_MODEL" \
"Create a file run.py containing exactly this one line: import cowsay
Then run it with: python3 run.py
It will fail with ModuleNotFoundError: No module named 'cowsay'.
Fix it by installing the package: pip install cowsay
Then run python3 run.py again so it exits with no error. Keep going until python3 run.py succeeds." 2>&1 | tail -12
echo "[run exit ${PIPESTATUS[0]}]"
echo "waiting 12s for async promotion..."
sleep 12

echo "stats: $(curl -s "$BASE/graphs/$GID/stats")"
echo "=== TOOL OUTCOMEs (did a bash call get classified failure?) ==="
grep -E 'TOOL OUTCOME' plugin-debug.log | tail -30
echo "=== promotion (failure_delta -> memory) ==="
grep -E 'DIAG|promotion-gate|inserted|extract call' plugin-debug.log | tail -10
echo "=== procedural recall ==="
curl -s -X POST "$BASE/graphs/$GID/retrieve" -H 'Content-Type: application/json' \
  -d '{"observation":"how do I fix ModuleNotFoundError when running the script","mode":"procedural_memory"}' | head -c 500
echo
kill "$SERVE_PID" 2>/dev/null; echo done
