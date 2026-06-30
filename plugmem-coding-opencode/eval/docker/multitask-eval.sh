#!/usr/bin/env bash
# Runs several diverse OpenCode agent sessions against a PERSISTENT opencode
# serve, accumulating a richer memory graph (multiple conventions + a
# failure->success) for inspecting in the Memory Inspector.
set -u
BASE="http://127.0.0.1:8000/api/v1"
GID="repo_opencode_local__work_proj"
PLUGIN="/app/PlugMem/plugmem-coding-opencode/dist/index.js"
OC_MODEL="${OC_MODEL:-local-qwen/unsloth/MiniMax-M2.7-GGUF:UD-Q4_K_XL}"
OPENCODE_BASE_URL="${OPENCODE_BASE_URL:-https://jizejtestblackwell01.ngrok.app/v1}"
OC_TIMEOUT="${OC_TIMEOUT:-420}"
case "$OC_MODEL" in local-qwen/*) LOCAL_ID="${OC_MODEL#local-qwen/}";; *) LOCAL_ID="placeholder";; esac

mkdir -p /work/proj && cd /work/proj || exit 1
rm -f plugin-debug.log
[ -d .git ] || git init -q
cat > opencode.json <<JSON
{ "\$schema":"https://opencode.ai/config.json",
  "provider":{"local-qwen":{"npm":"@ai-sdk/openai-compatible","name":"Local","options":{"baseURL":"${OPENCODE_BASE_URL}","apiKey":"EMPTY","headers":{"ngrok-skip-browser-warning":"true"}},"models":{"${LOCAL_ID}":{"name":"local"}}}},
  "model":"${OC_MODEL}", "plugin":["${PLUGIN}"] }
JSON

PLUGMEM_URL=http://127.0.0.1:8000 opencode serve --port 4096 --hostname 127.0.0.1 >/tmp/serve.log 2>&1 &
SERVE_PID=$!
for i in $(seq 1 30); do c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4096/ 2>/dev/null); [ "$c" != "000" ] && break; sleep 1; done
echo "opencode server up"

run_one () {
  echo "==================== $1 ===================="
  timeout "$OC_TIMEOUT" opencode run --attach http://127.0.0.1:4096 --model "$OC_MODEL" "$2" 2>&1 | tail -4
  echo "[exit ${PIPESTATUS[0]}]"
  sleep 10
  echo "stats: $(curl -s "$BASE/graphs/$GID/stats")"
}

run_one "S1 logging convention (correction + tool)" \
  "Stop using print statements for logging in this project. Use Python's logging module. Create a file logutil.py with a function get_logger(name) that returns a configured logger."
run_one "S2 linter convention (correction)" \
  "We use ruff for linting and formatting in this project, not black or flake8."
run_one "S3 paths convention (correction)" \
  "Always use pathlib.Path for filesystem paths in this project instead of os.path."
run_one "S4 failure->success (failure_delta)" \
  "Create a file check.py containing exactly one line: import cowsay. Run it with python3 check.py - it will fail with ModuleNotFoundError. Fix it by running pip install cowsay, then run python3 check.py again until it succeeds."

sleep 4
echo "=== FINAL stats: $(curl -s "$BASE/graphs/$GID/stats")"
kill "$SERVE_PID" 2>/dev/null; echo done
