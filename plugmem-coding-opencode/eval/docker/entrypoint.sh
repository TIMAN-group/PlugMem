#!/usr/bin/env bash
# Starts the PlugMem server, waits for health, then keeps the container alive
# so the eval can be driven via `docker exec`.
set -u

export CHROMA_MODE="${CHROMA_MODE:-ephemeral}"
# Bind 127.0.0.1 by default; set PLUGMEM_HOST=0.0.0.0 (with `docker run -p 8000:8000`)
# to reach the API + Memory Inspector (/inspector/) from the host browser.
HOST="${PLUGMEM_HOST:-127.0.0.1}"
echo "[entrypoint] LLM_BASE_URL=${LLM_BASE_URL:-<unset>}"
echo "[entrypoint] EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL:-<unset>}"
echo "[entrypoint] starting PlugMem server on ${HOST}:8000 (chroma=$CHROMA_MODE) ..."

uvicorn plugmem.api.app:app --host "$HOST" --port 8000 --log-level warning &
SERVER_PID=$!

for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[entrypoint] server died on startup"; exit 1; fi
  sleep 1
done

echo "[entrypoint] health: $(curl -s http://127.0.0.1:8000/api/v1/health)"
echo "[entrypoint] ready. Drive the eval with:"
echo "    docker exec -it <name> /app/PlugMem/plugmem-coding-opencode/eval/docker/run-eval.sh"
tail -f /dev/null
