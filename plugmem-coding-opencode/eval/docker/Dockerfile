# End-to-end eval image: OpenCode + the PlugMem plugin + the PlugMem server.
# Lets a real OpenCode agent session run against the plugin in isolation.
#
# Build (context = repo root):
#   docker build -f plugmem-coding-opencode/eval/docker/Dockerfile -t plugmem-eval .
# Run (pass your ngrok endpoints):
#   docker run -d --name plugmem-eval \
#     -e LLM_BASE_URL=https://jizejtestada00.ngrok.app/v1 \
#     -e LLM_MODEL=CalamitousFelicitousness/Qwen2.5-32B-Instruct-fp8-dynamic \
#     -e LLM_API_KEY=EMPTY \
#     -e EMBEDDING_BASE_URL=https://jizejtestada01.ngrok.app/v1/embeddings,https://jizejtestada02.ngrok.app/v1/embeddings \
#     -e EMBEDDING_MODEL=nvidia/NV-Embed-v2 \
#     plugmem-eval
#   docker exec -it plugmem-eval /app/PlugMem/plugmem-coding-opencode/eval/docker/run-eval.sh

FROM node:22-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip git curl unzip ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

# OpenCode CLI (standalone binary -> ~/.opencode/bin)
RUN curl -fsSL https://opencode.ai/install | bash
ENV PATH="/root/.opencode/bin:${PATH}"

# Isolated Python env for the PlugMem server (avoids Debian PEP-668 lockout)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app/PlugMem
COPY . /app/PlugMem

# PlugMem server + its deps
RUN pip install --no-cache-dir -e .

# Build the TS packages (core first so its .d.ts resolves for the adapter)
RUN cd plugmem-coding-core && npm install --no-audit --no-fund && npm run build \
 && cd ../plugmem-coding-opencode && npm install --no-audit --no-fund && npm run build

RUN chmod +x /app/PlugMem/plugmem-coding-opencode/eval/docker/entrypoint.sh \
             /app/PlugMem/plugmem-coding-opencode/eval/docker/run-eval.sh

WORKDIR /work
ENTRYPOINT ["/app/PlugMem/plugmem-coding-opencode/eval/docker/entrypoint.sh"]
