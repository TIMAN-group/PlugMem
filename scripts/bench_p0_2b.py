#!/usr/bin/env python3
"""P0-2B benchmark — legacy O(N) cosine loop vs numpy vectorized matrix path.

WHY (P0-2B): the old retrieve_semantic_nodes looped over every active node and
called get_similarity() one at a time (47K nodes here; 89K in production =>
3.79-5.85s cold).  The vectorized path collapses that into one float32 matrix
multiply + topk selection, keeping the BM25/credibility fusion untouched.

Read-only: opens persistent ChromaDB in-process; never talks to the 8089 HTTP
service.  Query vectors use the configured HTTP embedder when reachable, else
fall back offline to LocalDeterministicEmbeddingClient (dim-matched to stored
vectors) so the benchmark still runs.  Output: /tmp/bench_p0_2b_result.txt.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CHROMA_MODE", "persistent")
os.environ.setdefault("CHROMA_PATH", str(REPO_ROOT / "data" / "chroma"))
os.environ.pop("PLUGMEM_P0_2B_DISABLE", None)

import chromadb

from plugmem.clients.embedding import (
    EmbeddingClient,
    HTTPEmbeddingClient,
    LocalDeterministicEmbeddingClient,
)
from plugmem.clients.llm import LLMClient
from plugmem.core.memory_graph import MemoryGraph
from plugmem.graph_manager import GraphManager
from plugmem.storage.chroma import ChromaStorage

GRAPH = "brian-main"
QUERIES = [
    "澳洲航线船期怎么安排",
    "企查查按次计费多少钱一次",
    "公司有多少人",
    "QM部署在哪台服务器",
    "记忆系统检索入口是什么服务",
]
RESULT_PATH = Path("/tmp/bench_p0_2b_result.txt")
WARMUP = 2
RUNS = 5


class DummyLLM(LLMClient):
    """Benchmark-only LLM; retrieve_semantic_nodes never calls complete()."""

    def complete(self, messages, temperature=0, top_p=1.0, max_tokens=4096):
        return ""


def _percentile(samples: List[float], p: float) -> float:
    """Linear-interpolation percentile (numpy.percentile default method)."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _query_embedder(node_dim: int) -> EmbeddingClient:
    """Prefer the configured HTTP embedder; fall back offline to deterministic.

    P0-2B benchmark only.  The fallback keeps query dims aligned with stored
    bge-m3 vectors so the matrix path stays dimensionally valid.
    """
    base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1/embeddings")
    model = os.getenv("EMBEDDING_MODEL", "bge-m3")
    client = HTTPEmbeddingClient(
        base_url=base_url, model=model, timeout=5, max_retries=1, retry_delay=0.5,
    )
    try:
        probe = client.embed(QUERIES[0])
        if len(probe) == node_dim:
            print(f"[embed] HTTP embedder OK dim={len(probe)}")
            return client
        print(f"[embed] HTTP embedder dim mismatch ({len(probe)} != {node_dim}); fallback")
    except Exception as exc:  # noqa: BLE001
        print(f"[embed] HTTP embedder unavailable ({exc}); using deterministic fallback")
    return LocalDeterministicEmbeddingClient(dim=node_dim)


def main() -> int:
    chroma_client = chromadb.PersistentClient(
        path=os.environ["CHROMA_PATH"],
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    probe_col = chroma_client.get_collection(f"{GRAPH}_semantic")
    probe = probe_col.get(include=["embeddings"], limit=1)
    node_dim = int(len(probe["embeddings"][0])) if probe.get("embeddings") is not None and len(probe["embeddings"]) else 1024

    embedder = _query_embedder(node_dim)
    storage = ChromaStorage(client=chroma_client, embedding_client=embedder)
    manager = GraphManager(storage=storage, llm=DummyLLM(), embedder=embedder)
    graph: MemoryGraph = manager.get_graph(GRAPH)

    active_total = sum(1 for n in graph.semantic_nodes if n.is_active)
    emb_total = sum(1 for n in graph.semantic_nodes if n.is_active and n.embedding is not None)
    print(
        f"[graph] {GRAPH} loaded: {len(graph.semantic_nodes)} semantic "
        f"(active={active_total}, active+embedding={emb_total}, dim={node_dim})"
    )

    query_vecs = [embedder.embed(q) for q in QUERIES]

    def run_one(path: str, q: str, qv: List[float]) -> Tuple[float, List[int]]:
        os.environ["PLUGMEM_P0_2B_DISABLE"] = "1" if path == "old" else "0"
        t0 = time.perf_counter()
        result = graph.retrieve_semantic_nodes(
            semantic_memory={"semantic_memory": q, "tags": []},
            semantic_memory_embedding={"semantic_memory": qv, "tags": []},
            value_func_tag=graph.tag_relevant,
            value_func=graph.semantic_relevant,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms, [n.semantic_id for n in result[:5]]

    per_query: Dict[str, Dict[str, List[float]]] = {"old": {}, "new": {}}
    top5: Dict[str, Dict[str, List[int]]] = {"old": {}, "new": {}}
    for path in ("old", "new"):
        for q, qv in zip(QUERIES, query_vecs):
            for _ in range(WARMUP):
                run_one(path, q, qv)
            samples = []
            ids: List[int] = []
            for _ in range(RUNS):
                elapsed, ids = run_one(path, q, qv)
                samples.append(elapsed)
            per_query[path][q] = samples
            top5[path][q] = ids
            print(
                f"[{path}] {q[:14]:<14} p50={_percentile(samples, 50):7.1f}ms "
                f"p95={_percentile(samples, 95):7.1f}ms top5={ids}"
            )

    all_old = [v for s in per_query["old"].values() for v in s]
    all_new = [v for s in per_query["new"].values() for v in s]
    old_p50, old_p95 = _percentile(all_old, 50), _percentile(all_old, 95)
    new_p50, new_p95 = _percentile(all_new, 50), _percentile(all_new, 95)

    lines: List[str] = []
    lines.append("P0-2B numpy vectorized semantic candidates benchmark")
    lines.append(
        f"graph={GRAPH} semantic_nodes={len(graph.semantic_nodes)} "
        f"active={active_total} active+embedding={emb_total} dim={node_dim}"
    )
    lines.append(f"runs_per_query={RUNS} warmup={WARMUP}")
    lines.append("")
    lines.append("query | old p50 ms | old p95 ms | new p50 ms | new p95 ms | top5 consistent")
    lines.append("------|-----------:|-----------:|-----------:|-----------:|----------------")
    consistent = True
    for q in QUERIES:
        o50 = _percentile(per_query["old"][q], 50)
        o95 = _percentile(per_query["old"][q], 95)
        n50 = _percentile(per_query["new"][q], 50)
        n95 = _percentile(per_query["new"][q], 95)
        ok = set(top5["old"][q]) == set(top5["new"][q])
        consistent = consistent and ok
        lines.append(
            f"{q} | {o50:.1f} | {o95:.1f} | {n50:.1f} | {n95:.1f} | "
            f"{'PASS' if ok else 'FAIL'}"
        )
    lines.append("")
    lines.append(
        f"AGG old p50={old_p50:.1f}ms old p95={old_p95:.1f}ms "
        f"new p50={new_p50:.1f}ms new p95={new_p95:.1f}ms"
    )
    lines.append(f"CONSISTENCY={'100%' if consistent else 'FAILED'}")
    lines.append(f"P95_TARGET={'PASS' if new_p95 < 200 else 'FAIL'} (new p95 < 200ms)")

    RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {RESULT_PATH}")
    print("\n".join(lines))
    return 0 if (consistent and new_p95 < 200) else 1


if __name__ == "__main__":
    raise SystemExit(main())
