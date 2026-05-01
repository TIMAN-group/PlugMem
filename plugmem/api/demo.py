"""Demo seed data for the Memory Inspector.

Fake-but-realistic OpenClaw trajectory: a coding-assistant persona working on
a fictional `acme-api` FastAPI repo across five sessions, plus one personal
travel-planning session at the end (mirroring how real PlugMem graphs end
up domain-mixed via auto-remember).

The seed is fully self-contained — no LLM or embedder calls — and uses
deterministic fake embeddings so the data is reproducible across runs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from plugmem.clients.embedding import EmbeddingClient, LocalDeterministicEmbeddingClient
from plugmem.storage.chroma import ChromaStorage

DEMO_GRAPH_ID = "demo"

# Default embedder used when no real embedder is provided. Seed and retrieval
# MUST use the same embedder (otherwise dim mismatches in chroma) — the route
# passes the configured one via ``embedder=`` so production paths get real
# vectors and the demo CLI / no-config path still works.
_DEFAULT_EMBEDDER = LocalDeterministicEmbeddingClient()


# ------------------------------------------------------------------ #
# Persona script
# ------------------------------------------------------------------ #
#
# Sessions (chronological):
#   session-001  2026-03-15  CI is red — debug flaky pytest
#   session-002  2026-03-17  add /users endpoint
#   session-003  2026-03-22  refactor auth module to JWT
#   session-004  2026-03-25  staging deploy goes sideways
#   session-005  2026-04-02  user pivots: Paris trip planning
#
# All times are integer counters (matches MemoryGraph.semantic_time semantics).
# ------------------------------------------------------------------ #

TAGS: List[Dict] = [
    {"tag_id": 0, "tag": "python",          "importance": 4},
    {"tag_id": 1, "tag": "fastapi",         "importance": 5},
    {"tag_id": 2, "tag": "pytest",          "importance": 3},
    {"tag_id": 3, "tag": "auth",            "importance": 4},
    {"tag_id": 4, "tag": "deployment",      "importance": 3},
    {"tag_id": 5, "tag": "conventions",     "importance": 5},
    {"tag_id": 6, "tag": "user-preference", "importance": 4},
    {"tag_id": 7, "tag": "travel",          "importance": 2},
    {"tag_id": 8, "tag": "france",          "importance": 1},
]

# (semantic_id, text, [tag_ids], time, credibility, is_active, date, session_id)
SEMANTICS: List[tuple] = [
    (0,  "The acme-api project uses FastAPI 0.110+ with Pydantic v2.",
        [0, 1, 5], 0, 10, True,  "2026-03-15", "session-001"),
    (1,  "CI runs pytest with --cov on Python 3.11 and 3.12.",
        [0, 2, 5], 1, 10, True,  "2026-03-15", "session-001"),
    (2,  "The team prefers httpx.AsyncClient over requests for outbound HTTP.",
        [0, 5],    2,  9, True,  "2026-03-15", "session-001"),
    (3,  "Auth tokens are JWT (HS256) with 1-hour expiry, refreshed via /auth/refresh.",
        [3, 1],    3, 10, True,  "2026-03-22", "session-003"),
    (4,  "The auth module uses session cookies stored in Redis.",
        [3],       4,  2, False, "2026-03-15", "session-001"),
    (5,  "The user prefers verbose docstrings on public API functions.",
        [6, 5],    5,  8, True,  "2026-03-17", "session-002"),
    (6,  "Database migrations use Alembic; never edit a migration after merge.",
        [5],       6, 10, True,  "2026-03-22", "session-003"),
    (7,  "PRs require at least one reviewer and green CI before merge.",
        [5],       7,  9, True,  "2026-03-17", "session-002"),
    (8,  "Production deploys via GitHub Actions on push to main; staging on tag.",
        [4],       8, 10, True,  "2026-03-25", "session-004"),
    (9,  "The user goes by 'Sam' in chat; prefers concise replies, no preamble.",
        [6],       9, 10, True,  "2026-03-15", "session-001"),
    (10, "The acme-api repo lives at github.com/acme-corp/acme-api.",
        [],       10, 10, True,  "2026-03-15", "session-001"),
    (11, "Deployment uses docker-compose in staging and Kubernetes in prod.",
        [4],      11,  9, True,  "2026-03-25", "session-004"),
    (12, "Black formats Python at line-length 100; ruff is the linter.",
        [0, 5],   12, 10, True,  "2026-03-17", "session-002"),
    (13, "The user is planning a trip to Paris for the first week of May 2026.",
        [6, 7, 8], 13, 7, True,  "2026-04-02", "session-005"),
    (14, "Direct flights from SFO to CDG run about 11 hours non-stop.",
        [7],      14,  6, True,  "2026-04-02", "session-005"),
    (15, "The user prefers train over plane for journeys under four hours.",
        [6, 7],   15,  8, True,  "2026-04-02", "session-005"),
    (16, "Health check endpoint is /healthz, not /health.",
        [1, 5],   16,  9, True,  "2026-03-25", "session-004"),
]

# (subgoal_id, text, time)
SUBGOALS: List[tuple] = [
    (0, "add a FastAPI endpoint", 1),
    (1, "fix a flaky pytest",     2),
    (2, "deploy to staging",      3),
    (3, "plan a Paris trip",      4),
]

# (procedural_id, text, subgoal_id, return_value, time, session_id)
PROCEDURALS: List[tuple] = [
    (0, "Define route in routers/, add Pydantic request/response models in "
        "schemas/, register the router in main.py, write 2 pytest cases "
        "(happy path + 401 unauthorized), tag the OpenAPI section.",
        0, 1.0, 1, "session-002"),
    (1, "Identify the race in the async fixture (yield order vs event loop), "
        "switch to anyio markers, rerun with pytest-rerunfailures to confirm "
        "stability over 50 iterations.",
        1, 0.8, 2, "session-001"),
    (2, "Bump version in pyproject.toml, push annotated tag v0.x.y, GH Action "
        "builds the image, then `helm upgrade --install acme-api ./chart -n "
        "staging` and smoke-test /healthz.",
        2, 1.0, 3, "session-004"),
    (3, "Push directly to main without bumping the tag — caused an image "
        "rebuild loop in staging. Manual rollback to previous tag, opened "
        "post-mortem incident-042. Do not repeat.",
        2, 0.2, 3, "session-004"),
    (4, "Book SNCF train Paris→Lyon on TGV INOUI 2nd class, reserve hotel "
        "near Gare de Lyon, draft a 4-day itinerary with one day-trip to "
        "Versailles.",
        3, 0.9, 4, "session-005"),
]

# Recall audit log — what the agent asked about during each session.
# Cross-session lookups are intentional (e.g. session-003 retrieves the
# deactivated cookie-auth semantic from session-001 while migrating to JWT)
# so the Sessions view has interesting things to show.
RECALLS: List[Dict] = [
    # session-001: debugging flaky pytest
    {
        "endpoint": "retrieve",
        "ts": "2026-03-15T09:12:33+00:00",
        "session_id": "session-001",
        "observation": "tests timing out at 30s with anyio backend",
        "mode": "semantic_memory",
        "query_tags": ["pytest", "python"],
        "selected_sids": [1, 2],
        "selected_pids": [],
        "n_messages": 4,
        "graph_time": 1,
    },
    {
        "endpoint": "reason",
        "ts": "2026-03-15T09:34:18+00:00",
        "session_id": "session-001",
        "observation": "what's the procedure for fixing flaky pytest fixtures?",
        "mode": "procedural_memory",
        "query_tags": ["pytest"],
        "selected_sids": [],
        "selected_pids": [1],
        "n_messages": 5,
        "graph_time": 2,
    },
    # session-002: adding the /users endpoint
    {
        "endpoint": "retrieve",
        "ts": "2026-03-17T14:08:02+00:00",
        "session_id": "session-002",
        "observation": "convention for adding a new FastAPI route",
        "mode": "procedural_memory",
        "query_tags": ["fastapi", "conventions"],
        "selected_sids": [0, 5, 7, 12],
        "selected_pids": [0],
        "n_messages": 5,
        "graph_time": 3,
    },
    {
        "endpoint": "retrieve",
        "ts": "2026-03-17T14:55:41+00:00",
        "session_id": "session-002",
        "observation": "Pydantic version and stack details for acme-api",
        "mode": "semantic_memory",
        "query_tags": ["fastapi", "python"],
        "selected_sids": [0, 1, 2],
        "selected_pids": [],
        "n_messages": 4,
        "graph_time": 4,
    },
    # session-003: switching from cookies to JWT — pulls a deactivated fact
    {
        "endpoint": "retrieve",
        "ts": "2026-03-22T10:45:00+00:00",
        "session_id": "session-003",
        "observation": "what was our previous auth setup?",
        "mode": "semantic_memory",
        "query_tags": ["auth"],
        "selected_sids": [4],  # the deactivated cookie-auth fact
        "selected_pids": [],
        "n_messages": 4,
        "graph_time": 5,
    },
    {
        "endpoint": "reason",
        "ts": "2026-03-22T11:20:51+00:00",
        "session_id": "session-003",
        "observation": "how do I migrate an auth module from cookies to JWT bearer?",
        "mode": "semantic_memory",
        "query_tags": ["auth", "fastapi"],
        "selected_sids": [3, 4],
        "selected_pids": [],
        "n_messages": 5,
        "graph_time": 6,
    },
    # session-004: staging deploy + post-mortem
    {
        "endpoint": "retrieve",
        "ts": "2026-03-25T16:02:14+00:00",
        "session_id": "session-004",
        "observation": "what's the deploy process for staging?",
        "mode": "procedural_memory",
        "query_tags": ["deployment"],
        "selected_sids": [8, 11],
        "selected_pids": [2],
        "n_messages": 5,
        "graph_time": 7,
    },
    {
        "endpoint": "retrieve",
        "ts": "2026-03-25T16:48:09+00:00",
        "session_id": "session-004",
        "observation": "where is the health check endpoint?",
        "mode": "semantic_memory",
        "query_tags": ["fastapi", "conventions"],
        "selected_sids": [16],
        "selected_pids": [],
        "n_messages": 4,
        "graph_time": 8,
    },
    {
        "endpoint": "reason",
        "ts": "2026-03-25T18:22:07+00:00",
        "session_id": "session-004",
        "observation": "should I push to main without bumping the tag?",
        "mode": "procedural_memory",
        "query_tags": ["deployment"],
        "selected_sids": [],
        "selected_pids": [2, 3],  # surfaces both the right way and the cautionary one
        "n_messages": 5,
        "graph_time": 9,
    },
    # session-005: Paris trip — domain-mixed
    {
        "endpoint": "retrieve",
        "ts": "2026-04-02T19:11:55+00:00",
        "session_id": "session-005",
        "observation": "user travel preferences",
        "mode": "semantic_memory",
        "query_tags": ["user-preference", "travel"],
        "selected_sids": [13, 15],
        "selected_pids": [],
        "n_messages": 4,
        "graph_time": 10,
    },
    {
        "endpoint": "reason",
        "ts": "2026-04-02T19:30:42+00:00",
        "session_id": "session-005",
        "observation": "Paris to Lyon: train or plane?",
        "mode": "procedural_memory",
        "query_tags": ["travel", "france"],
        "selected_sids": [15],
        "selected_pids": [4],
        "n_messages": 5,
        "graph_time": 11,
    },
]

# (episodic_id, observation, action, time, session_id, subgoal_text)
EPISODICS: List[tuple] = [
    (0, "pytest -v: 3 failures in tests/test_auth.py, all timing out at 30s.",
        "Edit conftest.py to add the anyio_backend fixture and switch the "
        "client fixture to async.",
        0, "session-001", "fix a flaky pytest"),
    (1, "Tests pass locally but CI is still red on the same suite.",
        "Update .github/workflows/ci.yml to add pytest-rerunfailures and "
        "set --reruns 2.",
        1, "session-001", "fix a flaky pytest"),
    (2, "User: 'add a /users endpoint with create + list, paginated'.",
        "Create routers/users.py with POST and GET handlers; wire pagination "
        "via fastapi.Query.",
        2, "session-002", "add a FastAPI endpoint"),
    (3, "Need request/response models for the new routes.",
        "Add UserCreate, UserOut, UsersPage in schemas/users.py.",
        3, "session-002", "add a FastAPI endpoint"),
    (4, "User: 'switch auth from cookies to JWT bearer'.",
        "Replace cookie middleware with HTTPBearer; add /auth/refresh; "
        "rotate refresh tokens on use.",
        4, "session-003", ""),
    (5, "Need test coverage for the refresh-token rotation.",
        "Add test_refresh_token_rotation in tests/test_auth.py covering "
        "valid, expired, and reused refresh tokens.",
        5, "session-003", ""),
    (6, "Cut a release for staging.",
        "Tag v0.4.1, push, watch GH Action build + helm upgrade.",
        6, "session-004", "deploy to staging"),
    (7, "Image rebuild loop in staging — same tag rebuilt 4 times in 8 min.",
        "Manual rollback to v0.4.0, paged the on-call, opened incident-042.",
        7, "session-004", "deploy to staging"),
    (8, "Production /health returns 404 from the new ingress.",
        "Confirm endpoint is /healthz not /health; update probe in helm "
        "values.",
        8, "session-004", "deploy to staging"),
    (9, "User: 'thinking about Paris first week of May, advise'.",
        "Capture trip dates, ask about train vs flight preference, draft a "
        "rough 4-day itinerary skeleton.",
        9, "session-005", "plan a Paris trip"),
    (10, "User: 'should I fly or train Paris→Lyon?'.",
        "Recommend SNCF TGV INOUI: ~2 hours center-to-center, scenic, "
        "no security theatre.",
        10, "session-005", "plan a Paris trip"),
]


# ------------------------------------------------------------------ #
# Seed function
# ------------------------------------------------------------------ #


def _tag_text(tag_id: int) -> str:
    return next(t["tag"] for t in TAGS if t["tag_id"] == tag_id)


def _build_tag_to_semantics() -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {t["tag_id"]: [] for t in TAGS}
    for sid, _text, tag_ids, *_ in SEMANTICS:
        for tid in tag_ids:
            out[tid].append(sid)
    return out


# SEMANTICS tuple positions: (sid, text, tag_ids, time, cred, active, date, session_id)
# EPISODICS  tuple positions: (eid, obs, act, time, session_id, subgoal_text)


def _bro_semantic_ids(session_id: str, exclude: int, k: int = 3) -> List[int]:
    return [s[0] for s in SEMANTICS if s[7] == session_id and s[0] != exclude][:k]


def _episodics_in_session(session_id: str, k: int = 2) -> List[int]:
    return [e[0] for e in EPISODICS if e[4] == session_id][:k]


def _episodics_for_subgoal(subgoal_text: str, k: int = 3) -> List[int]:
    return [e[0] for e in EPISODICS if e[5] == subgoal_text][:k]


def seed_demo_graph(
    storage: ChromaStorage,
    graph_id: str = DEMO_GRAPH_ID,
    reset: bool = False,
    embedder: Optional[EmbeddingClient] = None,
) -> Dict[str, int]:
    """Populate ``graph_id`` with the realistic demo trajectory.

    Idempotent: if the graph already exists and ``reset`` is False, returns
    the existing stats without rewriting. ``reset=True`` deletes and recreates.

    ``embedder`` controls which client computes the stored vectors. Pass the
    same one retrieval will use, or chroma will reject queries with a dim
    mismatch. Defaults to LocalDeterministicEmbeddingClient.
    """
    embed = (embedder or _DEFAULT_EMBEDDER).embed

    if reset:
        storage.delete_graph(graph_id)

    if storage.graph_exists(graph_id):
        stats = storage.get_graph_stats(graph_id)
        if any(stats.values()):
            return stats
        # Empty collections from a previous failed seed — wipe and redo.
        storage.delete_graph(graph_id)

    storage.create_graph(graph_id)

    # 1. Episodic — pass an explicit embedding so the demo doesn't rely on
    # a configured embedding service. The inspector never queries episodic
    # by similarity; the value is just a placeholder.
    for eid, obs, act, t, session_id, subgoal in EPISODICS:
        storage.add_episodic(
            graph_id,
            episodic_id=eid,
            observation=obs,
            action=act,
            time=t,
            session_id=session_id,
            subgoal=subgoal,
            embedding=embed(f"{obs}\n{act}"),
        )

    # 2. Tags (with semantic_ids cross-reference)
    tag_to_sem = _build_tag_to_semantics()
    for tag in TAGS:
        storage.add_tag(
            graph_id,
            tag_id=tag["tag_id"],
            tag=tag["tag"],
            embedding=embed(tag["tag"]),
            semantic_ids=tag_to_sem.get(tag["tag_id"], []),
            importance=tag["importance"],
            time=0,
        )

    # 3. Semantic
    for sid, text, tag_ids, t, cred, active, date, session_id in SEMANTICS:
        storage.add_semantic(
            graph_id,
            semantic_id=sid,
            text=text,
            embedding=embed(text),
            tags=[_tag_text(tid) for tid in tag_ids],
            tag_ids=tag_ids,
            time=t,
            is_active=active,
            episodic_ids=_episodics_in_session(session_id),
            bro_semantic_ids=_bro_semantic_ids(session_id, exclude=sid),
            session_id=session_id,
            credibility=cred,
            date=date,
        )

    # 4. Subgoals (with procedural_ids cross-reference)
    for sg_id, sg_text, t in SUBGOALS:
        storage.add_subgoal(
            graph_id,
            subgoal_id=sg_id,
            subgoal=sg_text,
            embedding=embed(sg_text),
            procedural_ids=[p[0] for p in PROCEDURALS if p[2] == sg_id],
            time=t,
        )

    # 5. Procedurals
    for pid, text, sg_id, ret, t, session_id in PROCEDURALS:
        sg_text = next(s[1] for s in SUBGOALS if s[0] == sg_id)
        storage.add_procedural(
            graph_id,
            procedural_id=pid,
            text=text,
            embedding=embed(text),
            subgoal=sg_text,
            subgoal_id=sg_id,
            episodic_ids=_episodics_for_subgoal(sg_text),
            time=t,
            return_value=ret,
            session_id=session_id,
        )

    # 6. Recall audit log — what the agent looked up during each session.
    # Demonstrates cross-session retrieval (e.g. session-003 surfacing the
    # deactivated cookie-auth fact from session-001) so the Sessions view
    # has interesting things to show.
    for entry in RECALLS:
        storage.add_recall(
            graph_id,
            endpoint=entry["endpoint"],
            ts=entry["ts"],
            graph_time=entry.get("graph_time", 0),
            session_id=entry["session_id"],
            observation=entry["observation"],
            mode=entry["mode"],
            query_tags=entry.get("query_tags", []),
            selected_semantic_ids=entry.get("selected_sids", []),
            selected_procedural_ids=entry.get("selected_pids", []),
            n_messages=entry.get("n_messages", 0),
            embedding=embed(entry["observation"]),
        )

    return storage.get_graph_stats(graph_id)


# ------------------------------------------------------------------ #
# CLI: `python -m plugmem.api.demo [--reset] [--graph-id NAME]`
# ------------------------------------------------------------------ #

if __name__ == "__main__":  # pragma: no cover
    import argparse
    import os

    import chromadb

    p = argparse.ArgumentParser(description="Seed the PlugMem demo graph.")
    p.add_argument("--graph-id", default=DEMO_GRAPH_ID)
    p.add_argument("--reset", action="store_true",
                   help="Delete and recreate if the graph already exists.")
    p.add_argument("--chroma-path", default=os.getenv("CHROMA_PATH", "./data/chroma"))
    args = p.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_path)
    storage = ChromaStorage(client=client)
    stats = seed_demo_graph(storage, graph_id=args.graph_id, reset=args.reset)
    print(f"seeded graph_id={args.graph_id!r} at {args.chroma_path}: {stats}")
