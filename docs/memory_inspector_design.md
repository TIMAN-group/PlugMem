# PlugMem Memory Inspector — Design & Plan

Status: draft
Branch: `feat_visualization`

## 1. Motivation

PlugMem builds a multi-type memory graph (episodic, semantic, tag, subgoal,
procedural) backed by ChromaDB. Today, the only way to inspect what an agent
has actually remembered is to write Python against `GraphManager` or open the
ChromaDB store directly. For users running PlugMem with an agent harness like
OpenClaw, that workflow is too far from where the questions actually arise.

The questions agent users ask, in roughly descending frequency:

1. *"What did the agent remember from this session?"*
2. *"Why didn't `recall` return X when I asked about it?"*
3. *"This stored fact is wrong — can I disable it?"*
4. *"How is the graph growing? Are tags exploding? Is consolidation working?"*
5. *"Show me the structure of the graph."* (one-time, demo, paper figures)

A force-directed node-link visualization answers (5) but is the wrong tool
for (1)–(4). Past ~50 semantic nodes a hairball view becomes unreadable noise.
This document specifies a small web app — the **Memory Inspector** — whose
*headline* is browse + recall debugging, with the graph view as one tab among
several rather than the whole product.

## 2. Goals & Non-Goals

### Goals

- Local-first web UI accessible at a single URL after running the existing
  PlugMem service.
- Supports the four user questions above for graphs with up to ~10k semantic
  nodes without falling over.
- Pure JSON API + static frontend. The server has no rendering logic.
- Frontend is **swappable by theme** — same data, same DOM contracts, different
  CSS. The pixel-art ambition lives entirely in a future theme module; the
  base build ships with one neutral default theme.
- No build step. Vanilla JS + a single graph library via CDN. Read the source,
  edit, refresh.

### Non-Goals (for now)

- Multi-user / collaborative editing.
- Real-time streaming updates (re-render on graph mutation outside the
  inspector). Refresh-to-see-new is fine.
- Topology editing — creating new edges or new nodes by hand.
- Triggering full consolidation runs from the UI. (Already exposed via the
  existing `/api/v1/graphs/{id}/consolidate` endpoint; we won't duplicate.)
- Embedding visualization (UMAP / t-SNE). Possible later, deliberately skipped
  for v1 — it doesn't answer any of the five questions above.

## 3. User scenarios

The design must make these flows fast.

**S1. Post-session sanity check.**
*Sara just ran a 30-minute OpenClaw session that auto-remembered things via
plugin hooks. She opens the Inspector, picks her graph from a list, sees the
"recently added" facts at the top of Browse, skims tags. Done in 30 seconds.*

**S2. Recall debugging.**
*A `plugmem.recall` call returned irrelevant results. Sara opens Recall Trace,
pastes the same observation/goal, and sees: which mode was chosen, which tags
were picked by the planner, which candidate semantic nodes were considered,
their relevance / recency / importance / credibility scores, and which made
the final cut. The smoking gun is usually visible — wrong mode, missing tags,
or a low-credibility node winning over a better one.*

**S3. Bad-fact deactivation.**
*Sara spots a wrong fact in Browse. She clicks "deactivate." The semantic node
is marked `is_active=false`; future recalls skip it.*

**S4. Structure viewing / demo.**
*Sara wants a screenshot for a slide. She opens the Graph tab, picks a node
type filter (e.g. "semantic + tag"), exports PNG.*

## 4. Architecture

```
┌──────────────────────────────────────────┐     ┌─────────────────────┐
│  Browser (Inspector frontend)            │     │  PlugMem service    │
│                                          │     │  (FastAPI)          │
│  index.html                              │     │                     │
│  ├─ tabs: Browse | Recall | Graph        │ ──► │  /api/v1/graphs/... │
│  ├─ vanilla JS (no build step)           │     │  /inspector/...     │
│  └─ themes/{default,pixel,...}.css       │     │  (new viz routes,   │
│                                          │     │   serves static     │
│                                          │     │   assets too)       │
└──────────────────────────────────────────┘     └─────────────────────┘
                                                            │
                                                            ▼
                                              ┌─────────────────────────┐
                                              │  GraphManager           │
                                              │  ChromaStorage          │
                                              │  ChromaDB (persistent)  │
                                              └─────────────────────────┘
```

**Backend.** Extend the existing FastAPI app. No new service. New routes live
under `plugmem/api/routes/inspector.py`. Static assets are mounted via
`StaticFiles` at `/inspector/`. DI for `GraphManager` is already wired.

**Frontend.** A single-page app served from
`plugmem/api/static/inspector/index.html`. Three tabs, vanilla JS, no
bundler. Graph rendering: **cytoscape.js** via CDN — chosen over vis-network
because its styling system is JSON-based and trivially themable, which
matches the swappable-theme requirement. Theme switching via a CSS file
loaded from `themes/{name}.css` and a cytoscape style JSON
`themes/{name}.cy.json`.

**Auth.** Reuses `require_api_key`. For local-only use the API key can be
unset (existing behavior — auth is a no-op when `api_key` is empty). The
frontend reads the key from `localStorage` if present and adds the
`X-API-Key` header.

## 5. Data API

Where possible, reuse existing endpoints. Net-new ones below are minimal.

### Reused (already exist)

- `GET  /api/v1/graphs` — list graphs
- `GET  /api/v1/graphs/{id}/stats` — node counts per type
- `GET  /api/v1/graphs/{id}/nodes?node_type=&limit=&offset=` — paginated
  table data; already returns the right fields per type

### New — Browse / search

- `GET  /api/v1/graphs/{id}/search`
  - Query: `q` (string), `node_type` (default `semantic`),
    `limit` (default 50)
  - Behavior: substring match on text fields (semantic memory, tag, subgoal,
    procedural memory) — *not* embedding similarity. Cheap and predictable;
    that's what users expect from a search box. Embedding-based search lives
    in Recall Trace.

- `GET  /api/v1/graphs/{id}/node/{node_type}/{node_id}`
  - Returns one node with its full edges expanded one hop:
    semantic → tags + episodic + bro/son semantics; tag → semantics;
    subgoal → procedurals; procedural → subgoal + episodics; episodic →
    semantics that referenced it.

### New — Recall trace

- `POST /api/v1/graphs/{id}/recall_trace`
  - Body: `{ goal, subgoal, state, observation, time, task_type, mode? }`
  - Returns the same shape as `/retrieve` plus a **trace** object describing
    the intermediate retrieval state. This is the bit that needs work in
    `memory_graph.py` — see §7.

  Response shape:
  ```jsonc
  {
    "mode": "semantic_memory",
    "plan": { "next_subgoal": "...", "query_tags": ["..."] },
    "trace": {
      "tag_candidates": [
        { "tag": "...", "tag_id": 7, "relevance": 0.83,
          "value": 0.71, "selected": true }
      ],
      "semantic_topk_by_similarity": [
        { "semantic_id": 42, "text": "...", "similarity": 0.79 }
      ],
      "semantic_candidates": [
        { "semantic_id": 42, "text": "...", "tags": ["..."],
          "relevance": 0.79, "recency": 12, "importance": 4.5,
          "credibility": 10, "value": 0.62, "selected": true,
          "tag_votes": 3 }
      ],
      "procedural_candidates": [/* analogous */]
    },
    "selected": { "semantic_ids": [...], "procedural_ids": [...] },
    "rendered_prompt": [/* chat messages */]
  }
  ```

### New — Graph view

- `GET  /api/v1/graphs/{id}/topology`
  - Query: `include_episodic` (bool, default false),
    `node_limit` (default 500), `tag_min_importance` (default 0)
  - Returns nodes + edges in a cytoscape-friendly shape:
    ```jsonc
    {
      "nodes": [
        { "data": { "id": "sem-3", "type": "semantic",
                    "label": "...", "is_active": true,
                    "credibility": 10 } }
      ],
      "edges": [
        { "data": { "source": "sem-3", "target": "tag-7", "kind": "tagged" } }
      ]
    }
    ```

### New — Mutations (single-node, conservative)

- `PATCH /api/v1/graphs/{id}/semantic/{semantic_id}`
  - Body: `{ "is_active": false }` — only field accepted in v1.
  - Reuses `ChromaStorage.update_semantic` with `metadata_updates`.

That's the entire API surface for v1. Notice we do not need a "delete node"
or "edit text" endpoint — deactivation is sufficient and reversible.

## 6. Frontend layout

### Pages / tabs

`/inspector/?graph=<id>&tab=<browse|recall|graph>`

Top bar: graph picker (dropdown of `list_graphs()`), stats summary
(N semantic / N procedural / N tag / N subgoal / N episodic), theme picker.

**Browse tab**
- Type selector chips: semantic | procedural | tag | subgoal | episodic.
- Search box (substring) — hits `/search`.
- Sort: `time desc` (default), `time asc`, `credibility`.
- Filter: `is_active` (semantic only), `min credibility`.
- Table view with virtualized scroll for large graphs (simple windowing —
  render only visible rows).
- Click a row → side panel with full text + edges + "Deactivate" button
  (semantic only).

**Recall tab**
- Form: goal, subgoal, state, observation, time, task_type, mode override.
  All optional except observation.
- "Run trace" button → POST `/recall_trace`.
- Results panel: chosen mode + plan, then a sorted candidate table per node
  type (relevance, recency, importance, credibility, value, ✓/✗), and the
  rendered prompt as a collapsible block.

**Graph tab**
- Cytoscape canvas. Toggle node types, toggle edge kinds. Click a node to
  open the same side panel as Browse.
- Layout: `cose-bilkent` for ≤500 nodes, `concentric` fallback for larger.
- Export PNG button (cytoscape built-in).

### Theming

`themes/default.css` — typography, spacing, semantic color tokens
(`--node-semantic`, `--node-tag`, `--node-procedural`, `--node-subgoal`,
`--node-episodic`, `--bg`, `--fg`, `--accent`, `--border`).
`themes/default.cy.json` — cytoscape stylesheet matching the same tokens.

To add a theme (e.g. pixel-art) later:
- drop `themes/pixel.css` overriding the same tokens + adding any
  pixel-specific classes;
- drop `themes/pixel.cy.json` for cytoscape styles;
- add `pixel` to the theme picker.

No frontend code changes. That is the goal.

### File layout

```
plugmem/api/static/inspector/
  index.html
  app.js                  # tab routing, state, fetches
  browse.js               # Browse tab
  recall.js               # Recall tab
  graph.js                # Graph tab (cytoscape)
  api.js                  # tiny fetch wrapper, X-API-Key plumbing
  themes/
    default.css
    default.cy.json
docs/
  memory_inspector_design.md   # this file
```

## 7. Server-side work for Recall Trace

The trickiest piece. Today `MemoryGraph.retrieve_semantic_nodes` returns the
final node list and discards intermediates. The Recall Trace needs:

- the planner output (`get_plan` returns `(next_subgoal, query_tags)`);
- the retrieval mode (`get_mode`);
- per-tag-candidate `relevance` + final `value`;
- top-K by raw cosine similarity *before* tag voting;
- per-semantic-candidate components (relevance, recency, importance,
  credibility, tag_votes, final value, selected/not).

Two ways to surface this:

1. **Refactor `retrieve_semantic_nodes`** to optionally return a trace dict
   alongside the node list. Backward compatible if we make the trace an
   opt-in second return.
2. **Parallel `retrieve_semantic_nodes_with_trace`** that duplicates the
   logic. Worse — drift hazard.

Plan: option 1. Add an internal `_trace: Optional[dict] = None` parameter
that, when a dict is passed, gets populated as the function runs. Same for
`retrieve_procedural_nodes`. The API route owns the dict and returns it.
Production callers pass nothing; cost is one `if _trace is not None:` check
per phase.

## 8. Phasing

### Phase 1 — Browse + scaffolding *(highest utility / smallest risk)*
- `inspector.py` route module with `/search`, `/node/{type}/{id}`, and
  static-files mount.
- `app.js` with tab routing + graph picker + stats bar.
- `browse.js`: type chips, search, table, side panel, deactivation.
- `themes/default.{css,cy.json}`.
- Backend `PATCH .../semantic/{id}` for deactivation.

Definition of done: a user with a populated graph can find any node by text
and deactivate a semantic fact, with the change persisting in ChromaDB.

### Phase 2 — Recall Trace
- Refactor `retrieve_semantic_nodes` / `retrieve_procedural_nodes` to fill an
  optional trace dict.
- `POST /recall_trace` route.
- `recall.js`: form, candidate tables, prompt viewer.

Definition of done: paste an observation, see why each candidate did or
didn't make the final cut, including its score breakdown.

### Phase 3 — Graph view
- `GET /topology` endpoint with sensible default limits.
- `graph.js`: cytoscape integration, node/edge filters, side panel reuse,
  PNG export.

### Phase 4 — Theme system polish + first alternate theme
- Confirm token coverage by building a deliberately-different theme
  (e.g. high-contrast or pixel-art).
- Document the token contract in `docs/memory_inspector_theming.md`.

### Phase 5 (later, optional) — Session timeline
- Filter nodes by `session_id`, sort by time, render a vertical timeline.

## 9. Open questions

1. **Auth for static assets.** `require_api_key` currently guards every
   `/api/v1` route. The static asset mount at `/inspector/` will be
   unguarded — anyone reaching the host can load the SPA. The data behind it
   still needs the key. Acceptable? (For local-only use, yes.)
2. **Search scope.** Should `/search` cross node types in one call, or stay
   per-type as in the existing `/nodes` endpoint? Leaning per-type for v1.
3. **Recall trace and the LLM.** `get_plan` and `get_mode` make LLM calls.
   The trace endpoint will too. If the user is just exploring, that's fine,
   but we should clearly mark this as a paid call in the UI and offer a
   "skip plan, supply tags manually" toggle for fast iteration.
4. **Graph topology size.** What does cytoscape do with 5k nodes? Need to
   measure before promising the >10k goal. Likely fine for ≤2k; fallback to
   "browse only, graph view disabled" above a threshold.
5. **Episodic tab utility.** Episodic nodes are numerous and noisy. Worth
   a Browse chip? Probably yes, but hidden behind an "advanced" toggle.

## 10. Risks

- **Trace refactor regresses retrieval.** Mitigation: keep `_trace` strictly
  opt-in, never branch logic on it, add a unit test that asserts the
  no-trace return value is unchanged for a fixed graph.
- **Theme contract leaks implementation details.** Mitigation: define tokens
  up front and *only* use tokens in the default theme — no raw colors in
  base CSS.
- **Backend coupling.** Putting the inspector in the production FastAPI app
  means a frontend bug could in principle DoS the service. Mitigation: the
  read paths are already exposed; the only new write is single-field
  semantic deactivation.
