# PlugMem for Coding Agents — Design & Plan

Status: stages 0–5 landed; stages 6–7 pending
Owner: jizej
Last updated: 2026-05-03

## Goal

Adapt PlugMem to serve as **cross-session** long-term memory for coding agents
— effectively a *self-writing CLAUDE.md*. The agent learns project-specific
facts, debugging recipes, and personal preferences from its own traces, and
recalls them on subsequent sessions without a human curating a markdown file.

**Primary harness targets:** Claude Code and OpenCode. Both have rich,
documented hook surfaces (session lifecycle, per-tool, user-prompt) that
make the promotion-gate design feasible.

**Extended target:** OpenClaw. Limited hook surface
(`before_reset` / `before_compaction` only) means a degraded mode — no
in-session failure-delta detection, promotion runs only at session
boundaries. Ships behind the same core, via a thin adapter, but is not the
shape we optimize for.

## Non-goals

- **In-session compaction.** Coding harnesses (Claude Code, the OpenClaw
  coding harness) already do this; competing there is a worse problem against
  a stronger baseline.
- **Conversational chat memory.** Already covered by the existing OpenClaw
  plugin (`openclaw-plugmem-plugin/`).
- **Replacing CLAUDE.md as a contract surface.** Engineers should still be
  able to write authoritative rules in CLAUDE.md; PlugMem augments, not
  overrides. On conflict, CLAUDE.md wins.

## Framing

The conversational PlugMem story is "compress history, beat raw-history
baselines." That doesn't translate to coding because the baseline isn't raw
history — it's CLAUDE.md + filesystem + grep, which is already a compact
recall mechanism. The credible coding story is **learned memory that a
static CLAUDE.md can't capture**:

- Corrections the user gave but never bothered to write down.
- Debugging recipes for flaky tests / environment quirks.
- Evolving conventions (the codebase changed; CLAUDE.md drifted).
- Per-package gotchas surfaced only on contact.

The eval target is therefore: *after N sessions on a repo, the
auto-populated graph matches or beats a hand-written CLAUDE.md, with no
human curation.*

## Architecture

### Graph topology

Reuses existing primitives — no new infrastructure.

Graph IDs are **isolated per harness**. Memory does not flow between
Claude Code and OpenCode (and OpenClaw) — even for the same user on the
same repo. Rationale: extracted memories embed harness-specific quirks
(tool names, hook payload shapes, system-prompt conventions) that don't
generalize cleanly. Consolidating later is cheap if we change our mind;
un-mixing a polluted graph is not.

- **Per-repo graph (write target):**
  `defaultGraphId = repo://<harness>/<host>/<owner>/<repo>`. All
  auto-promoted writes go here. Captures repo-specific facts, conventions,
  debugging recipes — for one harness only.
- **Per-user graph (shared-read fan-in):**
  `sharedReadGraphIds = ["user://<harness>/<user-id>"]`. Captures personal
  idioms ("terse, no emojis"), preferred libraries/styles. Read-only on
  recall; written explicitly when the agent extracts a user-level rule.
- Identifying the repo: `git remote get-url origin` at session start,
  normalized (strip protocol/credentials, lowercase host). Forks share the
  upstream graph in read-only mode (open question — see below).
- The `<harness>` segment is set by the adapter and is not user-configurable
  at the core layer — preventing accidental cross-harness writes.

### Memory-type mapping

Deliberate inversion of conversational defaults: episodic dominates chat,
but is mostly noise in code.

| Type       | Coding content                       | Examples                                                                                          | Notes                                                          |
|------------|--------------------------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| Semantic   | Facts, rules, locations              | "Auth tests live in `auth/tests/`, run with `pytest -k auth`"; "Repo standard is `httpx`, not `requests`" | Highest signal; cheapest; most CLAUDE.md-like.                 |
| Procedural | Multi-step workflows that worked     | Debug recipe for flaky CI; release-cut steps; how to regenerate fixtures                          | Stored only after observed success.                            |
| Episodic   | Full session traces                  | Past session transcripts                                                                          | Lowest value here. Default OFF; retain only when explicitly tagged. |

### Promotion gate (the hard part)

Coding produces ~100× more low-signal events than chat, and bad memories
*actively harm* (a stale "always use X" leads to wrong code). Default to
*not* writing; require an explicit signal.

Promotion signals (any one is sufficient):

1. **Failure → success delta.** A tool call (test, build, command) failed,
   then a follow-up succeeded after agent action. The diff between attempts
   is a procedural-memory candidate. Skip if the fix was trivial (typo,
   syntax).
2. **Explicit user correction.** User message contains negation/correction
   patterns ("don't", "stop", "actually", "no, the right way is..."). Extract
   the rule as semantic memory. Use the existing memory-system feedback-rule
   structure (rule + `Why:` + `How to apply:`).
3. **Post-merge confirmation.** PR merged or user says "ship it" / "looks
   good" / "lgtm" — promote session decisions to semantic.
4. **Repeated lookup.** Same symbol grepped/read ≥ 3 times across sessions
   on the same repo. Pin location as semantic.
5. **Explicit `remember` tool call.** Agent or user invokes the tool
   directly. Trust, but tag as user-asserted with high confidence.

Anti-signals (never auto-promote):

- Bare tool-call logs.
- Successful commands with no preceding failure.
- Code the agent just wrote (it's already in git; let `git log` be the
  history).

The promotion step itself is an LLM call (cheap model) that reads a
candidate window and emits 0–N memory nodes. Default emission: 0.

### Recall trigger policy

Don't recall every turn. Three triggers:

1. **Session start.** Top-K from per-repo graph, filtered to
   high-confidence semantic memories. Inject into the system prompt as a
   "learned conventions" block. **Hard cap: ~500 exposed tokens.**
2. **Sub-task boundary.** When the user gives a new instruction (or the
   agent declares a new goal), recall against the task description.
3. **Tool-family pre-hook.** Before `pytest` / `npm test`, recall
   test-related memories. Before `git push`, recall release/CI memories.
   Optional, behind a config flag.

No history-keyed every-turn fan-out. Exposed-token budget is the
constraint.

## Hook surface

The plugin core depends on **abstract events**, not harness-native names.
A thin adapter per harness translates the harness's native event payloads
to this abstract interface. Core logic (promotion gate, recall policy,
graph-id derivation) is harness-agnostic.

### Abstract events

| Abstract event   | Used for                                                                  |
|------------------|---------------------------------------------------------------------------|
| `session_start`  | Identify repo; load recall block; inject into system prompt.              |
| `user_prompt`    | Detect correction patterns; sub-task boundary recall.                     |
| `pre_tool`       | Capture tool inputs for failure-delta detection; tool-family pre-recall.  |
| `post_tool`      | Detect failure→success transitions; trigger promotion-gate evaluation.   |
| `pre_compact`    | Promotion sweep before context compaction.                                |
| `session_end`    | Final promotion sweep; persist deferred candidates.                       |

### Per-harness mapping (verified Stage 0, 2026-05-01)

| Abstract event | Claude Code        | OpenCode                                      | OpenClaw (extended)  |
|----------------|--------------------|-----------------------------------------------|----------------------|
| `session_start`| `SessionStart`     | `event` filtered to `session.created`         | not exposed; emulate from first prompt |
| `user_prompt`  | `UserPromptSubmit` | `chat.message`                                | not exposed          |
| `pre_tool`     | `PreToolUse`       | `tool.execute.before`                         | not exposed          |
| `post_tool`    | `PostToolUse`      | `tool.execute.after`                          | not exposed          |
| `pre_compact`  | `PreCompact`       | `experimental.session.compacting`             | `before_compaction`  |
| `session_end`  | `Stop` / `SessionEnd` | `event` filtered to `session.idle`/`session.deleted` | `before_reset` |

**Context injection per harness:**
- Claude Code: `hookSpecificOutput.additionalContext` on stdout from
  `SessionStart` / `UserPromptSubmit`.
- OpenCode: push to `output.system: string[]` in
  `experimental.chat.system.transform`, or rewrite parts in `chat.message`.
- OpenClaw: no equivalent — recall block degrades to a no-op (or a
  `plugmem.recall` tool call surfaced to the agent).

**Transcript fallback paths** (when hook payloads are incomplete):
- Claude Code: `~/.claude/projects/<project>/<session>.jsonl` — every hook
  receives `transcript_path` pointing here.
- OpenCode: `~/.local/share/opencode/storage/session/...` and
  `message/<sessionID>/<messageID>.json`.

OpenClaw's adapter degrades to "promotion at session boundaries only" —
no failure-delta or correction detection. Acceptable for the extended
tier; not the shape we optimize against.

**Open verification item:** the Stage 0 audit reported that Claude Code's
`PostToolUse` fires only on success, with a separate `PostToolUseFailure`
event. This isn't matched by the current public hook docs as far as I've
confirmed and may be a confabulation. Treat success-vs-failure as a
heuristic on `tool_result` content / exit codes (per the
`outcome: "unknown"` allowance in the adapter interface) until verified
in implementation.

### Adapter interface

Each adapter:
1. Subscribes to its harness's native events (settings.json hooks for
   Claude Code, plugin/hook system for OpenCode, the existing `api.on(...)`
   for OpenClaw).
2. Normalizes payloads (message log, tool name, args, result, exit code)
   into the abstract event shape.
3. Calls into the shared core to run promotion-gate / recall logic.
4. Returns harness-shaped responses (Claude Code: JSON on stdout to
   inject context; OpenCode/OpenClaw: equivalent).

### Packages

- `plugmem-coding-core/` — harness-agnostic logic (promotion gate, recall
  policy, graph-id derivation, prompt assembly). HTTP client reused from
  the existing plugin.
- `plugmem-coding-claude-code/` — Claude Code adapter (hook scripts +
  plugin manifest).
- `plugmem-coding-opencode/` — OpenCode adapter.
- `plugmem-coding-openclaw/` — OpenClaw adapter (extended; reuses
  `client.ts` / `config.ts` from `openclaw-plugmem-plugin/`).

## Server-side changes

Most work is plugin-side. Server changes are minimal:

1. **Memory metadata for confidence/source.** Add `source`
   (`failure_delta` | `correction` | `merged` | `repeated_lookup` |
   `explicit`) and `confidence` (0–1) on insert. Used by recall to filter
   the session-start injection.
2. **Per-graph eviction policy.** Coding graphs need eviction (stale
   facts). Add a TTL/LRU policy at the storage layer, gated per-graph.
3. **Recall filter args.** Allow filtering by `source` / `confidence` /
   `type`. Verify how much is already supported in
   `plugmem/api/routes/retrieval.py` before extending.

No new memory types, no new storage backends.

## Evaluation

Per the saved benchmark feedback: token reporting **always splits exposed
(agent-facing) from internal (memory-system)** and never combines them.

### Baselines

1. **Empty.** No CLAUDE.md, no PlugMem. Cold start every session.
2. **Hand-written CLAUDE.md (strong baseline).** Engineer-curated for the
   target repo. This is the bar to clear.
3. **PlugMem learned (this work).** Cold start at session 1; graph
   populated via auto-promotion across N sessions.

### Metrics

- **Task success rate** on a held-out set of repo-specific tasks.
- **Exposed tokens to completion** (median, p90).
- **Internal tokens per session** (promotion-gate + recall LLM calls).
- **Redundant-exploration count:** number of grep/read calls for symbols
  already pinned in memory. Trends down → recall is working.
- **Repeated-correction count:** how often the user gives the same
  correction across sessions. Cleanest learning signal — should trend to 0.

### Repos

2–3 repos with distinct flavors:

- A medium Python service (well-defined test layout, learnable conventions).
- A polyglot project (cross-language gotchas).
- This repo (PlugMem itself) — meta-eval.

Each repo: 20–40 task instances, run sequentially so the graph accumulates.

### The crossover question

The interesting result isn't "PlugMem beats empty" (obvious). It's:
**at what session count does PlugMem-learned match hand-written CLAUDE.md?**
If never, the learning loop is broken. If at session 5, the value-prop is
strong.

## Implementation plan

Sequenced for progressive de-risking: ship on the most stable harness
first, validate the abstraction by porting, eval before optimization.

### Stage 0 — Adapter interface + harness-event audit ✅ (done 2026-05-01)

- Adapter interface drafted in `design_docs/adapter_interface.ts`
  (`SessionStart` / `UserPrompt` / `PreTool` / `PostTool` / `PreCompact` /
  `SessionEnd` events, `ContextInjection` return, `SessionState` for
  cross-event state).
- Per-harness mapping locked (see Hook surface section).
- Server-side gap surfaced — see Stage 2.
- Outstanding: `PostToolUse` failure-signal claim from the audit needs
  confirmation in implementation; design treats it as heuristic-only.

### Stage 1 — Core + Claude Code adapter scaffold (2–3 days)

- New packages `plugmem-coding-core/` and `plugmem-coding-claude-code/`.
- Repo-id derivation from `git remote get-url origin` (normalized).
- `session_start` → load recall block → inject into system prompt via
  Claude Code's hook-stdout context-injection convention.
- `session_end` / `pre_compact` → naïve full-trajectory remember (parity
  with existing OpenClaw plugin; not yet promotion-gated).
- Verify end-to-end loop on Claude Code: a fact stated in session 1
  surfaces in session 2.

### Stage 2 — Server-side metadata API ✅ (done 2026-05-01)

- `SemanticMemoryInput` / `ProceduralMemoryInput` extended with `source`
  (Literal: `failure_delta` | `correction` | `merged` | `repeated_lookup` |
  `explicit`) and `confidence` (float 0–1, default 0.5).
- Metadata threaded through `SemanticNode` / `ProceduralNode`,
  `MemoryGraph.insert`, chroma `add_semantic` / `add_procedural`, and
  `_load_semantic_nodes` / `_load_procedural_nodes` (round-trip on
  reload).
- `min_confidence` / `source_in` added to `RetrieveRequest` /
  `ReasonRequest` and threaded through `MemoryGraph.retrieve_memory` →
  `retrieve_semantic_nodes` / `retrieve_procedural_nodes`.
- **Filter applies at candidate-collection time**, not post-ranking.
  Earlier post-ranking attempt was wrong: it dropped high-confidence
  nodes that ranked outside the value-function top-K.
- Backwards-compat verified: existing callers with no metadata fields
  keep working unchanged. 9/9 prior memory + retrieval tests pass plus
  13 new tests (insert validation, round-trip, filter unit tests).

### Stage 3 — Promotion gate ✅ (done 2026-05-03)

Scope landed:
- **Failure-delta detector** (`recordPreTool` / `recordPostTool` in
  `plugmem-coding-core/src/promotion.ts`) pairs a `failure` outcome with
  a later `success` on the same tool name, within a 10-minute window,
  capped at 20 recent failures. Outcome `"unknown"` is conservatively
  ignored — no candidate emitted.
- **Correction-pattern detector** (`recordUserPrompt`) regex-matches
  negation/correction phrases (`don't`, `stop`, `actually`, `instead of`,
  `we use`, `the right way`, `should be|use|not`, `not X but Y`).
- **Disk-backed `SessionState`** (`plugmem-coding-claude-code/src/state.ts`)
  — required because CC hooks run as isolated processes; pre/post tool
  can't share an in-memory Map. Files at
  `~/.cache/plugmem/sessions/<sessionId>/<key>.json`.
- **LLM extractor** (`plugmem/inference/promotion.py`,
  `POST /api/v1/extract`) — receives candidates, emits structured
  semantic/procedural memories with `source` + `confidence`. Adapter
  posts the result to `/memories` itself; the server endpoint is pure
  extraction, no graph routing.
- **Promotion-gate runner** in core's `runPromotionGate`: drains
  candidates at `session_end` / `pre_compact`, calls `/extract`, inserts
  via `/memories` structured mode. **Replaces the Stage 1 naïve
  trajectory ingest entirely** — when there are no promotion candidates
  or the extractor returns `[]`, nothing is written.
- **CC hook registration** widened to include `UserPromptSubmit`,
  `PreToolUse`, `PostToolUse`.

Not landed (deferred):
- Post-merge confirmation (signal 3) — needs a Git status hook or
  user-language detection for "ship it" / "lgtm".
- Repeated lookup (signal 4) — needs cross-session counting; probably a
  background-task in the server.
- Explicit `remember` tool call (signal 5) — that's a tool surface, not
  a hook signal. Separate work.

Tests: 31 core unit tests (9 promotion-gate detectors + 7 core wiring +
15 prior) + 41 server tests pass. The `is_error` field on tool_result
content blocks is the actual Claude Code failure signal — confirmed
during build, replacing the audit's confabulated `PostToolUseFailure`
event.

### Stage 4 — Recall policy ✅ (done 2026-05-03)

Scope landed:
- **Session-start recall** moved from `/reason` (LLM synthesis) to
  `/retrieve` (vector retrieval only). Saves an LLM call on every
  session start; raw fact strings pack denser into the token budget than
  paraphrased prose. Format: `<plugmem-recall trigger="session-start"
  graph="...">…</plugmem-recall>`.
- **Sub-task recall** on `user_prompt` events — for prompts ≥ 30 chars
  by default, calls `/retrieve` against the prompt text and injects a
  `trigger="user-prompt"` block. Skipped for short ack-style prompts
  ("yes", "ok") which would burn cost without value.
- **Tool-family pre-recall** behind a config flag, default OFF. Caller
  provides a `mapping: Record<regex, query>` (e.g.,
  `{ "(?i)pytest|vitest": "test setup, fixtures" }`) — first match wins.
  Patterns support a leading `(?i)` / `(?m)` flag prefix (POSIX/Python
  convention) which is parsed out before constructing the JS RegExp.
- **Structured recall config** in `CreateCoreOptions` —
  `sessionStartRecall`, `userPromptRecall`, `toolFamilyRecall`. Each
  takes `enabled` / `minConfidence` / `sourceIn` / `charCap`, plus
  trigger-specific fields. Pass `false` to disable a trigger entirely.
- **`onPreTool` signature widened** to return `Promise<ContextInjection
  | null>` so adapters can wire tool-family recall into harness-native
  context-injection (CC: `hookSpecificOutput.additionalContext` from
  `PreToolUse`). OpenCode and OpenClaw adapters return null and discard
  for now; future adapters can plumb through if their harness supports it.

Tests: 11 new recall tests + 7 reworked core tests + 24 unchanged =
42 core tests pass; server tests unchanged.

### Stage 5 — Eval harness ✅ (skeleton done 2026-05-03)

Scope landed:
- **Phase-tagged token logging** (`plugmem/clients/llm.py` —
  `with_phase` context manager + `current_phase()` accessor). Each
  `_log_usage` JSONL record now carries a `phase` field. Wrapped call
  sites: `extract` (promotion-gate), `retrieve` (recall path),
  `reason` (final synthesis). Backwards-compat: untagged calls log
  `phase: "default"`.
- **Fixture format + types** in `plugmem-coding-core/src/eval/types.ts`
  — JSON schema for an event sequence (`session_start`, `user_prompt`,
  `pre_tool`, `post_tool`, `pre_compact`, `session_end`) plus per-trigger
  recall config overrides.
- **Session runner** in `src/eval/runner.ts` — walks events through
  *real* `createCore` against a live PlugMem service. In-memory
  `SessionState` (the harness is one long-lived process, no need for the
  CC-style disk backing). Truncates the fixture's graph before each run
  unless `--no-reset` is passed.
- **Metrics aggregator** in `src/eval/metrics.ts` —
  `summarizeTokenLog(path)` reads the server's `token_usage_file` JSONL
  and aggregates by `phase`. Produces `byPhase` totals + grand total.
- **CLI** in `src/eval/cli.ts` — `npm run eval -- run --fixture ...` and
  `... summarize --token-log ...`. Help on `--help`.
- **Sample fixture + README** in `eval/fixtures/smoke.json` and
  `eval/README.md`. Three-session smoke test exercising correction +
  failure-delta paths.

Scope cuts (deliberate):
- **No agent integration.** The runner fires hand-authored events, not
  events recorded from a real agent run. The "PlugMem vs hand-written
  CLAUDE.md" crossover gate from the original Stage 5 spec needs real
  Claude Code sessions producing event streams — that's the next layer
  on top of this harness, not Stage 5 itself.
- **No baseline-comparison automation.** Run the fixture once per
  baseline (empty / hand-written CLAUDE.md / PlugMem) and diff the
  reports manually. A multi-run aggregator is a useful addition once we
  have agent-driven event streams.
- **No success-rate measurement.** Task-completion correctness needs
  domain-specific assertions outside the scope of a fixture harness.
- **No plotting.** Reports are JSON; bring your own matplotlib.

What this stage does deliver: a repeatable, deterministic way to
exercise the entire memory loop (detectors → promotion → recall) end
to end, isolate per-phase token costs, and verify expectations about
graph state across sessions. Tests: 5 new eval tests + 47 prior core
tests + 4 phase-tagging tests pass.

The crossover gate is **not** evaluated by this harness — it requires
real agent runs. When that next layer is built, it should produce
fixtures (or fixture-equivalent event streams) that this harness can
replay against three different memory configurations.

### Stage 6 — OpenCode adapter (2–3 days)

- New package `plugmem-coding-opencode/` against the same core. Native TS
  plugin (not MCP); loaded via `.opencode/plugin/` or as an npm package
  referenced in `opencode.json`.
- Re-run the eval harness on OpenCode to confirm cross-harness parity.
- Surfaces any leakage of harness-specific assumptions in the core; fix
  there, not in the adapter.

### Stage 7 — Extended: OpenClaw adapter + tune (2–3 days)

- New package `plugmem-coding-openclaw/` against the same core. Degraded
  mode: promotion fires only at `before_reset` / `before_compaction`.
- Adjust promotion-gate prompts based on what's missed / what's noise
  across all three harnesses.
- Eviction policy if graphs grow unbounded.
- Crossover analysis: report session count at which PlugMem matches
  hand-written CLAUDE.md, per harness.

## Open questions

1. **User-level vs repo-level rules.** When the agent extracts a
   correction, how does it decide which graph to write to? LLM classifier
   eventually, but worth a heuristic first (rule references a file/symbol →
   repo; references "I"/"me"/personal preference → user).
2. **Graph-id collisions across forks/clones.** `git remote get-url
   origin` differs for forks. Need a normalization step (or explicit
   override) so a fork can share the upstream graph in read-only mode
   without the user reconfiguring.
3. **Retraction.** When code changes invalidate a memory ("auth lives in
   `auth/`" but the dir was just renamed), how do we evict? Stage 5
   question; not blocking earlier stages.
4. **Privacy.** Per-user shared graph crosses repo boundaries. If a memory
   captures content from a private repo and surfaces in a public one — leak.
   Probably needs per-graph visibility tags before this leaves single-user
   use.
5. **Conflict with CLAUDE.md.** When a learned memory contradicts a
   CLAUDE.md rule, the design says CLAUDE.md wins — but does the agent
   surface the conflict to the user, or silently drop the memory?
6. **Cross-harness consolidation (deferred).** Current design isolates
   per-harness — graph IDs include a `<harness>` segment, so a Claude
   Code session and an OpenCode session on the same repo populate
   separate graphs. Revisit only if (a) eval shows a clear win from
   sharing and (b) we have a normalization pass that scrubs
   harness-specific tool names / payload shapes from extracted memories
   before write. Until both, keep them separate.

## Implementation notes for future devs

These are non-obvious choices and gotchas that didn't fit cleanly into
the stage descriptions above. Read this section before changing the
relevant subsystem — most entries encode a decision someone already had
to walk back.

### Hook & adapter contract

- **CC hooks always `exit 0`.** Even on internal errors. A non-zero
  exit can block the user's tool call or session start, which is worse
  than missing a memory write. Errors go to stderr (gated by
  `PLUGMEM_DEBUG=1`); the user never sees them.
- **`Stop` is NOT registered as a session-end hook.** It fires every
  turn in Claude Code. Registering it would dump a trajectory per turn
  — registering `SessionEnd` (and `PreCompact` for long sessions) is
  intentional. The dispatcher *handles* `Stop` if it ever arrives, but
  `hooks.json` doesn't subscribe.
- **`outcome="unknown"` is the conservative default for tool results.**
  When the adapter can't decide success-vs-failure from `tool_result`
  shape, it sets `unknown` and the promotion gate ignores the event.
  Better to miss a candidate than to mis-pair one.
- **Single dispatcher binary.** `dist/bin/cc-hook.js` reads
  `hook_event_name` from stdin and switches internally. Adding a new
  abstract event = one `case` in the switch + a `hooks.json` entry; no
  new bin script.
- **`<harness>` segment in graph-ids is set by the adapter, not
  config.** The user can't accidentally point Claude Code at an
  OpenCode graph. Per the no-cross-harness-sharing decision, this is
  a hard invariant.

### Session state (Claude Code)

- **CC hooks are isolated processes** — each event spawns a new Node
  process. In-memory state is gone between `PreToolUse` and
  `PostToolUse`. This is *the* reason `SessionState` is disk-backed.
- **One file per key.** `~/.cache/plugmem/sessions/<sid>/<key>.json`,
  not one combined JSON. Two hooks firing concurrently (e.g., user
  prompt + post-tool) won't clobber each other.
- **Cleanup is best-effort.** `DiskSessionState.clear()` exists but
  isn't called yet — stale session dirs accumulate. A periodic prune
  (or session_end → clear) is a Stage 5 cleanup item.
- **`PLUGMEM_STATE_DIR` overrides the root** for testing/sandboxing.
  Useful when running multiple PlugMem instances on the same machine.

### Transcript parsing (Claude Code JSONL)

- **`thinking` content blocks are skipped.** They're verbose, often
  multi-paragraph, and don't generalize. The promotion gate can re-read
  the raw transcript file if it ever needs them.
- **Goal preference order:** `ai-title` line (CC's auto-generated
  session title) → first user message truncated to 200 chars →
  `"(no goal)"`. Don't infer goals from later turns; they're usually
  more about the current sub-task than the session.
- **Same-role messages collapse before pairing.** Multiple consecutive
  `user`-role lines (e.g., a real prompt followed by a tool_result
  message that CC also tags as `user`) get merged into one observation.
  Same for assistant turns spanning multiple JSONL records.
- **`is_error: true` on `tool_result` blocks is the actual failure
  signal** — not a separate `PostToolUseFailure` event (the Stage 0
  audit was wrong about that). The transcript parser already emits
  `[tool_result:error]` markers; Stage 3's failure-delta detector relies
  on `outcome` from the live hook payload, but post-mortem analysis
  should read `is_error` from the JSONL.

### Promotion gate

- **Runs at both `PreCompact` AND `SessionEnd`.** Long sessions with
  multiple compactions trigger the gate multiple times. Each run drains
  `candidates` fully (no double-promotion), but the LLM cost scales
  with compaction frequency. If this shows up as expensive in eval, add
  a debounce.
- **No fallback to naïve trajectory ingest.** Stage 1 used to dump the
  whole session as a trajectory if no promotion fired — Stage 3 removes
  that path entirely. Empty-candidate session = zero writes. This is
  deliberate: low-signal sessions shouldn't pollute the graph.
- **Failure-pairing window is 10 minutes, capped at 20 recent
  failures.** Both bounds are heuristic; tune in Stage 5. Larger window
  = more pairings but more false-positives (an unrelated success
  long after a failure shouldn't be a "fix").
- **`/extract` does not write to a graph.** The endpoint is pure
  extraction: candidates in, structured memories out. The adapter owns
  graph_id derivation and posts the result to `/memories` itself. This
  keeps server endpoints orthogonal — graph routing is a client
  concern.
- **Extractor prompt biases toward emitting fewer memories.** Bad
  memories actively harm coding agents (a stale "always use X" rule
  produces wrong code). The system prompt explicitly says: omit if the
  fix was trivial, the correction was ambiguous, or the rule may not
  generalize. Confidence < 0.5 → don't emit at all.
- **`source=None` for legacy / trajectory-derived memories.** Chroma
  metadata omits the `source` field when None (chroma rejects None
  values for some types and we don't want to spend a sentinel string on
  the common case). Consequence: `source_in: ["correction", ...]` queries
  naturally exclude legacy memories without an explicit "and not null"
  predicate.

### Recall (the three triggers)

- **`/retrieve`, not `/reason`.** All three triggers (session-start,
  user-prompt, tool-family) call `/retrieve` and format
  `variables.semantic_memory` + `variables.procedural_memory` into the
  block. `/reason` adds an LLM synthesis step on top — fine for chat
  use cases where the answer is a paraphrase, wasteful here where the
  agent's own LLM is doing the reasoning anyway.
- **No history-keyed every-turn fan-out.** The exposed-token budget is
  the constraint. Session-start fires once; user-prompt fires only on
  substantial prompts (≥ 30 chars by default); tool-family fires only
  on configured patterns. If you find yourself calling `/retrieve` more
  often than that, push back on the design.
- **Each trigger has its own char cap** (defaults: 2000 / 1000 / 800).
  Smaller for triggers that fire more often. The cap is hard — over-
  size injection wastes the agent's exposed context.
- **No dedup across triggers.** Same memory may surface in both the
  session-start block and a sub-task block — acceptable for now since
  the agent ignores duplicates cheaply. Tracking surfaced IDs in
  `SessionState` to dedupe is a Stage 5 candidate.
- **Tool-family regex patterns support a `(?flags)` prefix.** JS
  `RegExp` doesn't accept inline flags, so `compileFlagPrefixedRegex`
  parses a leading `(?i)` / `(?im)` / etc. into constructor flags. This
  matches POSIX/Python convention so users from those ecosystems aren't
  surprised. Plain regex (no prefix) works fine.
- **`PreToolUse` injection is CC-specific**, but the abstract event
  interface returns `ContextInjection | null` from `onPreTool` for
  forward-compat. OpenCode's `tool.execute.before` can mutate args but
  not inject context; that adapter will discard the return. OpenClaw
  doesn't have the hook at all.

### Filtering & retrieval

- **Metadata filters apply at candidate-collection time, not
  post-ranking.** Earlier post-ranking attempt was wrong: it dropped
  high-confidence nodes that the value function ranked outside the
  top-K similarity window. The filter must constrain *which nodes are
  considered*, not *which are returned*.
- **Filter is applied in both Phase 1 (similarity scan) and Phase 3
  (tag-vote union)** of `retrieve_semantic_nodes`. Tag voting can pull
  nodes that didn't pass the Phase 1 filter — re-applying at the union
  keeps the candidate set consistent.
- **`confidence` defaults to 0.5; `source` defaults to `None`.**
  Backwards-compat with the existing chat plugin, which never sets
  these fields. A 0.5 default lets `min_confidence: 0.5` still admit
  legacy memories — bump above 0.5 only when you specifically want to
  exclude them.

### Eval harness

- **The harness fires synthetic events, not real agent outputs.** The
  runner walks a JSON fixture through the production `createCore` against
  a live PlugMem service. This proves the loop works and isolates token
  costs, but it doesn't measure agent task-success — that's a separate
  layer that records real Claude Code transcripts and replays the
  abstract events here.
- **Phase tagging uses a `ContextVar`, not a `complete()` kwarg.** The
  `with_phase("...")` context manager in `plugmem/clients/llm.py` is
  read by `_log_usage` and stamped on each token-usage JSONL record.
  Avoids threading `phase` through every call signature; ContextVar
  makes it async-safe under FastAPI's request handlers.
- **The runner uses an in-memory `SessionState`**, not the CC adapter's
  disk backing. The harness is one long-lived process — pre/post tool
  events share state in-memory just fine. This is by design, but means
  the harness doesn't exercise `DiskSessionState`'s file-locking /
  concurrency edge cases.
- **`exposedChars` in the report ≠ token count.** It's the char total
  of all recall blocks injected to the agent during a session. A
  rough conversion is chars/4, but report exposed cost as chars in
  raw form and let downstream reporting do tokenization if needed.

### Server compatibility

- **`chromadb` v0.6 broke `list_collections()`.** It now returns names
  only, but the storage layer's `list_graphs()` calls `col.name` on
  the result expecting Collection objects. `test_list_graphs` and
  `test_health` (which probes via list_collections) fail because of
  this. Unrelated to anything in this design — fix at the storage layer
  when convenient.

### Forward compatibility for new harnesses

- **The `<harness>` segment is the *only* adapter-influenced part of
  the storage schema.** Everything else (graph topology, memory types,
  source/confidence enums, recall API) is harness-agnostic. Adding a
  fourth harness = new `<harness>` literal + new adapter package; no
  schema migration.
- **Each adapter owns its `SessionState` backend.** Disk for CC
  (isolated processes), in-memory Map for OpenCode (long-lived plugin),
  in-memory Map for OpenClaw. The core never assumes anything about
  state durability beyond the `SessionState` interface contract.
- **The abstract event interface in `plugmem-coding-core/src/adapter.ts`
  is the contract.** Adapters normalize harness-native payloads to it.
  When CC or OpenCode adds a new event type, decide first whether it
  maps to an existing abstract event before adding a new one — keeping
  the interface narrow forces adapter authors to make the same decision
  the same way.
