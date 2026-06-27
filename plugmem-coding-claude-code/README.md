# @plugmem/coding-claude-code

Claude Code plugin for PlugMem cross-session memory.

The agent learns project conventions, debugging recipes, and personal
preferences from your sessions and recalls them at the start of future
sessions — like a self-writing CLAUDE.md.

## Installation

See **[ONBOARDING.md](./ONBOARDING.md)** for the full walkthrough.

Quick version:

```bash
npm install
npm run build
export PLUGMEM_BASE_URL=http://localhost:8080
export PLUGMEM_API_KEY=...
claude --plugin-dir /absolute/path/to/this/dir
```

You'll need a running PlugMem service — see the main repo README for
service setup.

## Hooks installed

| Event | Behavior |
|---|---|
| `SessionStart` | Inject a recall block of high-confidence facts from the per-repo graph |
| `UserPromptSubmit` | Detect correction patterns; sub-task recall on substantial prompts |
| `PreToolUse` | Capture tool calls for failure→success delta detection; (optional) tool-family recall |
| `PostToolUse` | Pair successes with prior failures to extract debugging recipes |
| `PreCompact` | Drain promotion candidates and write structured memories before context is lost |
| `SessionEnd` | Final promotion sweep + state cleanup |

The dispatcher (`dist/bin/cc-hook.js`) handles all six events; routing
is by `hook_event_name` from the stdin payload.

## Configuration

All via environment variables. See ONBOARDING.md §3 for the full table.

Required: `PLUGMEM_BASE_URL`. Optional: `PLUGMEM_API_KEY`,
`PLUGMEM_DEBUG`, `PLUGMEM_STATE_DIR`, `PLUGMEM_KEEP_STATE`, etc.

If `PLUGMEM_BASE_URL` is unset, the dispatcher silently exits 0 — the
session works normally with no memory writes.

## Tests

```bash
npm test
```

39 tests cover normalize / respond / state / dispatcher.

## License

MIT
