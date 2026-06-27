// Abstract event interface consumed by plugmem-coding-core.
// Each harness adapter (Claude Code, OpenCode, OpenClaw) normalizes its
// native events into these shapes and calls the matching CoreCallbacks
// method. The core never knows which harness produced an event.
//
// Status: draft from Stage 0 audit (2026-05-01). Pre-implementation —
// tighten as Stage 1 reveals what actually needs to be on the wire.

export type Harness = "claude-code" | "opencode" | "openclaw";

// ---------------------------------------------------------------------
// Events (adapter → core)
// ---------------------------------------------------------------------

export interface SessionStartEvent {
  harness: Harness;
  sessionId: string;
  cwd: string;
  // Path to the persisted transcript on disk — used as a fallback when
  // a later hook payload is incomplete (Claude Code: ~/.claude/projects/...,
  // OpenCode: ~/.local/share/opencode/storage/...).
  transcriptPath?: string;
  // Claude Code-style source ("startup" | "resume" | "clear" | "compact").
  // Absent for harnesses that don't distinguish.
  source?: string;
}

export interface UserPromptEvent {
  harness: Harness;
  sessionId: string;
  prompt: string;
}

export interface PreToolEvent {
  harness: Harness;
  sessionId: string;
  toolName: string;
  toolInput: unknown;
  // Stable ID linking a pre/post pair for the same call.
  callId: string;
}

export interface PostToolEvent {
  harness: Harness;
  sessionId: string;
  toolName: string;
  toolInput: unknown;
  callId: string;
  toolResult: string;
  // Best-effort. Some harnesses don't surface a clean success/failure
  // bit — adapters fall back to heuristics (exit-code parsing, error-marker
  // sniffing, content shape). Core must tolerate "unknown".
  outcome: "success" | "failure" | "unknown";
}

export interface PreCompactEvent {
  harness: Harness;
  sessionId: string;
  // Inline message log if the harness provides it; else read transcriptPath.
  messages?: unknown[];
  transcriptPath?: string;
}

export interface SessionEndEvent {
  harness: Harness;
  sessionId: string;
  reason: "stop" | "session_end" | "reset" | "unknown";
  messages?: unknown[];
  transcriptPath?: string;
}

// ---------------------------------------------------------------------
// Core → adapter (return values)
// ---------------------------------------------------------------------

// Returned from session_start / user_prompt callbacks. Adapters wire this
// into the harness-native injection mechanism:
//   - Claude Code: hookSpecificOutput.additionalContext
//   - OpenCode: experimental.chat.system.transform → output.system.push(...)
//   - OpenClaw: n/a (no hook for this — degrades to no-op)
export interface ContextInjection {
  text: string;
  role: "system" | "user_context";
}

export interface CoreCallbacks {
  onSessionStart(e: SessionStartEvent): Promise<ContextInjection | null>;
  onUserPrompt(e: UserPromptEvent): Promise<ContextInjection | null>;
  onPreTool(e: PreToolEvent): Promise<void>;
  onPostTool(e: PostToolEvent): Promise<void>;
  onPreCompact(e: PreCompactEvent): Promise<void>;
  onSessionEnd(e: SessionEndEvent): Promise<void>;
}

// ---------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------

// Cross-event state, keyed by session. Backing store differs by harness:
//   - Claude Code: hooks run in isolated processes, so state is persisted
//     to disk (e.g., per-session file under ~/.cache/plugmem/) or to the
//     PlugMem service.
//   - OpenCode: long-lived plugin module — in-memory Map is fine.
//   - OpenClaw: long-lived plugin instance — in-memory.
// Failure-delta detection in particular needs this: the pre-tool event
// stores candidate args; the post-tool event reads them back to compute
// the delta when a retry succeeds.
export interface SessionState {
  get<T>(key: string): Promise<T | undefined>;
  set<T>(key: string, value: T): Promise<void>;
  del(key: string): Promise<void>;
}
