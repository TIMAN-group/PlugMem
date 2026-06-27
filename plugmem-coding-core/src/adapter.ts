// Abstract event interface consumed by the core. Each harness adapter
// (Claude Code, OpenCode, OpenClaw) normalizes its native events into
// these shapes and calls the matching CoreCallbacks method.

export type Harness = "claude-code" | "opencode" | "openclaw";

export interface SessionStartEvent {
  harness: Harness;
  sessionId: string;
  cwd: string;
  // Path to the persisted transcript on disk — used as a fallback when
  // a later hook payload is incomplete.
  transcriptPath?: string;
  // Claude Code-style source ("startup" | "resume" | "clear" | "compact").
  source?: string;
}

export interface UserPromptEvent {
  harness: Harness;
  sessionId: string;
  cwd: string;
  prompt: string;
}

export interface PreToolEvent {
  harness: Harness;
  sessionId: string;
  cwd: string;
  toolName: string;
  toolInput: unknown;
  callId: string;
}

export interface PostToolEvent {
  harness: Harness;
  sessionId: string;
  cwd: string;
  toolName: string;
  toolInput: unknown;
  callId: string;
  toolResult: string;
  // Best-effort. Adapters that don't have a clean success/failure signal
  // set this from heuristics; core tolerates "unknown".
  outcome: "success" | "failure" | "unknown";
}

export interface PreCompactEvent {
  harness: Harness;
  sessionId: string;
  cwd: string;
  messages?: unknown[];
  transcriptPath?: string;
}

export interface SessionEndEvent {
  harness: Harness;
  sessionId: string;
  cwd: string;
  reason: "stop" | "session_end" | "reset" | "unknown";
  messages?: unknown[];
  transcriptPath?: string;
}

// Returned from session_start / user_prompt callbacks. Adapters wire this
// into the harness-native injection mechanism.
export interface ContextInjection {
  text: string;
  role: "system" | "user_context";
}

export interface CoreCallbacks {
  onSessionStart(e: SessionStartEvent): Promise<ContextInjection | null>;
  onUserPrompt(e: UserPromptEvent): Promise<ContextInjection | null>;
  // PreTool may inject context for tool-family pre-recall. Adapters whose
  // harness can't inject before a tool call should discard the return.
  onPreTool(e: PreToolEvent): Promise<ContextInjection | null>;
  onPostTool(e: PostToolEvent): Promise<void>;
  onPreCompact(e: PreCompactEvent): Promise<void>;
  onSessionEnd(e: SessionEndEvent): Promise<void>;
}

// Cross-event state, keyed by session. Backing store differs by harness;
// the Claude Code adapter persists to disk because hooks are isolated
// processes; OpenCode/OpenClaw use in-memory.
export interface SessionState {
  get<T>(key: string): Promise<T | undefined>;
  set<T>(key: string, value: T): Promise<void>;
  del(key: string): Promise<void>;
}
