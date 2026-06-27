// src/normalize.ts
import type {
  SessionStartEvent as CoreSessionStartEvent,
  UserPromptEvent as CoreUserPromptEvent,
  PreToolEvent as CorePreToolEvent,
  PostToolEvent as CorePostToolEvent,
  PreCompactEvent as CorePreCompactEvent,
  SessionEndEvent as CoreSessionEndEvent,
} from "@plugmem/coding-core";

export function normalizeSessionStart(input: any, directory: string, sessionId?: string): CoreSessionStartEvent {
  return {
    harness: "opencode",
    sessionId: sessionId || "default-session",
    cwd: directory,
    source: "startup",
  };
}

export function normalizeUserPrompt(input: any, directory: string): CoreUserPromptEvent | null {
  if (input.message?.role !== "user") return null;
  return {
    harness: "opencode",
    sessionId: input.message?.sessionId || input.session?.id || "default-session", 
    cwd: directory,
    prompt: input.message?.content || "",
  };
}

export function normalizePreTool(input: any, directory: string, callId: string): CorePreToolEvent {
  return {
    harness: "opencode",
    sessionId: input.session?.id || "default-session",
    cwd: directory,
    toolName: input.tool?.name || "unknown",
    toolInput: input.tool?.args || {},
    callId: callId,
  };
}

export function normalizePostTool(input: any, directory: string, callId: string): CorePostToolEvent {
  const isFailure = input.tool?.exitCode !== 0 || String(input.tool?.result || "").toLowerCase().includes("error");
  
  return {
    harness: "opencode",
    sessionId: input.session?.id || "default-session",
    cwd: directory,
    toolName: input.tool?.name || "unknown",
    toolInput: input.tool?.args || {},
    callId: callId,
    toolResult: String(input.tool?.result || ""),
    outcome: isFailure ? "failure" : "success",
  };
}

export function normalizePreCompact(input: any, directory: string): CorePreCompactEvent {
  const evt = input.event || input;
  return {
    harness: "opencode",
    sessionId: evt.session?.id || "default-session",
    cwd: directory,
  };
}

export function normalizeSessionEnd(input: any, directory: string, sessionId?: string, reason: "stop" | "session_end" | "reset" = "session_end"): CoreSessionEndEvent {
  return {
    harness: "opencode",
    sessionId: sessionId || "default-session",
    cwd: directory,
    reason: reason,
  };
}

// ── tool.execute.after outcome parsing ──────────────────────────────
//
// OpenCode's `tool.execute.after` second argument is `{ title, output,
// metadata }` (see @opencode-ai/plugin Hooks). There is NO top-level
// `result`, `error`, or `exitCode` — the tool's textual result is
// `output.output` and tool-specific data (e.g. a shell exit code) lives in
// `output.metadata`. These helpers read that shape defensively and never
// throw, so a missing/odd payload degrades to "" / "success" instead of
// killing the post-tool detector.

const SHELL_TOOL_RE = /^(bash|shell|sh|zsh|exec|run|terminal|command)/i;

// Strong shell-failure markers. Deliberately NARROW: applied only to
// shell-family tools and only as a fallback, so a successful read/grep/build
// whose output merely contains the word "Error:" is never flagged. (The old
// implementation blind-scanned every tool's full output for "Error:".)
const SHELL_FAILURE_MARKERS: RegExp[] = [
  /\bcommand failed\b/i,
  /\bexit (?:code|status)\b[:=\s]*[1-9]/i,
  /\bnon-zero exit\b/i,
];

function firstNumber(...vals: unknown[]): number | undefined {
  for (const v of vals) if (typeof v === "number") return v;
  return undefined;
}

/** Best-effort textual result from an OpenCode after-hook `output`. */
export function extractResultText(output: unknown): string {
  if (output == null) return "";
  if (typeof output === "string") return output;
  if (typeof output === "object") {
    const o = output as Record<string, unknown>;
    if (typeof o.output === "string") return o.output; // real OpenCode shape
    if (typeof o.result === "string") return o.result; // legacy/defensive
  }
  try {
    return JSON.stringify(output) ?? "";
  } catch {
    return String(output);
  }
}

/**
 * Classify a tool outcome from the after-hook payload.
 *
 * Precedence: structured exit code → explicit error flag → shell-only text
 * markers → "success" (the hook fired, so the tool completed). Never returns
 * a failure from a blind substring scan of arbitrary tool output.
 */
export function sniffOutcome(
  toolName: string,
  output: unknown,
): "success" | "failure" | "unknown" {
  if (output && typeof output === "object") {
    const o = output as Record<string, any>;
    const meta = (o.metadata ?? {}) as Record<string, any>;

    // 1. Structured exit code (most reliable — no false positives).
    const exit = firstNumber(
      meta.exit, meta.exitCode, meta.exit_code, meta.code,
      o.exitCode, o.exit, // legacy/defensive shapes
    );
    if (exit !== undefined) return exit === 0 ? "success" : "failure";

    // 2. Explicit boolean error flag some tools/versions may surface.
    if (
      meta.error === true ||
      o.error === true ||
      o.isError === true ||
      o.is_error === true
    ) {
      return "failure";
    }
  }

  // 3. Shell-family text markers only.
  if (SHELL_TOOL_RE.test(toolName || "")) {
    const text = extractResultText(output);
    if (SHELL_FAILURE_MARKERS.some((re) => re.test(text))) return "failure";
  }

  // 4. The after-hook fired without a failure signal → tool completed.
  return "success";
}
