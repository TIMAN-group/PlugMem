// Translate Claude Code hook stdin payloads to abstract events.
//
// CC payload shapes (from Stage 0 audit):
//   common: { session_id, transcript_path, cwd, hook_event_name,
//             permission_mode }
//   SessionStart adds: { source: "startup"|"resume"|"clear"|"compact" }
//   UserPromptSubmit adds: { prompt }
//   PreToolUse adds: { tool_name, tool_input, tool_use_id }
//   PostToolUse adds: { tool_name, tool_input, tool_use_id, tool_result }
//   PreCompact / Stop / SessionEnd: just the common fields

import type {
  PreCompactEvent,
  PreToolEvent,
  PostToolEvent,
  SessionEndEvent,
  SessionStartEvent,
  UserPromptEvent,
} from "@plugmem/coding-core";

export interface CCStdinPayload {
  session_id: string;
  transcript_path?: string;
  cwd: string;
  hook_event_name: string;
  source?: string;
  prompt?: string;
  tool_name?: string;
  tool_input?: unknown;
  tool_use_id?: string;
  tool_result?: unknown;
}

const HARNESS = "claude-code" as const;

export function toSessionStart(p: CCStdinPayload): SessionStartEvent {
  return {
    harness: HARNESS,
    sessionId: p.session_id,
    cwd: p.cwd,
    transcriptPath: p.transcript_path,
    source: p.source,
  };
}

export function toUserPrompt(p: CCStdinPayload): UserPromptEvent {
  return {
    harness: HARNESS,
    sessionId: p.session_id,
    cwd: p.cwd,
    prompt: p.prompt ?? "",
  };
}

export function toPreTool(p: CCStdinPayload): PreToolEvent {
  return {
    harness: HARNESS,
    sessionId: p.session_id,
    cwd: p.cwd,
    toolName: p.tool_name ?? "",
    toolInput: p.tool_input,
    callId: p.tool_use_id ?? "",
  };
}

export function toPostTool(p: CCStdinPayload): PostToolEvent {
  // Outcome heuristic: tool_result is typically the message-block array used
  // in the JSONL, where individual blocks may carry is_error. Falling back
  // to "unknown" is fine for Stage 1 — Stage 3 will tighten this.
  const outcome = sniffOutcome(p.tool_result);
  return {
    harness: HARNESS,
    sessionId: p.session_id,
    cwd: p.cwd,
    toolName: p.tool_name ?? "",
    toolInput: p.tool_input,
    callId: p.tool_use_id ?? "",
    toolResult: stringifyResult(p.tool_result),
    outcome,
  };
}

export function toPreCompact(p: CCStdinPayload): PreCompactEvent {
  return {
    harness: HARNESS,
    sessionId: p.session_id,
    cwd: p.cwd,
    transcriptPath: p.transcript_path,
  };
}

export function toSessionEnd(p: CCStdinPayload): SessionEndEvent {
  // Both `Stop` and `SessionEnd` map here. CC's `Stop` fires every turn —
  // we treat it as session_end-equivalent for Stage 1 because that's the
  // last opportunity to ingest a trajectory before context is lost.
  // Stage 3 will distinguish them once promotion is gated.
  const reason: SessionEndEvent["reason"] =
    p.hook_event_name === "Stop"
      ? "stop"
      : p.hook_event_name === "SessionEnd"
        ? "session_end"
        : "unknown";
  return {
    harness: HARNESS,
    sessionId: p.session_id,
    cwd: p.cwd,
    reason,
    transcriptPath: p.transcript_path,
  };
}

function sniffOutcome(
  result: unknown,
): "success" | "failure" | "unknown" {
  if (result == null) return "unknown";
  if (Array.isArray(result)) {
    for (const block of result) {
      if (
        block &&
        typeof block === "object" &&
        (block as { is_error?: boolean }).is_error === true
      ) {
        return "failure";
      }
    }
    return "success";
  }
  return "unknown";
}

function stringifyResult(result: unknown): string {
  if (result == null) return "";
  if (typeof result === "string") return result;
  try {
    return JSON.stringify(result);
  } catch {
    return String(result);
  }
}
