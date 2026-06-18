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
