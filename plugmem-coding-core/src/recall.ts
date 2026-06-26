// Recall configuration and shared helpers used by all three recall
// triggers (session_start, user_prompt sub-task, pre_tool tool-family).
//
// All triggers share the same primitive: call /retrieve with metadata
// filters, format `variables.semantic_memory` + `variables.procedural_memory`
// into a recall block, return as ContextInjection (or null if empty).
//
// Vector retrieval (/retrieve), not LLM synthesis (/reason). Reasons:
//   - Per-prompt recall on every user message would burn LLM cost.
//   - Raw facts pack denser into a token budget than synthesized prose.
//   - The agent's own LLM is doing the reasoning anyway — no need to
//     pre-chew it.

import type { ContextInjection } from "./adapter.js";
import type { PlugMemClient } from "./client.js";
import type { MemorySourceWire } from "./types.js";

export interface RecallConfig {
  /** Default true. Disable to bypass this trigger entirely. */
  enabled?: boolean;
  /** Drop memories below this confidence. Default per-trigger. */
  minConfidence?: number;
  /** Restrict recall to memories with these sources. Default: all. */
  sourceIn?: MemorySourceWire[];
  /** Hard char cap on the injected block. Default per-trigger. */
  charCap?: number;
}

export interface SessionStartRecallConfig extends RecallConfig {
  /** Observation passed to /retrieve. Tune to surface session-relevant
   *  high-level conventions vs targeted facts. */
  query?: string;
}

export interface UserPromptRecallConfig extends RecallConfig {
  /** Skip recall when prompt length is below this (e.g., "yes" / "ok").
   *  Default 30 chars. */
  minPromptLength?: number;
}

export interface ToolFamilyRecallConfig extends RecallConfig {
  /** Map regex (matched against toolName) → recall query. Empty by default,
   *  which means tool-family recall is off until a caller configures it.
   *  Example: { "(?i)pytest|vitest": "test setup, fixtures, flaky tests" }. */
  mapping?: Record<string, string>;
}

// ── Defaults ─────────────────────────────────────────────────────────

export const DEFAULT_SESSION_START_RECALL: Required<SessionStartRecallConfig> = {
  enabled: true,
  // Surface high-quality memories, including legacy/None-source
  // (which pass the default 0.5 confidence). Bump above 0.5 to
  // exclude legacy.
  minConfidence: 0.5,
  sourceIn: undefined as unknown as MemorySourceWire[],
  charCap: 2000,
  query: "project conventions, facts, debugging recipes",
};

export const DEFAULT_USER_PROMPT_RECALL: Required<UserPromptRecallConfig> = {
  enabled: true,
  minConfidence: 0.6,
  sourceIn: undefined as unknown as MemorySourceWire[],
  charCap: 1000,
  minPromptLength: 30,
};

export const DEFAULT_TOOL_FAMILY_RECALL: Required<ToolFamilyRecallConfig> = {
  enabled: false, // explicit opt-in
  minConfidence: 0.6,
  sourceIn: undefined as unknown as MemorySourceWire[],
  charCap: 800,
  mapping: {},
};

// ── Resolve user-provided config against defaults ───────────────────

export function resolveSessionStartRecall(
  raw: SessionStartRecallConfig | false | undefined,
): Required<SessionStartRecallConfig> | null {
  if (raw === false) return null;
  return { ...DEFAULT_SESSION_START_RECALL, ...(raw ?? {}) };
}

export function resolveUserPromptRecall(
  raw: UserPromptRecallConfig | false | undefined,
): Required<UserPromptRecallConfig> | null {
  if (raw === false) return null;
  return { ...DEFAULT_USER_PROMPT_RECALL, ...(raw ?? {}) };
}

export function resolveToolFamilyRecall(
  raw: ToolFamilyRecallConfig | false | undefined,
): Required<ToolFamilyRecallConfig> | null {
  if (raw === false) return null;
  return { ...DEFAULT_TOOL_FAMILY_RECALL, ...(raw ?? {}) };
}

// ── Shared recall primitive ─────────────────────────────────────────

export interface DoRecallParams {
  client: PlugMemClient;
  graphId: string;
  observation: string;
  minConfidence?: number;
  sourceIn?: MemorySourceWire[];
  charCap: number;
  blockTitle: string;
  log: (msg: string, err?: unknown) => void;
}

/**
 * One recall call against /retrieve, formatted as a ContextInjection.
 * Returns null if the recall produced nothing useful (empty graph,
 * filter excluded everything, retrieve errored).
 */
export async function doRecall(
  p: DoRecallParams,
): Promise<ContextInjection | null> {
  let response;
  try {
    response = await p.client.retrieve(p.graphId, {
      observation: p.observation,
      mode: "semantic_memory",
      ...(p.minConfidence !== undefined ? { min_confidence: p.minConfidence } : {}),
      ...(p.sourceIn ? { source_in: p.sourceIn } : {}),
    } as Parameters<typeof p.client.retrieve>[1]);
  } catch (err) {
    p.log(`retrieve(${p.graphId}) failed`, err);
    return null;
  }

  const text = formatVariables(response.variables);
  if (!text) return null;

  const capped = capChars(text, p.charCap);
  return {
    role: "system",
    text: formatBlock(p.blockTitle, p.graphId, capped),
  };
}

// ── Tool-family matching ────────────────────────────────────────────

/**
 * Find the first regex in the mapping that matches the toolName, return
 * its associated recall query. Returns null if no rule matches.
 *
 * Patterns may use a leading `(?flags)` prefix (POSIX/Python convention
 * — e.g., `(?i)` for case-insensitive). JavaScript's RegExp constructor
 * doesn't accept inline flags, so we parse them out manually before
 * constructing.
 */
export function matchToolFamily(
  mapping: Record<string, string>,
  toolName: string,
): string | null {
  for (const [pattern, query] of Object.entries(mapping)) {
    const re = compileFlagPrefixedRegex(pattern);
    if (!re) continue;
    if (re.test(toolName)) return query;
  }
  return null;
}

function compileFlagPrefixedRegex(pattern: string): RegExp | null {
  let body = pattern;
  let flags = "";
  const m = /^\(\?([a-z]+)\)/.exec(pattern);
  if (m) {
    flags = m[1]!;
    body = pattern.slice(m[0].length);
  }
  try {
    return new RegExp(body, flags);
  } catch {
    return null;
  }
}

// ── Helpers ─────────────────────────────────────────────────────────

function formatVariables(variables: Record<string, unknown>): string {
  const parts: string[] = [];
  const sem = (variables.semantic_memory as string | undefined) ?? "";
  const proc = (variables.procedural_memory as string | undefined) ?? "";
  if (sem && sem !== "No relevant fact") parts.push(sem.trim());
  if (proc && proc !== "No relevant experiences") parts.push(proc.trim());
  return parts.join("\n\n").trim();
}

function capChars(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max - 1) + "…";
}

function formatBlock(title: string, graphId: string, text: string): string {
  return [
    `<plugmem-recall trigger="${title}" graph="${graphId}">`,
    text,
    `</plugmem-recall>`,
  ].join("\n");
}
