// Eval harness fixture and result types.
//
// A fixture describes a sequence of synthetic events fired through real
// `createCore` against a live PlugMem service. The runner records the
// outcome of each event (injection text, candidate state, graph stats)
// and writes a JSON report. Metrics aggregator can then split exposed
// vs internal token cost by reading the server's token-usage JSONL.
//
// The harness does NOT mock PlugMem — it talks HTTP to a real service.
// That's the point: we want to measure what we actually ship.

import type { Harness } from "../adapter.js";
import type { MemorySourceWire } from "../types.js";

export type EvalEventType =
  | "session_start"
  | "user_prompt"
  | "pre_tool"
  | "post_tool"
  | "pre_compact"
  | "session_end";

export interface EvalEventBase {
  type: EvalEventType;
  /** Optional human label for traceability in the output report. */
  label?: string;
}

export interface EvalSessionStart extends EvalEventBase {
  type: "session_start";
  /** Override session_start `source` field (Claude Code's startup/resume/clear/compact). */
  source?: string;
}

export interface EvalUserPrompt extends EvalEventBase {
  type: "user_prompt";
  prompt: string;
}

export interface EvalPreTool extends EvalEventBase {
  type: "pre_tool";
  toolName: string;
  toolInput?: unknown;
  callId: string;
}

export interface EvalPostTool extends EvalEventBase {
  type: "post_tool";
  toolName: string;
  toolInput?: unknown;
  callId: string;
  toolResult: string;
  outcome: "success" | "failure" | "unknown";
}

export interface EvalPreCompact extends EvalEventBase {
  type: "pre_compact";
}

export interface EvalSessionEnd extends EvalEventBase {
  type: "session_end";
  reason?: "stop" | "session_end" | "reset" | "unknown";
}

export type EvalEvent =
  | EvalSessionStart
  | EvalUserPrompt
  | EvalPreTool
  | EvalPostTool
  | EvalPreCompact
  | EvalSessionEnd;

export interface EvalSession {
  id: number | string;
  events: EvalEvent[];
}

export interface EvalRecallConfig {
  sessionStartRecall?: {
    enabled?: boolean;
    minConfidence?: number;
    sourceIn?: MemorySourceWire[];
    charCap?: number;
    query?: string;
  } | false;
  userPromptRecall?: {
    enabled?: boolean;
    minConfidence?: number;
    sourceIn?: MemorySourceWire[];
    charCap?: number;
    minPromptLength?: number;
  } | false;
  toolFamilyRecall?: {
    enabled?: boolean;
    minConfidence?: number;
    sourceIn?: MemorySourceWire[];
    charCap?: number;
    mapping?: Record<string, string>;
  } | false;
}

export interface EvalFixture {
  /** Human-readable fixture name. */
  name: string;
  /** PlugMem service URL. The runner expects this to already be running. */
  baseUrl: string;
  /** API key, if the service has auth enabled. */
  apiKey?: string;
  /** Graph ID to write to. The runner truncates the graph at the start
   *  of each run so multiple invocations don't pollute each other. */
  graphId: string;
  /** Harness identifier. Affects how repo-graph IDs would be derived,
   *  but for fixture-driven runs we override `graphId` directly. */
  harness: Harness;
  /** Working directory passed in events; not actually used to derive
   *  the graph since `graphId` is provided. */
  cwd: string;
  /** Recall config overrides. Defaults match the production core. */
  recall?: EvalRecallConfig;
  /** Sessions to run sequentially. */
  sessions: EvalSession[];
}

// ── Result shapes ───────────────────────────────────────────────────

export interface EvalEventResult {
  type: EvalEventType;
  label?: string;
  /** When the event returned a ContextInjection — captured here. */
  injection?: { text: string; charCount: number };
  /** Free-form notes (e.g., "skipped: prompt too short"). */
  note?: string;
}

export interface EvalSessionResult {
  id: number | string;
  events: EvalEventResult[];
  /** Graph stats after this session ran. */
  statsAfter: Record<string, number>;
  /** Promotions inserted during this session (count of semantic + procedural). */
  promotions: number;
  /** Total exposed-context chars injected this session. */
  exposedChars: number;
}

export interface EvalRunReport {
  fixture: string;
  startedAt: string;
  finishedAt: string;
  sessions: EvalSessionResult[];
  /** Sum across all sessions. */
  totalPromotions: number;
  totalExposedChars: number;
  /** Final graph stats. */
  finalStats: Record<string, number>;
}
