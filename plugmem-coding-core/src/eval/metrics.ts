// Read the server's token-usage JSONL and split costs by phase.
//
// Per the saved `feedback_bench_token_split` rule, we ALWAYS report
// exposed (agent-facing) and internal (memory-system) tokens separately.
// In the coding harness:
//   - exposed = recall blocks injected to the agent (NOT in this JSONL —
//     comes from the runner's report.totalExposedChars)
//   - internal = LLM calls the memory service made (this file)
//
// Phase taxonomy (set by `with_phase` on the server side):
//   extract     — promotion-gate extractor
//   retrieve    — inference/retrieving (sub-tasks of recall)
//   reason      — final reasoning synthesis (chat plugin only currently)
//   structuring — Memory.close() trajectory structuring (legacy)
//   default     — untagged

import { readFile } from "node:fs/promises";

export interface PhaseTotals {
  prompt: number;
  completion: number;
  total: number;
  count: number;
}

export interface InternalTokenSummary {
  byPhase: Record<string, PhaseTotals>;
  totals: PhaseTotals;
}

interface TokenLogEntry {
  phase?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

/**
 * Read a token_usage_file JSONL and aggregate by phase.
 * Returns empty totals if the file doesn't exist or is empty.
 */
export async function summarizeTokenLog(
  path: string,
): Promise<InternalTokenSummary> {
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch {
    return { byPhase: {}, totals: emptyPhaseTotals() };
  }

  const byPhase: Record<string, PhaseTotals> = {};
  const totals = emptyPhaseTotals();

  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let entry: TokenLogEntry;
    try {
      entry = JSON.parse(trimmed);
    } catch {
      continue;
    }

    const phase = entry.phase ?? "default";
    if (!byPhase[phase]) byPhase[phase] = emptyPhaseTotals();

    const prompt = numOrZero(entry.prompt_tokens);
    const completion = numOrZero(entry.completion_tokens);
    const total =
      numOrZero(entry.total_tokens) || prompt + completion;

    byPhase[phase].prompt += prompt;
    byPhase[phase].completion += completion;
    byPhase[phase].total += total;
    byPhase[phase].count += 1;

    totals.prompt += prompt;
    totals.completion += completion;
    totals.total += total;
    totals.count += 1;
  }

  return { byPhase, totals };
}

/**
 * Slice the log to entries written between two ISO timestamps. Useful for
 * isolating a single fixture run when the log is shared across runs.
 *
 * The log doesn't natively store timestamps — callers should rotate the
 * log per-run (or pass a fresh path) for clean isolation. This helper is
 * for cases where rotation isn't possible.
 */
export async function summarizeTokenLogBetween(
  path: string,
  _startedAt: string,
  _finishedAt: string,
): Promise<InternalTokenSummary> {
  // The token_usage_file format doesn't include timestamps today. For now
  // this just delegates to summarizeTokenLog. If we add timestamps later,
  // wire the filter here.
  return summarizeTokenLog(path);
}

function emptyPhaseTotals(): PhaseTotals {
  return { prompt: 0, completion: 0, total: 0, count: 0 };
}

function numOrZero(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}
