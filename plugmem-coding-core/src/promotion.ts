// Promotion-gate logic: detectors that observe events during a session
// and accumulate candidates, plus the session-end runner that drains
// candidates and asks the server's /extract endpoint to emit structured
// memories.
//
// Detectors require cross-event SessionState (failure-delta needs to
// remember that a tool failed before a success arrives) so they're a
// no-op when state is not available.

import type {
  PostToolEvent,
  PreToolEvent,
  SessionState,
  UserPromptEvent,
} from "./adapter.js";

// ---------------------------------------------------------------------
// Candidate / state shapes
// ---------------------------------------------------------------------

export type CandidateKind = "failure_delta" | "correction";

export interface Candidate {
  kind: CandidateKind;
  window: string;
  toolName?: string;
  ts: number;
}

interface PendingCall {
  toolName: string;
  toolInput: unknown;
  ts: number;
}

interface FailureRecord {
  toolName: string;
  toolInput: unknown;
  toolResult: string;
  ts: number;
}

const PENDING_KEY = "pending_calls";
const FAILURES_KEY = "recent_failures";
const CANDIDATES_KEY = "candidates";

// Cap on how many recent failures we hold for matching against a later
// success. Prevents the cache from growing unbounded across long sessions.
const MAX_RECENT_FAILURES = 20;

// Window after which a failure no longer counts as "recent" enough to
// pair with a success. 10 minutes is generous — most fix-iterate cycles
// happen in seconds.
const FAILURE_PAIRING_WINDOW_MS = 10 * 60 * 1000;

// Regex for correction patterns. Conservative — false-positives here mean
// noisy memories. Stage 5 tuning will tighten/loosen based on eval.
const CORRECTION_PATTERNS: RegExp[] = [
  /\b(don'?t|do not)\b/i,
  /\bstop\b/i,
  /\b(actually|instead of|rather than)\b/i,
  /\bno,?\s+(?:the|that|you|we)\b/i,
  /\bnot\s+\w+,?\s+(?:but|use)\b/i,
  /\bwe\s+(?:use|prefer|don'?t use|don't use)\b/i,
  /\bthe right way\b/i,
  /\bshould\s+(?:be|use|not)\b/i,
];

// ---------------------------------------------------------------------
// Detectors (called from CoreCallbacks)
// ---------------------------------------------------------------------

export async function recordPreTool(
  state: SessionState,
  e: PreToolEvent,
): Promise<void> {
  const pending =
    (await state.get<Record<string, PendingCall>>(PENDING_KEY)) ?? {};
  pending[e.callId] = {
    toolName: e.toolName,
    toolInput: e.toolInput,
    ts: Date.now(),
  };
  await state.set(PENDING_KEY, pending);
}

export async function recordPostTool(
  state: SessionState,
  e: PostToolEvent,
): Promise<void> {
  const pending =
    (await state.get<Record<string, PendingCall>>(PENDING_KEY)) ?? {};
  const call = pending[e.callId];
  delete pending[e.callId];
  await state.set(PENDING_KEY, pending);

  // We only act on outcomes we're sure about. "unknown" is conservative —
  // skip both the failure-recording and the success-pairing path.
  if (e.outcome === "failure") {
    await pushFailure(state, {
      toolName: e.toolName,
      toolInput: e.toolInput,
      toolResult: e.toolResult,
      ts: Date.now(),
    });
    return;
  }

  if (e.outcome !== "success") return;

  // Look for a recent failure on the same tool to pair with this success.
  const failures =
    (await state.get<FailureRecord[]>(FAILURES_KEY)) ?? [];
  const now = Date.now();
  const matchIndex = failures.findIndex(
    (f) =>
      f.toolName === e.toolName &&
      now - f.ts <= FAILURE_PAIRING_WINDOW_MS,
  );
  if (matchIndex < 0) return;

  const failure = failures[matchIndex]!;
  failures.splice(matchIndex, 1);
  await state.set(FAILURES_KEY, failures);

  const window = renderFailureDelta(failure, {
    toolName: e.toolName,
    toolInput: e.toolInput,
    toolResult: e.toolResult,
    callBefore: call,
  });
  await appendCandidate(state, {
    kind: "failure_delta",
    window,
    toolName: e.toolName,
    ts: now,
  });
}

export async function recordUserPrompt(
  state: SessionState,
  e: UserPromptEvent,
): Promise<void> {
  if (!matchesCorrectionPattern(e.prompt)) return;
  await appendCandidate(state, {
    kind: "correction",
    window: `User correction: ${e.prompt.slice(0, 1500)}`,
    ts: Date.now(),
  });
}

export function matchesCorrectionPattern(prompt: string): boolean {
  if (!prompt) return false;
  return CORRECTION_PATTERNS.some((re) => re.test(prompt));
}

// ---------------------------------------------------------------------
// State helpers
// ---------------------------------------------------------------------

async function pushFailure(
  state: SessionState,
  failure: FailureRecord,
): Promise<void> {
  const failures =
    (await state.get<FailureRecord[]>(FAILURES_KEY)) ?? [];
  failures.push(failure);
  if (failures.length > MAX_RECENT_FAILURES) {
    failures.splice(0, failures.length - MAX_RECENT_FAILURES);
  }
  await state.set(FAILURES_KEY, failures);
}

async function appendCandidate(
  state: SessionState,
  candidate: Candidate,
): Promise<void> {
  const list = (await state.get<Candidate[]>(CANDIDATES_KEY)) ?? [];
  list.push(candidate);
  await state.set(CANDIDATES_KEY, list);
}

export async function drainCandidates(
  state: SessionState,
): Promise<Candidate[]> {
  const list = (await state.get<Candidate[]>(CANDIDATES_KEY)) ?? [];
  await state.del(CANDIDATES_KEY);
  return list;
}

// ---------------------------------------------------------------------
// Window rendering
// ---------------------------------------------------------------------

function renderFailureDelta(
  failure: FailureRecord,
  success: {
    toolName: string;
    toolInput: unknown;
    toolResult: string;
    callBefore?: PendingCall;
  },
): string {
  const truncate = (s: string, n: number) =>
    s.length > n ? s.slice(0, n) + "…" : s;
  const inputStr = (v: unknown) => {
    try {
      return JSON.stringify(v);
    } catch {
      return String(v);
    }
  };
  return [
    `Tool: ${failure.toolName}`,
    `Failed call input: ${truncate(inputStr(failure.toolInput), 800)}`,
    `Failure output: ${truncate(failure.toolResult, 800)}`,
    `Successful call input: ${truncate(inputStr(success.toolInput), 800)}`,
    `Success output: ${truncate(success.toolResult, 800)}`,
  ].join("\n");
}
