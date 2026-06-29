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

export type CandidateKind = "failure_delta" | "correction" | "episodic";

export interface Candidate {
  kind: CandidateKind;
  window: string;
  toolName?: string;
  ts: number;
  /** Episodic step index where this candidate resolved. Mapped to a segment
   *  at drain time so the extracted memory grounds on its own trajectory split. */
  stepIndex: number;
  /** Filled at drain: the subgoal segment this candidate's memory grounds on. */
  segment: number;
}

/** One stored step of the session trajectory (the raw episodic substrate). */
interface StoredStep {
  observation: string;
  action: string;
}

/** One step of the drained trajectory. `segment` partitions the trajectory into
 *  subgoal units (split at user prompts + resolved candidates), so each
 *  procedural memory grounds on the contiguous sub-sequence it came from. */
export interface EpisodicStepRec extends StoredStep {
  segment: number;
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
const EPISODIC_KEY = "episodic_steps";
// Segment-start step indices: positions in the trajectory where a new subgoal
// begins. The trajectory is split here so each procedural grounds on its own
// sub-sequence — the coding-agent analog of the original PlugMem
// subgoal-similarity split. A boundary is added at every user prompt and after
// every resolved candidate (a failure→fix cycle / correction completing a
// subgoal). Index 0 is always an implicit boundary.
const BOUNDARIES_KEY = "episodic_boundaries";

// Cap on episodic steps held per drain (one trajectory) and per-field length,
// so episodic nodes stay bounded on long turns while remaining a readable log.
const MAX_EPISODIC_STEPS = 60;
const MAX_STEP_CHARS = 600;

const clipStep = (s: string, n = MAX_STEP_CHARS): string =>
  s.length > n ? s.slice(0, n) + "…" : s;
const asText = (v: unknown): string => {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v) ?? String(v);
  } catch {
    return String(v);
  }
};

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

const EPISODIC_PATTERNS: RegExp[] = [
  /\b(it works|looks good|perfect|we are done|finished|success)\b/i,
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

  // Record this tool step into the session trajectory (episodic substrate),
  // regardless of outcome — this is the raw "what the agent did" log.
  const stepIndex = await pushEpisodicStep(state, {
    observation: clipStep(`[${e.outcome}] ${e.toolResult}`),
    action: clipStep(`${e.toolName} ${asText(e.toolInput)}`),
  });

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
    stepIndex,
    segment: 0,
  });
  // The fix cycle completed a subgoal — the next step starts a new segment.
  await addBoundary(state, stepIndex + 1);
}

export async function recordUserPrompt(
  state: SessionState,
  e: UserPromptEvent,
): Promise<void> {
  // A new user request begins a new subgoal segment. The request itself is the
  // opening episodic step of that segment.
  const stepIndex = await pushEpisodicStep(state, {
    observation: clipStep(`User: ${e.prompt}`),
    action: "",
  });
  await addBoundary(state, stepIndex);

  if (matchesCorrectionPattern(e.prompt)) {
    await appendCandidate(state, {
      kind: "correction",
      window: `User correction: ${e.prompt.slice(0, 1500)}`,
      ts: Date.now(),
      stepIndex,
      segment: 0,
    });
  } else if (EPISODIC_PATTERNS.some((re) => re.test(e.prompt))) {
    await appendCandidate(state, {
      kind: "episodic",
      window: `Goal completed: ${e.prompt.slice(0, 1500)}`,
      ts: Date.now(),
      stepIndex,
      segment: 0,
    });
  }
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

/** Append a raw trajectory step; returns its absolute index in the trajectory. */
async function pushEpisodicStep(
  state: SessionState,
  step: StoredStep,
): Promise<number> {
  const list = (await state.get<StoredStep[]>(EPISODIC_KEY)) ?? [];
  list.push(step);
  await state.set(EPISODIC_KEY, list);
  return list.length - 1;
}

/** Record a segment-start boundary at the given step index (idempotent). */
async function addBoundary(state: SessionState, idx: number): Promise<void> {
  const b = (await state.get<number[]>(BOUNDARIES_KEY)) ?? [];
  if (!b.includes(idx)) {
    b.push(idx);
    await state.set(BOUNDARIES_KEY, b);
  }
}

/** Build the absolute-index → segment mapping for one drain.
 *
 *  Applies the trailing-window cap (keep the last MAX_EPISODIC_STEPS steps so a
 *  pathologically long turn stays bounded), then numbers contiguous segments
 *  from the boundaries that fall inside the window. Index 0 of the window is
 *  always a segment start, so segment numbers are dense (0..n-1) and line up
 *  with the trajectory list the gate builds. */
function buildSegmentMap(
  nSteps: number,
  rawBoundaries: number[],
): { offset: number; segmentOf: (absIdx: number) => number } {
  const offset = Math.max(0, nSteps - MAX_EPISODIC_STEPS);
  const starts = new Set<number>([offset]);
  for (const b of rawBoundaries) {
    if (b > offset && b < nSteps) starts.add(b);
  }
  const sorted = [...starts].sort((a, b) => a - b);
  const segmentOf = (absIdx: number): number => {
    if (absIdx <= offset) return 0;
    let seg = -1;
    for (const s of sorted) {
      if (s <= absIdx) seg += 1;
      else break;
    }
    return seg < 0 ? 0 : seg;
  };
  return { offset, segmentOf };
}

/** Drain promotion candidates, stamping each with the segment its evidence step
 *  falls in. Reads (does not clear) the trajectory + boundaries, so it must run
 *  before drainEpisodicSteps. */
export async function drainCandidates(
  state: SessionState,
): Promise<Candidate[]> {
  const list = (await state.get<Candidate[]>(CANDIDATES_KEY)) ?? [];
  const steps = (await state.get<StoredStep[]>(EPISODIC_KEY)) ?? [];
  const boundaries = (await state.get<number[]>(BOUNDARIES_KEY)) ?? [];
  await state.del(CANDIDATES_KEY);
  const { offset, segmentOf } = buildSegmentMap(steps.length, boundaries);
  return list
    // Drop candidates whose evidence step fell outside the trailing window.
    .filter((c) => c.stepIndex >= offset)
    .map((c) => ({ ...c, segment: segmentOf(c.stepIndex) }));
}

/** Drain the accumulated session trajectory (the episodic substrate), split into
 *  subgoal segments. Clears the trajectory + boundaries so a post-compact
 *  continuation starts fresh. Run after drainCandidates. */
export async function drainEpisodicSteps(
  state: SessionState,
): Promise<EpisodicStepRec[]> {
  const list = (await state.get<StoredStep[]>(EPISODIC_KEY)) ?? [];
  const boundaries = (await state.get<number[]>(BOUNDARIES_KEY)) ?? [];
  await state.del(EPISODIC_KEY);
  await state.del(BOUNDARIES_KEY);
  const { offset, segmentOf } = buildSegmentMap(list.length, boundaries);
  return list.slice(offset).map((s, i) => ({
    observation: s.observation,
    action: s.action,
    segment: segmentOf(offset + i),
  }));
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
