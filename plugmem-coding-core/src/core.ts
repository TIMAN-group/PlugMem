import type {
  ContextInjection,
  CoreCallbacks,
  PostToolEvent,
  PreCompactEvent,
  PreToolEvent,
  SessionEndEvent,
  SessionStartEvent,
  SessionState,
  UserPromptEvent,
} from "./adapter.js";
import { PlugMemClient } from "./client.js";
import {
  type CoreConfig,
  type ResolvedCoreConfig,
  resolveConfig,
} from "./config.js";
import {
  drainCandidates,
  recordPostTool,
  recordPreTool,
  recordUserPrompt,
  type Candidate,
} from "./promotion.js";
import {
  doRecall,
  matchToolFamily,
  resolveSessionStartRecall,
  resolveToolFamilyRecall,
  resolveUserPromptRecall,
  type SessionStartRecallConfig,
  type ToolFamilyRecallConfig,
  type UserPromptRecallConfig,
} from "./recall.js";
import { deriveRepoGraphId } from "./repo_id.js";
import {
  PlugMemError,
  type ExtractedMemory,
  type ProceduralMemoryInput,
  type SemanticMemoryInput,
} from "./types.js";

export interface CreateCoreOptions {
  config: CoreConfig;
  /** Session-start recall: top-K facts injected at session start.
   *  Pass `false` to disable. Default: enabled, char cap 2000. */
  sessionStartRecall?: SessionStartRecallConfig | false;
  /** Sub-task recall on each substantial user prompt.
   *  Pass `false` to disable. Default: enabled, char cap 1000. */
  userPromptRecall?: UserPromptRecallConfig | false;
  /** Tool-family pre-recall (e.g., before `pytest`).
   *  Default: disabled — set `enabled: true` and provide a `mapping`. */
  toolFamilyRecall?: ToolFamilyRecallConfig | false;
  /**
   * Cross-event SessionState. Required for promotion-gate detectors
   * (failure-delta, correction). When omitted, those detectors no-op
   * silently — Stage 1 behavior.
   */
  state?: (sessionId: string) => SessionState;
  /** Optional injection: caller can swap in a custom client (testing). */
  client?: PlugMemClient;
  /** Optional logger. Default writes to stderr; pass () => {} to silence. */
  log?: (msg: string, err?: unknown) => void;
}

interface Runtime {
  resolved: ResolvedCoreConfig;
  client: PlugMemClient;
  sessionStartRecall: ReturnType<typeof resolveSessionStartRecall>;
  userPromptRecall: ReturnType<typeof resolveUserPromptRecall>;
  toolFamilyRecall: ReturnType<typeof resolveToolFamilyRecall>;
  state: ((sessionId: string) => SessionState) | undefined;
  log: (msg: string, err?: unknown) => void;
}

export function createCore(opts: CreateCoreOptions): CoreCallbacks {
  const resolved = resolveConfig(opts.config);
  const client = opts.client ?? new PlugMemClient(resolved);
  const log =
    opts.log ??
    ((msg, err) => {
      const tail = err
        ? ` ${err instanceof Error ? err.message : String(err)}`
        : "";
      process.stderr.write(`[plugmem-coding-core] ${msg}${tail}\n`);
    });

  const runtime: Runtime = {
    resolved,
    client,
    sessionStartRecall: resolveSessionStartRecall(opts.sessionStartRecall),
    userPromptRecall: resolveUserPromptRecall(opts.userPromptRecall),
    toolFamilyRecall: resolveToolFamilyRecall(opts.toolFamilyRecall),
    state: opts.state,
    log,
  };

  return {
    onSessionStart: (e) => handleSessionStart(runtime, e),
    onUserPrompt: (e) => handleUserPrompt(runtime, e),
    onPreTool: (e) => handlePreTool(runtime, e),
    onPostTool: (e) => handlePostTool(runtime, e),
    onPreCompact: (e) => handlePreCompact(runtime, e),
    onSessionEnd: (e) => handleSessionEnd(runtime, e),
  };
}

// ── Session start ────────────────────────────────────────────────────

async function handleSessionStart(
  runtime: Runtime,
  event: SessionStartEvent,
): Promise<ContextInjection | null> {
  const cfg = runtime.sessionStartRecall;
  if (!cfg || !cfg.enabled) return null;

  const graphId = await deriveRepoGraphId(event.harness, event.cwd);

  if (!(await ensureNonEmptyGraph(runtime, graphId))) return null;

  return doRecall({
    client: runtime.client,
    graphId,
    observation: cfg.query,
    minConfidence: cfg.minConfidence,
    sourceIn: cfg.sourceIn,
    charCap: cfg.charCap,
    blockTitle: "session-start",
    log: runtime.log,
  });
}

// ── User prompt: promotion-gate record + sub-task recall ─────────────

async function handleUserPrompt(
  runtime: Runtime,
  event: UserPromptEvent,
): Promise<ContextInjection | null> {
  // 1. Promotion-gate side effect — independent of recall.
  const state = runtime.state?.(event.sessionId);
  if (state) {
    try {
      await recordUserPrompt(state, event);
    } catch (err) {
      runtime.log(`recordUserPrompt failed`, err);
    }
  }

  // 2. Sub-task recall.
  const cfg = runtime.userPromptRecall;
  if (!cfg || !cfg.enabled) return null;
  const prompt = event.prompt?.trim() ?? "";
  if (prompt.length < cfg.minPromptLength) return null;

  const graphId = await deriveRepoGraphId(event.harness, event.cwd);
  if (!(await ensureNonEmptyGraph(runtime, graphId))) return null;

  return doRecall({
    client: runtime.client,
    graphId,
    observation: prompt,
    minConfidence: cfg.minConfidence,
    sourceIn: cfg.sourceIn,
    charCap: cfg.charCap,
    blockTitle: "user-prompt",
    log: runtime.log,
  });
}

// ── Pre-tool: detector record + tool-family recall ───────────────────

async function handlePreTool(
  runtime: Runtime,
  event: PreToolEvent,
): Promise<ContextInjection | null> {
  // 1. Promotion-gate state record.
  const state = runtime.state?.(event.sessionId);
  if (state) {
    try {
      await recordPreTool(state, event);
    } catch (err) {
      runtime.log(`recordPreTool failed`, err);
    }
  }

  // 2. Tool-family recall.
  const cfg = runtime.toolFamilyRecall;
  if (!cfg || !cfg.enabled) return null;
  const query = matchToolFamily(cfg.mapping, event.toolName);
  if (!query) return null;

  const graphId = await deriveRepoGraphId(event.harness, event.cwd);
  if (!(await ensureNonEmptyGraph(runtime, graphId))) return null;

  return doRecall({
    client: runtime.client,
    graphId,
    observation: query,
    minConfidence: cfg.minConfidence,
    sourceIn: cfg.sourceIn,
    charCap: cfg.charCap,
    blockTitle: `tool:${event.toolName}`,
    log: runtime.log,
  });
}

// ── Post-tool: promotion detector only (no injection possible) ───────

async function handlePostTool(
  runtime: Runtime,
  event: PostToolEvent,
): Promise<void> {
  const state = runtime.state?.(event.sessionId);
  if (!state) return;
  try {
    await recordPostTool(state, event);
  } catch (err) {
    runtime.log(`recordPostTool failed`, err);
  }
}

// ── Session end / pre-compact: drain promotion candidates ────────────

async function handleSessionEnd(
  runtime: Runtime,
  event: SessionEndEvent,
): Promise<void> {
  await runPromotionGate(runtime, event.harness, event.cwd, event.sessionId);
}

async function handlePreCompact(
  runtime: Runtime,
  event: PreCompactEvent,
): Promise<void> {
  await runPromotionGate(runtime, event.harness, event.cwd, event.sessionId);
}

async function runPromotionGate(
  runtime: Runtime,
  harness: SessionEndEvent["harness"],
  cwd: string,
  sessionId: string,
): Promise<void> {
  const state = runtime.state?.(sessionId);
  if (!state) {
    return;
  }

  let candidates: Candidate[];
  try {
    candidates = await drainCandidates(state);
  } catch (err) {
    runtime.log(`drainCandidates failed`, err);
    return;
  }
  if (candidates.length === 0) return;

  let extracted: ExtractedMemory[];
  try {
    const r = await runtime.client.extract({
      candidates: candidates.map((c) => ({ kind: c.kind, window: c.window })),
    });
    extracted = r.memories;
  } catch (err) {
    runtime.log(`extract call failed`, err);
    return;
  }
  if (extracted.length === 0) {
    runtime.log(
      `promotion-gate: ${candidates.length} candidate(s) → 0 memories`,
    );
    return;
  }

  const semantic: SemanticMemoryInput[] = [];
  const procedural: ProceduralMemoryInput[] = [];
  for (const m of extracted) {
    if (m.type === "semantic") {
      semantic.push({
        semantic_memory: m.semantic_memory,
        tags: m.tags ?? [],
        source: m.source,
        confidence: m.confidence,
      });
    } else {
      procedural.push({
        subgoal: m.subgoal,
        procedural_memory: m.procedural_memory,
        source: m.source,
        confidence: m.confidence,
      });
    }
  }

  const graphId = await deriveRepoGraphId(harness, cwd);
  try {
    await runtime.client.ensureGraph(graphId);
    await runtime.client.insertMemories(graphId, {
      mode: "structured",
      ...(semantic.length ? { semantic } : {}),
      ...(procedural.length ? { procedural } : {}),
    });
    runtime.log(
      `promotion-gate: inserted ${semantic.length} semantic + ${procedural.length} procedural into ${graphId}`,
    );
  } catch (err) {
    if (err instanceof PlugMemError) {
      runtime.log(`insert into ${graphId} failed (${err.statusCode})`, err);
    } else {
      runtime.log(`insert into ${graphId} failed`, err);
    }
  }
}

// ── Helpers ──────────────────────────────────────────────────────────

async function ensureNonEmptyGraph(
  runtime: Runtime,
  graphId: string,
): Promise<boolean> {
  try {
    await runtime.client.ensureGraph(graphId);
  } catch (err) {
    runtime.log(`ensureGraph(${graphId}) failed`, err);
    return false;
  }
  try {
    const stats = await runtime.client.getStats(graphId);
    const total =
      (stats.stats.semantic ?? 0) +
      (stats.stats.procedural ?? 0) +
      (stats.stats.episodic ?? 0);
    return total > 0;
  } catch (err) {
    runtime.log(`getStats(${graphId}) failed`, err);
    return false;
  }
}
