// Session runner: walks a fixture's events through real `createCore`
// against a live PlugMem service. Captures per-event outcomes and
// per-session stats.
//
// State backend: in-memory Map (per-fixture-run) — we don't need
// disk-backed state here since the runner is a single long-lived
// process. The CC adapter's DiskSessionState exists because hooks run
// as isolated processes, which the eval harness deliberately doesn't.

import { createCore } from "../core.js";
import type { CoreCallbacks } from "../adapter.js";
import { PlugMemClient } from "../client.js";
import { resolveConfig } from "../config.js";
import type { SessionState } from "../adapter.js";
import type {
  EvalEvent,
  EvalEventResult,
  EvalFixture,
  EvalRunReport,
  EvalSession,
  EvalSessionResult,
} from "./types.js";

class MemoryState implements SessionState {
  private store = new Map<string, unknown>();
  async get<T>(key: string): Promise<T | undefined> {
    if (!this.store.has(key)) return undefined;
    return JSON.parse(JSON.stringify(this.store.get(key))) as T;
  }
  async set<T>(key: string, value: T): Promise<void> {
    this.store.set(key, JSON.parse(JSON.stringify(value)));
  }
  async del(key: string): Promise<void> {
    this.store.delete(key);
  }
}

export interface RunFixtureOptions {
  /** Truncate the graph (delete + recreate) before running. Default true.
   *  Set false to accumulate across multiple runs of the same fixture. */
  resetGraph?: boolean;
  /** Optional log sink; defaults to stderr writes prefixed with [eval]. */
  log?: (msg: string) => void;
}

export async function runFixture(
  fixture: EvalFixture,
  opts: RunFixtureOptions = {},
): Promise<EvalRunReport> {
  const log =
    opts.log ?? ((msg) => process.stderr.write(`[eval] ${msg}\n`));

  const cfg = resolveConfig({
    baseUrl: fixture.baseUrl,
    apiKey: fixture.apiKey,
  });
  const client = new PlugMemClient(cfg);

  // Reset state per session id (so failure-pairing windows don't leak).
  const stateBySession = new Map<string, SessionState>();
  const stateFactory = (sessionId: string): SessionState => {
    let s = stateBySession.get(sessionId);
    if (!s) {
      s = new MemoryState();
      stateBySession.set(sessionId, s);
    }
    return s;
  };

  // Translate fixture recall → CreateCoreOptions shape (just a passthrough,
  // but TS wants us to be explicit about the discriminated unions).
  const core: CoreCallbacks = createCore({
    config: { baseUrl: fixture.baseUrl, apiKey: fixture.apiKey },
    state: stateFactory,
    sessionStartRecall: fixture.recall?.sessionStartRecall,
    userPromptRecall: fixture.recall?.userPromptRecall,
    toolFamilyRecall: fixture.recall?.toolFamilyRecall,
    log: () => {}, // silence core's stderr; runner has its own log.
  });

  // Reset graph if asked.
  if (opts.resetGraph !== false) {
    try {
      await client.deleteGraph(fixture.graphId);
    } catch {
      // best-effort — graph may not exist
    }
  }
  await client.ensureGraph(fixture.graphId);

  const startedAt = new Date().toISOString();
  const sessions: EvalSessionResult[] = [];

  let totalPromotions = 0;
  let totalExposedChars = 0;

  for (const session of fixture.sessions) {
    log(`session ${session.id}: ${session.events.length} event(s)`);
    const beforeStats = await safeGetStats(client, fixture.graphId);

    const eventResults: EvalEventResult[] = [];
    let exposedChars = 0;

    for (const ev of session.events) {
      const r = await runEvent(core, fixture, session, ev);
      if (r.injection) exposedChars += r.injection.charCount;
      eventResults.push(r);
    }

    const afterStats = await safeGetStats(client, fixture.graphId);
    const promotions = countMemoryDelta(beforeStats, afterStats);

    sessions.push({
      id: session.id,
      events: eventResults,
      statsAfter: afterStats,
      promotions,
      exposedChars,
    });
    totalPromotions += promotions;
    totalExposedChars += exposedChars;
  }

  const finalStats = await safeGetStats(client, fixture.graphId);

  return {
    fixture: fixture.name,
    startedAt,
    finishedAt: new Date().toISOString(),
    sessions,
    totalPromotions,
    totalExposedChars,
    finalStats,
  };
}

async function runEvent(
  core: CoreCallbacks,
  fixture: EvalFixture,
  session: EvalSession,
  ev: EvalEvent,
): Promise<EvalEventResult> {
  const sid = String(session.id);
  const base = { type: ev.type, label: ev.label };

  try {
    switch (ev.type) {
      case "session_start": {
        const inj = await core.onSessionStart({
          harness: fixture.harness,
          sessionId: sid,
          cwd: fixture.cwd,
          source: ev.source,
        });
        return inj
          ? { ...base, injection: { text: inj.text, charCount: inj.text.length } }
          : { ...base };
      }
      case "user_prompt": {
        const inj = await core.onUserPrompt({
          harness: fixture.harness,
          sessionId: sid,
          cwd: fixture.cwd,
          prompt: ev.prompt,
        });
        return inj
          ? { ...base, injection: { text: inj.text, charCount: inj.text.length } }
          : { ...base };
      }
      case "pre_tool": {
        const inj = await core.onPreTool({
          harness: fixture.harness,
          sessionId: sid,
          cwd: fixture.cwd,
          toolName: ev.toolName,
          toolInput: ev.toolInput,
          callId: ev.callId,
        });
        return inj
          ? { ...base, injection: { text: inj.text, charCount: inj.text.length } }
          : { ...base };
      }
      case "post_tool": {
        await core.onPostTool({
          harness: fixture.harness,
          sessionId: sid,
          cwd: fixture.cwd,
          toolName: ev.toolName,
          toolInput: ev.toolInput,
          callId: ev.callId,
          toolResult: ev.toolResult,
          outcome: ev.outcome,
        });
        return { ...base };
      }
      case "pre_compact": {
        await core.onPreCompact({
          harness: fixture.harness,
          sessionId: sid,
          cwd: fixture.cwd,
        });
        return { ...base };
      }
      case "session_end": {
        await core.onSessionEnd({
          harness: fixture.harness,
          sessionId: sid,
          cwd: fixture.cwd,
          reason: ev.reason ?? "session_end",
        });
        return { ...base };
      }
    }
  } catch (err) {
    return {
      ...base,
      note: `error: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

async function safeGetStats(
  client: PlugMemClient,
  graphId: string,
): Promise<Record<string, number>> {
  try {
    const r = await client.getStats(graphId);
    return r.stats;
  } catch {
    return {};
  }
}

function countMemoryDelta(
  before: Record<string, number>,
  after: Record<string, number>,
): number {
  const keys = ["semantic", "procedural", "episodic"];
  let delta = 0;
  for (const k of keys) {
    delta += Math.max(0, (after[k] ?? 0) - (before[k] ?? 0));
  }
  return delta;
}
