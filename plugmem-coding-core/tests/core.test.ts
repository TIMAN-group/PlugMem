import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createCore } from "../src/core.js";
import { PlugMemClient } from "../src/client.js";
import {
  recordPostTool,
  recordPreTool,
  recordUserPrompt,
} from "../src/promotion.js";
import type { PostToolEvent, PreToolEvent } from "../src/adapter.js";
import type { SessionState } from "../src/adapter.js";

class MapState implements SessionState {
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

interface MockResponse {
  status: number;
  body: unknown;
}

function makeFetchStub(routes: Array<[RegExp | string, MockResponse]>) {
  const calls: Array<{ url: string; method: string; body: unknown }> = [];
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    let body: unknown = undefined;
    if (init?.body && typeof init.body === "string") {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    }
    calls.push({ url, method, body });

    for (const [matcher, response] of routes) {
      const matched =
        typeof matcher === "string"
          ? url.endsWith(matcher)
          : matcher.test(url);
      if (matched) {
        return new Response(JSON.stringify(response.body), {
          status: response.status,
          headers: { "Content-Type": "application/json" },
        });
      }
    }
    return new Response(JSON.stringify({ detail: "no route" }), {
      status: 500,
    });
  });
  return { fn, calls };
}

describe("createCore — session_start", () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("returns null injection when graph is empty", async () => {
    const { fn } = makeFetchStub([
      [/\/api\/v1\/graphs\/[^/]+$/, { status: 200, body: { graph_id: "x", stats: {} } }],
      [/\/stats$/, { status: 200, body: { graph_id: "x", stats: { semantic: 0 } } }],
    ]);
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      log: () => {},
    });
    const inj = await core.onSessionStart({
      harness: "claude-code",
      sessionId: "s1",
      cwd: "/tmp/no-such-repo-" + Math.random(),
    });
    expect(inj).toBeNull();
  });

  it("returns formatted injection when graph has memories", async () => {
    const { fn } = makeFetchStub([
      [/\/api\/v1\/graphs\/[^/]+$/, { status: 200, body: { graph_id: "x", stats: {} } }],
      [/\/stats$/, { status: 200, body: { graph_id: "x", stats: { semantic: 5 } } }],
      [
        /\/retrieve$/,
        {
          status: 200,
          body: {
            mode: "semantic_memory",
            reasoning_prompt: [],
            variables: {
              semantic_memory:
                "Fact 0: tests live in tests/, run with vitest",
              procedural_memory: "",
              episodic_memory: "",
            },
          },
        },
      ],
    ]);
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      log: () => {},
    });
    const inj = await core.onSessionStart({
      harness: "claude-code",
      sessionId: "s1",
      cwd: "/tmp/repo",
    });
    expect(inj).not.toBeNull();
    expect(inj!.role).toBe("system");
    expect(inj!.text).toContain("plugmem-recall");
    expect(inj!.text).toContain('trigger="session-start"');
    expect(inj!.text).toContain("vitest");
  });
});

describe("createCore — session_end without state", () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("is a no-op when no state factory is configured", async () => {
    const wrapped = vi.fn(
      async () =>
        new Response(JSON.stringify({}), { status: 200 }),
    );
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub", maxRetries: 0 },
      log: () => {},
    });
    await core.onSessionEnd({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      reason: "session_end",
    });
    // No HTTP calls — the only thing the runner could do is /extract or
    // /memories, and without state there are no candidates to drain.
    expect(wrapped).not.toHaveBeenCalled();
  });
});

describe("createCore — promotion gate at session_end", () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("calls /extract and inserts returned memories", async () => {
    const inserts: unknown[] = [];
    const wrapped = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (method === "POST" && /\/api\/v1\/extract$/.test(url)) {
        return new Response(
          JSON.stringify({
            memories: [
              {
                type: "semantic",
                semantic_memory: "Use httpx, not requests",
                tags: ["python", "convention"],
                source: "correction",
                confidence: 0.9,
              },
              {
                type: "procedural",
                subgoal: "fix import error in tests",
                procedural_memory: "pip install -e . then pytest",
                source: "failure_delta",
                confidence: 0.7,
              },
            ],
          }),
          { status: 200 },
        );
      }
      if (method === "GET" && /\/api\/v1\/graphs\/[^/]+$/.test(url)) {
        return new Response(
          JSON.stringify({ graph_id: "x", stats: {} }),
          { status: 200 },
        );
      }
      if (method === "POST" && /\/memories$/.test(url)) {
        inserts.push(JSON.parse(init!.body as string));
        return new Response(
          JSON.stringify({ status: "ok", stats: {} }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const sessionState = new MapState();
    // Pre-seed a correction candidate.
    await recordUserPrompt(sessionState, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      prompt: "actually, use httpx",
    });

    const core = createCore({
      config: { baseUrl: "http://stub", maxRetries: 0 },
      state: () => sessionState,
      log: () => {},
    });
    await core.onSessionEnd({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      reason: "session_end",
    });

    expect(inserts).toHaveLength(1);
    const ins = inserts[0] as {
      mode: string;
      semantic?: unknown[];
      procedural?: unknown[];
    };
    expect(ins.mode).toBe("structured");
    expect(ins.semantic).toBeDefined();
    expect(ins.semantic).toHaveLength(1);
    expect(ins.procedural).toBeDefined();
    expect(ins.procedural).toHaveLength(1);
  });

  it("filters out episodic candidates before /extract (no batch poisoning)", async () => {
    const { fn, calls } = makeFetchStub([
      [
        /\/api\/v1\/extract$/,
        {
          status: 200,
          body: {
            memories: [
              {
                type: "semantic",
                semantic_memory: "use httpx",
                tags: [],
                source: "correction",
                confidence: 0.9,
              },
            ],
          },
        },
      ],
      [/\/api\/v1\/graphs\/[^/]+$/, { status: 200, body: { graph_id: "x", stats: {} } }],
      [/\/memories$/, { status: 200, body: { status: "ok", stats: {} } }],
    ]);
    globalThis.fetch = fn as unknown as typeof fetch;

    const sessionState = new MapState();
    // A correction (sendable) and a completion phrase (episodic — must be dropped).
    await recordUserPrompt(sessionState, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      prompt: "actually, use httpx",
    });
    await recordUserPrompt(sessionState, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      prompt: "perfect, it works",
    });

    const core = createCore({
      config: { baseUrl: "http://stub", maxRetries: 0 },
      state: () => sessionState,
      log: () => {},
    });
    await core.onSessionEnd({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      reason: "session_end",
    });

    const extractCall = calls.find(
      (c) => c.method === "POST" && /\/api\/v1\/extract$/.test(c.url),
    );
    // /extract is still called — the episodic candidate must NOT block the batch.
    expect(extractCall).toBeDefined();
    const sentKinds = (extractCall!.body as { candidates: Array<{ kind: string }> })
      .candidates.map((c) => c.kind);
    expect(sentKinds).toContain("correction");
    expect(sentKinds).not.toContain("episodic");
    expect(sentKinds).toHaveLength(1);
  });

  it("inserts the episodic substrate (session-stamped) even when /extract returns []", async () => {
    const inserts: any[] = [];
    const wrapped = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (method === "POST" && /\/api\/v1\/extract$/.test(url)) {
        return new Response(JSON.stringify({ memories: [] }), { status: 200 });
      }
      if (method === "POST" && /\/memories$/.test(url)) {
        inserts.push(JSON.parse(init!.body as string));
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const sessionState = new MapState();
    // A user prompt records an episodic step (trajectory) + a correction candidate.
    await recordUserPrompt(sessionState, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      prompt: "stop doing that",
    });

    const core = createCore({
      config: { baseUrl: "http://stub", maxRetries: 0 },
      state: () => sessionState,
      log: () => {},
    });
    await core.onSessionEnd({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      reason: "session_end",
    });

    // /extract found nothing promotable, but the episodic substrate is still
    // inserted (stamped with the session id) with no semantic/procedural.
    expect(inserts).toHaveLength(1);
    const ins = inserts[0] as {
      mode: string;
      session_id?: string;
      episodic?: unknown[][];
      semantic?: unknown[];
      procedural?: unknown[];
    };
    expect(ins.mode).toBe("structured");
    expect(ins.session_id).toBe("s");
    expect(ins.episodic?.[0]?.length).toBe(1);
    expect(ins.semantic).toBeUndefined();
    expect(ins.procedural).toBeUndefined();
  });

  it("segments the trajectory and grounds procedural on its own segment", async () => {
    const inserts: any[] = [];
    const extractBodies: any[] = [];
    const wrapped = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (method === "POST" && /\/api\/v1\/extract$/.test(url)) {
        extractBodies.push(JSON.parse(init!.body as string));
        return new Response(
          JSON.stringify({
            memories: [
              {
                type: "procedural",
                subgoal: "fix the import error",
                procedural_memory: "pip install -e . before running tests",
                source: "failure_delta",
                confidence: 0.8,
              },
            ],
          }),
          { status: 200 },
        );
      }
      if (method === "POST" && /\/memories$/.test(url)) {
        inserts.push(JSON.parse(init!.body as string));
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const state = new MapState();
    const sid = "s";
    const cwd = "/tmp/repo";
    const pre = (toolInput: unknown, callId: string): PreToolEvent => ({
      harness: "claude-code", sessionId: sid, cwd, toolName: "Bash", toolInput, callId,
    });
    const post = (
      toolInput: unknown, callId: string, toolResult: string,
      outcome: PostToolEvent["outcome"],
    ): PostToolEvent => ({
      harness: "claude-code", sessionId: sid, cwd, toolName: "Bash",
      toolInput, callId, toolResult, outcome,
    });

    // Segment 0: request + a failing step.
    await recordUserPrompt(state, { harness: "claude-code", sessionId: sid, cwd, prompt: "add a logger" });
    await recordPreTool(state, pre({ command: "x" }, "c1"));
    await recordPostTool(state, post({ command: "x" }, "c1", "ImportError", "failure"));
    // Segment 1: request + the resolving success (failure_delta extracted here).
    await recordUserPrompt(state, { harness: "claude-code", sessionId: sid, cwd, prompt: "now fix the import" });
    await recordPreTool(state, pre({ command: "pip install -e ." }, "c2"));
    await recordPostTool(state, post({ command: "pip install -e ." }, "c2", "PASSED", "success"));

    const core = createCore({
      config: { baseUrl: "http://stub", maxRetries: 0 },
      state: () => state,
      log: () => {},
    });
    await core.onSessionEnd({ harness: "claude-code", sessionId: sid, cwd, reason: "session_end" });

    // One extract call (only segment 1 had a sendable candidate) and one insert.
    expect(extractBodies).toHaveLength(1);
    expect(inserts).toHaveLength(1);
    const ins = inserts[0] as {
      episodic: unknown[][];
      procedural: Array<{ trajectory_num?: number }>;
    };
    // Two episodic segments, two steps each (prompt + tool step).
    expect(ins.episodic).toHaveLength(2);
    expect(ins.episodic[0]).toHaveLength(2);
    expect(ins.episodic[1]).toHaveLength(2);
    // The procedural is grounded on segment 1 — the split it came from.
    expect(ins.procedural).toHaveLength(1);
    expect(ins.procedural[0].trajectory_num).toBe(1);
  });

  it("splits a single turn into segments at failure→fix boundaries", async () => {
    // One user prompt, two distinct fix cycles. Each procedural must ground on
    // its own sub-sequence — the split happens *within* a turn, which is what
    // surfaces under OpenCode's per-turn (session.idle) drain.
    const inserts: any[] = [];
    const extractBodies: any[] = [];
    let n = 0;
    const wrapped = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (method === "POST" && /\/api\/v1\/extract$/.test(url)) {
        extractBodies.push(JSON.parse(init!.body as string));
        n += 1;
        return new Response(
          JSON.stringify({
            memories: [
              {
                type: "procedural",
                subgoal: `subgoal ${n}`,
                procedural_memory: `recipe ${n}`,
                source: "failure_delta",
                confidence: 0.7,
              },
            ],
          }),
          { status: 200 },
        );
      }
      if (method === "POST" && /\/memories$/.test(url)) {
        inserts.push(JSON.parse(init!.body as string));
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const state = new MapState();
    const sid = "s";
    const cwd = "/tmp/repo";
    const pre = (callId: string): PreToolEvent => ({
      harness: "claude-code", sessionId: sid, cwd, toolName: "Bash", toolInput: {}, callId,
    });
    const post = (callId: string, r: string, o: PostToolEvent["outcome"]): PostToolEvent => ({
      harness: "claude-code", sessionId: sid, cwd, toolName: "Bash", toolInput: {}, callId, toolResult: r, outcome: o,
    });

    await recordUserPrompt(state, { harness: "claude-code", sessionId: sid, cwd, prompt: "set up the project" });
    // Cycle 1
    await recordPreTool(state, pre("a1"));
    await recordPostTool(state, post("a1", "boom", "failure"));
    await recordPreTool(state, pre("a2"));
    await recordPostTool(state, post("a2", "ok", "success"));
    // Cycle 2
    await recordPreTool(state, pre("b1"));
    await recordPostTool(state, post("b1", "boom again", "failure"));
    await recordPreTool(state, pre("b2"));
    await recordPostTool(state, post("b2", "ok", "success"));

    const core = createCore({
      config: { baseUrl: "http://stub", maxRetries: 0 },
      state: () => state,
      log: () => {},
    });
    await core.onSessionEnd({ harness: "claude-code", sessionId: sid, cwd, reason: "session_end" });

    // Two extract calls (one per segment) and one insert.
    expect(extractBodies).toHaveLength(2);
    expect(inserts).toHaveLength(1);
    const ins = inserts[0] as {
      episodic: unknown[][];
      procedural: Array<{ trajectory_num?: number }>;
    };
    // Segment 0 = prompt + cycle-1 (3 steps); segment 1 = cycle-2 (2 steps).
    expect(ins.episodic).toHaveLength(2);
    expect(ins.episodic[0]).toHaveLength(3);
    expect(ins.episodic[1]).toHaveLength(2);
    // Two procedurals, grounded on segments 0 and 1 respectively.
    expect(ins.procedural).toHaveLength(2);
    expect(ins.procedural.map((p) => p.trajectory_num).sort()).toEqual([0, 1]);
  });

  it("does not call /extract when there are no candidates", async () => {
    const wrapped = vi.fn(
      async () => new Response("{}", { status: 200 }),
    );
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub", maxRetries: 0 },
      state: () => new MapState(),
      log: () => {},
    });
    await core.onSessionEnd({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      reason: "session_end",
    });
    expect(wrapped).not.toHaveBeenCalled();
  });
});

describe("createCore — preserves PlugMemClient construction", () => {
  let originalFetch: typeof fetch;
  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("accepts an injected client", async () => {
    const wrapped = vi.fn(
      async () =>
        new Response(
          JSON.stringify({ graph_id: "x", stats: { semantic: 0 } }),
          { status: 200 },
        ),
    );
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const client = new PlugMemClient({
      baseUrl: "http://stub",
      timeoutMs: 1000,
      maxRetries: 0,
    });

    const core = createCore({
      config: { baseUrl: "http://stub" },
      client,
      log: () => {},
    });
    const inj = await core.onSessionStart({
      harness: "claude-code",
      sessionId: "s1",
      cwd: "/tmp/x",
    });
    expect(inj).toBeNull();
  });
});
