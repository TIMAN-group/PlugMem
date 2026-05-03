import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createCore } from "../src/core.js";
import { PlugMemClient } from "../src/client.js";
import {
  recordPostTool,
  recordPreTool,
  recordUserPrompt,
} from "../src/promotion.js";
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

  it("skips insert when /extract returns []", async () => {
    const inserts: unknown[] = [];
    const wrapped = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (method === "POST" && /\/api\/v1\/extract$/.test(url)) {
        return new Response(
          JSON.stringify({ memories: [] }),
          { status: 200 },
        );
      }
      if (method === "POST" && /\/memories$/.test(url)) {
        inserts.push(true);
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const sessionState = new MapState();
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

    expect(inserts).toHaveLength(0);
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
