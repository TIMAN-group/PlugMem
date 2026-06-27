import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createCore } from "../src/core.js";
import { matchToolFamily } from "../src/recall.js";

interface RouteResponse {
  status: number;
  body: unknown;
}

function buildStub(
  handler: (
    method: string,
    url: string,
    body: unknown,
  ) => RouteResponse | null,
) {
  const calls: Array<{ method: string; url: string; body: unknown }> = [];
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
    calls.push({ method, url, body });
    const r = handler(method, url, body);
    if (!r) {
      return new Response(JSON.stringify({ detail: "no route" }), {
        status: 500,
      });
    }
    return new Response(JSON.stringify(r.body), {
      status: r.status,
      headers: { "Content-Type": "application/json" },
    });
  });
  return { fn, calls };
}

function nonEmptyGraphRoutes(method: string, url: string) {
  if (method === "GET" && /\/api\/v1\/graphs\/[^/]+$/.test(url)) {
    return { status: 200, body: { graph_id: "x", stats: {} } };
  }
  if (/\/stats$/.test(url)) {
    return {
      status: 200,
      body: { graph_id: "x", stats: { semantic: 3 } },
    };
  }
  return null;
}

const RETRIEVE_BODY = {
  mode: "semantic_memory",
  reasoning_prompt: [],
  variables: {
    semantic_memory: "Fact 0: project uses httpx",
    procedural_memory: "",
    episodic_memory: "",
  },
};

describe("matchToolFamily", () => {
  it("returns the query for the first matching pattern", () => {
    const map = {
      "(?i)pytest|vitest": "test setup",
      "(?i)git\\s+push": "release flow",
    };
    expect(matchToolFamily(map, "Bash")).toBeNull();
    expect(matchToolFamily(map, "PYTEST")).toBe("test setup");
    expect(matchToolFamily(map, "vitest")).toBe("test setup");
    expect(matchToolFamily(map, "git push")).toBe("release flow");
  });

  it("skips invalid regexes and continues", () => {
    const map = {
      "[invalid(": "broken",
      "ok": "fallback",
    };
    expect(matchToolFamily(map, "ok")).toBe("fallback");
  });
});

describe("session-start recall config", () => {
  let originalFetch: typeof fetch;
  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("passes min_confidence and source_in to /retrieve", async () => {
    const { fn, calls } = buildStub((method, url) => {
      const ng = nonEmptyGraphRoutes(method, url);
      if (ng) return ng;
      if (method === "POST" && /\/retrieve$/.test(url)) {
        return { status: 200, body: RETRIEVE_BODY };
      }
      return null;
    });
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: {
        minConfidence: 0.8,
        sourceIn: ["correction", "explicit"],
      },
      log: () => {},
    });
    const inj = await core.onSessionStart({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
    });
    expect(inj).not.toBeNull();
    const retrieveCall = calls.find((c) => /\/retrieve$/.test(c.url))!;
    expect(retrieveCall.body).toMatchObject({
      min_confidence: 0.8,
      source_in: ["correction", "explicit"],
    });
  });

  it("returns null when sessionStartRecall is disabled", async () => {
    const { fn, calls } = buildStub(() => null);
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: false,
      log: () => {},
    });
    const inj = await core.onSessionStart({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
    });
    expect(inj).toBeNull();
    // Should not have hit any endpoint.
    expect(calls).toHaveLength(0);
  });
});

describe("user-prompt sub-task recall", () => {
  let originalFetch: typeof fetch;
  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("recalls against the prompt for substantial messages", async () => {
    const { fn, calls } = buildStub((method, url) => {
      const ng = nonEmptyGraphRoutes(method, url);
      if (ng) return ng;
      if (method === "POST" && /\/retrieve$/.test(url)) {
        return { status: 200, body: RETRIEVE_BODY };
      }
      return null;
    });
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: false,
      log: () => {},
    });
    const inj = await core.onUserPrompt({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      prompt: "fix the bug in the auth module please",
    });
    expect(inj).not.toBeNull();
    expect(inj!.text).toContain('trigger="user-prompt"');
    const retrieveCall = calls.find((c) => /\/retrieve$/.test(c.url))!;
    expect(retrieveCall.body).toMatchObject({
      observation: "fix the bug in the auth module please",
    });
  });

  it("skips recall for short prompts (yes/ok/etc)", async () => {
    const { fn, calls } = buildStub(() => null);
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: false,
      log: () => {},
    });
    const inj = await core.onUserPrompt({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      prompt: "yes",
    });
    expect(inj).toBeNull();
    expect(calls).toHaveLength(0);
  });

  it("respects minPromptLength override", async () => {
    const { fn, calls } = buildStub((method, url) => {
      const ng = nonEmptyGraphRoutes(method, url);
      if (ng) return ng;
      if (method === "POST" && /\/retrieve$/.test(url)) {
        return { status: 200, body: RETRIEVE_BODY };
      }
      return null;
    });
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: false,
      userPromptRecall: { minPromptLength: 5 },
      log: () => {},
    });
    const inj = await core.onUserPrompt({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      prompt: "longer prompt",
    });
    expect(inj).not.toBeNull();
    expect(calls.find((c) => /\/retrieve$/.test(c.url))).toBeDefined();
  });

  it("returns null when userPromptRecall is disabled", async () => {
    const { fn, calls } = buildStub(() => null);
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: false,
      userPromptRecall: false,
      log: () => {},
    });
    const inj = await core.onUserPrompt({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      prompt: "this is a substantial prompt that should trigger",
    });
    expect(inj).toBeNull();
    expect(calls).toHaveLength(0);
  });
});

describe("tool-family pre-recall", () => {
  let originalFetch: typeof fetch;
  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("is off by default", async () => {
    const { fn, calls } = buildStub(() => null);
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: false,
      log: () => {},
    });
    const inj = await core.onPreTool({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      toolName: "Bash",
      toolInput: { command: "pytest" },
      callId: "c1",
    });
    expect(inj).toBeNull();
    expect(calls).toHaveLength(0);
  });

  it("fires recall when toolName matches a configured pattern", async () => {
    const { fn, calls } = buildStub((method, url) => {
      const ng = nonEmptyGraphRoutes(method, url);
      if (ng) return ng;
      if (method === "POST" && /\/retrieve$/.test(url)) {
        return { status: 200, body: RETRIEVE_BODY };
      }
      return null;
    });
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: false,
      toolFamilyRecall: {
        enabled: true,
        mapping: { "(?i)pytest|vitest": "test setup, fixtures" },
      },
      log: () => {},
    });
    const inj = await core.onPreTool({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      toolName: "pytest",
      toolInput: {},
      callId: "c1",
    });
    expect(inj).not.toBeNull();
    expect(inj!.text).toContain('trigger="tool:pytest"');
    const retrieveCall = calls.find((c) => /\/retrieve$/.test(c.url))!;
    expect(retrieveCall.body).toMatchObject({
      observation: "test setup, fixtures",
    });
  });

  it("returns null for non-matching tool names", async () => {
    const { fn, calls } = buildStub(() => null);
    globalThis.fetch = fn as unknown as typeof fetch;

    const core = createCore({
      config: { baseUrl: "http://stub" },
      sessionStartRecall: false,
      toolFamilyRecall: {
        enabled: true,
        mapping: { "(?i)pytest": "test setup" },
      },
      log: () => {},
    });
    const inj = await core.onPreTool({
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp/repo",
      toolName: "Read",
      toolInput: {},
      callId: "c1",
    });
    expect(inj).toBeNull();
    expect(calls).toHaveLength(0);
  });
});
