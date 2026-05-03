import { describe, expect, it } from "vitest";

import {
  drainCandidates,
  matchesCorrectionPattern,
  recordPostTool,
  recordPreTool,
  recordUserPrompt,
} from "../src/promotion.js";
import type { SessionState } from "../src/adapter.js";

class MapState implements SessionState {
  private store = new Map<string, unknown>();

  async get<T>(key: string): Promise<T | undefined> {
    if (!this.store.has(key)) return undefined;
    // Round-trip through JSON to mimic disk-backed semantics (no shared refs).
    return JSON.parse(JSON.stringify(this.store.get(key))) as T;
  }
  async set<T>(key: string, value: T): Promise<void> {
    this.store.set(key, JSON.parse(JSON.stringify(value)));
  }
  async del(key: string): Promise<void> {
    this.store.delete(key);
  }
}

describe("matchesCorrectionPattern", () => {
  it("matches common negation phrases", () => {
    expect(matchesCorrectionPattern("don't use requests")).toBe(true);
    expect(matchesCorrectionPattern("Stop, that's wrong")).toBe(true);
    expect(matchesCorrectionPattern("actually we use httpx")).toBe(true);
    expect(matchesCorrectionPattern("instead of pip, use uv")).toBe(true);
    expect(matchesCorrectionPattern("we use the new API")).toBe(true);
    expect(matchesCorrectionPattern("not foo, but bar")).toBe(true);
    expect(matchesCorrectionPattern("the right way is to call X")).toBe(true);
    expect(matchesCorrectionPattern("should be httpx")).toBe(true);
  });

  it("does not match neutral prompts", () => {
    expect(matchesCorrectionPattern("add a test for the new feature")).toBe(
      false,
    );
    expect(matchesCorrectionPattern("can you summarize")).toBe(false);
    expect(matchesCorrectionPattern("")).toBe(false);
  });
});

describe("failure-delta detector", () => {
  it("emits a candidate when failure is followed by success on same tool", async () => {
    const state = new MapState();
    const sid = "s1";
    const cwd = "/tmp/x";

    await recordPreTool(state, {
      harness: "claude-code",
      sessionId: sid,
      cwd,
      toolName: "Bash",
      toolInput: { command: "pytest" },
      callId: "c1",
    });
    await recordPostTool(state, {
      harness: "claude-code",
      sessionId: sid,
      cwd,
      toolName: "Bash",
      toolInput: { command: "pytest" },
      callId: "c1",
      toolResult: "ImportError",
      outcome: "failure",
    });

    // No candidate yet — only failure recorded.
    let candidates = await drainCandidates(state);
    expect(candidates).toHaveLength(0);
    // Restore the failure (drain emptied it... but failures live under a
    // different key, so drainCandidates didn't touch them).
    // Actually drainCandidates only drains the candidates list, not failures.

    await recordPreTool(state, {
      harness: "claude-code",
      sessionId: sid,
      cwd,
      toolName: "Bash",
      toolInput: { command: "pip install -e . && pytest" },
      callId: "c2",
    });
    await recordPostTool(state, {
      harness: "claude-code",
      sessionId: sid,
      cwd,
      toolName: "Bash",
      toolInput: { command: "pip install -e . && pytest" },
      callId: "c2",
      toolResult: "PASSED",
      outcome: "success",
    });

    candidates = await drainCandidates(state);
    expect(candidates).toHaveLength(1);
    expect(candidates[0]!.kind).toBe("failure_delta");
    expect(candidates[0]!.toolName).toBe("Bash");
    expect(candidates[0]!.window).toContain("ImportError");
    expect(candidates[0]!.window).toContain("PASSED");
  });

  it("does not emit when only success occurs (no preceding failure)", async () => {
    const state = new MapState();
    await recordPreTool(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      toolName: "Bash",
      toolInput: {},
      callId: "c1",
    });
    await recordPostTool(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      toolName: "Bash",
      toolInput: {},
      callId: "c1",
      toolResult: "ok",
      outcome: "success",
    });
    expect(await drainCandidates(state)).toHaveLength(0);
  });

  it('does not emit on outcome="unknown" — conservative default', async () => {
    const state = new MapState();
    await recordPreTool(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      toolName: "Bash",
      toolInput: {},
      callId: "c1",
    });
    await recordPostTool(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      toolName: "Bash",
      toolInput: {},
      callId: "c1",
      toolResult: "?",
      outcome: "unknown",
    });
    // Add a success — no failure was recorded, so no pairing.
    await recordPostTool(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      toolName: "Bash",
      toolInput: {},
      callId: "c2",
      toolResult: "ok",
      outcome: "success",
    });
    expect(await drainCandidates(state)).toHaveLength(0);
  });

  it("does not pair a failure across different tool names", async () => {
    const state = new MapState();
    await recordPostTool(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      toolName: "Bash",
      toolInput: {},
      callId: "c1",
      toolResult: "fail",
      outcome: "failure",
    });
    await recordPostTool(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      toolName: "Read",
      toolInput: {},
      callId: "c2",
      toolResult: "ok",
      outcome: "success",
    });
    expect(await drainCandidates(state)).toHaveLength(0);
  });
});

describe("correction detector", () => {
  it("emits a candidate when prompt matches a correction pattern", async () => {
    const state = new MapState();
    await recordUserPrompt(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      prompt: "actually, we use httpx not requests",
    });
    const c = await drainCandidates(state);
    expect(c).toHaveLength(1);
    expect(c[0]!.kind).toBe("correction");
    expect(c[0]!.window).toContain("httpx");
  });

  it("ignores neutral prompts", async () => {
    const state = new MapState();
    await recordUserPrompt(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      prompt: "please add tests for the auth module",
    });
    expect(await drainCandidates(state)).toHaveLength(0);
  });
});

describe("drainCandidates", () => {
  it("clears the candidates list after returning it", async () => {
    const state = new MapState();
    await recordUserPrompt(state, {
      harness: "claude-code",
      sessionId: "s",
      cwd: "/tmp",
      prompt: "stop",
    });
    expect((await drainCandidates(state)).length).toBe(1);
    expect((await drainCandidates(state)).length).toBe(0);
  });
});
