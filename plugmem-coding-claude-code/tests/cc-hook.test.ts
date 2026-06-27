import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { runHook } from "../src/bin/cc-hook.js";
import type { CoreCallbacks } from "@plugmem/coding-core";

interface MockCallbacks extends CoreCallbacks {
  calls: { method: string; payload: unknown }[];
}

function mockCore(
  overrides: Partial<CoreCallbacks> = {},
): MockCallbacks {
  const calls: { method: string; payload: unknown }[] = [];
  const make = <T>(method: string, ret: T) =>
    vi.fn(async (e: unknown) => {
      calls.push({ method, payload: e });
      return ret;
    });

  return {
    onSessionStart: make("onSessionStart", null),
    onUserPrompt: make("onUserPrompt", null),
    onPreTool: make("onPreTool", null),
    onPostTool: make("onPostTool", undefined),
    onPreCompact: make("onPreCompact", undefined),
    onSessionEnd: make("onSessionEnd", undefined),
    ...overrides,
    calls,
  } as MockCallbacks;
}

describe("runHook — error handling", () => {
  it("returns exit 0 on malformed JSON", async () => {
    const r = await runHook("not-json", mockCore());
    expect(r.exitCode).toBe(0);
    expect(r.stdout).toBeNull();
  });

  it("returns exit 0 when hook_event_name is missing", async () => {
    const r = await runHook("{}", mockCore());
    expect(r.exitCode).toBe(0);
    expect(r.stdout).toBeNull();
  });

  it("returns exit 0 on unknown event name", async () => {
    const r = await runHook(
      JSON.stringify({ hook_event_name: "MadeUp", session_id: "s", cwd: "/x" }),
      mockCore(),
    );
    expect(r.exitCode).toBe(0);
    expect(r.stdout).toBeNull();
  });

  it("returns exit 0 when a callback throws", async () => {
    const core = mockCore({
      onSessionStart: vi.fn(async () => {
        throw new Error("boom");
      }),
    });
    const r = await runHook(
      JSON.stringify({ hook_event_name: "SessionStart", session_id: "s", cwd: "/x" }),
      core,
    );
    expect(r.exitCode).toBe(0);
  });
});

describe("runHook — dispatch by event", () => {
  it("dispatches SessionStart and serializes injection to stdout", async () => {
    const core = mockCore({
      onSessionStart: vi.fn(async () => ({
        role: "system",
        text: "hello agent",
      })),
    });
    const r = await runHook(
      JSON.stringify({
        hook_event_name: "SessionStart",
        session_id: "s",
        cwd: "/x",
        source: "startup",
      }),
      core,
    );
    expect(r.stdout).not.toBeNull();
    const parsed = JSON.parse(r.stdout!);
    expect(parsed.hookSpecificOutput.hookEventName).toBe("SessionStart");
    expect(parsed.hookSpecificOutput.additionalContext).toBe("hello agent");
  });

  it("dispatches UserPromptSubmit", async () => {
    const core = mockCore();
    await runHook(
      JSON.stringify({
        hook_event_name: "UserPromptSubmit",
        session_id: "s",
        cwd: "/x",
        prompt: "fix the bug",
      }),
      core,
    );
    const cm = core as MockCallbacks;
    expect(cm.calls.find((c) => c.method === "onUserPrompt")).toBeDefined();
  });

  it("dispatches PreToolUse and routes injection to stdout", async () => {
    const core = mockCore({
      onPreTool: vi.fn(async () => ({ role: "system", text: "tool tip" })),
    });
    const r = await runHook(
      JSON.stringify({
        hook_event_name: "PreToolUse",
        session_id: "s",
        cwd: "/x",
        tool_name: "Bash",
        tool_input: { command: "ls" },
        tool_use_id: "c1",
      }),
      core,
    );
    expect(r.stdout).not.toBeNull();
    expect(JSON.parse(r.stdout!).hookSpecificOutput.additionalContext).toBe(
      "tool tip",
    );
  });

  it("dispatches PostToolUse without stdout", async () => {
    const core = mockCore();
    const r = await runHook(
      JSON.stringify({
        hook_event_name: "PostToolUse",
        session_id: "s",
        cwd: "/x",
        tool_name: "Bash",
        tool_use_id: "c1",
        tool_result: [{ type: "tool_result", content: "ok", is_error: false }],
      }),
      core,
    );
    expect(r.stdout).toBeNull();
    const cm = core as MockCallbacks;
    expect(cm.calls.find((c) => c.method === "onPostTool")).toBeDefined();
  });

  it("dispatches PreCompact", async () => {
    const core = mockCore();
    const r = await runHook(
      JSON.stringify({
        hook_event_name: "PreCompact",
        session_id: "s",
        cwd: "/x",
        transcript_path: "/tmp/t.jsonl",
      }),
      core,
    );
    expect(r.stdout).toBeNull();
    const cm = core as MockCallbacks;
    expect(cm.calls.find((c) => c.method === "onPreCompact")).toBeDefined();
  });

  it("dispatches Stop and SessionEnd both to onSessionEnd", async () => {
    const core = mockCore();
    await runHook(
      JSON.stringify({
        hook_event_name: "Stop",
        session_id: "s1",
        cwd: "/x",
      }),
      core,
    );
    await runHook(
      JSON.stringify({
        hook_event_name: "SessionEnd",
        session_id: "s2",
        cwd: "/x",
      }),
      core,
    );
    const cm = core as MockCallbacks;
    const ends = cm.calls.filter((c) => c.method === "onSessionEnd");
    expect(ends).toHaveLength(2);
    expect((ends[0]!.payload as { reason: string }).reason).toBe("stop");
    expect((ends[1]!.payload as { reason: string }).reason).toBe("session_end");
  });
});

describe("runHook — state cleanup at session_end", () => {
  let stateRoot: string;

  beforeEach(async () => {
    stateRoot = join(tmpdir(), "plugmem-cleanup-" + Math.random().toString(36).slice(2));
    process.env.PLUGMEM_STATE_DIR = stateRoot;
    delete process.env.PLUGMEM_KEEP_STATE;
    await mkdir(stateRoot, { recursive: true });
  });
  afterEach(async () => {
    await rm(stateRoot, { recursive: true, force: true });
    delete process.env.PLUGMEM_STATE_DIR;
    delete process.env.PLUGMEM_KEEP_STATE;
  });

  it("clears the session dir after SessionEnd", async () => {
    // Pre-seed a state dir for the session.
    const sessionDir = join(stateRoot, "session-x");
    await mkdir(sessionDir, { recursive: true });
    await writeFile(join(sessionDir, "candidates.json"), "[]");

    await runHook(
      JSON.stringify({
        hook_event_name: "SessionEnd",
        session_id: "session-x",
        cwd: "/x",
      }),
      mockCore(),
    );

    const dirs = await readdir(stateRoot);
    expect(dirs).not.toContain("session-x");
  });

  it("preserves state when PLUGMEM_KEEP_STATE=1", async () => {
    process.env.PLUGMEM_KEEP_STATE = "1";

    const sessionDir = join(stateRoot, "keep-me");
    await mkdir(sessionDir, { recursive: true });
    await writeFile(join(sessionDir, "candidates.json"), "[]");

    await runHook(
      JSON.stringify({
        hook_event_name: "SessionEnd",
        session_id: "keep-me",
        cwd: "/x",
      }),
      mockCore(),
    );

    const dirs = await readdir(stateRoot);
    expect(dirs).toContain("keep-me");
  });

  it("doesn't clean up on non-end events", async () => {
    const sessionDir = join(stateRoot, "live-session");
    await mkdir(sessionDir, { recursive: true });
    await writeFile(join(sessionDir, "candidates.json"), "[]");

    await runHook(
      JSON.stringify({
        hook_event_name: "PreToolUse",
        session_id: "live-session",
        cwd: "/x",
        tool_name: "Bash",
        tool_input: {},
        tool_use_id: "c1",
      }),
      mockCore(),
    );

    const dirs = await readdir(stateRoot);
    expect(dirs).toContain("live-session");
  });
});
