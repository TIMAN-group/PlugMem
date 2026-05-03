import { describe, expect, it } from "vitest";

import {
  toPostTool,
  toPreCompact,
  toPreTool,
  toSessionEnd,
  toSessionStart,
  toUserPrompt,
} from "../src/normalize.js";

describe("toSessionStart", () => {
  it("maps the common CC payload fields", () => {
    const e = toSessionStart({
      session_id: "abc",
      cwd: "/repo",
      transcript_path: "/tmp/t.jsonl",
      hook_event_name: "SessionStart",
      source: "startup",
    });
    expect(e).toEqual({
      harness: "claude-code",
      sessionId: "abc",
      cwd: "/repo",
      transcriptPath: "/tmp/t.jsonl",
      source: "startup",
    });
  });

  it("source is optional", () => {
    const e = toSessionStart({
      session_id: "x",
      cwd: "/r",
      hook_event_name: "SessionStart",
    });
    expect(e.source).toBeUndefined();
  });
});

describe("toUserPrompt", () => {
  it("captures prompt", () => {
    const e = toUserPrompt({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "UserPromptSubmit",
      prompt: "fix the bug",
    });
    expect(e.prompt).toBe("fix the bug");
    expect(e.harness).toBe("claude-code");
  });

  it("defaults missing prompt to empty string", () => {
    const e = toUserPrompt({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "UserPromptSubmit",
    });
    expect(e.prompt).toBe("");
  });
});

describe("toPreTool", () => {
  it("maps tool_name / tool_input / tool_use_id", () => {
    const e = toPreTool({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "PreToolUse",
      tool_name: "Bash",
      tool_input: { command: "ls" },
      tool_use_id: "call-1",
    });
    expect(e.toolName).toBe("Bash");
    expect(e.toolInput).toEqual({ command: "ls" });
    expect(e.callId).toBe("call-1");
  });

  it("defaults missing fields to empty strings/undefined", () => {
    const e = toPreTool({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "PreToolUse",
    });
    expect(e.toolName).toBe("");
    expect(e.callId).toBe("");
    expect(e.toolInput).toBeUndefined();
  });
});

describe("toPostTool — outcome sniffing", () => {
  it("returns success for tool_result without is_error", () => {
    const e = toPostTool({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "PostToolUse",
      tool_name: "Bash",
      tool_use_id: "c1",
      tool_result: [{ type: "tool_result", content: "ok", is_error: false }],
    });
    expect(e.outcome).toBe("success");
  });

  it("returns failure when any block has is_error: true", () => {
    const e = toPostTool({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "PostToolUse",
      tool_name: "Bash",
      tool_use_id: "c1",
      tool_result: [
        { type: "tool_result", content: "boom", is_error: true },
      ],
    });
    expect(e.outcome).toBe("failure");
  });

  it("returns unknown for non-array tool_result", () => {
    const e = toPostTool({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "PostToolUse",
      tool_name: "Bash",
      tool_use_id: "c1",
      tool_result: "string-output",
    });
    expect(e.outcome).toBe("unknown");
  });

  it("returns unknown for missing tool_result", () => {
    const e = toPostTool({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "PostToolUse",
      tool_name: "Bash",
      tool_use_id: "c1",
    });
    expect(e.outcome).toBe("unknown");
  });

  it("stringifies tool_result for the toolResult field", () => {
    const e = toPostTool({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "PostToolUse",
      tool_name: "Bash",
      tool_use_id: "c1",
      tool_result: { foo: "bar" },
    });
    expect(e.toolResult).toContain("foo");
  });
});

describe("toPreCompact", () => {
  it("carries transcript_path", () => {
    const e = toPreCompact({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "PreCompact",
      transcript_path: "/tmp/t.jsonl",
    });
    expect(e.transcriptPath).toBe("/tmp/t.jsonl");
  });
});

describe("toSessionEnd", () => {
  it("maps Stop to reason=stop", () => {
    const e = toSessionEnd({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "Stop",
    });
    expect(e.reason).toBe("stop");
  });

  it("maps SessionEnd to reason=session_end", () => {
    const e = toSessionEnd({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "SessionEnd",
    });
    expect(e.reason).toBe("session_end");
  });

  it("falls back to unknown for other names", () => {
    const e = toSessionEnd({
      session_id: "s",
      cwd: "/r",
      hook_event_name: "Other",
    });
    expect(e.reason).toBe("unknown");
  });
});
