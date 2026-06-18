import { describe, it, expect, vi } from "vitest";
import {
  normalizeSessionStart,
  normalizeUserPrompt,
  normalizePreTool,
  normalizePostTool,
} from "../src/normalize.js";
import type {
  SessionCreatedEvent,
  ChatMessageEvent,
  ToolExecuteBeforeEvent,
  ToolExecuteAfterEvent,
} from "../src/types.js";

// Mock process.cwd to return a stable string for testing
vi.spyOn(process, "cwd").mockReturnValue("/mock/workspace");

describe("OpenCode Event Normalizers", () => {
  it("normalizes session.created", () => {
    const event: SessionCreatedEvent = {
      session: { id: "sess-123" },
      output: { system: [] },
    };
    
    const abstract = normalizeSessionStart(event);
    expect(abstract.sessionId).toBe("sess-123");
    expect(abstract.harness).toBe("opencode");
    expect(abstract.source).toBe("startup");
    expect(abstract.cwd).toBe("/mock/workspace");
  });

  it("normalizes chat.message and handles realistic nested sessionId", () => {
    const event: ChatMessageEvent = {
      message: {
        sessionId: "chat-456",
        id: "msg-1",
        role: "user",
        content: "Fix the bug",
      },
    };

    const abstract = normalizeUserPrompt(event);
    expect(abstract.sessionId).toBe("chat-456"); // Uses the nested ID
    expect(abstract.prompt).toBe("Fix the bug");
  });

  it("normalizes tool.execute.before", () => {
    const event: ToolExecuteBeforeEvent = {
      session: { id: "sess-123" },
      tool: {
        name: "readFile",
        args: { file: "test.ts" },
      },
    };

    const abstract = normalizePreTool(event, "call-789");
    expect(abstract.sessionId).toBe("sess-123");
    expect(abstract.toolName).toBe("readFile");
    expect(abstract.callId).toBe("call-789");
    expect(abstract.toolInput).toEqual({ file: "test.ts" });
  });

  it("normalizes tool.execute.after and correctly identifies success", () => {
    const event: ToolExecuteAfterEvent = {
      session: { id: "sess-123" },
      tool: {
        name: "readFile",
        args: { file: "test.ts" },
        result: "file contents here",
        exitCode: 0,
      },
    };

    const abstract = normalizePostTool(event, "call-789");
    expect(abstract.outcome).toBe("success");
    expect(abstract.toolResult).toBe("file contents here");
  });

  it("normalizes tool.execute.after and correctly identifies failure via exit code", () => {
    const event: ToolExecuteAfterEvent = {
      session: { id: "sess-123" },
      tool: {
        name: "bash",
        args: { command: "npm run test" },
        result: "test failed",
        exitCode: 1, // Non-zero indicates failure
      },
    };

    const abstract = normalizePostTool(event, "call-789");
    expect(abstract.outcome).toBe("failure");
  });

  it("normalizes tool.execute.after and correctly identifies failure via text output", () => {
    const event: ToolExecuteAfterEvent = {
      session: { id: "sess-123" },
      tool: {
        name: "python",
        args: { script: "script.py" },
        result: "Traceback (most recent call last): SyntaxError: invalid syntax",
        exitCode: 0, // Even if exit code is 0, 'error' in text flags failure
      },
    };

    const abstract = normalizePostTool(event, "call-789");
    expect(abstract.outcome).toBe("failure");
  });
});
