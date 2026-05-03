import { describe, expect, it } from "vitest";

import { parseTranscript } from "../src/transcript.js";

const lines = (objs: unknown[]) =>
  objs.map((o) => JSON.stringify(o)).join("\n");

describe("parseTranscript", () => {
  it("returns null for empty input", () => {
    expect(parseTranscript("")).toBeNull();
    expect(parseTranscript("\n\n")).toBeNull();
  });

  it("ignores non-message lines", () => {
    const jsonl = lines([
      { type: "permission-mode", permissionMode: "default" },
      { type: "file-history-snapshot", snapshot: {} },
    ]);
    expect(parseTranscript(jsonl)).toBeNull();
  });

  it("extracts (user, assistant) pair as one step", () => {
    const jsonl = lines([
      { type: "user", message: { role: "user", content: "what is X?" } },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "X is Y." }],
        },
      },
    ]);
    const t = parseTranscript(jsonl)!;
    expect(t.steps).toHaveLength(1);
    expect(t.steps[0]).toEqual({
      observation: "what is X?",
      action: "X is Y.",
    });
  });

  it("uses ai-title as goal when present", () => {
    const jsonl = lines([
      { type: "ai-title", aiTitle: "Refactor auth module" },
      { type: "user", message: { role: "user", content: "do the thing" } },
      {
        type: "assistant",
        message: { role: "assistant", content: [{ type: "text", text: "ok" }] },
      },
    ]);
    expect(parseTranscript(jsonl)!.goal).toBe("Refactor auth module");
  });

  it("falls back to first user message as goal", () => {
    const jsonl = lines([
      { type: "user", message: { role: "user", content: "first prompt" } },
      {
        type: "assistant",
        message: { role: "assistant", content: [{ type: "text", text: "ok" }] },
      },
    ]);
    expect(parseTranscript(jsonl)!.goal).toBe("first prompt");
  });

  it("collapses consecutive same-role messages", () => {
    const jsonl = lines([
      { type: "user", message: { role: "user", content: "hi" } },
      {
        type: "assistant",
        message: { role: "assistant", content: [{ type: "text", text: "a1" }] },
      },
      {
        type: "assistant",
        message: { role: "assistant", content: [{ type: "text", text: "a2" }] },
      },
      {
        type: "user",
        message: {
          role: "user",
          content: [
            { type: "tool_result", content: "result", is_error: false },
          ],
        },
      },
      {
        type: "assistant",
        message: { role: "assistant", content: [{ type: "text", text: "a3" }] },
      },
    ]);
    const t = parseTranscript(jsonl)!;
    expect(t.steps).toHaveLength(2);
    expect(t.steps[0]!.action).toContain("a1");
    expect(t.steps[0]!.action).toContain("a2");
    expect(t.steps[1]!.observation).toContain("[tool_result]");
    expect(t.steps[1]!.action).toBe("a3");
  });

  it("marks tool_result error blocks", () => {
    const jsonl = lines([
      { type: "user", message: { role: "user", content: "x" } },
      {
        type: "assistant",
        message: { role: "assistant", content: [{ type: "text", text: "ok" }] },
      },
      {
        type: "user",
        message: {
          role: "user",
          content: [{ type: "tool_result", content: "boom", is_error: true }],
        },
      },
      {
        type: "assistant",
        message: { role: "assistant", content: [{ type: "text", text: "fix" }] },
      },
    ]);
    const t = parseTranscript(jsonl)!;
    expect(t.steps[1]!.observation).toContain("[tool_result:error]");
  });

  it("skips thinking blocks", () => {
    const jsonl = lines([
      { type: "user", message: { role: "user", content: "hi" } },
      {
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "very long internal monologue ".repeat(50) },
            { type: "text", text: "actual reply" },
          ],
        },
      },
    ]);
    const t = parseTranscript(jsonl)!;
    expect(t.steps[0]!.action).toBe("actual reply");
  });
});
