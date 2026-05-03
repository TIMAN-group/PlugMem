import { describe, expect, it } from "vitest";

import { injectionToStdout } from "../src/respond.js";

describe("injectionToStdout", () => {
  it("returns null for null injection", () => {
    expect(injectionToStdout("SessionStart", null)).toBeNull();
  });

  it("returns null for empty-text injection", () => {
    expect(
      injectionToStdout("SessionStart", { role: "system", text: "" }),
    ).toBeNull();
  });

  it("returns hookSpecificOutput JSON for a real injection", () => {
    const out = injectionToStdout("UserPromptSubmit", {
      role: "system",
      text: "remember this",
    });
    expect(out).not.toBeNull();
    const parsed = JSON.parse(out!);
    expect(parsed).toEqual({
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: "remember this",
      },
    });
  });

  it("preserves multiline text", () => {
    const text = "line one\nline two\nline three";
    const out = injectionToStdout("SessionStart", {
      role: "system",
      text,
    });
    const parsed = JSON.parse(out!);
    expect(parsed.hookSpecificOutput.additionalContext).toBe(text);
  });
});
