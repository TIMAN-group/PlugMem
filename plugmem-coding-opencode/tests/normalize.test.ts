import { describe, it, expect } from "vitest";
import { sniffOutcome, extractResultText } from "../src/normalize.js";

// OpenCode's `tool.execute.after` second argument is `{ title, output,
// metadata }`. These tests pin the outcome heuristic against that real shape
// and guard the regression that motivated the fix: a blind substring scan of
// arbitrary tool output that misclassified successful reads/greps as failures.

describe("extractResultText", () => {
  it("returns a plain string result as-is", () => {
    expect(extractResultText("hello world")).toBe("hello world");
  });

  it("reads OpenCode's `output.output` field", () => {
    expect(
      extractResultText({ title: "bash", output: "done", metadata: {} }),
    ).toBe("done");
  });

  it("never throws on null/undefined", () => {
    expect(extractResultText(undefined)).toBe("");
    expect(extractResultText(null)).toBe("");
  });

  it("falls back to JSON for opaque objects without crashing", () => {
    expect(extractResultText({ foo: 1 })).toBe('{"foo":1}');
  });
});

describe("sniffOutcome", () => {
  it("marks a clean shell exit (metadata.exit === 0) as success", () => {
    expect(
      sniffOutcome("bash", { output: "ok", metadata: { exit: 0 } }),
    ).toBe("success");
  });

  it("marks a non-zero shell exit as failure", () => {
    expect(
      sniffOutcome("bash", { output: "boom", metadata: { exit: 1 } }),
    ).toBe("failure");
  });

  it("honors alternate exit-code keys (exitCode)", () => {
    expect(
      sniffOutcome("bash", { output: "boom", metadata: { exitCode: 2 } }),
    ).toBe("failure");
  });

  it("honors an explicit boolean error flag", () => {
    expect(
      sniffOutcome("edit", { output: "nope", metadata: { error: true } }),
    ).toBe("failure");
  });

  it("does NOT flag a successful read whose content contains 'Error:'", () => {
    // The core regression: reading a file that mentions errors is not a failure.
    expect(
      sniffOutcome("read", {
        output: "function f() { throw new Error: ... } // Error: handling",
        metadata: {},
      }),
    ).toBe("success");
  });

  it("does NOT flag a grep that returns lines containing 'Error:'", () => {
    expect(
      sniffOutcome("grep", {
        output: "app.ts:42: console.error('Error: boom')",
        metadata: {},
      }),
    ).toBe("success");
  });

  it("flags a shell failure via a narrow text marker when no exit code is present", () => {
    expect(
      sniffOutcome("bash", { output: "Command failed: npm test", metadata: {} }),
    ).toBe("failure");
  });

  it("does NOT apply shell text markers to non-shell tools", () => {
    expect(
      sniffOutcome("read", { output: "Command failed: legacy note", metadata: {} }),
    ).toBe("success");
  });

  it("defaults to success when the hook fired with no failure signal", () => {
    expect(sniffOutcome("read", { output: "file contents", metadata: {} })).toBe(
      "success",
    );
  });

  it("never throws on a malformed payload", () => {
    expect(() => sniffOutcome("bash", undefined)).not.toThrow();
    expect(sniffOutcome("bash", undefined)).toBe("success");
  });
});
