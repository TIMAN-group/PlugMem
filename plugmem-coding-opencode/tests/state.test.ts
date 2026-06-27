import { describe, it, expect } from "vitest";
import { getInMemoryState, clearSessionState } from "../src/state.js";

describe("in-memory SessionState", () => {
  it("get/set/del round-trips within a session", async () => {
    const s = getInMemoryState("s1");
    expect(await s.get("k")).toBeUndefined();
    await s.set("k", { a: 1 });
    expect(await s.get("k")).toEqual({ a: 1 });
    await s.del("k");
    expect(await s.get("k")).toBeUndefined();
  });

  it("isolates state across sessions", async () => {
    await getInMemoryState("a").set("x", 1);
    await getInMemoryState("b").set("x", 2);
    expect(await getInMemoryState("a").get("x")).toBe(1);
    expect(await getInMemoryState("b").get("x")).toBe(2);
  });

  it("clearSessionState wipes a session's bucket", async () => {
    const s = getInMemoryState("doomed");
    await s.set("candidates", [1, 2, 3]);
    expect(await s.get("candidates")).toEqual([1, 2, 3]);
    clearSessionState("doomed");
    // a fresh handle sees an empty bucket
    expect(await getInMemoryState("doomed").get("candidates")).toBeUndefined();
  });

  it("clearSessionState on an unknown session is a no-op", () => {
    expect(() => clearSessionState("never-existed")).not.toThrow();
  });
});
