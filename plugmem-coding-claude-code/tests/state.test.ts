import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mkdir, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// state.ts reads PLUGMEM_STATE_DIR at import time, so we must set it
// before importing. Use a sentinel base + per-test override.

const ROOT = join(tmpdir(), "plugmem-state-test-" + Math.random().toString(36).slice(2));
process.env.PLUGMEM_STATE_DIR = ROOT;

// Must import AFTER setting env so the module captures it.
import { DiskSessionState } from "../src/state.js";

describe("DiskSessionState", () => {
  beforeEach(async () => {
    await mkdir(ROOT, { recursive: true });
  });
  afterEach(async () => {
    await rm(ROOT, { recursive: true, force: true });
  });

  it("get returns undefined for missing key", async () => {
    const s = new DiskSessionState("sess-1");
    expect(await s.get("missing")).toBeUndefined();
  });

  it("set persists, get round-trips, del removes", async () => {
    const s = new DiskSessionState("sess-2");
    await s.set("foo", { a: 1, b: [2, 3] });
    expect(await s.get("foo")).toEqual({ a: 1, b: [2, 3] });
    await s.del("foo");
    expect(await s.get("foo")).toBeUndefined();
  });

  it("rejects path-unsafe keys", async () => {
    const s = new DiskSessionState("sess-3");
    await expect(s.set("../escape", "x")).rejects.toThrow();
    await expect(s.get("../escape")).rejects.toThrow();
    await expect(s.set("with space", "x")).rejects.toThrow();
  });

  it("sanitizes path-unsafe sessionId chars into the dir name", async () => {
    const s1 = new DiskSessionState("safe-id_123");
    const s2 = new DiskSessionState("ev/il:id");
    await s1.set("k", 1);
    await s2.set("k", 2);
    // Both writes succeed without escaping the root.
    const dirs = await readdir(ROOT);
    // The unsafe id is sanitized — no slashes leaked into a sub-tree.
    expect(dirs.length).toBe(2);
    expect(dirs.every((d) => !d.includes("/"))).toBe(true);
  });

  it("requires a non-empty sessionId", () => {
    expect(() => new DiskSessionState("")).toThrow();
  });

  it("clear removes the entire session dir", async () => {
    const s = new DiskSessionState("sess-clear");
    await s.set("a", 1);
    await s.set("b", 2);
    await s.clear();
    // After clear, get returns undefined for previously-set keys.
    expect(await s.get("a")).toBeUndefined();
    expect(await s.get("b")).toBeUndefined();
  });

  it("del on a missing key is a no-op (does not throw)", async () => {
    const s = new DiskSessionState("sess-del");
    await expect(s.del("never-set")).resolves.toBeUndefined();
  });
});
