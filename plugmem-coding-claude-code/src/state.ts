// Disk-backed SessionState for Claude Code.
//
// CC hooks run in isolated processes — pre/post tool, user prompt, and
// session_end can't share an in-memory Map. We persist per-session state
// to ~/.cache/plugmem/sessions/<sessionId>/<key>.json (one file per key
// so concurrent writes from different hook invocations don't collide).
//
// Cleanup: callers may invoke `clear()` from the session_end hook; if
// they don't, stale dirs accumulate. A periodic prune is out of scope
// for Stage 3.

import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

import type { SessionState } from "@plugmem/coding-core";

function defaultRoot(): string {
  return (
    process.env.PLUGMEM_STATE_DIR ??
    join(homedir(), ".cache", "plugmem", "sessions")
  );
}

export class DiskSessionState implements SessionState {
  private readonly dir: string;

  constructor(public readonly sessionId: string, root?: string) {
    if (!sessionId) throw new Error("DiskSessionState: sessionId required");
    // Replace any path-unsafe chars from arbitrary harness IDs.
    const safe = sessionId.replace(/[^A-Za-z0-9._-]/g, "_");
    this.dir = join(root ?? defaultRoot(), safe);
  }

  private keyPath(key: string): string {
    if (!/^[A-Za-z0-9._-]+$/.test(key)) {
      throw new Error(
        `SessionState key must match /^[A-Za-z0-9._-]+$/, got: ${key}`,
      );
    }
    return join(this.dir, `${key}.json`);
  }

  async get<T>(key: string): Promise<T | undefined> {
    try {
      const raw = await readFile(this.keyPath(key), "utf8");
      return JSON.parse(raw) as T;
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === "ENOENT") return undefined;
      throw err;
    }
  }

  async set<T>(key: string, value: T): Promise<void> {
    await mkdir(this.dir, { recursive: true });
    await writeFile(this.keyPath(key), JSON.stringify(value), "utf8");
  }

  async del(key: string): Promise<void> {
    try {
      await rm(this.keyPath(key));
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code !== "ENOENT") throw err;
    }
  }

  async clear(): Promise<void> {
    try {
      await rm(this.dir, { recursive: true, force: true });
    } catch {
      // best-effort
    }
  }
}
