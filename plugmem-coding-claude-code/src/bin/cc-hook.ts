#!/usr/bin/env node
// Single dispatcher for all Claude Code PlugMem hooks. Registered in
// hooks/hooks.json with the same command for every event; the event name
// in the stdin payload selects the handler.
//
// Config comes from env vars (Claude Code hooks inherit env from the
// parent process):
//
//   PLUGMEM_BASE_URL    (required)
//   PLUGMEM_API_KEY     (optional)
//   PLUGMEM_USER_ID     (optional)
//   PLUGMEM_TIMEOUT_MS  (optional, default 30000)
//   PLUGMEM_MAX_RETRIES (optional, default 3)
//   PLUGMEM_DEBUG       (optional, "1" to log to stderr)
//   PLUGMEM_STATE_DIR   (optional, default ~/.cache/plugmem/sessions)
//   PLUGMEM_KEEP_STATE  (optional, "1" to skip cleanup at session_end —
//                        debug aid for inspecting candidate state after a run)
//
// On error, we log to stderr and exit 0 — failing the hook would block
// the user's session, which is worse than missing a memory write.

import { createCore, type CoreCallbacks } from "@plugmem/coding-core";

import {
  toPreCompact,
  toPostTool,
  toPreTool,
  toSessionEnd,
  toSessionStart,
  toUserPrompt,
  type CCStdinPayload,
} from "../normalize.js";
import { injectionToStdout } from "../respond.js";
import { DiskSessionState } from "../state.js";

export async function runHook(
  stdinJson: string,
  core?: CoreCallbacks,
): Promise<{ stdout: string | null; exitCode: number }> {
  let payload: CCStdinPayload;
  try {
    payload = JSON.parse(stdinJson);
  } catch (err) {
    debug(`failed to parse stdin: ${(err as Error).message}`);
    return { stdout: null, exitCode: 0 };
  }

  if (!payload.hook_event_name) {
    debug("missing hook_event_name");
    return { stdout: null, exitCode: 0 };
  }

  const callbacks = core ?? buildCore();
  if (!callbacks) {
    return { stdout: null, exitCode: 0 };
  }

  try {
    switch (payload.hook_event_name) {
      case "SessionStart": {
        const inj = await callbacks.onSessionStart(toSessionStart(payload));
        return {
          stdout: injectionToStdout(payload.hook_event_name, inj),
          exitCode: 0,
        };
      }
      case "UserPromptSubmit": {
        const inj = await callbacks.onUserPrompt(toUserPrompt(payload));
        return {
          stdout: injectionToStdout(payload.hook_event_name, inj),
          exitCode: 0,
        };
      }
      case "PreToolUse": {
        const inj = await callbacks.onPreTool(toPreTool(payload));
        return {
          stdout: injectionToStdout(payload.hook_event_name, inj),
          exitCode: 0,
        };
      }
      case "PostToolUse":
        await callbacks.onPostTool(toPostTool(payload));
        return { stdout: null, exitCode: 0 };
      case "PreCompact":
        await callbacks.onPreCompact(toPreCompact(payload));
        return { stdout: null, exitCode: 0 };
      case "Stop":
      case "SessionEnd":
        await callbacks.onSessionEnd(toSessionEnd(payload));
        await maybeCleanupState(payload);
        return { stdout: null, exitCode: 0 };
      default:
        debug(`unhandled hook_event_name: ${payload.hook_event_name}`);
        return { stdout: null, exitCode: 0 };
    }
  } catch (err) {
    debug(`hook handler threw: ${(err as Error).message}`);
    return { stdout: null, exitCode: 0 };
  }
}

function buildCore(): CoreCallbacks | null {
  const baseUrl = process.env.PLUGMEM_BASE_URL;
  if (!baseUrl) {
    debug("PLUGMEM_BASE_URL not set — disabling hook");
    return null;
  }
  return createCore({
    config: {
      baseUrl,
      apiKey: process.env.PLUGMEM_API_KEY,
      userId: process.env.PLUGMEM_USER_ID,
      timeoutMs: numEnv("PLUGMEM_TIMEOUT_MS"),
      maxRetries: numEnv("PLUGMEM_MAX_RETRIES"),
    },
    state: (sessionId) => new DiskSessionState(sessionId),
    log: process.env.PLUGMEM_DEBUG === "1" ? undefined : () => {},
  });
}

function numEnv(name: string): number | undefined {
  const v = process.env[name];
  if (!v) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function debug(msg: string): void {
  if (process.env.PLUGMEM_DEBUG === "1") {
    process.stderr.write(`[plugmem-cc-hook] ${msg}\n`);
  }
}

async function maybeCleanupState(payload: CCStdinPayload): Promise<void> {
  if (process.env.PLUGMEM_KEEP_STATE === "1") return;
  if (!payload.session_id) return;
  try {
    await new DiskSessionState(payload.session_id).clear();
  } catch (err) {
    debug(`state cleanup failed: ${(err as Error).message}`);
  }
}

// Run when invoked as a CLI (not when imported for testing).
const isMain = import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  let buf = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => {
    buf += chunk;
  });
  process.stdin.on("end", () => {
    runHook(buf)
      .then(({ stdout, exitCode }) => {
        if (stdout) process.stdout.write(stdout);
        process.exit(exitCode);
      })
      .catch((err) => {
        debug(`top-level catch: ${(err as Error).message}`);
        process.exit(0);
      });
  });
}
