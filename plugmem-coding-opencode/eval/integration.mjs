// End-to-end integration harness for the OpenCode adapter.
//
// Drives the BUILT plugin (../dist/index.js) with real OpenCode-shaped events
// against a running PlugMem server, then verifies recall. Exercises the actual
// runtime paths that unit tests can't: the tool.execute.after outcome parser,
// failure->success pairing, per-event-family session-id resolution, the
// episodic filter, and the session.idle transcript-replay dedup.
//
// Requires a live PlugMem server with a real LLM + embedder. Usage:
//   1. cd plugmem-coding-opencode && npm run build
//   2. start the server (e.g. CHROMA_MODE=ephemeral uvicorn plugmem.api.app:app --port 8077)
//   3. PLUGMEM_PORT=8077 node eval/integration.mjs
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const PORT = process.env.PLUGMEM_PORT || "8077";
const BASE = `http://127.0.0.1:${PORT}/api/v1`;
process.env.PLUGMEM_URL = `http://127.0.0.1:${PORT}`;
delete process.env.PLUGMEM_API_KEY;

const PLUGIN_PATH = resolve(dirname(fileURLToPath(import.meta.url)), "../dist/index.js");
const SID = "harness-session-1";

let pass = 0, fail = 0;
const check = (name, ok, detail = "") => {
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? " — " + detail : ""}`);
  ok ? pass++ : fail++;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const J = async (p, opts) => (await fetch(`${BASE}${p}`, opts)).json();

async function waitHealth(timeout = 60000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    try {
      const h = await J("/health");
      if (h.llm_available && h.embedding_available && h.chroma_available) return h;
    } catch {}
    await sleep(1500);
  }
  throw new Error("server not healthy — is it running with a real LLM + embedder?");
}

// Full user-message history the transcript fallback re-reads on every idle.
const HISTORY = [
  { info: { id: "m1", role: "user", sessionID: SID },
    parts: [{ type: "text", text: "Don't use the requests library here — we standardized on httpx for all HTTP calls." }] },
  { info: { id: "m2", role: "user", sessionID: SID },
    parts: [{ type: "text", text: "perfect, it works" }] },
];

async function statsOf(gid) {
  return (await J(`/graphs/${encodeURIComponent(gid)}/stats`)).stats || {};
}
async function retrieve(gid, obs, mode) {
  const r = await J(`/graphs/${encodeURIComponent(gid)}/retrieve`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ observation: obs, goal: obs, mode }),
  });
  const v = r.variables || {};
  return ((v.semantic_memory || "") + " " + (v.procedural_memory || "")).toLowerCase();
}

async function main() {
  console.log("== health =="); console.log(" ", JSON.stringify(await waitHealth()));

  const { PlugMemPlugin } = await import(pathToFileURL(PLUGIN_PATH).href);
  const ctx = {
    directory: "C:/eval/integ-harness", // no git remote -> deterministic local graph id
    project: {}, $: {}, worktree: {},
    client: { session: {
      get: async () => ({ system: [] }),
      update: async () => {},
      messages: async () => HISTORY, // fallback re-reads this every idle
    } },
  };
  const hooks = await PlugMemPlugin(ctx);
  const ev = (type, properties) => hooks.event({ event: { type, properties } });

  console.log("\n== firing a realistic OpenCode session ==");
  await ev("session.created", { info: { id: SID } });

  // failure -> success on the same tool (bash), real after-hook shape
  await hooks["tool.execute.before"]({ tool: "bash", sessionID: SID, callID: "c1" }, { args: { command: "pytest -q" } });
  await hooks["tool.execute.after"]({ tool: "bash", sessionID: SID, callID: "c1", args: { command: "pytest -q" } },
    { title: "pytest", output: "ModuleNotFoundError: No module named 'plugmem'", metadata: { exit: 1 } });
  await hooks["tool.execute.before"]({ tool: "bash", sessionID: SID, callID: "c2" }, { args: { command: "pip install -e . && pytest -q" } });
  await hooks["tool.execute.after"]({ tool: "bash", sessionID: SID, callID: "c2", args: { command: "pip install -e . && pytest -q" } },
    { title: "pytest", output: "92 passed", metadata: { exit: 0 } });

  // live user prompts (real shape: session id on info/part)
  await ev("message.updated", { info: { id: "m1", role: "user", sessionID: SID } });
  await ev("message.part.updated", { part: { messageID: "m1", sessionID: SID, text: HISTORY[0].parts[0].text } });
  await ev("message.updated", { info: { id: "m2", role: "user", sessionID: SID } });
  await ev("message.part.updated", { part: { messageID: "m2", sessionID: SID, text: HISTORY[1].parts[0].text } });

  console.log("  session.idle #1 (drain + 32B extract + insert)...");
  await ev("session.idle", { info: { id: SID } });

  console.log("\n== verification ==");
  const graphs = await J("/graphs");
  const gid = (graphs.graphs || []).find((g) => g.startsWith("repo_opencode_local"));
  check("plugin created a per-repo graph", !!gid, gid || "(none)");
  if (!gid) return finish();

  const s1 = await statsOf(gid);
  console.log("  stats after idle #1:", JSON.stringify(s1));
  check("memories written despite episodic candidate (batch not poisoned)",
    (s1.semantic || 0) + (s1.procedural || 0) > 0, `${s1.semantic || 0} sem + ${s1.procedural || 0} proc`);
  check("no episodic node created", (s1.episodic || 0) === 0);
  check("correction deduped to exactly one semantic (no live/fallback double)", (s1.semantic || 0) === 1,
    `${s1.semantic || 0} semantic`);

  const httpx = await retrieve(gid, "Which HTTP client should I use to call a REST API?", "semantic_memory");
  check("recall surfaces the httpx correction (session-id path)", httpx.includes("httpx"), httpx.trim().slice(0, 80));
  const recipe = await retrieve(gid, "Tests fail with ModuleNotFoundError. How do I run them?", "procedural_memory");
  check("recall surfaces the failure->success recipe (outcome-parser path)",
    recipe.includes("pip"), recipe.trim().slice(0, 80));

  // --- Medium-fix validation: repeated session.idle must not duplicate ---
  console.log("\n  session.idle #2 (no new events; transcript fallback re-reads full history)...");
  await ev("session.idle", { info: { id: SID } });
  const s2 = await statsOf(gid);
  console.log("  stats after idle #2:", JSON.stringify(s2));
  check("repeated session.idle does NOT duplicate memories (transcript-replay dedup)",
    (s2.semantic || 0) === (s1.semantic || 0) && (s2.procedural || 0) === (s1.procedural || 0),
    `idle#1 ${s1.semantic}/${s1.procedural} -> idle#2 ${s2.semantic}/${s2.procedural}`);

  finish();
}

function finish() {
  console.log(`\n== RESULT: ${pass} passed, ${fail} failed ==`);
  process.exit(fail ? 1 : 0);
}
main().catch((e) => { console.error("HARNESS ERROR:", e); process.exit(2); });
