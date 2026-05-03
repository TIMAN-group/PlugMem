import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { runFixture } from "../src/eval/runner.js";
import { summarizeTokenLog } from "../src/eval/metrics.js";
import type { EvalFixture } from "../src/eval/types.js";

// ── Token-log summarizer ────────────────────────────────────────────

describe("summarizeTokenLog", () => {
  let dir: string;
  beforeEach(async () => {
    dir = join(tmpdir(), "plugmem-eval-test-" + Math.random().toString(36).slice(2));
    await mkdir(dir, { recursive: true });
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("returns empty totals for missing file", async () => {
    const out = await summarizeTokenLog(join(dir, "no.jsonl"));
    expect(out.totals.total).toBe(0);
    expect(out.totals.count).toBe(0);
    expect(Object.keys(out.byPhase)).toHaveLength(0);
  });

  it("aggregates by phase", async () => {
    const lines = [
      { phase: "extract", prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
      { phase: "extract", prompt_tokens: 80, completion_tokens: 40, total_tokens: 120 },
      { phase: "retrieve", prompt_tokens: 30, completion_tokens: 10, total_tokens: 40 },
      { prompt_tokens: 5, completion_tokens: 5, total_tokens: 10 }, // no phase → "default"
    ];
    const path = join(dir, "tokens.jsonl");
    await writeFile(path, lines.map((l) => JSON.stringify(l)).join("\n"));

    const out = await summarizeTokenLog(path);
    expect(out.totals.total).toBe(320);
    expect(out.totals.count).toBe(4);
    expect(out.byPhase.extract!.total).toBe(270);
    expect(out.byPhase.extract!.count).toBe(2);
    expect(out.byPhase.retrieve!.total).toBe(40);
    expect(out.byPhase.default!.total).toBe(10);
  });

  it("skips unparseable lines and continues", async () => {
    const path = join(dir, "tokens.jsonl");
    await writeFile(
      path,
      [
        JSON.stringify({ phase: "extract", total_tokens: 10 }),
        "garbage{{",
        "",
        JSON.stringify({ phase: "extract", total_tokens: 20 }),
      ].join("\n"),
    );
    const out = await summarizeTokenLog(path);
    expect(out.byPhase.extract!.total).toBe(30);
    expect(out.byPhase.extract!.count).toBe(2);
  });

  it("computes total_tokens from prompt+completion when missing", async () => {
    const path = join(dir, "tokens.jsonl");
    await writeFile(
      path,
      JSON.stringify({ phase: "extract", prompt_tokens: 7, completion_tokens: 3 }),
    );
    const out = await summarizeTokenLog(path);
    expect(out.byPhase.extract!.total).toBe(10);
  });
});

// ── runFixture ─────────────────────────────────────────────────────

describe("runFixture", () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("walks events and reports per-session stats", async () => {
    let semantic = 0;
    let procedural = 0;

    const wrapped = vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";

      if (method === "DELETE" && /\/api\/v1\/graphs\/[^/]+$/.test(url)) {
        return new Response("{}", { status: 200 });
      }
      if (method === "GET" && /\/api\/v1\/graphs\/[^/]+$/.test(url)) {
        return new Response(JSON.stringify({ graph_id: "x", stats: {} }), {
          status: 200,
        });
      }
      if (/\/stats$/.test(url)) {
        return new Response(
          JSON.stringify({
            graph_id: "x",
            stats: { semantic, procedural, episodic: 0 },
          }),
          { status: 200 },
        );
      }
      // /retrieve always returns a non-empty hit (so injection fires).
      if (method === "POST" && /\/retrieve$/.test(url)) {
        return new Response(
          JSON.stringify({
            mode: "semantic_memory",
            reasoning_prompt: [],
            variables: {
              semantic_memory: "Fact: tests live in tests/",
              procedural_memory: "",
              episodic_memory: "",
            },
          }),
          { status: 200 },
        );
      }
      // /extract returns one semantic memory.
      if (method === "POST" && /\/api\/v1\/extract$/.test(url)) {
        return new Response(
          JSON.stringify({
            memories: [
              {
                type: "semantic",
                semantic_memory: "Use httpx",
                tags: [],
                source: "correction",
                confidence: 0.9,
              },
            ],
          }),
          { status: 200 },
        );
      }
      // /memories acks the insert and bumps our counter.
      if (method === "POST" && /\/memories$/.test(url)) {
        const body = JSON.parse(init!.body as string);
        if (body.semantic) semantic += body.semantic.length;
        if (body.procedural) procedural += body.procedural.length;
        return new Response(JSON.stringify({ status: "ok", stats: {} }), {
          status: 200,
        });
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = wrapped as unknown as typeof fetch;

    const fixture: EvalFixture = {
      name: "smoke",
      baseUrl: "http://stub",
      graphId: "eval://test",
      harness: "claude-code",
      cwd: "/tmp/eval",
      sessions: [
        {
          id: 1,
          events: [
            { type: "session_start" },
            {
              type: "user_prompt",
              prompt: "actually, we use httpx not requests",
            },
            { type: "session_end" },
          ],
        },
        {
          id: 2,
          events: [
            { type: "session_start" },
            { type: "session_end" },
          ],
        },
      ],
    };

    const report = await runFixture(fixture, { log: () => {} });

    // Two sessions in the report.
    expect(report.sessions).toHaveLength(2);
    // Session 1: session_start (graph empty initially → no inject) + user_prompt
    // (correction so promotion-gate records, but stats start at 0 → inject
    // returns null because ensureNonEmptyGraph short-circuits) + session_end
    // (drains candidates → /extract → /memories with 1 semantic).
    expect(report.sessions[0]!.promotions).toBe(1);
    // Session 2: session_start now has 1 memory in the graph → recall hits
    // and injects.
    const startEvent = report.sessions[1]!.events[0]!;
    expect(startEvent.type).toBe("session_start");
    expect(startEvent.injection).toBeDefined();
    expect(startEvent.injection!.text).toContain("plugmem-recall");
    // No new promotions in session 2 (no candidates).
    expect(report.sessions[1]!.promotions).toBe(0);

    // Final tally.
    expect(report.totalPromotions).toBe(1);
    expect(report.finalStats.semantic).toBe(1);
  });
});
