import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  createPlugMemPlugin,
  messagesToTrajectory,
  parseSessionJsonl,
} from "../src/index.js";

// ── Mock fetch ───────────────────────────────────────────────────────

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as unknown as Response;
}

// ── Helpers ──────────────────────────────────────────────────────────

interface RegisteredTool {
  name: string;
  description: string;
  parameters: unknown;
  execute: (
    id: string,
    params: Record<string, unknown>,
  ) => Promise<{ content: Array<{ type: string; text?: string }> }>;
}

type HookHandler = (...args: unknown[]) => Promise<void> | void;

function activatePlugin(
  defaultGraphId?: string,
  opts?: {
    autoRemember?:
      | false
      | { onSessionReset?: boolean; onCompaction?: boolean; minSteps?: number };
    sharedReadGraphIds?: string[];
  },
) {
  const tools: Record<string, RegisteredTool> = {};
  const hooks: Record<string, HookHandler[]> = {};
  const api = {
    registerTool(tool: RegisteredTool) {
      tools[tool.name] = tool;
    },
    on(hookName: string, handler: HookHandler) {
      if (!hooks[hookName]) hooks[hookName] = [];
      hooks[hookName].push(handler);
    },
  };

  const plugin = createPlugMemPlugin({
    baseUrl: "http://localhost:8080",
    apiKey: "key",
    defaultGraphId,
    sharedReadGraphIds: opts?.sharedReadGraphIds,
    maxRetries: 0,
    autoRemember: opts?.autoRemember,
  });
  plugin.activate(api);

  return { tools, hooks };
}

function getText(result: { content: Array<{ text?: string }> }): string {
  return result.content.map((c) => c.text ?? "").join("");
}

// ── Tests ────────────────────────────────────────────────────────────

describe("OpenClaw plugin", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("registers both tools", () => {
    const { tools } = activatePlugin("default-graph");
    expect(tools["plugmem.remember"]).toBeDefined();
    expect(tools["plugmem.recall"]).toBeDefined();
  });

  // ── plugmem.remember ─────────────────────────────────────────────

  describe("plugmem.remember", () => {
    it("stores semantic memory from free text", async () => {
      const { tools } = activatePlugin("g1");
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ status: "ok", stats: { semantic: 5 } }),
      );

      const result = await tools["plugmem.remember"].execute("call-1", {
        text: "The capital of France is Paris",
        tags: ["geography"],
      });

      const text = getText(result);
      expect(text).toContain("Remembered");
      expect(text).toContain("5 semantic");

      const sent = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(sent.mode).toBe("structured");
      expect(sent.semantic[0].semantic_memory).toBe(
        "The capital of France is Paris",
      );
      expect(sent.semantic[0].tags).toEqual(["geography"]);
    });

    it("stores trajectory from goal + steps", async () => {
      const { tools } = activatePlugin("g1");
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          status: "ok",
          stats: { episodic: 2, semantic: 1, procedural: 1 },
        }),
      );

      const result = await tools["plugmem.remember"].execute("call-2", {
        goal: "navigate to settings",
        steps: [
          { observation: "on home page", action: "click settings" },
          { observation: "on settings page", action: "done" },
        ],
      });

      const text = getText(result);
      expect(text).toContain("Stored trajectory (2 steps)");

      const sent = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(sent.mode).toBe("trajectory");
    });

    it("returns guidance when nothing to store", async () => {
      const { tools } = activatePlugin("g1");

      const result = await tools["plugmem.remember"].execute("call-3", {});
      expect(getText(result)).toContain("Nothing to store");
    });

    it("requires graph_id when no default", async () => {
      const { tools } = activatePlugin(); // no default
      const result = await tools["plugmem.remember"].execute("call-4", {
        text: "some fact",
      });
      expect(getText(result)).toContain("graph_id is required");
    });

    it("uses explicit graph_id over default", async () => {
      const { tools } = activatePlugin("default-graph");
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ status: "ok", stats: {} }),
      );

      await tools["plugmem.remember"].execute("call-5", {
        graph_id: "custom-graph",
        text: "a fact",
      });

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain("/graphs/custom-graph/memories");
    });
  });

  // ── plugmem.recall ───────────────────────────────────────────────

  describe("plugmem.recall", () => {
    it("returns LLM-synthesized reasoning by default", async () => {
      const { tools } = activatePlugin("g1");
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          mode: "semantic_memory",
          reasoning: "Paris is the capital of France.",
          reasoning_prompt: [{ role: "user", content: "..." }],
        }),
      );

      const result = await tools["plugmem.recall"].execute("call-6", {
        observation: "What is the capital of France?",
      });

      const text = getText(result);
      expect(text).toContain("[semantic_memory]");
      expect(text).toContain("Paris is the capital of France.");

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain("/reason");
    });

    it("returns raw retrieval prompt when raw=true", async () => {
      const { tools } = activatePlugin("g1");
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          mode: "semantic_memory",
          reasoning_prompt: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: "Relevant facts: Paris is capital..." },
          ],
          variables: {},
        }),
      );

      const result = await tools["plugmem.recall"].execute("call-7", {
        observation: "capital of France?",
        raw: true,
      });

      const text = getText(result);
      expect(text).toContain("[semantic_memory]");
      expect(text).toContain("**system**");
      expect(text).toContain("**user**");

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain("/retrieve");
    });

    it("passes mode through to the API", async () => {
      const { tools } = activatePlugin("g1");
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          mode: "procedural_memory",
          reasoning: "Step 1...",
          reasoning_prompt: [],
        }),
      );

      await tools["plugmem.recall"].execute("call-8", {
        observation: "how to deploy?",
        mode: "procedural_memory",
      });

      const sent = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(sent.mode).toBe("procedural_memory");
    });

    it("handles API errors gracefully", async () => {
      const { tools } = activatePlugin("g1");
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: "Graph not found" }),
        text: () => Promise.resolve('{"detail":"Graph not found"}'),
      } as unknown as Response);

      const result = await tools["plugmem.recall"].execute("call-9", {
        observation: "anything",
      });

      const text = getText(result);
      expect(text).toContain("PlugMem error (404)");
    });
  });

  // ── Shared-read fan-out ──────────────────────────────────────────

  describe("plugmem.recall with sharedReadGraphIds", () => {
    it("fans out reason across primary + shared graphs", async () => {
      const { tools } = activatePlugin("agent-1", {
        sharedReadGraphIds: ["user-facts"],
      });
      mockFetch
        .mockResolvedValueOnce(
          jsonResponse({
            mode: "episodic_memory",
            reasoning: "Last session you deployed to staging.",
            reasoning_prompt: [],
          }),
        )
        .mockResolvedValueOnce(
          jsonResponse({
            mode: "semantic_memory",
            reasoning: "User prefers blue/green deploys.",
            reasoning_prompt: [],
          }),
        );

      const result = await tools["plugmem.recall"].execute("call-fan-1", {
        observation: "how should I deploy?",
      });

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(mockFetch.mock.calls[0][0]).toContain(
        "/graphs/agent-1/reason",
      );
      expect(mockFetch.mock.calls[1][0]).toContain(
        "/graphs/user-facts/reason",
      );

      const text = getText(result);
      expect(text).toContain("[graph:agent-1 | episodic_memory]");
      expect(text).toContain("Last session you deployed to staging.");
      expect(text).toContain("[graph:user-facts | semantic_memory]");
      expect(text).toContain("User prefers blue/green deploys.");
    });

    it("fans out retrieve when raw=true", async () => {
      const { tools } = activatePlugin("agent-1", {
        sharedReadGraphIds: ["user-facts"],
      });
      mockFetch
        .mockResolvedValueOnce(
          jsonResponse({
            mode: "semantic_memory",
            reasoning_prompt: [
              { role: "user", content: "Facts: staging uses k8s" },
            ],
            variables: {},
          }),
        )
        .mockResolvedValueOnce(
          jsonResponse({
            mode: "semantic_memory",
            reasoning_prompt: [
              { role: "user", content: "Facts: user owns repo X" },
            ],
            variables: {},
          }),
        );

      const result = await tools["plugmem.recall"].execute("call-fan-2", {
        observation: "context?",
        raw: true,
      });

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(mockFetch.mock.calls[0][0]).toContain(
        "/graphs/agent-1/retrieve",
      );
      expect(mockFetch.mock.calls[1][0]).toContain(
        "/graphs/user-facts/retrieve",
      );

      const text = getText(result);
      expect(text).toContain("[graph:agent-1 | semantic_memory]");
      expect(text).toContain("staging uses k8s");
      expect(text).toContain("[graph:user-facts | semantic_memory]");
      expect(text).toContain("user owns repo X");
    });

    it("returns partial results when a shared graph fails", async () => {
      const { tools } = activatePlugin("agent-1", {
        sharedReadGraphIds: ["missing-graph"],
      });
      mockFetch
        .mockResolvedValueOnce(
          jsonResponse({
            mode: "semantic_memory",
            reasoning: "Primary answer.",
            reasoning_prompt: [],
          }),
        )
        .mockResolvedValueOnce({
          ok: false,
          status: 404,
          json: () => Promise.resolve({ detail: "Graph not found" }),
          text: () => Promise.resolve('{"detail":"Graph not found"}'),
        } as unknown as Response);

      const result = await tools["plugmem.recall"].execute("call-fan-3", {
        observation: "anything",
      });

      const text = getText(result);
      expect(text).toContain("[graph:agent-1 | semantic_memory]");
      expect(text).toContain("Primary answer.");
      expect(text).toContain("[graph:missing-graph | error]");
      expect(text).toContain("PlugMem error (404)");
    });

    it("dedupes when primary graph also appears in shared list", async () => {
      const { tools } = activatePlugin("agent-1", {
        sharedReadGraphIds: ["agent-1", "user-facts"],
      });
      mockFetch
        .mockResolvedValueOnce(
          jsonResponse({
            mode: "semantic_memory",
            reasoning: "A",
            reasoning_prompt: [],
          }),
        )
        .mockResolvedValueOnce(
          jsonResponse({
            mode: "semantic_memory",
            reasoning: "B",
            reasoning_prompt: [],
          }),
        );

      await tools["plugmem.recall"].execute("call-fan-4", {
        observation: "x",
      });

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(mockFetch.mock.calls[0][0]).toContain("/graphs/agent-1/");
      expect(mockFetch.mock.calls[1][0]).toContain("/graphs/user-facts/");
    });

    it("preserves single-graph output shape when sharedReadGraphIds is empty", async () => {
      const { tools } = activatePlugin("agent-1", { sharedReadGraphIds: [] });
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          mode: "semantic_memory",
          reasoning: "Solo answer.",
          reasoning_prompt: [],
        }),
      );

      const result = await tools["plugmem.recall"].execute("call-fan-5", {
        observation: "x",
      });

      const text = getText(result);
      // Old format — no [graph:...] prefix.
      expect(text).toBe("[semantic_memory] Solo answer.");
    });
  });

  // ── messagesToTrajectory ─────────────────────────────────────────

  describe("messagesToTrajectory", () => {
    it("pairs simple user/assistant messages", () => {
      const steps = messagesToTrajectory([
        { role: "user", content: "What is 2+2?" },
        { role: "assistant", content: [{ type: "text", text: "4" }], stopReason: "stop" },
        { role: "user", content: "And 3+3?" },
        { role: "assistant", content: [{ type: "text", text: "6" }], stopReason: "stop" },
      ]);
      expect(steps).toEqual([
        { observation: "What is 2+2?", action: "4" },
        { observation: "And 3+3?", action: "6" },
      ]);
    });

    it("unfolds agentic turn: user → assistant(toolCall) → toolResult → assistant(stop)", () => {
      const steps = messagesToTrajectory([
        { role: "user", content: "Find the bug in app.ts" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me read the file..." },
            { type: "toolCall", id: "t1", name: "read", arguments: { file_path: "/src/app.ts" } },
          ],
          stopReason: "toolUse",
        },
        {
          role: "toolResult",
          toolCallId: "t1",
          toolName: "read",
          content: [{ type: "text", text: "const x = null; x.foo();" }],
          isError: false,
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "Found null pointer bug on line 1." }],
          stopReason: "stop",
        },
      ]);

      expect(steps).toHaveLength(2);
      // Step 0: user observation → assistant reads file
      expect(steps[0].observation).toBe("Find the bug in app.ts");
      expect(steps[0].action).toContain("Let me read the file...");
      expect(steps[0].action).toContain("[read(");
      // Step 1: tool result as observation → assistant conclusion
      expect(steps[1].observation).toContain("[read result]");
      expect(steps[1].observation).toContain("const x = null");
      expect(steps[1].action).toContain("Found null pointer bug");
    });

    it("handles multi-tool agentic turn", () => {
      const steps = messagesToTrajectory([
        { role: "user", content: "Fix the typo" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Reading file..." },
            { type: "toolCall", id: "t1", name: "read", arguments: { file_path: "a.ts" } },
          ],
          stopReason: "toolUse",
        },
        {
          role: "toolResult",
          toolCallId: "t1",
          toolName: "read",
          content: [{ type: "text", text: "const msg = 'helo';" }],
          isError: false,
        },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Fixing typo..." },
            {
              type: "toolCall",
              id: "t2",
              name: "edit",
              arguments: { file_path: "a.ts", old_string: "helo", new_string: "hello" },
            },
          ],
          stopReason: "toolUse",
        },
        {
          role: "toolResult",
          toolCallId: "t2",
          toolName: "edit",
          content: [{ type: "text", text: "File edited successfully" }],
          isError: false,
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "Fixed the typo: helo → hello." }],
          stopReason: "stop",
        },
      ]);

      expect(steps).toHaveLength(3);
      // Step 0: user → read
      expect(steps[0].observation).toBe("Fix the typo");
      expect(steps[0].action).toContain("[read(");
      // Step 1: read result → edit
      expect(steps[1].observation).toContain("[read result]");
      expect(steps[1].action).toContain("[edit(");
      // Step 2: edit result → conclusion
      expect(steps[2].observation).toContain("[edit result]");
      expect(steps[2].action).toContain("Fixed the typo");
    });

    it("handles bashExecution messages", () => {
      const steps = messagesToTrajectory([
        { role: "user", content: "Run the tests" },
        {
          role: "assistant",
          content: [
            { type: "toolCall", id: "t1", name: "bash", arguments: { command: "npm test" } },
          ],
          stopReason: "toolUse",
        },
        {
          role: "bashExecution",
          command: "npm test",
          output: "3 passed, 0 failed",
          exitCode: 0,
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "All tests pass." }],
          stopReason: "stop",
        },
      ]);

      expect(steps).toHaveLength(2);
      expect(steps[1].observation).toContain("[bash]");
      expect(steps[1].observation).toContain("npm test");
      expect(steps[1].observation).toContain("3 passed");
    });

    it("includes exit code for failed bash", () => {
      const steps = messagesToTrajectory([
        { role: "user", content: "Build it" },
        {
          role: "assistant",
          content: [
            { type: "toolCall", id: "t1", name: "bash", arguments: { command: "make" } },
          ],
          stopReason: "toolUse",
        },
        {
          role: "bashExecution",
          command: "make",
          output: "error: missing header",
          exitCode: 2,
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "Build failed." }],
          stopReason: "stop",
        },
      ]);

      expect(steps[1].observation).toContain("[bash exit=2]");
    });

    it("marks error tool results", () => {
      const steps = messagesToTrajectory([
        { role: "user", content: "Read the file" },
        {
          role: "assistant",
          content: [
            { type: "toolCall", id: "t1", name: "read", arguments: { file_path: "missing.ts" } },
          ],
          stopReason: "toolUse",
        },
        {
          role: "toolResult",
          toolCallId: "t1",
          toolName: "read",
          content: [{ type: "text", text: "File not found" }],
          isError: true,
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "File doesn't exist." }],
          stopReason: "stop",
        },
      ]);

      expect(steps[1].observation).toContain("[ERROR from read]");
    });

    it("merges consecutive user messages", () => {
      const steps = messagesToTrajectory([
        { role: "user", content: "Hello" },
        { role: "user", content: "Can you help?" },
        { role: "assistant", content: [{ type: "text", text: "Sure!" }], stopReason: "stop" },
      ]);
      expect(steps).toEqual([
        { observation: "Hello\nCan you help?", action: "Sure!" },
      ]);
    });

    it("skips system and compactionSummary messages", () => {
      const steps = messagesToTrajectory([
        { role: "system", content: "You are helpful." },
        { role: "compactionSummary", summary: "Prior conversation..." },
        { role: "user", content: "Hi" },
        { role: "assistant", content: [{ type: "text", text: "Hello!" }], stopReason: "stop" },
      ]);
      expect(steps).toEqual([{ observation: "Hi", action: "Hello!" }]);
    });

    it("handles array content blocks in user messages", () => {
      const steps = messagesToTrajectory([
        {
          role: "user",
          content: [
            { type: "text", text: "Part 1" },
            { type: "image" },
            { type: "text", text: "Part 2" },
          ],
        },
        { role: "assistant", content: [{ type: "text", text: "Got it" }], stopReason: "stop" },
      ]);
      expect(steps).toEqual([
        { observation: "Part 1\nPart 2", action: "Got it" },
      ]);
    });

    it("drops trailing user message without a response", () => {
      const steps = messagesToTrajectory([
        { role: "user", content: "Hi" },
        { role: "assistant", content: [{ type: "text", text: "Hello" }], stopReason: "stop" },
        { role: "user", content: "Bye" },
      ]);
      expect(steps).toEqual([{ observation: "Hi", action: "Hello" }]);
    });

    it("returns empty for empty input", () => {
      expect(messagesToTrajectory([])).toEqual([]);
    });
  });

  // ── parseSessionJsonl ─────────────────────────────────────────────

  describe("parseSessionJsonl", () => {
    it("extracts messages from JSONL lines", () => {
      const jsonl = [
        JSON.stringify({ type: "session", version: 3, id: "s1" }),
        JSON.stringify({
          type: "message",
          id: "001",
          parentId: null,
          message: { role: "user", content: "Hello" },
        }),
        JSON.stringify({
          type: "message",
          id: "002",
          parentId: "001",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Hi!" }],
            stopReason: "stop",
          },
        }),
        JSON.stringify({ type: "compaction", id: "003", summary: "..." }),
      ].join("\n");

      const messages = parseSessionJsonl(jsonl);
      expect(messages).toHaveLength(2);
      expect(messages[0].role).toBe("user");
      expect(messages[1].role).toBe("assistant");
    });

    it("skips malformed lines", () => {
      const jsonl = "not json\n" + JSON.stringify({
        type: "message",
        id: "001",
        message: { role: "user", content: "Hi" },
      });
      const messages = parseSessionJsonl(jsonl);
      expect(messages).toHaveLength(1);
    });
  });

  // ── Auto-remember hooks ──────────────────────────────────────────

  describe("auto-remember hooks", () => {
    it("registers before_reset and before_compaction hooks by default", () => {
      const { hooks } = activatePlugin("g1");
      expect(hooks["before_reset"]).toHaveLength(1);
      expect(hooks["before_compaction"]).toHaveLength(1);
    });

    it("does not register hooks when autoRemember is false", () => {
      const { hooks } = activatePlugin("g1", { autoRemember: false });
      expect(hooks["before_reset"]).toBeUndefined();
      expect(hooks["before_compaction"]).toBeUndefined();
    });

    it("registers only before_reset when onCompaction is false", () => {
      const { hooks } = activatePlugin("g1", {
        autoRemember: { onSessionReset: true, onCompaction: false },
      });
      expect(hooks["before_reset"]).toHaveLength(1);
      expect(hooks["before_compaction"]).toBeUndefined();
    });

    it("sends trajectory to PlugMem on before_reset", async () => {
      const { hooks } = activatePlugin("g1");
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ status: "ok", stats: { episodic: 2 } }),
      );

      const handler = hooks["before_reset"][0];
      await handler(
        {
          messages: [
            { role: "user", content: "Deploy the app" },
            {
              role: "assistant",
              content: [{ type: "text", text: "Running deploy script..." }],
              stopReason: "stop",
            },
            { role: "user", content: "Check status" },
            {
              role: "assistant",
              content: [{ type: "text", text: "Deploy complete." }],
              stopReason: "stop",
            },
          ],
        },
        { sessionId: "s1" },
      );

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const [url, init] = mockFetch.mock.calls[0];
      expect(url).toContain("/graphs/g1/memories");

      const sent = JSON.parse(init.body);
      expect(sent.mode).toBe("trajectory");
      expect(sent.steps).toHaveLength(2);
      expect(sent.steps[0].observation).toBe("Deploy the app");
      expect(sent.goal).toContain("Deploy the app");
    });

    it("sends trajectory to PlugMem on before_compaction", async () => {
      const { hooks } = activatePlugin("g1");
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ status: "ok", stats: {} }),
      );

      const handler = hooks["before_compaction"][0];
      await handler(
        {
          messageCount: 10,
          messages: [
            { role: "user", content: "Fix the bug" },
            {
              role: "assistant",
              content: [{ type: "text", text: "Found the issue in auth.ts" }],
              stopReason: "stop",
            },
            { role: "user", content: "Apply the fix" },
            {
              role: "assistant",
              content: [{ type: "text", text: "Fixed and tests pass." }],
              stopReason: "stop",
            },
          ],
        },
        { sessionId: "s2" },
      );

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const sent = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(sent.mode).toBe("trajectory");
      expect(sent.steps).toHaveLength(2);
    });

    it("skips auto-remember when messages are empty", async () => {
      const { hooks } = activatePlugin("g1");

      await hooks["before_reset"][0]({ messages: [] }, { sessionId: "s3" });
      await hooks["before_reset"][0]({}, { sessionId: "s4" });

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("skips auto-remember when too few steps", async () => {
      const { hooks } = activatePlugin("g1", {
        autoRemember: { minSteps: 3 },
      });

      await hooks["before_reset"][0](
        {
          messages: [
            { role: "user", content: "Hi" },
            {
              role: "assistant",
              content: [{ type: "text", text: "Hello!" }],
              stopReason: "stop",
            },
          ],
        },
        { sessionId: "s5" },
      );

      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("skips auto-remember when no defaultGraphId", async () => {
      const { hooks } = activatePlugin(); // no default graph
      if (hooks["before_reset"]) {
        await hooks["before_reset"][0](
          {
            messages: [
              { role: "user", content: "Hi" },
              {
                role: "assistant",
                content: [{ type: "text", text: "Hello!" }],
                stopReason: "stop",
              },
            ],
          },
          { sessionId: "s6" },
        );
      }
      expect(mockFetch).not.toHaveBeenCalled();
    });

    it("does not throw on API failure (best-effort)", async () => {
      const { hooks } = activatePlugin("g1");
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      mockFetch.mockRejectedValueOnce(new Error("network down"));

      await hooks["before_reset"][0](
        {
          messages: [
            { role: "user", content: "Do something" },
            {
              role: "assistant",
              content: [{ type: "text", text: "Done" }],
              stopReason: "stop",
            },
            { role: "user", content: "More" },
            {
              role: "assistant",
              content: [{ type: "text", text: "Done again" }],
              stopReason: "stop",
            },
          ],
        },
        { sessionId: "s7" },
      );

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining("[plugmem] auto-remember"),
        expect.any(String),
      );
      consoleSpy.mockRestore();
    });
  });
});
