import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { PlugMemClient } from "../src/client.js";
import { PlugMemError, PlugMemConnectionError } from "../src/types.js";

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

function errorResponse(body: unknown, status: number): Response {
  return {
    ok: false,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as unknown as Response;
}

// ── Tests ────────────────────────────────────────────────────────────

describe("PlugMemClient", () => {
  let client: PlugMemClient;

  beforeEach(() => {
    mockFetch.mockReset();
    client = new PlugMemClient({
      baseUrl: "http://localhost:8080",
      apiKey: "test-key",
      maxRetries: 1,
      timeoutMs: 5000,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ── Headers ──────────────────────────────────────────────────────

  it("sends API key in X-API-Key header", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ graphs: [] }));

    await client.listGraphs();

    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers["X-API-Key"]).toBe("test-key");
  });

  it("omits API key header when not configured", async () => {
    const noAuthClient = new PlugMemClient({ baseUrl: "http://localhost:8080" });
    mockFetch.mockResolvedValueOnce(jsonResponse({ graphs: [] }));

    await noAuthClient.listGraphs();

    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers["X-API-Key"]).toBeUndefined();
  });

  // ── Graph CRUD ───────────────────────────────────────────────────

  describe("graphs", () => {
    it("creates a graph", async () => {
      const body = { graph_id: "test-graph", stats: {} };
      mockFetch.mockResolvedValueOnce(jsonResponse(body, 201));

      const result = await client.createGraph("test-graph");
      expect(result).toEqual(body);

      const [url, init] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8080/api/v1/graphs");
      expect(init.method).toBe("POST");
      expect(JSON.parse(init.body)).toEqual({ graph_id: "test-graph" });
    });

    it("lists graphs", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ graphs: ["g1", "g2"] }),
      );

      const result = await client.listGraphs();
      expect(result.graphs).toEqual(["g1", "g2"]);
    });

    it("gets a graph", async () => {
      const body = { graph_id: "g1", stats: { semantic: 5 } };
      mockFetch.mockResolvedValueOnce(jsonResponse(body));

      const result = await client.getGraph("g1");
      expect(result).toEqual(body);
    });

    it("deletes a graph", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse(undefined, 204));

      await expect(client.deleteGraph("g1")).resolves.toBeUndefined();

      const [url, init] = mockFetch.mock.calls[0];
      expect(url).toBe("http://localhost:8080/api/v1/graphs/g1");
      expect(init.method).toBe("DELETE");
    });

    it("gets stats", async () => {
      const body = { graph_id: "g1", stats: { semantic: 3, procedural: 2 } };
      mockFetch.mockResolvedValueOnce(jsonResponse(body));

      const result = await client.getStats("g1");
      expect(result.stats.semantic).toBe(3);
    });

    it("gets nodes with query params", async () => {
      const body = {
        graph_id: "g1",
        node_type: "episodic",
        count: 1,
        nodes: [{ episodic_id: 0 }],
      };
      mockFetch.mockResolvedValueOnce(jsonResponse(body));

      const result = await client.getNodes("g1", "episodic", 10, 5);
      expect(result.node_type).toBe("episodic");

      const [url] = mockFetch.mock.calls[0];
      expect(url).toContain("node_type=episodic");
      expect(url).toContain("limit=10");
      expect(url).toContain("offset=5");
    });
  });

  // ── Memory Insertion ─────────────────────────────────────────────

  describe("memories", () => {
    it("inserts trajectory", async () => {
      const resp = { status: "ok", stats: { episodic: 2 } };
      mockFetch.mockResolvedValueOnce(jsonResponse(resp));

      const result = await client.insertTrajectory("g1", "do something", [
        { observation: "saw X", action: "did Y" },
        { observation: "saw Z", action: "did W" },
      ]);
      expect(result.stats.episodic).toBe(2);

      const sent = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(sent.mode).toBe("trajectory");
      expect(sent.goal).toBe("do something");
      expect(sent.steps).toHaveLength(2);
    });

    it("inserts structured semantic memories", async () => {
      const resp = { status: "ok", stats: { semantic: 1 } };
      mockFetch.mockResolvedValueOnce(jsonResponse(resp));

      const result = await client.insertStructured("g1", {
        semantic: [{ semantic_memory: "the sky is blue", tags: ["fact"] }],
      });
      expect(result.stats.semantic).toBe(1);

      const sent = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(sent.mode).toBe("structured");
    });
  });

  // ── Retrieval & Reasoning ────────────────────────────────────────

  describe("retrieval", () => {
    it("retrieves memories", async () => {
      const resp = {
        mode: "semantic_memory",
        reasoning_prompt: [{ role: "user", content: "context..." }],
        variables: {},
      };
      mockFetch.mockResolvedValueOnce(jsonResponse(resp));

      const result = await client.retrieve("g1", {
        observation: "what color is the sky?",
      });
      expect(result.mode).toBe("semantic_memory");
    });

    it("reasons over memories", async () => {
      const resp = {
        mode: "semantic_memory",
        reasoning: "The sky is blue based on stored facts.",
        reasoning_prompt: [{ role: "user", content: "context..." }],
      };
      mockFetch.mockResolvedValueOnce(jsonResponse(resp));

      const result = await client.reason("g1", {
        observation: "what color is the sky?",
      });
      expect(result.reasoning).toContain("blue");
    });

    it("consolidates", async () => {
      const resp = { status: "ok", stats: { merged: 2 } };
      mockFetch.mockResolvedValueOnce(jsonResponse(resp));

      const result = await client.consolidate("g1", {
        merge_threshold: 0.7,
      });
      expect(result.status).toBe("ok");
    });
  });

  // ── Health ───────────────────────────────────────────────────────

  it("checks health", async () => {
    const resp = {
      status: "ok",
      version: "0.1.0",
      llm_available: true,
      embedding_available: true,
      chroma_available: true,
    };
    mockFetch.mockResolvedValueOnce(jsonResponse(resp));

    const result = await client.healthCheck();
    expect(result.status).toBe("ok");

    const [url] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/health");
  });

  // ── Error handling ───────────────────────────────────────────────

  describe("error handling", () => {
    it("throws PlugMemError on 404", async () => {
      mockFetch.mockResolvedValueOnce(
        errorResponse({ detail: "Graph not found" }, 404),
      );

      try {
        await client.getGraph("missing");
        expect.unreachable("should have thrown");
      } catch (err) {
        expect(err).toBeInstanceOf(PlugMemError);
        expect((err as PlugMemError).statusCode).toBe(404);
        expect((err as PlugMemError).body).toEqual({ detail: "Graph not found" });
      }
    });

    it("retries on 503 then succeeds", async () => {
      mockFetch
        .mockResolvedValueOnce(errorResponse({ detail: "unavailable" }, 503))
        .mockResolvedValueOnce(jsonResponse({ graphs: [] }));

      const result = await client.listGraphs();
      expect(result.graphs).toEqual([]);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it("retries on network error then succeeds", async () => {
      mockFetch
        .mockRejectedValueOnce(new TypeError("fetch failed"))
        .mockResolvedValueOnce(jsonResponse({ graphs: [] }));

      const result = await client.listGraphs();
      expect(result.graphs).toEqual([]);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it("throws PlugMemConnectionError after exhausting retries", async () => {
      mockFetch
        .mockRejectedValueOnce(new TypeError("fetch failed"))
        .mockRejectedValueOnce(new TypeError("fetch failed"));

      await expect(client.listGraphs()).rejects.toThrow(
        PlugMemConnectionError,
      );
    });

    it("does not retry on 400", async () => {
      mockFetch.mockResolvedValueOnce(
        errorResponse({ detail: "bad request" }, 400),
      );

      await expect(client.createGraph()).rejects.toThrow(PlugMemError);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  // ── URL normalization ────────────────────────────────────────────

  it("strips trailing slashes from baseUrl", async () => {
    const c = new PlugMemClient({ baseUrl: "http://localhost:8080///" });
    mockFetch.mockResolvedValueOnce(jsonResponse({ graphs: [] }));

    await c.listGraphs();

    const [url] = mockFetch.mock.calls[0];
    expect(url).toBe("http://localhost:8080/api/v1/graphs");
  });
});
