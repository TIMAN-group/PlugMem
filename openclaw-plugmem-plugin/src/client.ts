import type { PlugMemPluginConfig } from "./config.js";
import { resolveConfig } from "./config.js";
import type {
  ConsolidateRequest,
  ConsolidateResponse,
  GraphResponse,
  GraphListResponse,
  HealthResponse,
  MemoryInsertRequest,
  MemoryInsertResponse,
  NodeListResponse,
  ReasonRequest,
  ReasonResponse,
  RetrieveRequest,
  RetrieveResponse,
  StatsResponse,
} from "./types.js";
import { PlugMemError, PlugMemConnectionError } from "./types.js";

// ── Helpers ──────────────────────────────────────────────────────────

const RETRYABLE_STATUS_CODES = new Set([408, 429, 502, 503, 504]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Client ───────────────────────────────────────────────────────────

export class PlugMemClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  constructor(config: PlugMemPluginConfig) {
    const resolved = resolveConfig(config);
    this.baseUrl = resolved.baseUrl;
    this.apiKey = resolved.apiKey;
    this.timeoutMs = resolved.timeoutMs;
    this.maxRetries = resolved.maxRetries;
  }

  // ── Low-level request ────────────────────────────────────────────

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }

    let lastError: unknown;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      if (attempt > 0) {
        // Exponential backoff: 500ms, 1s, 2s, ...
        await sleep(500 * 2 ** (attempt - 1));
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      try {
        const response = await fetch(url, {
          method,
          headers,
          body: body !== undefined ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timer);

        if (response.ok) {
          // 204 No Content
          if (response.status === 204) {
            return undefined as T;
          }
          return (await response.json()) as T;
        }

        // Parse error body for context
        let errorBody: unknown;
        try {
          errorBody = await response.json();
        } catch {
          errorBody = await response.text().catch(() => null);
        }

        // Retry on transient errors
        if (
          RETRYABLE_STATUS_CODES.has(response.status) &&
          attempt < this.maxRetries
        ) {
          lastError = new PlugMemError(
            `${method} ${path} failed with ${response.status}`,
            response.status,
            errorBody,
          );
          continue;
        }

        throw new PlugMemError(
          `${method} ${path} failed with ${response.status}`,
          response.status,
          errorBody,
        );
      } catch (err) {
        clearTimeout(timer);

        if (err instanceof PlugMemError) {
          throw err;
        }

        // Network / timeout errors are retryable
        lastError = err;
        if (attempt < this.maxRetries) {
          continue;
        }

        throw new PlugMemConnectionError(
          `${method} ${path} failed after ${this.maxRetries + 1} attempts`,
          lastError,
        );
      }
    }

    // Should not reach here, but TypeScript needs it
    throw new PlugMemConnectionError(
      `${method} ${path} failed after ${this.maxRetries + 1} attempts`,
      lastError,
    );
  }

  // ── Graph CRUD ───────────────────────────────────────────────────

  async createGraph(graphId?: string): Promise<GraphResponse> {
    return this.request<GraphResponse>("POST", "/api/v1/graphs", {
      graph_id: graphId,
    });
  }

  async listGraphs(): Promise<GraphListResponse> {
    return this.request<GraphListResponse>("GET", "/api/v1/graphs");
  }

  async getGraph(graphId: string): Promise<GraphResponse> {
    return this.request<GraphResponse>("GET", `/api/v1/graphs/${graphId}`);
  }

  async deleteGraph(graphId: string): Promise<void> {
    await this.request<void>("DELETE", `/api/v1/graphs/${graphId}`);
  }

  async getStats(graphId: string): Promise<StatsResponse> {
    return this.request<StatsResponse>(
      "GET",
      `/api/v1/graphs/${graphId}/stats`,
    );
  }

  async getNodes(
    graphId: string,
    nodeType: string = "semantic",
    limit: number = 50,
    offset: number = 0,
  ): Promise<NodeListResponse> {
    const params = new URLSearchParams({
      node_type: nodeType,
      limit: String(limit),
      offset: String(offset),
    });
    return this.request<NodeListResponse>(
      "GET",
      `/api/v1/graphs/${graphId}/nodes?${params}`,
    );
  }

  // ── Memory Insertion ─────────────────────────────────────────────

  async insertMemories(
    graphId: string,
    request: MemoryInsertRequest,
  ): Promise<MemoryInsertResponse> {
    return this.request<MemoryInsertResponse>(
      "POST",
      `/api/v1/graphs/${graphId}/memories`,
      request,
    );
  }

  async insertTrajectory(
    graphId: string,
    goal: string,
    steps: Array<{ observation: string; action: string }>,
  ): Promise<MemoryInsertResponse> {
    return this.insertMemories(graphId, { mode: "trajectory", goal, steps });
  }

  async insertStructured(
    graphId: string,
    memories: Omit<
      import("./types.js").StructuredInsertRequest,
      "mode"
    >,
  ): Promise<MemoryInsertResponse> {
    return this.insertMemories(graphId, { mode: "structured", ...memories });
  }

  // ── Retrieval & Reasoning ────────────────────────────────────────

  async retrieve(
    graphId: string,
    query: RetrieveRequest,
  ): Promise<RetrieveResponse> {
    return this.request<RetrieveResponse>(
      "POST",
      `/api/v1/graphs/${graphId}/retrieve`,
      query,
    );
  }

  async reason(
    graphId: string,
    query: ReasonRequest,
  ): Promise<ReasonResponse> {
    return this.request<ReasonResponse>(
      "POST",
      `/api/v1/graphs/${graphId}/reason`,
      query,
    );
  }

  async consolidate(
    graphId: string,
    options?: ConsolidateRequest,
  ): Promise<ConsolidateResponse> {
    return this.request<ConsolidateResponse>(
      "POST",
      `/api/v1/graphs/${graphId}/consolidate`,
      options ?? {},
    );
  }

  // ── Health ───────────────────────────────────────────────────────

  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/health");
  }
}
