import type { ResolvedCoreConfig } from "./config.js";
import {
  PlugMemConnectionError,
  PlugMemError,
  type ExtractRequest,
  type ExtractResponse,
  type GraphResponse,
  type GraphListResponse,
  type HealthResponse,
  type MemoryInsertRequest,
  type MemoryInsertResponse,
  type ReasonRequest,
  type ReasonResponse,
  type RetrieveRequest,
  type RetrieveResponse,
  type StatsResponse,
} from "./types.js";

const RETRYABLE_STATUS_CODES = new Set([408, 429, 502, 503, 504]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export class PlugMemClient {
  constructor(private readonly cfg: ResolvedCoreConfig) {}

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.cfg.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    if (this.cfg.apiKey) headers["X-API-Key"] = this.cfg.apiKey;

    let lastError: unknown;

    for (let attempt = 0; attempt <= this.cfg.maxRetries; attempt++) {
      if (attempt > 0) await sleep(500 * 2 ** (attempt - 1));

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.cfg.timeoutMs);

      try {
        const response = await fetch(url, {
          method,
          headers,
          body: body !== undefined ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });
        clearTimeout(timer);

        if (response.ok) {
          if (response.status === 204) return undefined as T;
          return (await response.json()) as T;
        }

        let errorBody: unknown;
        try {
          errorBody = await response.json();
        } catch {
          errorBody = await response.text().catch(() => null);
        }

        if (
          RETRYABLE_STATUS_CODES.has(response.status) &&
          attempt < this.cfg.maxRetries
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
        if (err instanceof PlugMemError) throw err;

        lastError = err;
        if (attempt < this.cfg.maxRetries) continue;

        throw new PlugMemConnectionError(
          `${method} ${path} failed after ${this.cfg.maxRetries + 1} attempts`,
          lastError,
        );
      }
    }

    throw new PlugMemConnectionError(
      `${method} ${path} failed after ${this.cfg.maxRetries + 1} attempts`,
      lastError,
    );
  }

  async createGraph(graphId?: string): Promise<GraphResponse> {
    return this.request<GraphResponse>("POST", "/api/v1/graphs", {
      graph_id: graphId,
    });
  }

  async getGraph(graphId: string): Promise<GraphResponse> {
    return this.request<GraphResponse>(
      "GET",
      `/api/v1/graphs/${encodeURIComponent(graphId)}`,
    );
  }

  async listGraphs(): Promise<GraphListResponse> {
    return this.request<GraphListResponse>("GET", "/api/v1/graphs");
  }

  async deleteGraph(graphId: string): Promise<void> {
    await this.request<void>(
      "DELETE",
      `/api/v1/graphs/${encodeURIComponent(graphId)}`,
    );
  }

  // Idempotent: returns the graph if it exists, otherwise creates it.
  // Graph IDs derived from repo URLs may not exist on first session;
  // core uses this on session_start.
  async ensureGraph(graphId: string): Promise<GraphResponse> {
    try {
      return await this.getGraph(graphId);
    } catch (err) {
      if (err instanceof PlugMemError && err.statusCode === 404) {
        return this.createGraph(graphId);
      }
      throw err;
    }
  }

  async getStats(graphId: string): Promise<StatsResponse> {
    return this.request<StatsResponse>(
      "GET",
      `/api/v1/graphs/${encodeURIComponent(graphId)}/stats`,
    );
  }

  async insertMemories(
    graphId: string,
    request: MemoryInsertRequest,
  ): Promise<MemoryInsertResponse> {
    return this.request<MemoryInsertResponse>(
      "POST",
      `/api/v1/graphs/${encodeURIComponent(graphId)}/memories`,
      request,
    );
  }

  async retrieve(
    graphId: string,
    query: RetrieveRequest,
  ): Promise<RetrieveResponse> {
    return this.request<RetrieveResponse>(
      "POST",
      `/api/v1/graphs/${encodeURIComponent(graphId)}/retrieve`,
      query,
    );
  }

  async reason(
    graphId: string,
    query: ReasonRequest,
  ): Promise<ReasonResponse> {
    return this.request<ReasonResponse>(
      "POST",
      `/api/v1/graphs/${encodeURIComponent(graphId)}/reason`,
      query,
    );
  }

  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/health");
  }

  async extract(request: ExtractRequest): Promise<ExtractResponse> {
    return this.request<ExtractResponse>(
      "POST",
      "/api/v1/extract",
      request,
    );
  }
}
