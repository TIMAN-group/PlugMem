// Core config: just the bits needed to talk to the PlugMem service.
// Adapter-specific config (recall token caps, auto-remember flags, hook
// scopes) lives in the adapter packages.

export interface CoreConfig {
  baseUrl: string;
  apiKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  // User identifier for the per-user shared-read graph. If omitted, the
  // user-level graph is not consulted on recall.
  userId?: string;
}

export interface ResolvedCoreConfig {
  baseUrl: string;
  apiKey?: string;
  timeoutMs: number;
  maxRetries: number;
  userId?: string;
}

const DEFAULTS = {
  timeoutMs: 30_000,
  maxRetries: 3,
} as const;

export function resolveConfig(raw: CoreConfig): ResolvedCoreConfig {
  if (!raw.baseUrl) {
    throw new Error("CoreConfig.baseUrl is required");
  }
  return {
    baseUrl: raw.baseUrl.replace(/\/+$/, ""),
    apiKey: raw.apiKey,
    timeoutMs: raw.timeoutMs ?? DEFAULTS.timeoutMs,
    maxRetries: raw.maxRetries ?? DEFAULTS.maxRetries,
    userId: raw.userId,
  };
}
