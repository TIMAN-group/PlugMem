export interface AutoRememberConfig {
  /** Auto-store session trajectory on session reset/new (default: true) */
  onSessionReset?: boolean;
  /** Auto-store messages before context compaction (default: true) */
  onCompaction?: boolean;
  /** Minimum number of message pairs to trigger auto-remember (default: 2) */
  minSteps?: number;
}

export interface PlugMemPluginConfig {
  /** PlugMem service URL (e.g. http://localhost:8080) */
  baseUrl: string;
  /** API key for service authentication */
  apiKey?: string;
  /** Default graph ID used when not specified per-call */
  defaultGraphId?: string;
  /**
   * Extra graph IDs that are always included in `plugmem.recall` fan-out.
   * Typically a user-level semantic graph shared across agents.
   * Writes (`plugmem.remember`, auto-remember) never target these graphs.
   */
  sharedReadGraphIds?: string[];
  /** Request timeout in milliseconds (default: 30000) */
  timeoutMs?: number;
  /** Maximum retry attempts for transient failures (default: 3) */
  maxRetries?: number;
  /** Auto-remember configuration (default: enabled) */
  autoRemember?: AutoRememberConfig | false;
}

const DEFAULTS = {
  timeoutMs: 30_000,
  maxRetries: 3,
} as const;

const AUTO_REMEMBER_DEFAULTS: Required<AutoRememberConfig> = {
  onSessionReset: true,
  onCompaction: true,
  minSteps: 2,
};

export interface ResolvedConfig {
  baseUrl: string;
  apiKey?: string;
  defaultGraphId?: string;
  sharedReadGraphIds: string[];
  timeoutMs: number;
  maxRetries: number;
  autoRemember: Required<AutoRememberConfig> | false;
}

export function resolveConfig(raw: PlugMemPluginConfig): ResolvedConfig {
  return {
    baseUrl: raw.baseUrl.replace(/\/+$/, ""),
    apiKey: raw.apiKey,
    defaultGraphId: raw.defaultGraphId,
    sharedReadGraphIds: raw.sharedReadGraphIds ?? [],
    timeoutMs: raw.timeoutMs ?? DEFAULTS.timeoutMs,
    maxRetries: raw.maxRetries ?? DEFAULTS.maxRetries,
    autoRemember:
      raw.autoRemember === false
        ? false
        : { ...AUTO_REMEMBER_DEFAULTS, ...raw.autoRemember },
  };
}
