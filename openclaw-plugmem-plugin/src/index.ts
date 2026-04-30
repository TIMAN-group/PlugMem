import { Type } from "@sinclair/typebox";
import { PlugMemClient } from "./client.js";
import type { PlugMemPluginConfig, ResolvedConfig } from "./config.js";
import { resolveConfig } from "./config.js";
import type {
  PlugMemError,
  ReasonResponse,
  RetrieveResponse,
  TrajectoryStep,
} from "./types.js";

// ── OpenClaw SDK type stubs ──────────────────────────────────────────
// Minimal interfaces matching the OpenClaw plugin-sdk API.
// When the real SDK is installed these are satisfied by its exports.

interface ContentBlock {
  type: string;
  text?: string;
}

interface ToolDefinition {
  name: string;
  description: string;
  parameters: unknown;
  execute: (
    id: string,
    params: Record<string, unknown>,
  ) => Promise<{ content: ContentBlock[] }>;
}

// Hook event types matching OpenClaw's typed hook system

interface BeforeResetEvent {
  sessionFile?: string;
  messages?: SessionMessage[];
  reason?: string;
}

interface BeforeCompactionEvent {
  messageCount: number;
  compactingCount?: number;
  tokenCount?: number;
  messages?: SessionMessage[];
  sessionFile?: string;
}

interface HookContext {
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  workspaceDir?: string;
}

/** Content block inside an assistant message. */
interface MessageContentBlock {
  type: string;         // "text", "toolCall", "thinking", "image"
  text?: string;        // for type: "text"
  thinking?: string;    // for type: "thinking"
  id?: string;          // for type: "toolCall"
  name?: string;        // for type: "toolCall"
  arguments?: Record<string, unknown>; // for type: "toolCall"
}

/** A message from the OpenClaw session transcript (AgentMessage union). */
interface SessionMessage {
  role?: string;        // "user" | "assistant" | "toolResult" | "bashExecution" | "compactionSummary" | ...
  content?: string | MessageContentBlock[];
  // toolResult fields
  toolCallId?: string;
  toolName?: string;
  isError?: boolean;
  // assistant fields
  stopReason?: string;  // "stop" | "toolUse" | "length" | "error" | "aborted"
  // bashExecution fields
  command?: string;
  output?: string;
  exitCode?: number;
  // compactionSummary fields
  summary?: string;
  // catch-all for fields we don't use
  [key: string]: unknown;
}

type HookHandler<E> = (event: E, ctx: HookContext) => Promise<void> | void;

interface OpenClawPluginApi {
  registerTool(tool: ToolDefinition): void;
  on(hookName: "before_reset", handler: HookHandler<BeforeResetEvent>): void;
  on(
    hookName: "before_compaction",
    handler: HookHandler<BeforeCompactionEvent>,
  ): void;
  on(hookName: string, handler: HookHandler<unknown>): void;
}

interface PluginEntry {
  id: string;
  name: string;
  description: string;
  version: string;
  activate: (api: OpenClawPluginApi) => void;
}

// Re-export client for direct usage outside OpenClaw
export { PlugMemClient } from "./client.js";
export type { PlugMemPluginConfig } from "./config.js";
export * from "./types.js";

// ── Tool helpers ─────────────────────────────────────────────────────

function textContent(text: string): { content: ContentBlock[] } {
  return { content: [{ type: "text", text }] };
}

function errorContent(err: unknown): { content: ContentBlock[] } {
  const plugMemErr = err as PlugMemError;
  if (plugMemErr.statusCode) {
    return textContent(
      `PlugMem error (${plugMemErr.statusCode}): ${plugMemErr.message}`,
    );
  }
  return textContent(
    `PlugMem error: ${err instanceof Error ? err.message : String(err)}`,
  );
}

function requireGraphId(
  params: Record<string, unknown>,
  defaultGraphId?: string,
): string {
  const graphId = (params.graph_id as string) ?? defaultGraphId;
  if (!graphId) {
    throw Object.assign(
      new Error(
        "graph_id is required (pass it explicitly or configure defaultGraphId)",
      ),
      { statusCode: 400 },
    );
  }
  return graphId;
}

// ── Message → Trajectory extraction ──────────────────────────────────

/** Extract text-only content from a message (string or content-block array). */
function extractText(msg: SessionMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content
      .filter((b) => b.type === "text" && b.text)
      .map((b) => b.text!)
      .join("\n");
  }
  return "";
}

/** Format tool calls from an assistant message's content blocks. */
function formatToolCalls(msg: SessionMessage): string {
  if (!Array.isArray(msg.content)) return "";
  const calls = msg.content.filter((b) => b.type === "toolCall" && b.name);
  if (calls.length === 0) return "";
  return calls
    .map((tc) => {
      const args = tc.arguments ?? {};
      // Compact display: tool(key=val, ...)
      const argStr = Object.entries(args)
        .map(([k, v]) => {
          const vs = typeof v === "string" ? truncate(v, 80) : JSON.stringify(v);
          return `${k}=${vs}`;
        })
        .join(", ");
      return `[${tc.name}(${argStr})]`;
    })
    .join(" ");
}

/** Build a full action description from an assistant message (text + tool calls). */
function formatAssistantAction(msg: SessionMessage): string {
  const text = extractText(msg);
  const tools = formatToolCalls(msg);
  return [text, tools].filter(Boolean).join("\n");
}

/** Format a toolResult message as an observation. */
function formatToolResult(msg: SessionMessage): string {
  const label = msg.isError ? `[ERROR from ${msg.toolName}]` : `[${msg.toolName} result]`;
  const text = extractText(msg);
  return text ? `${label} ${truncate(text, 500)}` : label;
}

/** Format a bashExecution message as an observation. */
function formatBashResult(msg: SessionMessage): string {
  const cmd = msg.command ? truncate(msg.command, 120) : "bash";
  const prefix = msg.exitCode !== undefined && msg.exitCode !== 0
    ? `[bash exit=${msg.exitCode}]`
    : "[bash]";
  const output = msg.output ? truncate(msg.output, 500) : "";
  return `${prefix} ${cmd}${output ? "\n" + output : ""}`;
}

/**
 * Convert OpenClaw session messages into PlugMem (observation, action) steps.
 *
 * An OpenClaw agentic turn looks like:
 *   user → assistant(toolCalls) → toolResult → assistant(toolCalls) → toolResult → assistant(stop)
 *
 * We unfold this into multiple PlugMem steps:
 *   Step 0: obs = user message,       action = assistant decision + tool calls
 *   Step 1: obs = tool results,       action = next assistant decision + tool calls
 *   Step 2: obs = tool results,       action = final assistant response
 *
 * This gives PlugMem's structuring pipeline (subgoal detection, reward eval,
 * state tracking) meaningful content at each step.
 */
export function messagesToTrajectory(
  messages: SessionMessage[],
): TrajectoryStep[] {
  const steps: TrajectoryStep[] = [];

  // Accumulate the current observation (may be user msg or tool results)
  let pendingObservation: string | null = null;
  // Accumulate tool results between assistant messages
  const toolResultBuffer: string[] = [];

  for (const msg of messages) {
    const role = msg.role;

    if (role === "user") {
      // Flush any buffered tool results as an unclosed observation
      // (shouldn't normally happen — means assistant didn't respond)
      flushToolResults();

      const text = extractText(msg);
      if (!text) continue;

      // Merge consecutive user messages
      if (pendingObservation !== null) {
        pendingObservation += "\n" + text;
      } else {
        pendingObservation = text;
      }
    } else if (role === "assistant") {
      // Flush buffered tool results into the pending observation
      flushToolResults();

      const action = formatAssistantAction(msg);
      if (!action) continue;

      if (pendingObservation !== null) {
        steps.push({ observation: pendingObservation, action });
        pendingObservation = null;
      }
      // If stopReason is "toolUse", the next messages will be tool results
      // which become the observation for the next step.
      // (pendingObservation stays null — tool results will fill it)
    } else if (role === "toolResult") {
      toolResultBuffer.push(formatToolResult(msg));
    } else if (role === "bashExecution") {
      toolResultBuffer.push(formatBashResult(msg));
    }
    // skip: system, compactionSummary, branchSummary, custom, etc.
  }

  return steps;

  function flushToolResults(): void {
    if (toolResultBuffer.length === 0) return;
    const combined = toolResultBuffer.join("\n");
    toolResultBuffer.length = 0;
    if (pendingObservation !== null) {
      pendingObservation += "\n" + combined;
    } else {
      pendingObservation = combined;
    }
  }
}

/**
 * Parse an OpenClaw JSONL session file into SessionMessage[].
 * Each line is a JSON object; we extract the `message` field from `type: "message"` entries.
 */
export function parseSessionJsonl(content: string): SessionMessage[] {
  const messages: SessionMessage[] = [];
  for (const line of content.split("\n")) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === "message" && entry.message) {
        messages.push(entry.message as SessionMessage);
      }
    } catch {
      // skip malformed lines
    }
  }
  return messages;
}

/**
 * Infer a goal from the first user message in the session.
 * Falls back to a generic label.
 */
function inferGoal(messages: SessionMessage[]): string {
  for (const msg of messages) {
    if (msg.role === "user") {
      const text = extractText(msg);
      if (text) return truncate(text, 200);
    }
  }
  return "Agent session";
}

// ── Auto-remember handler ────────────────────────────────────────────

async function autoRemember(
  client: PlugMemClient,
  config: ResolvedConfig,
  messages: SessionMessage[] | undefined,
  source: string,
): Promise<void> {
  if (!messages || messages.length === 0) return;
  if (!config.defaultGraphId) return; // can't auto-remember without a target graph
  if (config.autoRemember === false) return;

  const steps = messagesToTrajectory(messages);
  if (steps.length < config.autoRemember.minSteps) return;

  const goal = inferGoal(messages);

  try {
    await client.insertTrajectory(config.defaultGraphId, goal, steps);
  } catch (err) {
    // Auto-remember is best-effort — log but don't crash the session
    console.error(
      `[plugmem] auto-remember (${source}) failed:`,
      err instanceof Error ? err.message : err,
    );
  }
}

/**
 * Read session messages from a JSONL file path.
 * Used as a fallback when hook events don't include messages directly
 * (e.g. before_compaction in current OpenClaw versions).
 */
async function readSessionFile(
  sessionFile: string | undefined,
): Promise<SessionMessage[] | undefined> {
  if (!sessionFile) return undefined;
  try {
    // Dynamic import — node:fs is available in OpenClaw's Node runtime
    const fs = await import("node:fs/promises");
    const content = await fs.readFile(sessionFile, "utf-8");
    return parseSessionJsonl(content);
  } catch {
    return undefined;
  }
}

// ── Plugin definition ────────────────────────────────────────────────

export function createPlugMemPlugin(config: PlugMemPluginConfig): PluginEntry {
  const resolved = resolveConfig(config);
  const client = new PlugMemClient(config);
  const defaultGraphId = resolved.defaultGraphId;

  return {
    id: "plugmem",
    name: "PlugMem",
    description:
      "Long-term memory for LLM agents — store and recall experiences across sessions",
    version: "0.1.0",

    activate(api: OpenClawPluginApi) {
      // ── plugmem.remember ─────────────────────────────────────────
      api.registerTool({
        name: "plugmem.remember",
        description:
          "Store information in long-term memory. Accepts either free-text " +
          "(stored as a semantic memory with optional tags) or a full " +
          "trajectory of observation/action steps that will be structured " +
          "by the PlugMem service into semantic, procedural, and episodic memories.",
        parameters: Type.Object({
          graph_id: Type.Optional(
            Type.String({
              description: "Memory graph ID (uses default if omitted)",
            }),
          ),
          // -- Simple semantic insertion (most common for agents) --
          text: Type.Optional(
            Type.String({ description: "Free-text information to remember" }),
          ),
          tags: Type.Optional(
            Type.Array(Type.String(), {
              description: "Category tags for the memory",
            }),
          ),
          // -- Full trajectory insertion --
          goal: Type.Optional(
            Type.String({
              description: "Task goal (required for trajectory mode)",
            }),
          ),
          steps: Type.Optional(
            Type.Array(
              Type.Object({
                observation: Type.String(),
                action: Type.String(),
              }),
              {
                description:
                  "Observation/action pairs (required for trajectory mode)",
              },
            ),
          ),
        }),

        async execute(_id, params) {
          try {
            const graphId = requireGraphId(params, defaultGraphId);

            // Trajectory mode: goal + steps
            if (params.steps && params.goal) {
              const result = await client.insertTrajectory(
                graphId,
                params.goal as string,
                params.steps as Array<{
                  observation: string;
                  action: string;
                }>,
              );
              return textContent(
                `Stored trajectory (${(params.steps as unknown[]).length} steps). ` +
                  `Graph now has: ${formatStats(result.stats)}`,
              );
            }

            // Semantic mode: text (+ optional tags)
            if (params.text) {
              const result = await client.insertStructured(graphId, {
                semantic: [
                  {
                    semantic_memory: params.text as string,
                    tags: (params.tags as string[]) ?? [],
                  },
                ],
              });
              return textContent(
                `Remembered: "${truncate(params.text as string, 80)}". ` +
                  `Graph now has: ${formatStats(result.stats)}`,
              );
            }

            return textContent(
              "Nothing to store. Provide either `text` (semantic memory) " +
                "or `goal` + `steps` (trajectory).",
            );
          } catch (err) {
            return errorContent(err);
          }
        },
      });

      // ── plugmem.recall ───────────────────────────────────────────
      api.registerTool({
        name: "plugmem.recall",
        description:
          "Retrieve relevant memories from long-term storage. Returns " +
          "LLM-synthesized reasoning over the most relevant memories " +
          "matching the query. Use this when you need to recall past " +
          "experiences, facts, or procedures.",
        parameters: Type.Object({
          graph_id: Type.Optional(
            Type.String({
              description: "Memory graph ID (uses default if omitted)",
            }),
          ),
          observation: Type.String({
            description:
              "Current observation or question to recall memories for",
          }),
          goal: Type.Optional(
            Type.String({ description: "Current task goal for context" }),
          ),
          mode: Type.Optional(
            Type.Union(
              [
                Type.Literal("semantic_memory"),
                Type.Literal("episodic_memory"),
                Type.Literal("procedural_memory"),
              ],
              {
                description:
                  "Force a retrieval mode, or omit to let the service auto-detect",
              },
            ),
          ),
          raw: Type.Optional(
            Type.Boolean({
              description:
                "If true, return raw retrieval prompt instead of LLM reasoning (default: false)",
              default: false,
            }),
          ),
        }),

        async execute(_id, params) {
          try {
            const primaryGraphId = requireGraphId(params, defaultGraphId);
            const readGraphIds = dedupe([
              primaryGraphId,
              ...resolved.sharedReadGraphIds,
            ]);
            const query = {
              observation: params.observation as string,
              goal: params.goal as string | undefined,
              mode: params.mode as
                | "semantic_memory"
                | "episodic_memory"
                | "procedural_memory"
                | undefined,
            };

            // Single-graph path — preserve exact output shape for callers
            // that don't configure sharedReadGraphIds.
            if (readGraphIds.length === 1) {
              if (params.raw) {
                const result = await client.retrieve(readGraphIds[0], query);
                return textContent(
                  `[${result.mode}] Retrieved memories:\n\n` +
                    formatRetrievalPrompt(result),
                );
              }
              const result = await client.reason(readGraphIds[0], query);
              return textContent(`[${result.mode}] ${result.reasoning}`);
            }

            // Multi-graph fan-out. A failure on any single graph (e.g. a
            // stale sharedReadGraphIds entry) does not abort the call —
            // successful graphs still return their memories.
            if (params.raw) {
              const settled = await Promise.allSettled(
                readGraphIds.map((gid) => client.retrieve(gid, query)),
              );
              return textContent(
                formatFanOut(readGraphIds, settled, (r) =>
                  formatRetrievalPrompt(r),
                ),
              );
            }
            const settled = await Promise.allSettled(
              readGraphIds.map((gid) => client.reason(gid, query)),
            );
            return textContent(
              formatFanOut(readGraphIds, settled, (r) => r.reasoning),
            );
          } catch (err) {
            return errorContent(err);
          }
        },
      });

      // ── Auto-remember hooks ──────────────────────────────────────

      if (resolved.autoRemember !== false) {
        // Before session reset (/new, /reset) — capture the full session
        if (resolved.autoRemember.onSessionReset) {
          api.on("before_reset", async (event) => {
            await autoRemember(
              client,
              resolved,
              event.messages,
              "session_reset",
            );
          });
        }

        // Before context compaction — capture messages about to be lost.
        // Note: current OpenClaw versions don't populate event.messages for
        // this hook, so we fall back to reading the JSONL session file.
        if (resolved.autoRemember.onCompaction) {
          api.on("before_compaction", async (event) => {
            const messages =
              event.messages ?? (await readSessionFile(event.sessionFile));
            await autoRemember(client, resolved, messages, "compaction");
          });
        }
      }
    },
  };
}

// ── Utility ──────────────────────────────────────────────────────────

function formatStats(stats: Record<string, number>): string {
  return Object.entries(stats)
    .map(([k, v]) => `${v} ${k}`)
    .join(", ");
}

function truncate(s: string, len: number): string {
  return s.length <= len ? s : s.slice(0, len - 1) + "\u2026";
}

function dedupe<T>(xs: T[]): T[] {
  return Array.from(new Set(xs));
}

function formatRetrievalPrompt(result: RetrieveResponse): string {
  return result.reasoning_prompt
    .map((m) => `**${m.role}**: ${m.content}`)
    .join("\n\n");
}

function formatFailure(reason: unknown): string {
  const err = reason as PlugMemError;
  if (err && typeof err.statusCode === "number") {
    return `PlugMem error (${err.statusCode}): ${err.message}`;
  }
  return `PlugMem error: ${reason instanceof Error ? reason.message : String(reason)}`;
}

function formatFanOut<T extends RetrieveResponse | ReasonResponse>(
  graphIds: string[],
  settled: PromiseSettledResult<T>[],
  renderBody: (value: T) => string,
): string {
  return settled
    .map((s, i) => {
      const gid = graphIds[i];
      if (s.status === "fulfilled") {
        return `[graph:${gid} | ${s.value.mode}]\n${renderBody(s.value)}`;
      }
      return `[graph:${gid} | error]\n${formatFailure(s.reason)}`;
    })
    .join("\n\n---\n\n");
}

// ── Default export for OpenClaw plugin loader ────────────────────────

export default createPlugMemPlugin;
