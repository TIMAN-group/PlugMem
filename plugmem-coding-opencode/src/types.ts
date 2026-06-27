// src/types.ts

/**
 * OpenCode Plugin API Interfaces
 * Updated to reflect the actual OpenCode plugin architecture as defined in the docs.
 */

export interface PluginContext {
  project: any;
  client: any;
  $: any;
  directory: string;
  worktree: any;
}

export interface PluginHooks {
  "session.created"?: (input: any, output: any) => Promise<void>;
  "session.compacted"?: (input: any, output: any) => Promise<void>;
  "session.idle"?: (input: any, output: any) => Promise<void>;
  "session.deleted"?: (input: any, output: any) => Promise<void>;
  "tool.execute.before"?: (input: any, output: any) => Promise<void>;
  "tool.execute.after"?: (input: any, output: any) => Promise<void>;
  "message.updated"?: (input: any, output: any) => Promise<void>;
  "message.part.updated"?: (input: any, output: any) => Promise<void>;
  event?: (input: { event: { type: string; data?: unknown; session?: { id: string } } }) => Promise<void>;
}

export type Plugin = (ctx: PluginContext) => Promise<PluginHooks>;
