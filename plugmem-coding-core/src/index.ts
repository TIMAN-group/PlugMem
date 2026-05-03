export type {
  ContextInjection,
  CoreCallbacks,
  Harness,
  PreCompactEvent,
  PreToolEvent,
  PostToolEvent,
  SessionEndEvent,
  SessionStartEvent,
  SessionState,
  UserPromptEvent,
} from "./adapter.js";

export { PlugMemClient } from "./client.js";
export { resolveConfig, type CoreConfig, type ResolvedCoreConfig } from "./config.js";
export {
  deriveRepoGraphId,
  deriveUserGraphId,
  parseGitUrl,
  type RepoIdent,
} from "./repo_id.js";
export { parseTranscript, type Trajectory } from "./transcript.js";
export { createCore, type CreateCoreOptions } from "./core.js";
export {
  matchToolFamily,
  type RecallConfig,
  type SessionStartRecallConfig,
  type ToolFamilyRecallConfig,
  type UserPromptRecallConfig,
} from "./recall.js";
export {
  drainCandidates,
  matchesCorrectionPattern,
  recordPostTool,
  recordPreTool,
  recordUserPrompt,
  type Candidate,
  type CandidateKind,
} from "./promotion.js";
export {
  PlugMemError,
  PlugMemConnectionError,
  type ExtractedMemory,
  type ExtractedSemanticMemory,
  type ExtractedProceduralMemory,
} from "./types.js";
