// PlugMem REST API request/response types.
// Mirrors plugmem/api/schemas.py.

export interface GraphCreateRequest {
  graph_id?: string;
}

export interface GraphResponse {
  graph_id: string;
  stats: Record<string, number>;
}

export interface GraphListResponse {
  graphs: string[];
}

export interface StatsResponse {
  graph_id: string;
  stats: Record<string, number>;
}

export interface NodeListResponse {
  graph_id: string;
  node_type: string;
  count: number;
  nodes: Record<string, unknown>[];
}

export interface TrajectoryStep {
  observation: string;
  action: string;
}

export interface SemanticMemoryInput {
  semantic_memory: string;
  tags?: string[];
  source?: MemorySourceWire;
  confidence?: number;
}

export interface ProceduralMemoryInput {
  subgoal: string;
  procedural_memory: string;
  return?: number;
  source?: MemorySourceWire;
  confidence?: number;
}

export interface EpisodicStep {
  observation?: string;
  action?: string;
  subgoal?: string;
  state?: string;
  reward?: string;
  time?: string | number;
}

export interface TrajectoryInsertRequest {
  mode: "trajectory";
  goal: string;
  steps: TrajectoryStep[];
}

export interface StructuredInsertRequest {
  mode: "structured";
  episodic?: EpisodicStep[][];
  semantic?: SemanticMemoryInput[];
  procedural?: ProceduralMemoryInput[];
}

export type MemoryInsertRequest =
  | TrajectoryInsertRequest
  | StructuredInsertRequest;

export interface MemoryInsertResponse {
  status: string;
  stats: Record<string, number>;
}

export interface RetrieveRequest {
  observation: string;
  goal?: string;
  subgoal?: string;
  state?: string;
  task_type?: string;
  time?: string;
  mode?: "semantic_memory" | "episodic_memory" | "procedural_memory" | null;
  min_confidence?: number;
  source_in?: MemorySourceWire[];
}

export interface RetrieveResponse {
  mode: string;
  reasoning_prompt: Array<{ role: string; content: string }>;
  variables: Record<string, unknown>;
}

export interface ReasonRequest {
  observation: string;
  goal?: string;
  subgoal?: string;
  state?: string;
  task_type?: string;
  time?: string;
  mode?: "semantic_memory" | "episodic_memory" | "procedural_memory" | null;
  min_confidence?: number;
  source_in?: MemorySourceWire[];
}

export interface ReasonResponse {
  mode: string;
  reasoning: string;
  reasoning_prompt: Array<{ role: string; content: string }>;
}

export interface HealthResponse {
  status: string;
  version: string;
  llm_available: boolean;
  embedding_available: boolean;
  chroma_available: boolean;
}

// ── Promotion-gate extraction ───────────────────────────────────────

export type CandidateKindWire = "failure_delta" | "correction";

export interface CandidateWire {
  kind: CandidateKindWire;
  window: string;
}

export interface ExtractRequest {
  candidates: CandidateWire[];
}

export type MemorySourceWire =
  | "failure_delta"
  | "correction"
  | "merged"
  | "repeated_lookup"
  | "explicit";

export interface ExtractedSemanticMemory {
  type: "semantic";
  semantic_memory: string;
  tags?: string[];
  source: MemorySourceWire;
  confidence: number;
}

export interface ExtractedProceduralMemory {
  type: "procedural";
  subgoal: string;
  procedural_memory: string;
  source: MemorySourceWire;
  confidence: number;
}

export type ExtractedMemory =
  | ExtractedSemanticMemory
  | ExtractedProceduralMemory;

export interface ExtractResponse {
  memories: ExtractedMemory[];
}

export class PlugMemError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly body?: unknown,
  ) {
    super(message);
    this.name = "PlugMemError";
  }
}

export class PlugMemConnectionError extends Error {
  constructor(
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = "PlugMemConnectionError";
  }
}
