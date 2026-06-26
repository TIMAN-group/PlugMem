// Parse Claude Code's session JSONL into a (goal, steps) trajectory
// suitable for trajectory-mode insert.
//
// Each line is one of many record types (user, assistant, attachment,
// system, ai-title, etc.). We only consume `user` and `assistant` records;
// `ai-title` is used as the goal when present.

import type { TrajectoryStep } from "./types.js";

export interface Trajectory {
  goal: string;
  steps: TrajectoryStep[];
}

interface MessageBlock {
  type?: string;
  text?: string;
  thinking?: string;
  name?: string;
  input?: unknown;
  content?: unknown;
  is_error?: boolean;
}

interface RawMessage {
  role?: string;
  content?: string | MessageBlock[];
}

interface RawLine {
  type?: string;
  message?: RawMessage;
  aiTitle?: string;
}

const TOOL_USE_PREVIEW = 1000;
const TOOL_RESULT_PREVIEW = 2000;

export function parseTranscript(jsonl: string): Trajectory | null {
  const messages: Array<{ role: "user" | "assistant"; text: string }> = [];
  let goal = "";

  for (const line of jsonl.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    let obj: RawLine;
    try {
      obj = JSON.parse(trimmed);
    } catch {
      continue;
    }

    if (obj.type === "ai-title" && obj.aiTitle && !goal) {
      goal = obj.aiTitle;
      continue;
    }

    if (obj.type !== "user" && obj.type !== "assistant") continue;

    const text = renderMessage(obj.message);
    if (!text.trim()) continue;

    messages.push({ role: obj.type, text });
  }

  if (messages.length === 0) return null;

  if (!goal) {
    const firstUser = messages.find((m) => m.role === "user");
    goal = firstUser ? truncate(firstUser.text, 200) : "(no goal)";
  }

  // Collapse consecutive same-role messages (multi-block exchanges).
  const grouped: typeof messages = [];
  for (const m of messages) {
    const last = grouped[grouped.length - 1];
    if (last && last.role === m.role) {
      last.text += "\n\n" + m.text;
    } else {
      grouped.push({ ...m });
    }
  }

  const steps: TrajectoryStep[] = [];
  let pendingObs = "";
  for (const m of grouped) {
    if (m.role === "user") {
      pendingObs = m.text;
    } else {
      steps.push({
        observation: pendingObs || "(no observation)",
        action: m.text,
      });
      pendingObs = "";
    }
  }

  if (steps.length === 0) return null;
  return { goal, steps };
}

function renderMessage(message?: RawMessage): string {
  if (!message) return "";
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content)) return "";

  const parts: string[] = [];
  for (const block of message.content) {
    if (!block || typeof block !== "object") continue;
    switch (block.type) {
      case "text":
        if (block.text) parts.push(block.text);
        break;
      case "thinking":
        // Skip — too verbose for naïve trajectory. Promotion gate
        // (Stage 3) can re-read the raw transcript if needed.
        break;
      case "tool_use": {
        const inputStr = block.input !== undefined
          ? JSON.stringify(block.input).slice(0, TOOL_USE_PREVIEW)
          : "";
        parts.push(`[tool:${block.name ?? "?"}] ${inputStr}`);
        break;
      }
      case "tool_result": {
        const content = typeof block.content === "string"
          ? block.content
          : JSON.stringify(block.content);
        const marker = block.is_error ? "[tool_result:error]" : "[tool_result]";
        parts.push(`${marker} ${content.slice(0, TOOL_RESULT_PREVIEW)}`);
        break;
      }
    }
  }
  return parts.join("\n\n");
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
