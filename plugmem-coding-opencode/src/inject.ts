// src/inject.ts
import type { ContextInjection } from "@plugmem/coding-core";

export function injectSessionStart(output: any, injection: ContextInjection | null): void {
  if (!injection) return;
  output.system = output.system || [];
  output.system.push(injection.text);
}

export function injectUserPrompt(output: any, injection: ContextInjection | null): void {
  // OpenCode lacks a clean way to append context to a user prompt after it's been updated.
  // We do not inject here.
}

export function injectPreTool(output: any, injection: ContextInjection | null): void {
  if (!injection) return;
  output.args = output.args || {};
  output.args._plugmem_context = injection.text;
}
