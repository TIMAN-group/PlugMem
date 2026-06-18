// src/state.ts
import type { SessionState } from "@plugmem/coding-core";

/**
 * OpenCode runs plugins as long-lived modules, unlike Claude Code which 
 * spins up isolated processes for every hook. Therefore, we do not need 
 * to serialize state to the disk. A simple in-memory Map is sufficient 
 * and much faster.
 * 
 * The outer map is keyed by `sessionId`. 
 * The inner map is keyed by the state `key` used by @plugmem/coding-core.
 */
const globalState = new Map<string, Map<string, any>>();

/**
 * Returns a SessionState implementation scoped to a specific session ID.
 * This is passed to the @plugmem/coding-core `createCore` factory.
 */
export function getInMemoryState(sessionId: string): SessionState {
  // Ensure the session has a bucket in our global state
  if (!globalState.has(sessionId)) {
    globalState.set(sessionId, new Map<string, any>());
  }

  const sessionBucket = globalState.get(sessionId)!;

  return {
    async get<T>(key: string): Promise<T | undefined> {
      return sessionBucket.get(key) as T | undefined;
    },
    
    async set<T>(key: string, value: T): Promise<void> {
      sessionBucket.set(key, value);
    },
    
    async del(key: string): Promise<void> {
      sessionBucket.delete(key);
    }
  };
}

/**
 * Helper to wipe a session's state from memory when OpenCode 
 * fires a session.deleted or session.idle event.
 */
export function clearSessionState(sessionId: string): void {
  globalState.delete(sessionId);
}
