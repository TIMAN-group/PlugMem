import { createCore } from "@plugmem/coding-core";
import { getInMemoryState, clearSessionState } from "./state.js";
import {
  normalizeSessionStart,
  normalizeUserPrompt,
  normalizePreTool,
  normalizePostTool,
  normalizePreCompact,
  normalizeSessionEnd,
  sniffOutcome,
  extractResultText,
} from "./normalize.js";
import type { Plugin, PluginContext, PluginHooks } from "./types.js";
import * as fs from "fs";

function logDebug(msg: string) {
  fs.appendFileSync("plugin-debug.log", `[${new Date().toISOString()}] ${msg}\n`);
}

export function getSessionId(event: any): string {
  // Session id lives in a different place per event family (OpenCode SDK):
  //   message.part.updated -> properties.part.sessionID
  //   message.updated      -> properties.info.sessionID  (Message.sessionID)
  //   session.*            -> properties.info.id          (Session.id)
  // Checking only info.id/sessionID (the old behavior) made user-prompt events
  // fall back to "default-session", so correction/episodic candidates were
  // bucketed away from the session the promotion gate drains — and silently lost.
  const p = event?.properties ?? {};
  return (
    p.part?.sessionID ||
    p.info?.sessionID ||
    p.info?.id ||
    p.sessionID ||
    "default-session"
  );
}

export const PlugMemPlugin: Plugin = async (ctx: PluginContext): Promise<PluginHooks> => {
  logDebug("==== PLUGMEM ADAPTER INITIALIZED ====");
  logDebug(`ENV API KEY: ${process.env.PLUGMEM_API_KEY}`);

  const core = createCore({
    config: {
      baseUrl: process.env.PLUGMEM_URL || "http://127.0.0.1:8000",
      apiKey: process.env.PLUGMEM_API_KEY,
      userId: process.env.USER || "opencode-user", 
    },
    state: getInMemoryState,
    log: (msg, err) => {
      logDebug(`CORE LOG: ${msg} ${err ? String(err) : ""}`);
    }
  });

  // Marks which user-message ids the live `message.part.updated` path has seen,
  // so we know which parts belong to a user turn.
  const userMessageIds = new Set<string>();

  // Persistent per-session set of user-message ids already fed to onUserPrompt.
  // session.idle fires after every turn and the transcript fallback re-reads the
  // FULL history each time; without this, every correction/episodic would be
  // re-extracted on every idle (and again vs. the live event). markProcessed
  // returns true only the first time a given (session, message) is seen.
  const processedPromptIds = new Map<string, Set<string>>();
  const markProcessed = (sessionId: string, msgId: string): boolean => {
    let seen = processedPromptIds.get(sessionId);
    if (!seen) { seen = new Set<string>(); processedPromptIds.set(sessionId, seen); }
    if (seen.has(msgId)) return false;
    seen.add(msgId);
    return true;
  };

  return {
    'tool.execute.before': async ({ tool, sessionID, callID }: any, { args }: any) => {
       logDebug(`TOOL BEFORE: ${tool} ${callID}`);
       try {
         await core.onPreTool({
            harness: "opencode",
            sessionId: sessionID || "default-session",
            cwd: ctx.directory,
            toolName: tool,
            toolInput: typeof args === 'string' ? args : JSON.stringify(args),
            callId: callID || ""
         });
       } catch (err: any) { logDebug(`ERROR IN PRE TOOL: ${err.message}`); }
    },
    // OpenCode signature: (input: { tool, sessionID, callID, args }, output: { title, output, metadata }).
    // `args` is on the FIRST arg; the result text is `output.output` (NOT a
    // `result`/`error` field, which is why the old `{ args, result, error }`
    // destructure read undefined and `resStr.includes(...)` threw — silently
    // killing every post-tool detection).
    'tool.execute.after': async ({ tool, sessionID, callID, args }: any, output: any) => {
       logDebug(`TOOL AFTER: ${tool} ${callID}`);
       try {
         const resStr = extractResultText(output);
         const outcome = sniffOutcome(tool, output);

         await core.onPostTool({
            harness: "opencode",
            sessionId: sessionID || "default-session",
            cwd: ctx.directory,
            toolName: tool,
            toolInput: args === undefined ? "" : typeof args === 'string' ? args : JSON.stringify(args),
            callId: callID || "",
            toolResult: resStr,
            outcome
         });
       } catch (err: any) { logDebug(`ERROR IN POST TOOL: ${err.message}`); }
    },
    event: async ({ event }: any) => {
      if (!event || !event.type) return;
      logDebug(`EVENT RECEIVED: ${event.type}`);

      if (event.type === "session.created") {
        logDebug(`Processing session.created`);
        try {
          const sessionId = getSessionId(event);
          const abstractEvent = normalizeSessionStart(event, ctx.directory, sessionId);
          const injection = await core.onSessionStart(abstractEvent);
          
          if (injection && sessionId !== "default-session" && ctx.client?.session?.update) {
             const sess = await ctx.client.session.get(sessionId);
             let sys = sess?.system || [];
             if (!Array.isArray(sys)) sys = [sys];
             sys.push(injection.text);
             await ctx.client.session.update(sessionId, { system: sys });
          }
        } catch (err: any) {
          logDebug(`ERROR IN SESSION CREATED: ${err.stack || err}`);
        }
      } else if (event.type === "message.updated") {
         const role = event.properties?.info?.role;
         const msgId = event.properties?.info?.id;
         if (role === "user" && msgId) {
            userMessageIds.add(msgId);
         }
      } else if (event.type === "message.part.updated") {
         const msgId = event.properties?.part?.messageID;
         const text = event.properties?.part?.text;
         if (msgId && text && userMessageIds.has(msgId)) {
            const sessionId = getSessionId(event);
            // Process each user message once; streaming re-fires and the
            // session-end transcript fallback are deduped via processedPromptIds.
            if (markProcessed(sessionId, msgId)) {
              logDebug(`Processing user prompt: ${text}`);
              try {
                await core.onUserPrompt({
                    harness: "opencode",
                    sessionId,
                    cwd: ctx.directory,
                    prompt: text || ""
                });
              } catch (err: any) {
                logDebug(`ERROR IN USER PROMPT: ${err.stack || err}`);
              }
            }
            userMessageIds.delete(msgId);
         }
      } else if (event.type === "session.idle" || event.type === "session.deleted") {
        logDebug(`Processing session end via event type: ${event.type}`);
        try {
          const sessionId = getSessionId(event);
          
          // Transcript fallback: catch user prompts missed by live events.
          // Deduped via processedPromptIds so re-reading the full history on
          // every session.idle doesn't re-extract the same messages.
          if (ctx.client?.session?.messages) {
             try {
               const history = await ctx.client.session.messages({ path: { id: sessionId } });
               if (Array.isArray(history)) {
                 for (const msg of history) {
                   if (msg.info?.role === "user") {
                     const mid = msg.info?.id;
                     const textPart = msg.parts?.find((p: any) => p.type === "text" || p.text);
                     if (textPart && textPart.text && (!mid || markProcessed(sessionId, mid))) {
                       await core.onUserPrompt({
                         harness: "opencode",
                         sessionId,
                         cwd: ctx.directory,
                         prompt: textPart.text
                       });
                     }
                   }
                 }
               }
             } catch (err: any) {
               logDebug(`WARNING: Transcript fallback failed: ${err.message}`);
             }
          }

          const abstractEvent = normalizeSessionEnd(event, ctx.directory, sessionId, "session_end");
          await core.onSessionEnd(abstractEvent);

          // session.idle fires after every turn (session continues), so only
          // release per-session state on a real session.deleted.
          if (event.type === "session.deleted") {
            clearSessionState(sessionId);
            processedPromptIds.delete(sessionId);
          }
          logDebug(`Session end processing completed successfully`);
        } catch (err: any) {
          logDebug(`ERROR IN SESSION END: ${err.stack || err}`);
        }
      }
    }
  };
};
