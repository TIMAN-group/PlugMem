import { createCore } from "@plugmem/coding-core";
import { getInMemoryState, clearSessionState } from "./state.js";
import {
  normalizeSessionStart,
  normalizeUserPrompt,
  normalizePreTool,
  normalizePostTool,
  normalizePreCompact,
  normalizeSessionEnd,
} from "./normalize.js";
import type { Plugin, PluginContext, PluginHooks } from "./types.js";
import * as fs from "fs";

function logDebug(msg: string) {
  fs.appendFileSync("plugin-debug.log", `[${new Date().toISOString()}] ${msg}\n`);
}

function getSessionId(event: any): string {
  return event.properties?.info?.id || event.properties?.sessionID || "default-session";
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

  const userMessageIds = new Set<string>();

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
    'tool.execute.after': async ({ tool, sessionID, callID }: any, { args, result }: any) => {
       logDebug(`TOOL AFTER: ${tool} ${callID}`);
       try {
         await core.onPostTool({
            harness: "opencode",
            sessionId: sessionID || "default-session",
            cwd: ctx.directory,
            toolName: tool,
            toolInput: typeof args === 'string' ? args : JSON.stringify(args),
            callId: callID || "",
            toolResult: typeof result === 'string' ? result : JSON.stringify(result),
            outcome: "success"
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
            logDebug(`Processing user prompt: ${text}`);
            try {
              await core.onUserPrompt({
                  harness: "opencode",
                  sessionId: getSessionId(event),
                  cwd: ctx.directory,
                  prompt: text || ""
              });
              userMessageIds.delete(msgId);
            } catch (err: any) {
              logDebug(`ERROR IN USER PROMPT: ${err.stack || err}`);
            }
         }
      } else if (event.type === "session.idle" || event.type === "session.deleted") {
        logDebug(`Processing session end via event type: ${event.type}`);
        try {
          const sessionId = getSessionId(event);
          
          // FETCH FULL TRANSCRIPT FALLBACK
          if (ctx.client?.session?.messages) {
             try {
               const history = await ctx.client.session.messages({ path: { id: sessionId } });
               if (Array.isArray(history)) {
                 for (const msg of history) {
                   if (msg.info?.role === "user") {
                     const textPart = msg.parts?.find((p: any) => p.type === "text" || p.text);
                     if (textPart && textPart.text) {
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
          logDebug(`Session end processing completed successfully`);
        } catch (err: any) {
          logDebug(`ERROR IN SESSION END: ${err.stack || err}`);
        }
      }
    }
  };
};
