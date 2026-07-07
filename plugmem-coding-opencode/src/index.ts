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
  const id = (
    p.part?.sessionID ||
    p.info?.sessionID ||
    p.info?.id ||
    p.sessionID ||
    "default-session"
  );
  if (id === "default-session") return id;
  
  // Clean off any url encoding if present
  let cleanId = id;
  try { cleanId = decodeURIComponent(cleanId); } catch {}
  try { cleanId = decodeURIComponent(cleanId); } catch {}
  
  if (!cleanId.startsWith("ses")) return "default-session";
  if (cleanId.includes("{") || cleanId.includes("%7B")) return "default-session";
  if (cleanId.includes("}")) return "default-session";
  
  return cleanId;
}

function loadEnvFile() {
  if (fs.existsSync(".env")) {
    try {
      const content = fs.readFileSync(".env", "utf-8");
      for (const line of content.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const match = trimmed.match(/^([^=]+)=(.*)$/);
        if (match) {
          const key = match[1].trim();
          let value = match[2].trim();
          if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
          }
          process.env[key] = value;
        }
      }
    } catch (e) {
      logDebug(`Failed to parse .env file: ${String(e)}`);
    }
  }
}

export const PlugMemPlugin: Plugin = async (ctx: PluginContext): Promise<PluginHooks> => {
  loadEnvFile();
  logDebug("==== PLUGMEM ADAPTER INITIALIZED ====");
  logDebug(`ENV API KEY: ${process.env.PLUGMEM_API_KEY}`);

  const core = createCore({
    config: {
      baseUrl: process.env.PLUGMEM_URL || process.env.PLUGMEM_BASE_URL || "http://127.0.0.1:8077",
      apiKey: process.env.PLUGMEM_API_KEY || "dev-key-change-me",
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
  let activeSessionId = "default-session";
  let hasInjected = false;
  let realSessionIdFromTool: string | null = null;
  
  // Deduplication caches for OpenCode's double-fire bug
  const processedToolCalls = new Set<string>();
  const seenEvents = new Set<string>();
  // Cooldown map: sessionId -> last extraction timestamp (prevents double-fire
  // from session.idle+session.deleted, but allows re-extraction after new user input)
  const lastExtractionTime = new Map<string, number>();
  const EXTRACTION_COOLDOWN_MS = 5000; // 5 second cooldown


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
      if (callID && processedToolCalls.has(`before:${callID}`)) return;
      if (callID) processedToolCalls.add(`before:${callID}`);
      
      logDebug(`TOOL BEFORE: ${tool} sessionID=${sessionID}`);
      // Capture real session ID from tool hooks (OpenCode resolves the template here)
      if (sessionID && sessionID.startsWith("ses_")) {
        realSessionIdFromTool = sessionID;
      }
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
       if (callID && processedToolCalls.has(`after:${callID}`)) return;
       if (callID) processedToolCalls.add(`after:${callID}`);
       
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
      
      // Deduplicate identical events that fire within the same session
      const eventHash = `${event.type}:${JSON.stringify(event.properties || {})}`;
      if (seenEvents.has(eventHash)) return;
      seenEvents.add(eventHash);
      
      // Keep set from growing infinitely
      if (seenEvents.size > 1000) seenEvents.clear();

      logDebug(`EVENT RECEIVED: ${event.type} props=${JSON.stringify(event.properties || {}).substring(0, 300)}`);

      if (event.type === "session.created") {
          logDebug(`Processing session.created RAW sessionID=${event.properties?.sessionID} info.id=${event.properties?.info?.id}`);
          try {
              const sessionId = getSessionId(event);
              logDebug(`session.created resolved sessionId=${sessionId}`);
              activeSessionId = sessionId;
              
              if (sessionId !== "default-session") {
                  hasInjected = true;
                  logDebug(`Attempting injection for ${sessionId}`);
                  const abstractEvent = normalizeSessionStart(event, ctx.directory, sessionId);
                  const injection = await core.onSessionStart(abstractEvent);
                  if (injection && ctx.client?.session?.update) {
                      try {
                          const sess = await ctx.client.session.get(sessionId);
                          let sys = sess?.system || [];
                          if (!Array.isArray(sys)) sys = [sys];
                          sys.push(injection.text);
                          await ctx.client.session.update(sessionId, { system: sys });
                          logDebug(`SUCCESS: Injected into ${sessionId}`);
                      } catch (fetchErr) {
                          logDebug(`WARNING: Could not fetch/update session. Error: ${fetchErr}`);
                      }
                  }
              } else {
                  logDebug(`session.created got default-session, will try deferred injection later`);
              }
          }
          catch (err: any) {
              logDebug(`ERROR IN SESSION CREATED: ${err.stack || err}`);
          }
      } else if (event.type === "message.updated") {
          const role = event.properties?.info?.role;
          const msgId = event.properties?.info?.id;
          if (role === "user" && msgId) {
              userMessageIds.add(msgId);
          }
          
          // Deferred injection: try event ID first, then fall back to tool-captured ID
          let realId = getSessionId(event);
          if (realId === "default-session" && realSessionIdFromTool) {
              realId = realSessionIdFromTool;
              logDebug(`Using tool-captured session ID: ${realId}`);
          }
          if (!hasInjected && realId !== "default-session") {
              hasInjected = true;
              logDebug(`Running deferred injection for real session ID: ${realId}`);
              try {
                  const abstractEvent = normalizeSessionStart(event, ctx.directory, realId);
                  const injection = await core.onSessionStart(abstractEvent);
                  if (injection && ctx.client?.session?.update) {
                      const sess = await ctx.client.session.get(realId);
                      let sys = sess?.system || [];
                      if (!Array.isArray(sys)) sys = [sys];
                      sys.push(injection.text);
                      await ctx.client.session.update(realId, { system: sys });
                      logDebug(`Successfully injected prompt into ${realId}`);
                  }
              } catch (e) {
                  logDebug(`Deferred injection failed: ${e}`);
              }
          }
      } else if (event.type === "message.part.updated") {
         const msgId = event.properties?.part?.messageID;
         const text = event.properties?.part?.text;
         if (msgId && text && userMessageIds.has(msgId)) {
            let cleanText = text;
            // Strip automated system instructions that OpenCode appends to user messages
            cleanText = cleanText.replace(/continue if you have next steps.*?/gi, "");
            cleanText = cleanText.replace(/stop and ask for clarification.*?/gi, "");
            cleanText = cleanText.trim();

            const sessionId = getSessionId(event);
            if (cleanText.length > 0 && markProcessed(sessionId, msgId)) {
              logDebug(`Processing user prompt: ${cleanText}`);
              try {
                await core.onUserPrompt({
                    harness: "opencode",
                    sessionId,
                    cwd: ctx.directory,
                    prompt: cleanText
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
          const now = Date.now();
          const lastTime = lastExtractionTime.get(sessionId) || 0;
          
          if (now - lastTime < EXTRACTION_COOLDOWN_MS) {
              logDebug(`Skipping extraction for ${sessionId} (cooldown: ${now - lastTime}ms ago)`);
          } else {
              lastExtractionTime.set(sessionId, now);
              
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
                           let cleanText = textPart.text;
                           cleanText = cleanText.replace(/continue if you have next steps.*?/gi, "");
                           cleanText = cleanText.replace(/stop and ask for clarification.*?/gi, "");
                           cleanText = cleanText.trim();
                           if (cleanText.length > 0) {
                             await core.onUserPrompt({
                               harness: "opencode",
                               sessionId,
                               cwd: ctx.directory,
                               prompt: cleanText
                             });
                           }
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
          }

          // session.idle fires after every turn (session continues), so only
          // release per-session state on a real session.deleted.
          if (event.type === "session.deleted") {
            clearSessionState(sessionId);
            processedPromptIds.delete(sessionId);
            userMessageIds.clear();
          }
        } catch (err: any) {
          logDebug(`ERROR IN SESSION IDLE/DELETED: ${err.stack || err}`);
        }
      }
    },
    dispose: async () => {
      logDebug(`Processing plugin shutdown hook (dispose)`);
      try {
        await core.onSessionEnd({
          harness: "opencode",
          sessionId: activeSessionId,
          cwd: ctx.directory,
          reason: "session_end"
        });
        logDebug(`Shutdown extraction completed`);
      } catch (err: any) {
        logDebug(`ERROR IN SHUTDOWN: ${err.message}`);
      }
    }
  } as any;
};
