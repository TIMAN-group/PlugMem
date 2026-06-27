import createPlugMemPlugin from "../src/index.js";
import type { OpenCodePluginApi, OpenCodeHookMap } from "../src/types.js";

// 1. Build a Fake OpenCode API
const hooks: Partial<OpenCodeHookMap> = {};

const fakeApi: OpenCodePluginApi = {
  on(eventName: any, callback: any) {
    console.log(`[Fake OpenCode] Plugin registered listener for: ${eventName}`);
    hooks[eventName as keyof OpenCodeHookMap] = callback;
  }
};

async function runMockSession() {
  console.log("--- Starting End-to-End Mock ---");

  // 2. Load the Plugin
  createPlugMemPlugin(fakeApi);

  // Helper to safely trigger a hook
  const fire = async <K extends keyof OpenCodeHookMap>(eventName: K, payload: any) => {
    const callback = hooks[eventName];
    if (callback) {
      console.log(`\n[Fake OpenCode] Firing ${eventName}...`);
      try {
        await (callback as any)(payload);
      } catch (err: any) {
        // We catch connection errors because we don't have the real PlugMem Python server running right now
        console.log(`[Core Brain Warning] ${err.message}`);
      }
    } else {
      console.warn(`[Fake OpenCode] No listener for ${eventName}`);
    }
  };

  const sessionId = "test-session-999";
  const sessionObj = { id: sessionId };

  // 3. Simulate an entire session!
  
  // A. Session Starts
  const sessionPayload = { session: sessionObj, output: { system: [] } };
  await fire("session.created", sessionPayload);
  console.log("   -> Output System Array:", sessionPayload.output.system);

  // B. User Prompts
  const messagePayload = { 
    message: { sessionId, id: "msg-1", role: "user", content: "Write a python script" } 
  };
  await fire("chat.message", messagePayload);
  console.log("   -> Mutated User Message:", messagePayload.message.content);

  // C. Tool Executes
  await fire("tool.execute.before", {
    session: sessionObj,
    tool: { name: "bash", args: { command: "python script.py" } }
  });

  await fire("tool.execute.after", {
    session: sessionObj,
    tool: { name: "bash", args: { command: "python script.py" }, result: "Success!", exitCode: 0 }
  });

  // D. Session Ends
  await fire("session.idle", { session: sessionObj, reason: "idle" });

  console.log("\n--- End-to-End Mock Finished Successfully! ---");
}

runMockSession().catch(console.error);
