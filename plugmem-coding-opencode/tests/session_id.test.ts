import { describe, it, expect } from "vitest";
import { getSessionId } from "../src/index.js";

// Regression: the session id lives in a different place per OpenCode event
// family. Reading only properties.info.id/sessionID made message.part.updated
// fall back to "default-session", so user-prompt candidates (corrections /
// episodics) were bucketed away from the session the promotion gate drains and
// were silently lost.

describe("getSessionId", () => {
  it("message.part.updated -> properties.part.sessionID", () => {
    const e = { type: "message.part.updated",
      properties: { part: { messageID: "m1", sessionID: "sess-A", text: "hi" } } };
    expect(getSessionId(e)).toBe("sess-A");
  });

  it("message.updated -> properties.info.sessionID (not the message id)", () => {
    const e = { type: "message.updated",
      properties: { info: { id: "m1", role: "user", sessionID: "sess-B" } } };
    expect(getSessionId(e)).toBe("sess-B");
  });

  it("session.* -> properties.info.id (Session.id)", () => {
    const e = { type: "session.idle", properties: { info: { id: "sess-C" } } };
    expect(getSessionId(e)).toBe("sess-C");
  });

  it("tool/session events with a top-level sessionID still resolve", () => {
    expect(getSessionId({ properties: { sessionID: "sess-D" } })).toBe("sess-D");
  });

  it("falls back to default-session when nothing is present", () => {
    expect(getSessionId({ properties: {} })).toBe("default-session");
    expect(getSessionId({})).toBe("default-session");
  });

  it("a message part and its session.idle resolve to the SAME id", () => {
    const sid = "sess-shared";
    const part = { type: "message.part.updated",
      properties: { part: { messageID: "m9", sessionID: sid, text: "x" } } };
    const idle = { type: "session.idle", properties: { info: { id: sid } } };
    expect(getSessionId(part)).toBe(getSessionId(idle));
  });
});
