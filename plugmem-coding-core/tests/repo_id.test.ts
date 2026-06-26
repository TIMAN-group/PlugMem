import { describe, expect, it } from "vitest";

import { deriveUserGraphId, parseGitUrl } from "../src/repo_id.js";

describe("parseGitUrl", () => {
  it("parses scp-style ssh", () => {
    expect(parseGitUrl("git@github.com:owner/repo.git")).toEqual({
      host: "github.com",
      owner: "owner",
      repo: "repo",
    });
  });

  it("parses https with .git suffix", () => {
    expect(parseGitUrl("https://github.com/owner/repo.git")).toEqual({
      host: "github.com",
      owner: "owner",
      repo: "repo",
    });
  });

  it("strips https credentials and lowercases host", () => {
    expect(
      parseGitUrl("https://user:token@GitHub.com/owner/repo"),
    ).toEqual({ host: "github.com", owner: "owner", repo: "repo" });
  });

  it("parses ssh:// scheme", () => {
    expect(parseGitUrl("ssh://git@gitlab.example.com:2222/group/sub/repo.git")).toEqual({
      host: "gitlab.example.com",
      owner: "group/sub",
      repo: "repo",
    });
  });

  it("preserves nested groups in owner", () => {
    expect(parseGitUrl("https://gitlab.com/group/sub/team/proj.git")).toEqual({
      host: "gitlab.com",
      owner: "group/sub/team",
      repo: "proj",
    });
  });

  it("returns null for unrecognized inputs", () => {
    expect(parseGitUrl("")).toBeNull();
    expect(parseGitUrl("not-a-url")).toBeNull();
    expect(parseGitUrl("https://github.com/")).toBeNull();
    expect(parseGitUrl("https://github.com/owner")).toBeNull();
  });
});

describe("deriveUserGraphId", () => {
  it("includes harness segment", () => {
    expect(deriveUserGraphId("claude-code", "alice")).toBe(
      "user://claude-code/alice",
    );
    expect(deriveUserGraphId("opencode", "alice")).toBe(
      "user://opencode/alice",
    );
  });
});
