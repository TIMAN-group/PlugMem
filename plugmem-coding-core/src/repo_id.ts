import { exec } from "node:child_process";
import { promisify } from "node:util";
import type { Harness } from "./adapter.js";

const execAsync = promisify(exec);

export interface RepoIdent {
  host: string;
  owner: string;
  repo: string;
}

/**
 * Parse a git remote URL into host/owner/repo.
 *
 * Supports:
 *   - https://[user[:pw]@]host/owner/repo[.git]
 *   - ssh://[user@]host[:port]/owner/repo[.git]
 *   - git@host:owner/repo[.git]   (scp-style ssh)
 *   - git://host/owner/repo[.git]
 *
 * Returns null on unrecognized shapes (don't guess).
 */
export function parseGitUrl(raw: string): RepoIdent | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;

  // Scheme URLs first (https://, ssh://, git://, http://) — must be
  // checked before the scp-style fallback because scp's regex would
  // otherwise greedily match ssh URLs and capture the port into the path.
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed)) {
    let url: URL;
    try {
      url = new URL(trimmed);
    } catch {
      return null;
    }
    return splitHostPath(url.host, url.pathname);
  }

  // scp-style: git@host:owner/repo[.git]
  const scp = /^[^@\s]+@([^:\s]+):([^\s]+)$/.exec(trimmed);
  if (scp) {
    return splitHostPath(scp[1]!, scp[2]!);
  }

  return null;
}

function splitHostPath(host: string, path: string): RepoIdent | null {
  const cleanHost = host.toLowerCase().replace(/:\d+$/, ""); // drop port
  const cleanPath = path.replace(/\.git$/i, "").replace(/^\/+|\/+$/g, "");
  const parts = cleanPath.split("/").filter(Boolean);
  if (parts.length < 2) return null;
  // owner = first segment, repo = last segment. Some hosts (gitlab) allow
  // groups/subgroups in between — collapse them into the owner string with
  // slashes preserved so the graph id round-trips uniquely.
  const repo = parts[parts.length - 1]!;
  const owner = parts.slice(0, -1).join("/");
  return { host: cleanHost, owner, repo };
}

async function readGitRemote(cwd: string): Promise<string | null> {
  try {
    const { stdout } = await execAsync("git remote get-url origin", {
      cwd,
      timeout: 2000,
    });
    return stdout.trim() || null;
  } catch {
    return null;
  }
}

/**
 * Per-repo graph ID — write target.
 * Includes the harness segment so memories are isolated per harness.
 */
export async function deriveRepoGraphId(
  harness: Harness,
  cwd: string,
): Promise<string> {
  const remote = await readGitRemote(cwd);
  if (remote) {
    const ident = parseGitUrl(remote);
    if (ident) {
      return `repo://${harness}/${ident.host}/${ident.owner}/${ident.repo}`;
    }
  }
  // Fallback: identify by absolute path. Marker segment "local" so it
  // can never collide with a real host name.
  return `repo://${harness}/local${cwd}`;
}

/**
 * Per-user graph ID — read-only fan-in for personal idioms / preferences.
 * Includes the harness segment per design (no cross-harness sharing).
 */
export function deriveUserGraphId(harness: Harness, userId: string): string {
  return `user://${harness}/${userId}`;
}
