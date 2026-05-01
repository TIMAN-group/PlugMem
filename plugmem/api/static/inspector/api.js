// Tiny fetch wrapper for the PlugMem API.
//
// - X-API-Key is read from localStorage (key: "plugmem_api_key") if set.
// - JSON requests/responses by default.
// - Errors surface as thrown Error with .status, .body for callers to handle.

const BASE = "/api/v1";
const KEY_STORAGE = "plugmem_api_key";

export function getApiKey() {
  return localStorage.getItem(KEY_STORAGE) || "";
}

export function setApiKey(key) {
  if (key) localStorage.setItem(KEY_STORAGE, key);
  else localStorage.removeItem(KEY_STORAGE);
}

async function request(method, path, { query, body } = {}) {
  let url = BASE + path;
  if (query) {
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === null || v === "") continue;
      params.append(k, v);
    }
    const qs = params.toString();
    if (qs) url += `?${qs}`;
  }

  const headers = { "Accept": "application/json" };
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const res = await fetch(url, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  let payload = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    payload = await res.json().catch(() => null);
  } else {
    payload = await res.text().catch(() => null);
  }

  if (!res.ok) {
    const detail = (payload && payload.detail) || payload || res.statusText;
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = res.status;
    err.body = payload;
    throw err;
  }
  return payload;
}

export const api = {
  listGraphs: () => request("GET", "/graphs"),
  getStats: (gid) => request("GET", `/graphs/${encodeURIComponent(gid)}/stats`),
  search: (gid, { q, node_type, limit, only_active } = {}) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/search`, {
      query: { q, node_type, limit, only_active },
    }),
  getNode: (gid, type, id) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/node/${type}/${id}`),
  patchSemantic: (gid, sid, body) =>
    request("PATCH", `/graphs/${encodeURIComponent(gid)}/semantic/${sid}`, { body }),
  seedDemo: ({ graph_id, reset } = {}) =>
    request("POST", "/demo/seed", { query: { graph_id, reset } }),
  recallTrace: (gid, body) =>
    request("POST", `/graphs/${encodeURIComponent(gid)}/recall_trace`, { body }),
  topology: (gid, { include_episodic, include_inactive, node_limit, tag_min_importance } = {}) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/topology`, {
      query: { include_episodic, include_inactive, node_limit, tag_min_importance },
    }),
  listSessions: (gid) =>
    request("GET", `/graphs/${encodeURIComponent(gid)}/sessions`),
  sessionTimeline: (gid, sessionId) =>
    request(
      "GET",
      `/graphs/${encodeURIComponent(gid)}/sessions/${encodeURIComponent(sessionId)}`,
    ),
};
