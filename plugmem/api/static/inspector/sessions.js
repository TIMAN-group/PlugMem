// Sessions tab: pick a session id from the sidebar, render a chronological
// timeline of inserts + recalls for that session.
//
// Inserts come from the graph's node metadata (semantic/procedural/episodic
// with matching session_id). Recalls come from the per-graph recall_audit
// collection. The /sessions/{id} endpoint merges them server-side.

import { api } from "./api.js";

export function mountSessions({ container, getGraphId, toast }) {
  const els = {
    list:        container.querySelector("#sessions-list"),
    empty:       container.querySelector("#sessions-empty"),
    header:      container.querySelector("#sessions-header"),
    timeline:    container.querySelector("#sessions-timeline"),
    placeholder: container.querySelector("#sessions-placeholder"),
    detail:      container.querySelector("#sessions-detail"),
    detailTitle: container.querySelector("#sessions-detail-title"),
    detailBody:  container.querySelector("#sessions-detail-body"),
    detailClose: container.querySelector("#sessions-detail-close"),
  };

  const state = {
    sessions: [],
    selectedSessionId: null,
    selectedEventKey: null,
  };

  async function loadSessions() {
    const gid = getGraphId();
    if (!gid) {
      state.sessions = [];
      renderList();
      clearTimeline("Pick a graph.");
      return;
    }
    try {
      const res = await api.listSessions(gid);
      state.sessions = res.sessions || [];
      renderList();
      if (state.sessions.length === 0) {
        clearTimeline("No sessions yet.");
        return;
      }
      // Auto-pick the first session if none selected (or stale).
      if (!state.selectedSessionId || !state.sessions.includes(state.selectedSessionId)) {
        await selectSession(state.sessions[0]);
      } else {
        await selectSession(state.selectedSessionId);
      }
    } catch (err) {
      toast(`sessions: ${err.message}`, "error");
    }
  }

  function renderList() {
    els.list.innerHTML = "";
    if (state.sessions.length === 0) {
      els.empty.hidden = false;
      return;
    }
    els.empty.hidden = true;
    for (const sid of state.sessions) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "session-btn";
      btn.textContent = sid;
      btn.setAttribute("aria-pressed", sid === state.selectedSessionId ? "true" : "false");
      btn.addEventListener("click", () => void selectSession(sid));
      els.list.appendChild(btn);
    }
  }

  async function selectSession(sessionId) {
    const gid = getGraphId();
    if (!gid) return;
    state.selectedSessionId = sessionId;
    state.selectedEventKey = null;
    closeDetail();
    for (const b of els.list.querySelectorAll(".session-btn")) {
      b.setAttribute("aria-pressed", b.textContent === sessionId ? "true" : "false");
    }
    els.placeholder.hidden = true;
    els.timeline.innerHTML = "<p class='hint'>Loading…</p>";
    try {
      const res = await api.sessionTimeline(gid, sessionId);
      renderTimeline(res);
    } catch (err) {
      els.timeline.innerHTML = `<p class='hint'>error: ${escapeHtml(err.message)}</p>`;
      toast(`timeline: ${err.message}`, "error");
    }
  }

  function clearTimeline(message) {
    state.selectedSessionId = null;
    els.timeline.innerHTML = "";
    els.header.hidden = true;
    els.placeholder.hidden = false;
    els.placeholder.textContent = message;
  }

  function renderTimeline(res) {
    els.placeholder.hidden = true;
    els.timeline.innerHTML = "";

    // Header summary
    els.header.hidden = false;
    const counts = countBy(res.events, (e) =>
      e.kind === "recall" ? "recall" : `insert:${e.node_type}`,
    );
    els.header.innerHTML = "";
    const title = document.createElement("h2");
    title.className = "sessions-title";
    title.textContent = res.session_id;
    els.header.appendChild(title);
    const subtitle = document.createElement("div");
    subtitle.className = "sessions-subtitle";
    const order = ["insert:semantic", "insert:procedural", "insert:episodic", "recall"];
    const labels = {
      "insert:semantic": "semantic",
      "insert:procedural": "procedural",
      "insert:episodic": "episodic",
      "recall": "recalls",
    };
    const parts = order
      .filter((k) => counts[k])
      .map((k) => `${counts[k]} ${labels[k]}`);
    subtitle.textContent = parts.join(" · ") + ` · ${res.count} total`;
    els.header.appendChild(subtitle);

    if (res.events.length === 0) {
      const e = document.createElement("p");
      e.className = "hint";
      e.textContent = "No events in this session yet.";
      els.timeline.appendChild(e);
      return;
    }

    for (const ev of res.events) {
      els.timeline.appendChild(renderEventRow(ev));
    }
  }

  function renderEventRow(ev) {
    const row = document.createElement("div");
    row.className = `timeline-row ${ev.kind === "recall" ? "kind-recall" : `kind-insert type-${ev.node_type}`}`;
    const key = eventKey(ev);
    row.dataset.eventKey = key;
    if (state.selectedEventKey === key) row.classList.add("selected");

    // left rail with time marker
    const rail = document.createElement("div");
    rail.className = "timeline-rail";
    const time = document.createElement("span");
    time.className = "timeline-time";
    time.textContent = `t${ev.time}`;
    rail.appendChild(time);
    if (ev.kind === "recall" && ev.ts) {
      const ts = document.createElement("span");
      ts.className = "timeline-ts";
      ts.textContent = ev.ts.replace("T", " ").slice(0, 16);
      rail.appendChild(ts);
    }
    row.appendChild(rail);

    // body
    const body = document.createElement("div");
    body.className = "timeline-body";
    body.appendChild(renderEventBody(ev));
    row.appendChild(body);

    row.addEventListener("click", () => onEventClick(ev, row));
    return row;
  }

  function renderEventBody(ev) {
    const wrap = document.createElement("div");
    wrap.className = "timeline-card";

    const head = document.createElement("div");
    head.className = "timeline-head";
    if (ev.kind === "insert") {
      const pill = document.createElement("span");
      pill.className = `node-id-pill type-${ev.node_type}`;
      pill.textContent = `#${ev.node_id}`;
      head.appendChild(pill);
      const tag = document.createElement("span");
      tag.className = "timeline-kind";
      tag.textContent = ev.node_type;
      head.appendChild(tag);
      if (ev.node_type === "semantic" && ev.is_active === false) {
        const inactive = document.createElement("span");
        inactive.className = "badge inactive";
        inactive.textContent = "inactive";
        head.appendChild(inactive);
      }
    } else {
      const pill = document.createElement("span");
      pill.className = "timeline-recall-pill";
      pill.textContent = ev.endpoint;
      head.appendChild(pill);
      const mode = document.createElement("span");
      mode.className = "timeline-kind";
      mode.textContent = ev.mode || "";
      head.appendChild(mode);
    }
    wrap.appendChild(head);

    const text = document.createElement("div");
    text.className = "timeline-text";
    text.textContent = ev.kind === "insert"
      ? (ev.label || ev.text || "")
      : (ev.observation || "");
    wrap.appendChild(text);

    if (ev.kind === "recall") {
      const sids = ev.selected_semantic_ids || [];
      const pids = ev.selected_procedural_ids || [];
      if (sids.length || pids.length) {
        const refs = document.createElement("div");
        refs.className = "timeline-refs";
        for (const id of sids) refs.appendChild(refPill("semantic", id));
        for (const id of pids) refs.appendChild(refPill("procedural", id));
        wrap.appendChild(refs);
      }
      if ((ev.query_tags || []).length) {
        const tags = document.createElement("div");
        tags.className = "cell-tags timeline-tags";
        for (const t of ev.query_tags) {
          const p = document.createElement("span");
          p.className = "tag-pill";
          p.textContent = t;
          tags.appendChild(p);
        }
        wrap.appendChild(tags);
      }
    }
    return wrap;
  }

  function refPill(type, id) {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = `node-id-pill type-${type} timeline-ref`;
    pill.textContent = `#${id}`;
    pill.title = `Open ${type} #${id}`;
    pill.addEventListener("click", (e) => {
      e.stopPropagation();
      void openNodeDetail(type, id);
    });
    return pill;
  }

  function onEventClick(ev, row) {
    state.selectedEventKey = eventKey(ev);
    for (const r of els.timeline.querySelectorAll(".timeline-row.selected")) {
      r.classList.remove("selected");
    }
    row.classList.add("selected");
    if (ev.kind === "insert") {
      void openNodeDetail(ev.node_type, ev.node_id);
    } else {
      openRecallDetail(ev);
    }
  }

  async function openNodeDetail(type, nodeId) {
    const gid = getGraphId();
    if (!gid) return;
    els.detail.hidden = false;
    els.detailTitle.innerHTML = "";
    const pill = document.createElement("span");
    pill.className = `node-id-pill type-${type}`;
    pill.textContent = `#${nodeId}`;
    els.detailTitle.appendChild(pill);
    const sub = document.createElement("span");
    sub.style.marginLeft = "8px";
    sub.textContent = type;
    els.detailTitle.appendChild(sub);
    els.detailBody.innerHTML = "<p class='hint'>Loading…</p>";
    try {
      const res = await api.getNode(gid, type, nodeId);
      renderNodeDetail(res);
    } catch (err) {
      els.detailBody.innerHTML = `<p class='hint'>error: ${escapeHtml(err.message)}</p>`;
    }
  }

  function renderNodeDetail(res) {
    const { node_type, node, edges } = res;
    const body = els.detailBody;
    body.innerHTML = "";

    body.appendChild(section("Text", textBlock(primaryText(node_type, node))));

    const meta = metaRow(node_type, node);
    if (meta) body.appendChild(section("Meta", meta));

    for (const [name, list] of Object.entries(edges || {})) {
      if (!Array.isArray(list) || list.length === 0) continue;
      body.appendChild(section(`${name} (${list.length})`, edgeList(list)));
    }
  }

  function openRecallDetail(ev) {
    els.detail.hidden = false;
    els.detailTitle.innerHTML = "";
    const pill = document.createElement("span");
    pill.className = "timeline-recall-pill";
    pill.textContent = ev.endpoint;
    els.detailTitle.appendChild(pill);
    const sub = document.createElement("span");
    sub.style.marginLeft = "8px";
    sub.textContent = `recall #${ev.recall_id ?? "?"}`;
    els.detailTitle.appendChild(sub);

    const body = els.detailBody;
    body.innerHTML = "";
    body.appendChild(section("Observation", textBlock(ev.observation || "")));

    const meta = document.createElement("div");
    meta.className = "detail-meta";
    if (ev.mode) meta.appendChild(badge(`mode: ${ev.mode}`));
    if (ev.next_subgoal) meta.appendChild(badge(`next_subgoal: ${truncate(ev.next_subgoal, 40)}`));
    if (ev.ts) meta.appendChild(badge(`ts: ${ev.ts}`));
    meta.appendChild(badge(`graph_time: ${ev.time}`));
    meta.appendChild(badge(`messages: ${ev.n_messages ?? 0}`));
    body.appendChild(section("Meta", meta));

    if ((ev.query_tags || []).length) {
      const tagWrap = document.createElement("div");
      tagWrap.className = "cell-tags";
      for (const t of ev.query_tags) {
        const p = document.createElement("span");
        p.className = "tag-pill";
        p.textContent = t;
        tagWrap.appendChild(p);
      }
      body.appendChild(section("Query tags", tagWrap));
    }

    const sids = ev.selected_semantic_ids || [];
    const pids = ev.selected_procedural_ids || [];
    if (sids.length || pids.length) {
      const refs = document.createElement("div");
      refs.className = "timeline-refs";
      for (const id of sids) refs.appendChild(refPill("semantic", id));
      for (const id of pids) refs.appendChild(refPill("procedural", id));
      body.appendChild(section("Selected", refs));
    }
  }

  function closeDetail() {
    els.detail.hidden = true;
  }

  // helpers ---

  function eventKey(ev) {
    return ev.kind === "insert"
      ? `insert-${ev.node_type}-${ev.node_id}`
      : `recall-${ev.recall_id}`;
  }

  function countBy(arr, fn) {
    const out = {};
    for (const item of arr) {
      const k = fn(item);
      out[k] = (out[k] || 0) + 1;
    }
    return out;
  }

  function section(title, content) {
    const sec = document.createElement("div");
    sec.className = "detail-section";
    const h = document.createElement("h3");
    h.textContent = title;
    sec.appendChild(h);
    if (content instanceof Node) sec.appendChild(content);
    return sec;
  }
  function textBlock(text) {
    const pre = document.createElement("pre");
    pre.textContent = text || "(empty)";
    return pre;
  }
  function primaryText(type, node) {
    if (type === "semantic" || type === "procedural") return node.text;
    if (type === "tag") return node.tag;
    if (type === "subgoal") return node.subgoal;
    if (type === "episodic") {
      const parts = [];
      if (node.observation) parts.push(`obs: ${node.observation}`);
      if (node.action) parts.push(`act: ${node.action}`);
      return parts.join("\n\n");
    }
    return "";
  }
  function metaRow(type, node) {
    const wrap = document.createElement("div");
    wrap.className = "detail-meta";
    const items = [];
    if (type === "semantic") {
      items.push(["state", node.is_active ? "active" : "inactive"]);
      items.push(["credibility", node.credibility]);
      items.push(["time", node.time]);
      if (node.session_id) items.push(["session", node.session_id]);
    } else if (type === "procedural") {
      items.push(["return", node.return]);
      items.push(["time", node.time]);
      if (node.session_id) items.push(["session", node.session_id]);
    } else if (type === "episodic") {
      if (node.session_id) items.push(["session", node.session_id]);
      if (node.subgoal) items.push(["subgoal", node.subgoal]);
      items.push(["time", node.time]);
    }
    if (!items.length) return null;
    for (const [k, v] of items) {
      wrap.appendChild(badge(`${k}: ${v}`));
    }
    return wrap;
  }
  function edgeList(list) {
    const wrap = document.createElement("div");
    wrap.className = "detail-edge-list";
    for (const item of list) {
      const node = document.createElement("div");
      node.className = "detail-edge";
      const targetType = inferEdgeType(item);
      const idLabel = item.id ?? "";
      const text =
        item.text || item.tag || item.subgoal || item.observation || item.action || "";
      const pill = document.createElement("span");
      pill.className = `node-id-pill type-${targetType}`;
      pill.textContent = `#${idLabel}`;
      node.appendChild(pill);
      const span = document.createElement("span");
      span.style.marginLeft = "6px";
      span.textContent = truncate(text, 120);
      node.appendChild(span);
      node.addEventListener("click", () => void openNodeDetail(targetType, idLabel));
      wrap.appendChild(node);
    }
    return wrap;
  }
  function inferEdgeType(item) {
    if ("semantic_id" in item) return "semantic";
    if ("tag_id" in item) return "tag";
    if ("subgoal_id" in item) return "subgoal";
    if ("procedural_id" in item) return "procedural";
    if ("episodic_id" in item) return "episodic";
    return "semantic";
  }
  function badge(text) {
    const b = document.createElement("span");
    b.className = "badge";
    b.textContent = text;
    return b;
  }
  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // wire close
  els.detailClose.addEventListener("click", closeDetail);

  return {
    refresh() {
      void loadSessions();
    },
  };
}
