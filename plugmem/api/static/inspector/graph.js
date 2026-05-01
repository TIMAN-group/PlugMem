// Graph tab: cytoscape-rendered topology view.
//
// Cytoscape is loaded from CDN as a global (`window.cytoscape`) — no module
// import. This file only runs when the Graph tab is selected.

import { api } from "./api.js";

const NODE_TYPES = ["semantic", "procedural", "tag", "subgoal", "episodic"];

const EDGE_KINDS = [
  { kind: "tagged",        label: "tagged" },
  { kind: "related",       label: "related (bro)" },
  { kind: "derived_from",  label: "derived from" },
  { kind: "evidenced_by",  label: "evidenced by" },
  { kind: "grouped_by",    label: "grouped by" },
  { kind: "from_session",  label: "from session" },
];

function readVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function buildStylesheet() {
  // Re-read CSS variables every render so theme switches stick.
  const c = {
    semantic:   readVar("--node-semantic")   || "#3a6df0",
    procedural: readVar("--node-procedural") || "#2e8b57",
    tag:        readVar("--node-tag")        || "#d97b2a",
    subgoal:    readVar("--node-subgoal")    || "#7b5ea7",
    episodic:   readVar("--node-episodic")   || "#6c757d",
    fg:         readVar("--fg")              || "#1a1a1a",
    fgMuted:    readVar("--fg-muted")        || "#5a5a63",
    border:     readVar("--border-strong")   || "#c8c8cf",
    bg:         readVar("--bg-elev")         || "#ffffff",
  };
  return [
    {
      selector: "node",
      style: {
        "background-color": c.semantic,
        "label": "data(label)",
        "color": c.fg,
        "font-size": 10,
        "text-valign": "bottom",
        "text-halign": "center",
        "text-margin-y": 4,
        "text-wrap": "ellipsis",
        "text-max-width": 140,
        "border-width": 1,
        "border-color": c.border,
        "width": 22,
        "height": 22,
      },
    },
    { selector: "node.semantic",   style: { "background-color": c.semantic,   "shape": "round-rectangle", "width": 28, "height": 16 } },
    { selector: "node.procedural", style: { "background-color": c.procedural, "shape": "diamond",         "width": 24, "height": 24 } },
    { selector: "node.tag",        style: { "background-color": c.tag,        "shape": "round-tag",       "width": 26, "height": 16 } },
    { selector: "node.subgoal",    style: { "background-color": c.subgoal,    "shape": "round-hexagon",   "width": 26, "height": 22 } },
    { selector: "node.episodic",   style: { "background-color": c.episodic,   "shape": "ellipse",         "width": 14, "height": 14, "font-size": 9 } },
    {
      selector: "node.inactive",
      style: {
        "opacity": 0.45,
        "border-style": "dashed",
      },
    },
    {
      selector: "node:selected",
      style: {
        "border-width": 3,
        "border-color": c.fg,
      },
    },
    {
      selector: "edge",
      style: {
        "width": 1,
        "line-color": c.fgMuted,
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "target-arrow-color": c.fgMuted,
        "arrow-scale": 0.8,
        "opacity": 0.6,
      },
    },
    { selector: "edge.tagged",       style: { "line-color": c.tag,        "target-arrow-color": c.tag,        "line-style": "solid"  } },
    { selector: "edge.related",      style: { "line-color": c.semantic,   "target-arrow-shape": "none",       "line-style": "dashed" } },
    { selector: "edge.derived_from", style: { "line-color": c.semantic,   "target-arrow-color": c.semantic,   "line-style": "solid"  } },
    { selector: "edge.evidenced_by", style: { "line-color": c.episodic,   "target-arrow-color": c.episodic,   "line-style": "dotted" } },
    { selector: "edge.grouped_by",   style: { "line-color": c.subgoal,    "target-arrow-color": c.subgoal,    "line-style": "solid"  } },
    { selector: "edge.from_session", style: { "line-color": c.episodic,   "target-arrow-color": c.episodic,   "line-style": "dotted" } },
    {
      selector: "node.dim, edge.dim",
      style: { "opacity": 0.08 },
    },
    {
      selector: "node.highlight",
      style: {
        "border-width": 3,
        "border-color": c.fg,
        "opacity": 1,
      },
    },
    {
      selector: "edge.highlight",
      style: { "opacity": 1, "width": 2 },
    },
  ];
}

function layoutOptions(name) {
  const common = { animate: false, fit: true, padding: 30 };
  if (name === "cose") {
    return {
      ...common,
      name: "cose",
      randomize: true,
      idealEdgeLength: 110,
      nodeOverlap: 8,
      gravity: 60,
      numIter: 1200,
      coolingFactor: 0.99,
    };
  }
  if (name === "concentric") {
    return {
      ...common,
      name: "concentric",
      minNodeSpacing: 30,
      concentric: (n) => {
        const t = n.data("type");
        return ({ tag: 4, subgoal: 3, semantic: 2, procedural: 2, episodic: 1 }[t] ?? 0);
      },
      levelWidth: () => 1,
    };
  }
  if (name === "breadthfirst") {
    return { ...common, name: "breadthfirst", spacingFactor: 1.2 };
  }
  return { ...common, name: "grid" };
}

export function mountGraph({ container, getGraphId, toast, onTheme }) {
  const els = {
    typeRow:       container.querySelector("#graph-type-toggles"),
    inactiveCheck: container.querySelector("#graph-include-inactive"),
    layoutSel:     container.querySelector("#graph-layout"),
    nodeLimit:     container.querySelector("#graph-node-limit"),
    refresh:       container.querySelector("#graph-refresh"),
    fit:           container.querySelector("#graph-fit"),
    exportBtn:     container.querySelector("#graph-export"),
    counts:        container.querySelector("#graph-counts"),
    legend:        container.querySelector("#graph-legend"),
    canvas:        container.querySelector("#graph-canvas"),
    empty:         container.querySelector("#graph-empty"),
    detail:        container.querySelector("#graph-detail"),
    detailTitle:   container.querySelector("#graph-detail-title"),
    detailBody:    container.querySelector("#graph-detail-body"),
    detailClose:   container.querySelector("#graph-detail-close"),
  };

  // Defaults: hide episodic since it's noisy. Active = visible in graph.
  const state = {
    typeActive: { semantic: true, procedural: true, tag: true, subgoal: true, episodic: false },
    cy: null,
    lastPayload: null,
  };

  function renderTypeToggles() {
    els.typeRow.innerHTML = "";
    for (const t of NODE_TYPES) {
      const btn = document.createElement("button");
      btn.className = `chip type-${t}`;
      btn.textContent = t;
      btn.setAttribute("aria-pressed", state.typeActive[t] ? "true" : "false");
      btn.addEventListener("click", () => {
        const prev = state.typeActive[t];
        state.typeActive[t] = !prev;
        btn.setAttribute("aria-pressed", state.typeActive[t] ? "true" : "false");
        // Toggling episodic on requires a server fetch — episodics aren't
        // included by default. Other types are always fetched, so just
        // toggle visibility client-side.
        if (t === "episodic" && state.typeActive[t]) {
          void load();
        } else {
          applyVisibility();
        }
      });
      els.typeRow.appendChild(btn);
    }
  }

  const EDGE_STYLE = {
    tagged:        { color: "--node-tag",      style: "solid"  },
    related:       { color: "--node-semantic", style: "dashed" },
    derived_from:  { color: "--node-semantic", style: "solid"  },
    evidenced_by:  { color: "--node-episodic", style: "dotted" },
    grouped_by:    { color: "--node-subgoal",  style: "solid"  },
    from_session:  { color: "--node-episodic", style: "dotted" },
  };

  function renderLegend() {
    els.legend.innerHTML = "";
    const head = document.createElement("div");
    head.style.fontWeight = "600";
    head.textContent = "Edges";
    els.legend.appendChild(head);
    for (const { kind, label } of EDGE_KINDS) {
      const cfg = EDGE_STYLE[kind];
      const row = document.createElement("div");
      row.className = "graph-legend-row";
      const line = document.createElement("span");
      line.className = "graph-legend-line";
      line.style.borderTopStyle = cfg.style;
      line.style.borderTopColor = readVar(cfg.color);
      row.appendChild(line);
      const lbl = document.createElement("span");
      lbl.textContent = label;
      row.appendChild(lbl);
      els.legend.appendChild(row);
    }
  }

  function ensureCy() {
    if (state.cy) return state.cy;
    if (typeof window.cytoscape !== "function") {
      toast("cytoscape failed to load", "error");
      return null;
    }
    state.cy = window.cytoscape({
      container: els.canvas,
      elements: [],
      style: buildStylesheet(),
      wheelSensitivity: 0.2,
      minZoom: 0.1,
      maxZoom: 4,
    });
    state.cy.on("tap", "node", (evt) => openDetail(evt.target));
    state.cy.on("tap", (evt) => {
      if (evt.target === state.cy) {
        clearHighlight();
        closeDetail();
      }
    });
    return state.cy;
  }

  function applyVisibility() {
    const cy = state.cy;
    if (!cy) return;
    cy.batch(() => {
      cy.nodes().forEach((n) => {
        const t = n.data("type");
        n.style("display", state.typeActive[t] ? "element" : "none");
      });
      cy.edges().forEach((e) => {
        const s = e.source(), t = e.target();
        const visible = s.style("display") !== "none" && t.style("display") !== "none";
        e.style("display", visible ? "element" : "none");
      });
    });
  }

  async function load() {
    const gid = getGraphId();
    if (!gid) {
      els.empty.hidden = false;
      els.empty.textContent = "Pick a graph and click Refresh.";
      els.counts.textContent = "";
      if (state.cy) state.cy.elements().remove();
      return;
    }
    els.empty.hidden = false;
    els.empty.textContent = "Loading…";
    els.refresh.disabled = true;
    try {
      const limit = Math.max(10, parseInt(els.nodeLimit.value, 10) || 500);
      const payload = await api.topology(gid, {
        include_episodic: state.typeActive.episodic ? true : false,
        include_inactive: !!els.inactiveCheck.checked,
        node_limit: limit,
      });
      state.lastPayload = payload;
      renderCounts(payload);
      drawElements(payload);
      els.empty.hidden = (payload.nodes || []).length > 0;
      if (!(payload.nodes || []).length) {
        els.empty.textContent = "Graph is empty.";
      }
    } catch (err) {
      els.empty.hidden = false;
      els.empty.textContent = `Error: ${err.message}`;
      toast(`topology: ${err.message}`, "error");
    } finally {
      els.refresh.disabled = false;
    }
  }

  function renderCounts(payload) {
    const c = payload.counts || {};
    const order = ["semantic", "procedural", "tag", "subgoal", "episodic"];
    const parts = order
      .filter((k) => k in c && c[k] > 0)
      .map((k) => `${k}: ${c[k]}`);
    parts.push(`edges: ${c.total_edges ?? 0}`);
    let line = parts.join(" · ");
    if (payload.truncated) {
      line += `  (truncated to ${payload.node_limit})`;
    }
    els.counts.textContent = line;
  }

  function drawElements(payload) {
    const cy = ensureCy();
    if (!cy) return;
    const nodes = (payload.nodes || []).map((n) => ({ ...n, group: "nodes" }));
    const edges = (payload.edges || []).map((e) => ({ ...e, group: "edges" }));
    cy.elements().remove();
    cy.add(nodes);
    cy.add(edges);
    applyVisibility();
    runLayout();
  }

  function runLayout() {
    const cy = state.cy;
    if (!cy) return;
    const visible = cy.elements(":visible");
    if (visible.length === 0) return;
    visible.layout(layoutOptions(els.layoutSel.value)).run();
  }

  function clearHighlight() {
    const cy = state.cy;
    if (!cy) return;
    cy.elements().removeClass("dim highlight");
  }

  function highlightNeighborhood(node) {
    const cy = state.cy;
    if (!cy) return;
    const nbh = node.closedNeighborhood();
    cy.batch(() => {
      cy.elements().addClass("dim").removeClass("highlight");
      nbh.removeClass("dim").addClass("highlight");
    });
  }

  async function openDetail(node) {
    highlightNeighborhood(node);
    els.detail.hidden = false;
    const data = node.data();
    els.detailTitle.innerHTML = "";
    const pill = document.createElement("span");
    pill.className = `node-id-pill type-${data.type}`;
    pill.textContent = `#${data.node_id}`;
    els.detailTitle.appendChild(pill);
    const sub = document.createElement("span");
    sub.style.marginLeft = "8px";
    sub.textContent = data.type;
    els.detailTitle.appendChild(sub);

    els.detailBody.innerHTML = "<p class='hint'>Loading…</p>";
    const gid = getGraphId();
    if (!gid) return;
    try {
      const res = await api.getNode(gid, data.type, data.node_id);
      renderDetail(res);
    } catch (err) {
      els.detailBody.innerHTML = `<p class='hint'>error: ${escapeHtml(err.message)}</p>`;
    }
  }

  function renderDetail(res) {
    const { node_type, node, edges } = res;
    const body = els.detailBody;
    body.innerHTML = "";

    body.appendChild(detailSection("Text", textBlock(primaryText(node_type, node))));

    const meta = metaRow(node_type, node);
    if (meta) body.appendChild(detailSection("Meta", meta));

    for (const [name, list] of Object.entries(edges || {})) {
      if (!Array.isArray(list) || list.length === 0) continue;
      body.appendChild(detailSection(`${name} (${list.length})`, edgeList(name, list)));
    }
  }

  function closeDetail() {
    els.detail.hidden = true;
  }

  function detailSection(title, content) {
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
    } else if (type === "tag") {
      items.push(["importance", node.importance]);
      items.push(["semantics", node.n_semantics]);
    } else if (type === "subgoal") {
      items.push(["activated", node.activated]);
      items.push(["procedurals", node.n_procedurals]);
    } else if (type === "episodic") {
      if (node.session_id) items.push(["session", node.session_id]);
      items.push(["time", node.time]);
    }
    if (!items.length) return null;
    for (const [k, v] of items) {
      const pill = document.createElement("span");
      pill.className = "badge";
      pill.textContent = `${k}: ${v}`;
      wrap.appendChild(pill);
    }
    return wrap;
  }

  function edgeList(_name, list) {
    const wrap = document.createElement("div");
    wrap.className = "detail-edge-list";
    for (const item of list) {
      const node = document.createElement("div");
      node.className = "detail-edge";
      const targetType = inferEdgeType(item);
      const idLabel = item.id ?? "";
      const text =
        item.text ||
        item.tag ||
        item.subgoal ||
        item.observation ||
        item.action ||
        "";
      const pill = document.createElement("span");
      pill.className = `node-id-pill type-${targetType}`;
      pill.textContent = `#${idLabel}`;
      node.appendChild(pill);
      const span = document.createElement("span");
      span.style.marginLeft = "6px";
      span.textContent = truncate(text, 120);
      node.appendChild(span);
      node.addEventListener("click", () => {
        const cy = state.cy;
        if (!cy) return;
        const target = cy.getElementById(uid(targetType, idLabel));
        if (target && target.length) {
          target.emit("tap");
          cy.animate({ center: { eles: target }, zoom: 1.2 }, { duration: 250 });
        }
      });
      wrap.appendChild(node);
    }
    return wrap;
  }

  function uid(type, id) {
    return ({ semantic: "sem", tag: "tag", procedural: "proc", subgoal: "sg", episodic: "epis" }[type] ?? type) + "-" + id;
  }
  function inferEdgeType(item) {
    if ("semantic_id" in item) return "semantic";
    if ("tag_id" in item) return "tag";
    if ("subgoal_id" in item) return "subgoal";
    if ("procedural_id" in item) return "procedural";
    if ("episodic_id" in item) return "episodic";
    return "semantic";
  }
  function truncate(text, n) {
    if (!text) return "";
    return text.length > n ? text.slice(0, n - 1) + "…" : text;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // Wire controls
  renderTypeToggles();
  renderLegend();
  els.refresh.addEventListener("click", load);
  els.inactiveCheck.addEventListener("change", load);
  els.fit.addEventListener("click", () => state.cy && state.cy.fit(undefined, 30));
  els.layoutSel.addEventListener("change", runLayout);
  els.exportBtn.addEventListener("click", () => {
    if (!state.cy) return;
    const png = state.cy.png({ full: true, scale: 2, bg: readVar("--bg") || "#fafafa" });
    const a = document.createElement("a");
    a.href = png;
    a.download = `plugmem-graph-${getGraphId() || "graph"}.png`;
    a.click();
  });
  els.detailClose.addEventListener("click", () => {
    clearHighlight();
    closeDetail();
  });

  if (typeof onTheme === "function") {
    onTheme(() => {
      renderLegend();
      if (!state.cy) return;
      state.cy.style(buildStylesheet());
    });
  }

  return {
    refresh() {
      closeDetail();
      void load();
    },
    relayout() {
      // called when the tab becomes visible — cytoscape needs to recalc its
      // canvas size if it was hidden during init.
      if (!state.cy) return;
      state.cy.resize();
      state.cy.fit(undefined, 30);
    },
  };
}
