// Browse tab: type chips, search, table, side panel, deactivation.

import { api } from "./api.js";

const NODE_TYPES = ["semantic", "procedural", "tag", "subgoal", "episodic"];

const COLUMNS = {
  semantic: [
    { key: "id", label: "id", render: idPill },
    { key: "text", label: "fact", render: textCell },
    { key: "tags", label: "tags", render: tagsCell },
    { key: "credibility", label: "cred" },
    { key: "time", label: "t" },
    { key: "is_active", label: "state", render: stateBadge },
  ],
  procedural: [
    { key: "id", label: "id", render: idPill },
    { key: "text", label: "experience", render: textCell },
    { key: "subgoals", label: "subgoal", render: subgoalsCell },
    { key: "return", label: "ret" },
    { key: "time", label: "t" },
  ],
  tag: [
    { key: "id", label: "id", render: idPill },
    { key: "tag", label: "tag" },
    { key: "importance", label: "imp" },
    { key: "n_semantics", label: "→ sem" },
    { key: "time", label: "t" },
  ],
  subgoal: [
    { key: "id", label: "id", render: idPill },
    { key: "subgoal", label: "subgoal", render: textCell },
    { key: "n_procedurals", label: "→ proc" },
    { key: "time", label: "t" },
  ],
  episodic: [
    { key: "id", label: "id", render: idPill },
    { key: "observation", label: "observation", render: textCell },
    { key: "action", label: "action", render: textCell },
    { key: "time", label: "t" },
  ],
};

function idPill(value, row, type) {
  const span = document.createElement("span");
  span.className = `node-id-pill type-${type}`;
  span.textContent = `#${value}`;
  return span;
}
function textCell(value) {
  const span = document.createElement("span");
  span.className = "cell-text";
  span.textContent = value || "";
  span.title = value || "";
  return span;
}
function tagsCell(value) {
  const wrap = document.createElement("span");
  wrap.className = "cell-tags";
  for (const t of value || []) {
    const pill = document.createElement("span");
    pill.className = "tag-pill";
    pill.textContent = t;
    wrap.appendChild(pill);
  }
  return wrap;
}
function subgoalsCell(value) {
  return tagsCellGeneric(value, "node-subgoal");
}
function tagsCellGeneric(value, _color) {
  const wrap = document.createElement("span");
  wrap.className = "cell-tags";
  for (const t of value || []) {
    const pill = document.createElement("span");
    pill.className = "tag-pill";
    pill.textContent = t;
    wrap.appendChild(pill);
  }
  return wrap;
}
function stateBadge(value) {
  const span = document.createElement("span");
  span.className = `badge ${value ? "ok" : "inactive"}`;
  span.textContent = value ? "active" : "inactive";
  return span;
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export function mountBrowse({ container, getGraphId, toast }) {
  const els = {
    chipRow: container.querySelector("#browse-type-chips"),
    search: container.querySelector("#browse-search"),
    onlyActive: container.querySelector("#browse-only-active"),
    thead: container.querySelector("#browse-thead"),
    tbody: container.querySelector("#browse-tbody"),
    empty: container.querySelector("#browse-empty"),
    hint: container.querySelector("#browse-result-hint"),
    splitWrap: container.querySelector(".split"),
    detail: container.querySelector("#detail-panel"),
    detailTitle: container.querySelector("#detail-title"),
    detailBody: container.querySelector("#detail-body"),
    detailClose: container.querySelector("#detail-close"),
  };

  const state = {
    type: "semantic",
    q: "",
    onlyActive: false,
    rows: [],
    selectedKey: null,
  };

  function renderChips() {
    els.chipRow.innerHTML = "";
    for (const t of NODE_TYPES) {
      const btn = document.createElement("button");
      btn.className = `chip type-${t}`;
      btn.textContent = t;
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", t === state.type ? "true" : "false");
      btn.addEventListener("click", () => {
        state.type = t;
        for (const c of els.chipRow.querySelectorAll(".chip")) {
          c.setAttribute("aria-selected", c.textContent === t ? "true" : "false");
        }
        closeDetail();
        void load();
      });
      els.chipRow.appendChild(btn);
    }
  }

  function renderHeader() {
    const cols = COLUMNS[state.type];
    els.thead.innerHTML = "";
    for (const col of cols) {
      const th = document.createElement("th");
      th.textContent = col.label;
      els.thead.appendChild(th);
    }
  }

  function renderTable(rows) {
    const cols = COLUMNS[state.type];
    els.tbody.innerHTML = "";
    if (!rows.length) {
      els.empty.hidden = false;
      return;
    }
    els.empty.hidden = true;
    for (const row of rows) {
      const tr = document.createElement("tr");
      const rowKey = `${state.type}-${row.id}`;
      tr.dataset.key = rowKey;
      if (state.type === "semantic" && row.is_active === false) tr.classList.add("inactive");
      if (state.selectedKey === rowKey) tr.classList.add("selected");
      for (const col of cols) {
        const td = document.createElement("td");
        const v = row[col.key];
        if (col.render) {
          const node = col.render(v, row, state.type);
          if (node instanceof Node) td.appendChild(node);
          else td.textContent = node == null ? "" : String(node);
        } else {
          td.textContent = v == null ? "" : String(v);
        }
        tr.appendChild(td);
      }
      tr.addEventListener("click", () => openDetail(row.id));
      els.tbody.appendChild(tr);
    }
  }

  async function load() {
    const gid = getGraphId();
    renderHeader();
    if (!gid) {
      state.rows = [];
      renderTable([]);
      els.hint.textContent = "Pick a graph.";
      return;
    }
    try {
      const res = await api.search(gid, {
        q: state.q,
        node_type: state.type,
        limit: 200,
        only_active: state.onlyActive ? true : undefined,
      });
      state.rows = res.nodes || [];
      const total = res.count;
      const shown = state.rows.length;
      els.hint.textContent =
        total === 0
          ? "no matches"
          : shown < total
          ? `showing ${shown} of ${total}`
          : `${total} total`;
      renderTable(state.rows);
    } catch (err) {
      toast(`search: ${err.message}`, "error");
    }
  }

  async function openDetail(id) {
    const gid = getGraphId();
    if (!gid) return;
    state.selectedKey = `${state.type}-${id}`;
    for (const tr of els.tbody.querySelectorAll("tr")) {
      tr.classList.toggle("selected", tr.dataset.key === state.selectedKey);
    }
    els.splitWrap.classList.add("has-detail");
    els.detail.hidden = false;
    els.detailTitle.innerHTML = "";
    els.detailBody.innerHTML = "<p class='hint'>Loading…</p>";

    try {
      const res = await api.getNode(gid, state.type, id);
      renderDetail(res);
    } catch (err) {
      els.detailBody.innerHTML = `<p class='hint'>error: ${escapeHtml(err.message)}</p>`;
    }
  }

  function renderDetail(res) {
    const { node_type, node, edges } = res;
    els.detailTitle.innerHTML = "";
    const pill = idPill(node.id, node, node_type);
    els.detailTitle.appendChild(pill);
    const label = document.createElement("span");
    label.style.marginLeft = "8px";
    label.textContent = node_type;
    els.detailTitle.appendChild(label);

    const body = els.detailBody;
    body.innerHTML = "";

    body.appendChild(detailSection("Text", textBlock(primaryText(node_type, node))));

    const meta = metaRow(node_type, node);
    if (meta) body.appendChild(detailSection("Meta", meta));

    for (const [name, list] of Object.entries(edges || {})) {
      if (!Array.isArray(list) || list.length === 0) continue;
      body.appendChild(detailSection(`${name} (${list.length})`, edgeList(name, list)));
    }

    if (node_type === "semantic") {
      body.appendChild(actionsRow(node));
    }
  }

  function actionsRow(node) {
    const wrap = document.createElement("div");
    wrap.className = "detail-actions";
    const btn = document.createElement("button");
    btn.className = node.is_active ? "btn btn-danger" : "btn btn-primary";
    btn.textContent = node.is_active ? "Deactivate" : "Reactivate";
    btn.addEventListener("click", async () => {
      const gid = getGraphId();
      if (!gid) return;
      btn.disabled = true;
      try {
        await api.patchSemantic(gid, node.id, { is_active: !node.is_active });
        toast(`semantic ${node.id} ${node.is_active ? "deactivated" : "reactivated"}`);
        // refresh row + detail
        await load();
        await openDetail(node.id);
      } catch (err) {
        toast(`update: ${err.message}`, "error");
      } finally {
        btn.disabled = false;
      }
    });
    wrap.appendChild(btn);
    return wrap;
  }

  function closeDetail() {
    els.splitWrap.classList.remove("has-detail");
    els.detail.hidden = true;
    state.selectedKey = null;
    for (const tr of els.tbody.querySelectorAll("tr.selected")) {
      tr.classList.remove("selected");
    }
  }

  function detailSection(title, content) {
    const sec = document.createElement("div");
    sec.className = "detail-section";
    const h = document.createElement("h3");
    h.textContent = title;
    sec.appendChild(h);
    if (content instanceof Node) sec.appendChild(content);
    else if (typeof content === "string") {
      const p = document.createElement("p");
      p.textContent = content;
      sec.appendChild(p);
    }
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
      if (node.tags?.length) items.push(["tags", node.tags.length]);
      if (node.session_id) items.push(["session", node.session_id]);
      if (node.date) items.push(["date", node.date]);
    } else if (type === "procedural") {
      items.push(["return", node.return]);
      items.push(["time", node.time]);
      items.push(["episodics", node.n_episodics]);
    } else if (type === "tag") {
      items.push(["importance", node.importance]);
      items.push(["semantics", node.n_semantics]);
      items.push(["time", node.time]);
    } else if (type === "subgoal") {
      items.push(["activated", node.activated]);
      items.push(["procedurals", node.n_procedurals]);
      items.push(["time", node.time]);
    } else if (type === "episodic") {
      if (node.subgoal) items.push(["subgoal", node.subgoal]);
      if (node.session_id) items.push(["session", node.session_id]);
      if (node.state) items.push(["state", node.state]);
      if (node.reward) items.push(["reward", node.reward]);
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
      node.innerHTML = "";
      const pill = idPill(idLabel, item, targetType);
      node.appendChild(pill);
      const span = document.createElement("span");
      span.style.marginLeft = "6px";
      span.textContent = truncate(text, 140);
      node.appendChild(span);
      node.addEventListener("click", () => {
        state.type = targetType;
        for (const c of els.chipRow.querySelectorAll(".chip")) {
          c.setAttribute("aria-selected", c.textContent === targetType ? "true" : "false");
        }
        renderHeader();
        void load();
        void openDetail(idLabel);
      });
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

  function truncate(text, n) {
    if (!text) return "";
    return text.length > n ? text.slice(0, n - 1) + "…" : text;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // wire controls
  renderChips();
  renderHeader();

  els.search.addEventListener("input", debounce((e) => {
    state.q = e.target.value;
    void load();
  }, 200));

  els.onlyActive.addEventListener("change", (e) => {
    state.onlyActive = !!e.target.checked;
    void load();
  });

  els.detailClose.addEventListener("click", closeDetail);

  return {
    refresh() {
      closeDetail();
      void load();
    },
  };
}
