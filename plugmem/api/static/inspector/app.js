// Inspector shell: tab routing, graph picker, stats bar, theme switcher.
//
// State lives in URL query params:
//   ?graph=<id>&tab=<browse|recall|graph>&theme=<default|...>
// so refreshing or sharing a link preserves view.

import { api, getApiKey, setApiKey } from "./api.js";
import { mountBrowse } from "./browse.js";
import { mountRecall } from "./recall.js";
import { mountGraph } from "./graph.js";
import { mountSessions } from "./sessions.js";

const TABS = ["browse", "recall", "graph", "sessions"];
const DEFAULT_TAB = "browse";

const state = {
  graphs: [],
  graphId: null,
  tab: DEFAULT_TAB,
  theme: "default",
  stats: null,
};

const els = {
  graphPicker: document.getElementById("graph-picker"),
  themePicker: document.getElementById("theme-picker"),
  themeLink: document.getElementById("theme-link"),
  statsBar: document.getElementById("stats-bar"),
  tabButtons: document.querySelectorAll(".tab"),
  tabPanels: {
    browse: document.getElementById("tab-browse"),
    recall: document.getElementById("tab-recall"),
    graph: document.getElementById("tab-graph"),
    sessions: document.getElementById("tab-sessions"),
  },
  toast: document.getElementById("toast"),
  apiKeyBtn: document.getElementById("api-key-btn"),
  emptyBanner: document.getElementById("empty-banner"),
  emptyBannerBtn: document.getElementById("empty-banner-btn"),
};

let toastTimer = null;
export function toast(msg, kind = "info") {
  els.toast.textContent = msg;
  els.toast.className = `toast ${kind}`;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 4000);
}

function readUrl() {
  const params = new URLSearchParams(location.search);
  const tab = params.get("tab");
  if (tab && TABS.includes(tab)) state.tab = tab;
  const theme = params.get("theme");
  if (theme) state.theme = theme;
  const gid = params.get("graph");
  if (gid) state.graphId = gid;
}

function writeUrl() {
  const params = new URLSearchParams();
  if (state.graphId) params.set("graph", state.graphId);
  if (state.tab && state.tab !== DEFAULT_TAB) params.set("tab", state.tab);
  if (state.theme && state.theme !== "default") params.set("theme", state.theme);
  const qs = params.toString();
  const url = qs ? `?${qs}` : location.pathname;
  history.replaceState(null, "", url);
}

const themeListeners = [];
function onTheme(cb) {
  themeListeners.push(cb);
  // Fire immediately so late-registering listeners (mountGraph runs after
  // initial applyTheme) get the current theme without an extra event.
  cb(state.theme);
}

function applyTheme(name) {
  state.theme = name;
  els.themeLink.href = `themes/${name}.css`;
  els.themePicker.value = name;
  document.documentElement.dataset.theme = name;
  writeUrl();
  // Wait one tick so the new stylesheet is applied before listeners read vars.
  setTimeout(() => { for (const cb of themeListeners) cb(name); }, 50);
}

function selectTab(name) {
  if (!TABS.includes(name)) name = DEFAULT_TAB;
  state.tab = name;
  for (const btn of els.tabButtons) {
    const active = btn.dataset.tab === name;
    btn.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const [key, panel] of Object.entries(els.tabPanels)) {
    panel.hidden = key !== name;
  }
  writeUrl();
  if (name === "browse") refreshBrowse();
  if (name === "graph" && graphHandle) {
    if (!graphHasLoaded) {
      graphHasLoaded = true;
      refreshGraph();
    } else {
      graphHandle.relayout();
    }
  }
  if (name === "sessions" && sessionsHandle) {
    if (!sessionsHasLoaded) {
      sessionsHasLoaded = true;
      refreshSessions();
    }
  }
}

let graphHasLoaded = false;
let sessionsHasLoaded = false;

function renderStats(stats) {
  if (!stats) {
    els.statsBar.innerHTML = "";
    return;
  }
  const order = ["semantic", "tag", "procedural", "subgoal", "episodic"];
  const parts = order
    .filter((k) => k in stats)
    .map((k) => `<span class="stat-pair"><span class="stat-key">${k}</span><span class="stat-val">${stats[k]}</span></span>`);
  els.statsBar.innerHTML = parts.join("");
}

async function loadStats() {
  if (!state.graphId) {
    state.stats = null;
    renderStats(null);
    return;
  }
  try {
    const res = await api.getStats(state.graphId);
    state.stats = res.stats || {};
  } catch (err) {
    state.stats = null;
    toast(`stats: ${err.message}`, "error");
  }
  renderStats(state.stats);
}

async function loadGraphs() {
  try {
    const res = await api.listGraphs();
    state.graphs = res.graphs || [];
  } catch (err) {
    state.graphs = [];
    toast(`graphs: ${err.message}`, "error");
  }

  els.graphPicker.innerHTML = "";
  els.emptyBanner.hidden = state.graphs.length > 0;
  if (state.graphs.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no graphs)";
    els.graphPicker.appendChild(opt);
    state.graphId = null;
    return;
  }

  for (const gid of state.graphs) {
    const opt = document.createElement("option");
    opt.value = gid;
    opt.textContent = gid;
    els.graphPicker.appendChild(opt);
  }

  if (!state.graphId || !state.graphs.includes(state.graphId)) {
    state.graphId = state.graphs[0];
  }
  els.graphPicker.value = state.graphId;
}

let browseHandle = null;
let recallHandle = null;
let graphHandle = null;
let sessionsHandle = null;
function refreshBrowse() {
  if (!browseHandle) return;
  browseHandle.refresh({ graphId: state.graphId });
}
function refreshRecall() {
  if (!recallHandle) return;
  recallHandle.refresh({ graphId: state.graphId });
}
function refreshGraph() {
  if (!graphHandle) return;
  graphHandle.refresh({ graphId: state.graphId });
}
function refreshSessions() {
  if (!sessionsHandle) return;
  sessionsHandle.refresh({ graphId: state.graphId });
}

async function onGraphChange(gid) {
  state.graphId = gid || null;
  writeUrl();
  await loadStats();
  refreshBrowse();
  refreshRecall();
  refreshGraph();
  if (state.tab === "sessions") refreshSessions();
}

function bindControls() {
  els.graphPicker.addEventListener("change", (e) => onGraphChange(e.target.value));
  els.themePicker.addEventListener("change", (e) => applyTheme(e.target.value));
  for (const btn of els.tabButtons) {
    btn.addEventListener("click", () => selectTab(btn.dataset.tab));
  }
  els.emptyBannerBtn.addEventListener("click", async () => {
    els.emptyBannerBtn.disabled = true;
    try {
      const res = await api.seedDemo();
      toast(`demo graph '${res.graph_id}' loaded`);
      state.graphId = res.graph_id;
      await loadGraphs();
      await loadStats();
      refreshBrowse();
      if (state.tab === "graph") {
        graphHasLoaded = true;
        refreshGraph();
      }
    } catch (err) {
      toast(`seed: ${err.message}`, "error");
    } finally {
      els.emptyBannerBtn.disabled = false;
    }
  });
  els.apiKeyBtn.addEventListener("click", () => {
    const cur = getApiKey();
    const next = prompt("X-API-Key (leave blank to clear):", cur);
    if (next === null) return;
    setApiKey(next.trim());
    toast(next.trim() ? "API key saved" : "API key cleared");
    void boot();
  });
}

async function boot() {
  readUrl();
  applyTheme(state.theme);
  await loadGraphs();
  await loadStats();

  browseHandle = mountBrowse({
    container: els.tabPanels.browse,
    getGraphId: () => state.graphId,
    toast,
  });
  recallHandle = mountRecall({
    container: els.tabPanels.recall,
    getGraphId: () => state.graphId,
    toast,
  });
  graphHandle = mountGraph({
    container: els.tabPanels.graph,
    getGraphId: () => state.graphId,
    toast,
    onTheme,
  });
  sessionsHandle = mountSessions({
    container: els.tabPanels.sessions,
    getGraphId: () => state.graphId,
    toast,
  });

  selectTab(state.tab);
}

bindControls();
boot();
