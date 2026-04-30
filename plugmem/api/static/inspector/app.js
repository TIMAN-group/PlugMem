// Inspector shell: tab routing, graph picker, stats bar, theme switcher.
//
// State lives in URL query params:
//   ?graph=<id>&tab=<browse|recall|graph>&theme=<default|...>
// so refreshing or sharing a link preserves view.

import { api, getApiKey, setApiKey } from "./api.js";
import { mountBrowse } from "./browse.js";

const TABS = ["browse", "recall", "graph"];
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
  },
  toast: document.getElementById("toast"),
  apiKeyBtn: document.getElementById("api-key-btn"),
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

function applyTheme(name) {
  state.theme = name;
  els.themeLink.href = `themes/${name}.css`;
  els.themePicker.value = name;
  writeUrl();
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
}

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
function refreshBrowse() {
  if (!browseHandle) return;
  browseHandle.refresh({ graphId: state.graphId });
}

async function onGraphChange(gid) {
  state.graphId = gid || null;
  writeUrl();
  await loadStats();
  refreshBrowse();
}

function bindControls() {
  els.graphPicker.addEventListener("change", (e) => onGraphChange(e.target.value));
  els.themePicker.addEventListener("change", (e) => applyTheme(e.target.value));
  for (const btn of els.tabButtons) {
    btn.addEventListener("click", () => selectTab(btn.dataset.tab));
  }
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

  selectTab(state.tab);
}

bindControls();
boot();
