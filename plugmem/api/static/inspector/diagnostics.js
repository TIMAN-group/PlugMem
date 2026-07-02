import { api } from "./api.js";

export function mountDiagnostics({ container, toast }) {
  const els = {
    summary: container.querySelector("#diag-summary"),
    tbody: container.querySelector("#diag-tbody"),
    empty: container.querySelector("#diag-empty"),
    refresh: container.querySelector("#diag-refresh"),
    detail: container.querySelector("#diag-detail"),
    detailTitle: container.querySelector("#diag-detail-title"),
    detailBody: container.querySelector("#diag-detail-body"),
    detailClose: container.querySelector("#diag-detail-close"),
  };

  let selectedTaskId = null;
  let currentGraphId = null;

  async function loadStats() {
    try {
      const stats = await api.getDiagnosticsStats(currentGraphId);
      renderStats(stats);
    } catch (err) {
      toast(`stats: ${err.message}`, "error");
    }
  }

  function renderStats(stats) {
    if (!stats) {
      els.summary.innerHTML = `<div class="hint">No stats available.</div>`;
      return;
    }

    const cards = [
      { label: "Total LLM Calls", val: stats.total_llm_calls || 0 },
      { label: "Total Tokens", val: (stats.total_tokens || 0).toLocaleString() },
      { label: "Avg Latency (LLM)", val: `${(stats.avg_latency_sec || 0).toFixed(3)}s` },
      { label: "RAM Usage RSS", val: `${(stats.latest_ram_usage_mb || 0).toFixed(1)} MB` },
      { label: "DB Disk Usage", val: `${(stats.latest_db_disk_usage_mb || 0).toFixed(2)} MB` },
    ];

    els.summary.innerHTML = cards
      .map(
        (c) => `
        <div class="diag-card">
          <div class="diag-card-label">${c.label}</div>
          <div class="diag-card-val">${c.val}</div>
        </div>
      `
      )
      .join("");
  }

  async function loadLogs() {
    try {
      const logs = await api.getDiagnosticsLogs(200, currentGraphId);
      renderLogs(logs);
    } catch (err) {
      toast(`logs: ${err.message}`, "error");
    }
  }

  function renderLogs(logs) {
    els.tbody.innerHTML = "";
    els.empty.hidden = logs.length > 0;

    for (const log of logs) {
      const tr = document.createElement("tr");
      tr.className = "clickable-row";
      if (log.task_id === selectedTaskId) {
        tr.classList.add("selected");
      }

      const ts = log.timestamp ? log.timestamp.split("T")[1].substring(0, 8) : "";
      const latency = log.latency_sec ? `${log.latency_sec.toFixed(3)}s` : "0.000s";
      const tokens = log.total_prompt_tokens + log.total_completion_tokens;
      const statusClass = log.status_code >= 400 ? "status-err" : "status-ok";

      tr.innerHTML = `
        <td>${ts}</td>
        <td><span class="method-badge">${log.method}</span></td>
        <td><span class="endpoint-text">${log.endpoint}</span></td>
        <td><span class="status-badge ${statusClass}">${log.status_code}</span></td>
        <td>${latency}</td>
        <td>${tokens.toLocaleString()}</td>
      `;

      tr.addEventListener("click", () => {
        const rows = els.tbody.querySelectorAll("tr");
        for (const r of rows) r.classList.remove("selected");
        tr.classList.add("selected");
        showDetail(log);
      });

      els.tbody.appendChild(tr);
    }
  }

  function showDetail(log) {
    selectedTaskId = log.task_id;
    els.detailTitle.textContent = `Request Detail: ${log.task_id.substring(0, 12)}...`;
    
    // Clean objects for render
    const displayLog = { ...log };
    delete displayLog.task_id;

    let html = `
      <div class="detail-section">
        <h3>Metadata</h3>
        <table class="detail-meta-table">
          <tr><th>Timestamp</th><td>${log.timestamp}</td></tr>
          <tr><th>Method</th><td>${log.method}</td></tr>
          <tr><th>Endpoint</th><td>${log.endpoint}</td></tr>
          <tr><th>Status Code</th><td>${log.status_code}</td></tr>
          <tr><th>Latency</th><td>${log.latency_sec.toFixed(4)}s</td></tr>
        </table>
      </div>
      
      <div class="detail-section">
        <h3>Tokens & Memory</h3>
        <table class="detail-meta-table">
          <tr><th>Prompt Tokens</th><td>${log.total_prompt_tokens}</td></tr>
          <tr><th>Completion Tokens</th><td>${log.total_completion_tokens}</td></tr>
          <tr><th>RAM Usage</th><td>${log.ram_usage_mb.toFixed(2)} MB</td></tr>
          <tr><th>DB Size</th><td>${log.db_disk_usage_mb.toFixed(2)} MB</td></tr>
        </table>
      </div>
    `;

    if (log.memory_retrieved) {
      html += `
        <div class="detail-section">
          <h3>Memory Retrieved</h3>
          <pre class="code-wrap">${escapeHtml(log.memory_retrieved)}</pre>
        </div>
      `;
    }

    if (log.agent_output) {
      html += `
        <div class="detail-section">
          <h3>Agent Output</h3>
          <pre class="code-wrap">${escapeHtml(log.agent_output)}</pre>
        </div>
      `;
    }

    if (log.llm_calls && log.llm_calls.length > 0) {
      html += `
        <div class="detail-section">
          <h3>LLM Calls (${log.llm_calls.length})</h3>
          ${log.llm_calls.map((c, i) => `
            <div class="sub-card">
              <strong>Call #${i + 1} - ${c.model}</strong>
              <div class="hint">Prompt: ${c.prompt_tokens}t | Completion: ${c.completion_tokens}t | Latency: ${c.latency_sec.toFixed(3)}s</div>
            </div>
          `).join("")}
        </div>
      `;
    }

    if (log.retrieval_calls && log.retrieval_calls.length > 0) {
      html += `
        <div class="detail-section">
          <h3>Retrieval Calls (${log.retrieval_calls.length})</h3>
          ${log.retrieval_calls.map((c, i) => `
            <div class="sub-card">
              <strong>Call #${i + 1} - ${c.mode}</strong>
              <div class="hint">Latency: ${c.latency_sec.toFixed(3)}s</div>
            </div>
          `).join("")}
        </div>
      `;
    }

    els.detailBody.innerHTML = html;
    els.detail.hidden = false;
  }

  function closeDetail() {
    selectedTaskId = null;
    els.detail.hidden = true;
    const rows = els.tbody.querySelectorAll("tr");
    for (const r of rows) r.classList.remove("selected");
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  els.refresh.addEventListener("click", () => {
    void loadStats();
    void loadLogs();
  });
  els.detailClose.addEventListener("click", closeDetail);

  // Expose handles
  return {
    refresh({ graphId } = {}) {
      currentGraphId = graphId || null;
      closeDetail();
      void loadStats();
      void loadLogs();
    },
  };
}
