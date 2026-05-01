// Recall-trace tab: submit observation/tags/mode → render full retrieval trace.

import { api } from "./api.js";

export function mountRecall({ container, getGraphId, toast }) {
  const els = {
    observation: container.querySelector("#recall-observation"),
    mode: container.querySelector("#recall-mode"),
    tags: container.querySelector("#recall-tags"),
    subgoal: container.querySelector("#recall-subgoal"),
    goal: container.querySelector("#recall-goal"),
    state: container.querySelector("#recall-state"),
    autoPlan: container.querySelector("#recall-auto-plan"),
    runBtn: container.querySelector("#recall-run"),
    empty: container.querySelector("#recall-empty"),
    results: container.querySelector("#recall-results"),
  };

  function parseTags(s) {
    return (s || "")
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);
  }

  async function run() {
    const gid = getGraphId();
    if (!gid) {
      toast("pick a graph first", "warn");
      return;
    }
    const observation = els.observation.value.trim();
    if (!observation) {
      toast("observation is required", "warn");
      return;
    }

    const body = {
      observation,
      mode: els.mode.value,
      auto_plan: els.autoPlan.checked,
    };
    const tagsStr = els.tags.value.trim();
    if (tagsStr) body.query_tags = parseTags(tagsStr);
    const subgoal = els.subgoal.value.trim();
    if (subgoal) body.next_subgoal = subgoal;
    const goal = els.goal.value.trim();
    if (goal) body.goal = goal;
    const state = els.state.value.trim();
    if (state) body.state = state;

    els.runBtn.disabled = true;
    els.empty.hidden = true;
    els.results.hidden = false;
    els.results.innerHTML = "<p class='hint'>Running…</p>";

    try {
      const res = await api.recallTrace(gid, body);
      renderResults(res);
    } catch (err) {
      els.results.innerHTML = "";
      els.empty.hidden = false;
      els.empty.textContent = `Error: ${err.message}`;
      toast(`recall: ${err.message}`, "error");
    } finally {
      els.runBtn.disabled = false;
    }
  }

  function renderResults(res) {
    els.results.innerHTML = "";
    els.results.appendChild(planSection(res));
    if (res.trace?.semantic && Object.keys(res.trace.semantic).length) {
      els.results.appendChild(semanticSection(res.trace.semantic));
    }
    if (res.trace?.procedural && Object.keys(res.trace.procedural).length) {
      els.results.appendChild(proceduralSection(res.trace.procedural));
    }
    els.results.appendChild(promptSection(res.rendered_prompt));
  }

  function section(title) {
    const s = document.createElement("div");
    s.className = "recall-section";
    const h = document.createElement("h3");
    h.textContent = title;
    s.appendChild(h);
    return s;
  }

  function planSection(res) {
    const s = section("Plan");
    const row = document.createElement("div");
    row.className = "plan-row";
    const src = res.plan?.source || {};
    row.appendChild(badge(`mode: ${res.mode}`, src.mode));
    row.appendChild(badge(`subgoal: ${truncate(res.plan?.next_subgoal || "(none)", 60)}`, src.next_subgoal));
    const tags = (res.plan?.query_tags || []).join(", ") || "(none)";
    row.appendChild(badge(`tags: ${tags}`, src.query_tags));
    s.appendChild(row);

    const meta = document.createElement("div");
    meta.className = "plan-row";
    meta.appendChild(metaBadge("selected semantics", (res.selected?.semantic_ids || []).length));
    meta.appendChild(metaBadge("selected procedurals", (res.selected?.procedural_ids || []).length));
    s.appendChild(meta);

    return s;
  }

  function badge(text, source) {
    const b = document.createElement("span");
    b.className = `badge src-${source || "default"}`;
    b.textContent = text;
    if (source) b.title = `source: ${source}`;
    return b;
  }
  function metaBadge(k, v) {
    const b = document.createElement("span");
    b.className = "badge";
    b.textContent = `${k}: ${v}`;
    return b;
  }

  function semanticSection(t) {
    const s = section("Semantic retrieval");

    if (Array.isArray(t.tag_candidates) && t.tag_candidates.length) {
      const sub = document.createElement("h3");
      sub.textContent = "Tag candidates";
      sub.style.marginTop = "8px";
      s.appendChild(sub);
      s.appendChild(table(
        ["", "query", "tag", "rel", "rec", "imp", "value"],
        t.tag_candidates,
        (r) => [
          r.selected ? "✓" : "",
          r.query_tag,
          r.tag,
          fmt(r.relevance),
          r.recency,
          fmt(r.importance),
          fmt(r.value),
        ],
        (r) => r.selected,
        { textCols: [1, 2] }
      ));
    }

    if (Array.isArray(t.semantic_topk_by_similarity) && t.semantic_topk_by_similarity.length) {
      const sub = document.createElement("h3");
      sub.textContent = `Top-${t.semantic_topk_by_similarity.length} by raw similarity (Phase 1)`;
      sub.style.marginTop = "8px";
      s.appendChild(sub);
      s.appendChild(table(
        ["#", "text", "similarity"],
        t.semantic_topk_by_similarity,
        (r) => [r.semantic_id, r.text, fmt(r.similarity)],
        () => false,
        { textCols: [1] }
      ));
    }

    if (Array.isArray(t.semantic_candidates) && t.semantic_candidates.length) {
      const sub = document.createElement("h3");
      sub.textContent = `Final candidates (k=${t.k}, threshold=${fmt(t.value_threshold)})`;
      sub.style.marginTop = "8px";
      s.appendChild(sub);
      s.appendChild(table(
        ["", "#", "text", "rel", "rec", "imp", "cred", "votes", "value"],
        t.semantic_candidates,
        (r) => [
          r.selected ? "✓" : "",
          r.semantic_id,
          r.text,
          fmt(r.relevance),
          r.recency,
          fmt(r.importance),
          r.credibility,
          r.tag_votes,
          fmt(r.value),
        ],
        (r) => r.selected,
        { textCols: [2] }
      ));
    }
    return s;
  }

  function proceduralSection(t) {
    const s = section("Procedural retrieval");

    if (t.subgoal_match) {
      const m = document.createElement("p");
      m.style.color = "var(--fg-muted)";
      m.style.fontSize = "var(--fs-small)";
      m.style.margin = "0 0 8px";
      m.textContent = `query: ${t.subgoal_query} → matched subgoal #${t.subgoal_match.subgoal_id} (${t.subgoal_match.subgoal})`;
      s.appendChild(m);
    } else {
      const m = document.createElement("p");
      m.style.color = "var(--fg-faint)";
      m.style.fontSize = "var(--fs-small)";
      m.textContent = `query: ${t.subgoal_query} → no subgoal matched.`;
      s.appendChild(m);
    }

    if (Array.isArray(t.procedural_candidates) && t.procedural_candidates.length) {
      s.appendChild(table(
        ["", "#", "text", "rel", "rec", "ret", "value"],
        t.procedural_candidates,
        (r) => [
          r.selected ? "✓" : "",
          r.procedural_id,
          r.text,
          fmt(r.relevance),
          r.recency,
          fmt(r.return),
          fmt(r.value),
        ],
        (r) => r.selected,
        { textCols: [2] }
      ));
    }
    return s;
  }

  function promptSection(messages) {
    const s = section("Rendered reasoning prompt");
    const wrap = document.createElement("div");
    wrap.className = "prompt-block";
    if (!Array.isArray(messages) || !messages.length) {
      wrap.textContent = "(no prompt)";
    } else {
      for (const m of messages) {
        const div = document.createElement("div");
        div.className = "prompt-msg";
        const role = document.createElement("div");
        role.className = "role";
        role.textContent = m.role;
        const body = document.createElement("div");
        body.textContent = m.content;
        div.appendChild(role);
        div.appendChild(body);
        wrap.appendChild(div);
      }
    }
    s.appendChild(wrap);
    return s;
  }

  function table(headers, rows, render, isSelected, opts = {}) {
    const t = document.createElement("table");
    t.className = "score-table";
    const thead = document.createElement("thead");
    const trh = document.createElement("tr");
    const textCols = new Set(opts.textCols || []);
    headers.forEach((h, i) => {
      const th = document.createElement("th");
      th.textContent = h;
      if (textCols.has(i)) th.classList.add("col-text");
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    t.appendChild(thead);
    const tbody = document.createElement("tbody");
    for (const row of rows) {
      const tr = document.createElement("tr");
      tr.classList.add(isSelected(row) ? "selected" : "unselected");
      const cells = render(row);
      cells.forEach((c, i) => {
        const td = document.createElement("td");
        if (textCols.has(i)) {
          td.classList.add("col-text");
          const span = document.createElement("span");
          span.className = "cell-text";
          span.textContent = c == null ? "" : String(c);
          span.title = span.textContent;
          td.appendChild(span);
        } else if (c === "✓") {
          td.classList.add("selected-mark");
          td.textContent = c;
        } else {
          td.textContent = c == null ? "" : String(c);
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    t.appendChild(tbody);
    return t;
  }

  function fmt(x) {
    if (typeof x !== "number") return x ?? "";
    if (!isFinite(x)) return String(x);
    return x.toFixed(3);
  }
  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  els.runBtn.addEventListener("click", run);
  els.observation.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  });

  return {
    refresh() {
      // Form persists between graph switches; clear results so stale data
      // doesn't suggest the new graph was queried.
      els.results.innerHTML = "";
      els.results.hidden = true;
      els.empty.hidden = false;
      els.empty.textContent = "Submit a query to see the trace.";
    },
  };
}
