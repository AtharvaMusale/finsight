/* FinSight UI. Vanilla JS, no dependencies.
 *
 * Safety: the answer text comes from a model reading untrusted filings, so nothing from the server
 * is ever put into the page as HTML. Everything is built with createElement and textContent. */
"use strict";
(() => {
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];

  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "style") n.style.cssText = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return n;
  }
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: fine */ } },
  };
  const fmtMs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : (+ms).toFixed(ms < 10 ? 1 : 0) + " ms");
  const pause = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ------------------------------------------------------------------ theme, scroll, reveal */
  const root = document.documentElement;
  const saved = store.get("finsight-theme");
  root.dataset.theme = saved || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  $("#theme").addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    store.set("finsight-theme", root.dataset.theme);
  });

  const progress = $("#progress");
  const onScroll = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    progress.style.setProperty("--read", max > 0 ? Math.min(1, scrollY / max) : 0);
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }, { threshold: 0.08 });
    $$(".reveal").forEach((n) => io.observe(n));
    const links = $$('.nav nav a[href^="#"]:not(.btn)');
    const spy = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
      }
    }, { rootMargin: "-40% 0px -55% 0px" });
    $$("main section[id]").forEach((s) => spy.observe(s));
  } else {
    $$(".reveal").forEach((n) => n.classList.add("in"));
  }

  /* ------------------------------------------------------------------ pipeline strip */
  const STAGES = [
    { id: "plan", sub: "rules + llm", k: "llm" },
    { id: "retrieve", sub: "pinecone", k: "store" },
    { id: "facts", sub: "guarded sql", k: "code" },
    { id: "compose", sub: "cited answer", k: "llm" },
    { id: "verify", sub: "3 layers", k: "code" },
    { id: "revise", sub: "if needed", k: "llm" },
    { id: "finalize", sub: "caveat + note", k: "code" },
  ];
  const strip = $("#strip");
  for (const s of STAGES) {
    strip.append(el("div", { class: `stage k-${s.k}`, "data-stage": s.id }, s.id, el("small", { text: s.sub })));
  }
  function setStages(map) {
    $$(".stage", strip).forEach((n) => {
      n.classList.remove("active", "done", "skipped");
      const st = map[n.dataset.stage];
      if (st) n.classList.add(st);
    });
  }
  const finalStages = (resp) => {
    const route = resp.route;
    const v = resp.verification;
    return {
      plan: "done",
      retrieve: route === "facts" ? "skipped" : "done",
      facts: route === "text" ? "skipped" : "done",
      compose: "done",
      verify: v ? "done" : "skipped",
      revise: v && v.revisions > 0 ? "done" : "skipped",
      finalize: "done",
    };
  };
  let timers = [];
  const clearTimers = () => { timers.forEach(clearTimeout); timers = []; };
  function animateWhileWaiting() {
    clearTimers();
    const frames = [
      { plan: "active" },
      { plan: "done", retrieve: "active", facts: "active" },
      { plan: "done", retrieve: "done", facts: "done", compose: "active" },
      { plan: "done", retrieve: "done", facts: "done", compose: "done", verify: "active" },
    ];
    frames.forEach((f, i) => timers.push(setTimeout(() => setStages(f), i * 1100)));
  }

  /* ------------------------------------------------------------------ state */
  const EXAMPLES = [
    { route: "text", q: "What drove growth in Azure and other cloud services at Microsoft in fiscal 2025?" },
    { route: "facts", q: "What was NVIDIA data center revenue in its latest fiscal year?" },
    { route: "both", q: "How did Apple's revenue change between fiscal 2023 and fiscal 2024, and what did the filing say about why?" },
    { route: "comparison", q: "Compare Microsoft and NVIDIA revenue growth in fiscal 2025." },
  ];
  let mode = "sample";
  let sample = null;
  let status = null;
  let busy = false;

  const form = $("#ask-form"), qEl = $("#q"), goBtn = $("#go"), resultEl = $("#result"), noteEl = $("#mode-note");

  async function loadSample() {
    if (sample) return sample;
    const r = await fetch("sample.json");
    if (!r.ok) throw new Error("sample missing");
    sample = await r.json();
    return sample;
  }

  /* ------------------------------------------------------------------ result rendering */
  function citedText(text) {
    const frag = document.createDocumentFragment();
    let last = 0;
    for (const m of text.matchAll(/\[([CFK]\d+)\]/g)) {
      frag.append(document.createTextNode(text.slice(last, m.index)));
      const label = m[1];
      frag.append(el("button", {
        class: `cite ${label[0]}`, type: "button", title: "Show the evidence for " + label,
        onclick: () => flash(label),
      }, label));
      last = m.index + m[0].length;
    }
    frag.append(document.createTextNode(text.slice(last)));
    return frag;
  }
  function flash(label) {
    const card = document.getElementById("card-" + label);
    if (!card) return;
    card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1400);
  }
  const kv = (obj) => el("dl", { class: "kv" }, Object.entries(obj).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: v == null ? "null" : String(v) })]));

  function renderResult(resp) {
    const v = resp.verification;
    const verTag = v == null ? el("span", { class: "tag" }, el("i", { class: "dot" }), "verifier off")
      : v.ok ? el("span", { class: "tag good" }, el("i", { class: "dot ok" }), "verified")
        : el("span", { class: "tag warn" }, el("i", { class: "dot warn" }), `${v.issues.length} unresolved issue${v.issues.length === 1 ? "" : "s"}`);
    const meta = el("div", { class: "meta" },
      el("span", { class: "tag" }, "route: ", el("b", { text: resp.route })),
      verTag,
      v && el("span", { class: "tag", text: `${v.claims_checked} claim${v.claims_checked === 1 ? "" : "s"} checked` }),
      v && el("span", { class: "tag", text: `${v.revisions} revision${v.revisions === 1 ? "" : "s"}` }),
      v && el("span", { class: "tag", text: v.critic_ran ? "critic ran" : "critic skipped" }),
      el("button", { class: "tag link", type: "button", title: "Open this trace", onclick: () => { document.getElementById("trace").scrollIntoView({ behavior: "smooth" }); } }, "trace ", resp.trace_id.slice(0, 8), " ↓"));

    const cards = [];
    for (const [label, s] of Object.entries(resp.sources || {})) {
      cards.push(el("div", { class: "card-box", id: "card-" + label }, el("h4", { text: label + " · filing text" }), kv({ source: s.source, accession: s.accession_no })));
    }
    for (const [label, row] of Object.entries(resp.facts || {})) {
      cards.push(el("div", { class: "card-box", id: "card-" + label }, el("h4", { text: label + " · database row" }), kv(row)));
    }
    for (const [label, c] of Object.entries(resp.calculations || {})) {
      cards.push(el("div", { class: "card-box", id: "card-" + label }, el("h4", { text: label + " · calculation" }),
        kv({ label: c.label, value: c.value, unit: c.unit, formula: c.formula, from: (c.from || []).join(", ") })));
    }
    if (v && v.issues.length) {
      cards.push(el("div", { class: "card-box" }, el("h4", { text: "verifier issues" }),
        el("ul", {}, v.issues.map((i) => el("li", { text: `${i.kind}: ${i.detail}` })))));
    }
    const warn = [...(resp.invalid_citations || []).map((c) => "invalid citation " + c), ...(resp.warnings || [])];
    if (warn.length) {
      cards.push(el("div", { class: "card-box" }, el("h4", { text: "warnings" }), el("ul", {}, warn.map((w) => el("li", { text: String(w) })))));
    }

    resultEl.replaceChildren(
      meta,
      el("p", { class: "answer" }, citedText(resp.answer || "")),
      cards.length ? el("div", { class: "cols" }, cards) : null,
      el("p", { class: "disclaimer", text: resp.disclaimer || "" }),
    );
  }

  function renderError(err) {
    const agentsDown = status ? status.agents.filter((a) => a.configured && !a.up).map((a) => a.name) : [];
    let msg, hint = "";
    if (err.network) {
      msg = "Could not reach the server.";
      hint = "Start it with: uv run uvicorn finsight.api.app:app --host 127.0.0.1 --port 8000";
    } else if (err.status === 422) {
      msg = "The question must be between 3 and 500 characters.";
    } else if (err.status === 504) {
      msg = "The request timed out.";
    } else if (err.status === 502) {
      msg = "The server could not finish this question.";
      hint = agentsDown.length
        ? `A2A agent${agentsDown.length > 1 ? "s" : ""} not running: ${agentsDown.join(", ")}. Start the workers (Ways to use, A2A agents), or remove the FINSIGHT_A2A_*_URL lines from .env.`
        : "Check the keys in .env and that the filings were ingested. The server log has the details.";
    } else {
      msg = `The server answered with an error (${err.status}).`;
    }
    resultEl.replaceChildren(el("div", { class: "error", role: "alert" }, msg,
      hint && el("small", { text: hint }), err.detail && el("small", { text: String(err.detail) })));
  }

  /* ------------------------------------------------------------------ asking */
  async function liveAsk(question) {
    let r;
    try {
      r = await fetch("/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question }) });
    } catch {
      throw { network: true };
    }
    if (!r.ok) {
      let detail = "";
      try { const j = await r.json(); detail = typeof j.detail === "string" ? j.detail : ""; } catch { /* no body */ }
      throw { status: r.status, detail };
    }
    return r.json();
  }

  async function run(question) {
    if (busy) return;
    busy = true;
    goBtn.disabled = true;
    goBtn.textContent = "Running…";
    resultEl.replaceChildren(el("p", { class: "empty", text: "Running the pipeline…" }));
    try {
      let resp;
      if (mode === "sample") {
        const s = await loadSample();
        const frames = [
          { plan: "active" },
          { plan: "done", retrieve: "active" },
          { plan: "done", retrieve: "done", compose: "active" },
          { plan: "done", retrieve: "done", compose: "done", verify: "active" },
          { plan: "done", retrieve: "done", compose: "done", verify: "done", revise: "active" },
          { plan: "done", retrieve: "done", compose: "done", verify: "active", revise: "done" },
        ];
        for (const f of frames) { setStages(f); await pause(420); }
        resp = s.response;
        setStages(finalStages(resp));
        renderResult(resp);
        showTrace(s.trace, "Sample run");
      } else {
        animateWhileWaiting();
        resp = await liveAsk(question);
        clearTimers();
        setStages(finalStages(resp));
        renderResult(resp);
        loadRecent(false);
        openTraceById(resp.trace_id, false);
      }
    } catch (err) {
      clearTimers();
      setStages({});
      renderError(err && (err.status || err.network) ? err : { status: 0, detail: "" });
    } finally {
      busy = false;
      goBtn.disabled = false;
      goBtn.textContent = "Ask";
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = (mode === "sample" ? EXAMPLES[0].q : qEl.value).trim();
    if (q.length < 3) { qEl.focus(); return; }
    run(q);
  });
  qEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit(); });

  const chipsEl = $("#chips");
  function renderChips() {
    chipsEl.replaceChildren(...EXAMPLES.map((ex, i) => {
      const locked = mode === "sample" && i > 0;
      return el("button", {
        type: "button", class: "tag chip", "aria-disabled": locked ? "true" : null,
        style: locked ? "opacity:.45" : null, title: locked ? "Switch to Live mode to ask this one" : "Use this question",
        onclick: () => {
          if (locked) { noteEl.textContent = "Sample mode replays one recorded run. Switch to Live to ask a different question."; return; }
          qEl.value = ex.q; qEl.focus();
        },
      }, el("b", { text: ex.route }), " · ", ex.q.length > 46 ? ex.q.slice(0, 44) + "…" : ex.q);
    }));
  }

  function setMode(next) {
    mode = next;
    $("#mode-sample").setAttribute("aria-pressed", String(next === "sample"));
    $("#mode-live").setAttribute("aria-pressed", String(next === "live"));
    noteEl.className = "note";
    if (next === "sample") {
      qEl.value = EXAMPLES[0].q;
      qEl.readOnly = true;
      noteEl.textContent = "Sample mode replays a real recorded run (from the README) with no network and no cost.";
    } else {
      qEl.readOnly = false;
      if (qEl.value === EXAMPLES[0].q) qEl.value = "";
      noteEl.classList.add("warn");
      const missing = status ? [
        !status.keys.anthropic && "ANTHROPIC_API_KEY", !status.keys.pinecone && "PINECONE_API_KEY", !status.facts_db && "ingested filings",
      ].filter(Boolean) : [];
      noteEl.textContent = "Live mode runs the real pipeline: each question makes model and search calls, which cost a few cents."
        + (missing.length ? ` This server reports missing: ${missing.join(", ")}.` : "");
    }
    renderChips();
  }
  $("#mode-sample").addEventListener("click", () => setMode("sample"));
  $("#mode-live").addEventListener("click", () => setMode("live"));

  /* ------------------------------------------------------------------ trace viewer */
  const traceMain = $("#trace-main"), tlist = $("#tlist"), tnote = $("#tnote");
  const markCurrent = (id) => $$(".titem", tlist).forEach((b) => {
    if (b.dataset.id === id) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
  });
  const kindOf = (name) => (name === "llm" || /compose|plan|revise/.test(name) ? "var(--llm)"
    : /search|retrieve|pinecone/.test(name) ? "var(--store)" : "var(--code)");

  function showTrace(trace, title) {
    const spans = [...trace.spans].sort((a, b) => (a.ts || 0) - (b.ts || 0));
    const byId = new Map(spans.map((s) => [s.span, s]));
    const kids = new Map();
    for (const s of spans) {
      const p = s.parent && byId.has(s.parent) ? s.parent : null;
      if (!kids.has(p)) kids.set(p, []);
      kids.get(p).push(s);
    }
    const ordered = [];
    const walk = (parent, depth) => (kids.get(parent) || []).forEach((s) => { ordered.push({ s, depth }); walk(s.span, depth + 1); });
    walk(null, 0);

    const t0 = Math.min(...spans.map((s) => s.ts * 1000));
    const end = Math.max(...spans.map((s) => s.ts * 1000 + (s.ms || 0)));
    const total = Math.max(end - t0, 0.001);
    const services = [...new Set(spans.map((s) => s.service))];
    const detail = el("div", { class: "detail", hidden: true });

    const rows = ordered.map(({ s, depth }) => {
      const left = ((s.ts * 1000 - t0) / total) * 100;
      const width = Math.max(((s.ms || 0) / total) * 100, 0.8);
      const row = el("button", { class: "wf-row" + (s.ok === false ? " err" : ""), type: "button", style: `--k:${kindOf(s.name)}`,
        onclick: () => {
          $$(".wf-row", traceMain).forEach((r) => r.removeAttribute("aria-current"));
          row.setAttribute("aria-current", "true");
          detail.hidden = false;
          detail.replaceChildren(el("div", { class: "meta", style: "margin-bottom:10px" },
            el("span", { class: "tag", text: s.name }), el("span", { class: "tag", text: s.service }),
            el("span", { class: "tag", text: fmtMs(s.ms || 0) }),
            s.ok === false ? el("span", { class: "tag bad", text: "error: " + s.error }) : el("span", { class: "tag good", text: "ok" })),
            Object.keys(s.attrs || {}).length ? kv(s.attrs) : el("p", { class: "note", text: "No attributes." }));
        } },
        el("span", { class: "wf-name", style: `padding-left:${depth * 16}px` }, el("i"), s.name),
        el("span", { class: "wf-track" }, el("span", { class: "wf-bar", style: `left:${left}%;width:${Math.min(width, 100 - left)}%` })),
        el("span", { class: "wf-ms", text: fmtMs(s.ms || 0) }));
      return row;
    });

    traceMain.replaceChildren(
      el("div", { class: "trace-top" },
        el("span", { text: `${title || "trace"} · ${trace.trace_id}` }),
        el("span", { class: "meta" }, el("span", { class: "tag", text: `${spans.length} spans` }), el("span", { class: "tag", text: `wall ${fmtMs(total)}` }),
          services.map((x) => el("span", { class: "tag", text: x })),
          spans.some((s) => s.ok === false) ? el("span", { class: "tag bad", text: "errors" }) : el("span", { class: "tag good", text: "errors 0" }))),
      el("div", { class: "wf", role: "list" }, rows),
      detail,
    );
    $("#tid").value = trace.trace_id;
  }

  async function openTraceById(id, announce = true) {
    tnote.textContent = "";
    try {
      const r = await fetch("/traces/" + encodeURIComponent(id));
      if (!r.ok) {
        tnote.textContent = r.status === 404 ? "No spans found for that ID (tracing may be off, or the trace is older than three days)."
          : r.status === 422 ? "That does not look like a trace ID." : `Server error (${r.status}).`;
        return;
      }
      showTrace(await r.json(), "trace");
      markCurrent(id);
      if (announce) traceMain.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } catch {
      tnote.textContent = "Could not reach the server.";
    }
  }
  $("#tid-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const id = $("#tid").value.trim();
    if (id) openTraceById(id);
  });

  async function loadRecent() {
    const items = [el("button", { class: "titem", type: "button", "data-id": "sample", "aria-current": "true",
      onclick: async () => { const s = await loadSample(); showTrace(s.trace, "Sample run"); markCurrent("sample"); } },
      el("b", { text: "Sample run" }), "recorded · text · 1 revision")];
    try {
      const r = await fetch("/traces");
      if (r.ok) {
        const j = await r.json();
        if (!j.tracing) tnote.textContent = "Tracing is off on this server (FINSIGHT_TRACING=false).";
        for (const t of j.traces) {
          items.push(el("button", { class: "titem", type: "button", "data-id": t.trace_id, onclick: () => openTraceById(t.trace_id) },
            el("b", { text: `${t.route || "?"} · ${t.ms != null ? fmtMs(t.ms) : "?"}${t.verified === false ? " · unverified" : ""}` }),
            `${t.ts ? new Date(t.ts * 1000).toLocaleTimeString() : ""} · ${t.trace_id.slice(0, 8)}`));
        }
        if (j.tracing && !j.traces.length) tnote.textContent = "No live questions traced yet. Ask one in Live mode.";
      }
    } catch { /* server unreachable: the sample entry still works */ }
    tlist.replaceChildren(...items);
  }

  /* ------------------------------------------------------------------ status */
  const sysEl = $("#sys");
  function sysCard(title, ...rows) { return el("div", {}, el("h3", { text: title }), ...rows); }
  const row = (dot, label, value) => el("div", { class: "row" }, el("i", { class: "dot " + dot }), label && el("span", { text: label }), el("b", { text: value }));

  function renderStatus(s) {
    status = s;
    const keysOk = s.keys.anthropic && s.keys.pinecone && s.facts_db;
    const downs = s.agents.filter((a) => a.configured && !a.up);
    $("#badge-dot").className = "dot " + (downs.length ? "bad" : keysOk ? "ok" : "warn");
    $("#badge-text").textContent = downs.length ? `Server up · A2A agent down: ${downs.map((a) => a.name).join(", ")}`
      : keysOk ? `Server ready · ${s.mode} · ${s.model.split("/").pop()}`
        : "Server up · setup incomplete (see System)";

    sysEl.replaceChildren(
      sysCard("Model and pipeline", row("ok", "model", s.model.split("/").pop()), row(s.verify ? "ok" : "warn", "verifier", s.verify ? `on · up to ${s.max_revisions} revision${s.max_revisions === 1 ? "" : "s"}` : "off")),
      sysCard("Credentials", row(s.keys.anthropic ? "ok" : "bad", "Anthropic key", s.keys.anthropic ? "present" : "missing"),
        row(s.keys.pinecone ? "ok" : "bad", "Pinecone key", s.keys.pinecone ? "present" : "missing"),
        row(s.langsmith ? "ok" : "", "LangSmith export", s.langsmith ? "on" : "off")),
      sysCard("Data and tracing", row(s.facts_db ? "ok" : "bad", "filings database", s.facts_db ? "found" : "not found, run ingestion"), row(s.tracing ? "ok" : "warn", "local tracing", s.tracing ? "on" : "off")),
      sysCard("A2A agents · " + s.mode, ...s.agents.map((a) => row(!a.configured ? "" : a.up ? "ok" : "bad", a.name, !a.configured ? "in-process" : a.up ? "running" : "not running"))),
    );
    $("#a2a-live").replaceChildren(...s.agents.map((a) => el("li", {}, el("code", { text: a.name }), ": ",
      !a.configured ? "in-process (no URL set)" : a.up ? "running at " + a.url : "configured at " + a.url + " but not running")));
    if (mode === "live") setMode("live");
  }

  async function loadStatus() {
    try {
      const r = await fetch("/status");
      if (!r.ok) throw new Error(String(r.status));
      renderStatus(await r.json());
    } catch {
      $("#badge-dot").className = "dot bad";
      $("#badge-text").textContent = "Server unreachable. Sample mode still works.";
      sysEl.replaceChildren(sysCard("Unreachable", row("bad", "", "Could not read /status"), el("p", { class: "note", text: "Start the server with: uv run uvicorn finsight.api.app:app --host 127.0.0.1 --port 8000" })));
    }
  }
  $("#refresh").addEventListener("click", loadStatus);

  /* ------------------------------------------------------------------ tabs and copy buttons */
  const tabs = $$('[role="tab"]');
  function selectTab(tab, focus) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((t, i) => {
    t.tabIndex = i === 0 ? 0 : -1;
    t.addEventListener("click", () => selectTab(t, false));
    t.addEventListener("keydown", (e) => {
      const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (to == null) return;
      e.preventDefault();
      selectTab(tabs[(to + tabs.length) % tabs.length], true);
    });
  });

  $$(".code").forEach((box) => {
    const pre = $("pre", box), cap = $(".cap", box);
    if (!pre || !cap) return;
    const btn = el("button", { class: "copy", type: "button" }, "Copy");
    btn.addEventListener("click", async () => {
      const text = pre.textContent;
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        const ta = el("textarea", { style: "position:fixed;opacity:0" }); ta.value = text;
        document.body.append(ta); ta.select();
        try { document.execCommand("copy"); } catch { /* nothing more to try */ }
        ta.remove();
      }
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = "Copy"; }, 1400);
    });
    cap.append(btn);
  });

  /* ------------------------------------------------------------------ boot */
  setMode("sample");
  loadRecent();
  loadStatus();
  loadSample().then((s) => showTrace(s.trace, "Sample run")).catch(() => {
    traceMain.replaceChildren(el("p", { class: "empty", text: "Sample data could not be loaded." }));
  });
})();
