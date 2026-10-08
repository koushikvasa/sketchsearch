"use strict";
// Safety sweep (the agent), view switching and demo mode. Loaded after app.js and case.js and shares their
// helpers ($, api, icon, whereLabel, toggleCase, ...).

const agent = {
  running: false, markdown: "", abort: null, bg: null, timer: null, started: 0,
  goal: "", summary: "", titles: {}, planned: 0, searches: 0, checked: 0, live: [], lastReport: null, now: "",
};
const VERDICT_ICON = { YES: "check", NO: "x", UNSURE: "circle-help", SKIPPED: "skip-forward" };
const FALLBACK_GOAL = "Find risky moments between people and the forklift.";
const SWEEPS = [
  { id: "forklift", icon: "forklift", title: "People near the forklift", sub: "Workers walking close to a forklift", goal: FALLBACK_GOAL },
  { id: "robot", icon: "bot", title: "Workers in a robot's path", sub: "People walking toward or across a mobile robot",
    goal: "Find moments where a worker walks into the path of a mobile robot." },
  { id: "alone", icon: "user", title: "Working alone", sub: "One person in an aisle with nobody nearby",
    goal: "Find moments where a worker is alone in an aisle with nobody nearby." },
  { id: "crowd", icon: "users", title: "Crowding in aisles", sub: "Three or more people bunched together",
    goal: "Find moments where three or more people crowd together in an aisle." },
];

function demoMode() {
  return $("#demoMode").checked;
}

// ---------------------------------------------------------------------------
// Views + demo toggle
// ---------------------------------------------------------------------------
function showView(view) {
  $("#searchView").hidden = view !== "search";
  $("#agentView").hidden = view !== "agent";
  document.body.classList.toggle("view-sweep", view === "agent");
  for (const t of document.querySelectorAll(".tab")) {
    const on = t.dataset.view === view;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", String(on));
  }
  if (view !== "search") {
    const current = cards.find((c) => c.featured);
    if (current) pauseCard(current);
  }
  if (view === "search") window.dispatchEvent(new Event("resize")); // canvas may have been hidden
}

function initViews() {
  for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => showView(t.dataset.view));
  $("#sweepCta").addEventListener("click", () => showView("agent"));
  let saved = false;
  try { saved = localStorage.getItem("sketchsearch.demo") === "1"; } catch { /* storage blocked */ }
  setDemo(saved); // app.js: the presenter menu owns the toggle, the badge and saving it
}

// ---------------------------------------------------------------------------
// Mini sketch canvases
// ---------------------------------------------------------------------------
function drawMini(cv, sketch) {
  const dpr = window.devicePixelRatio || 1;
  const w = 192, h = 108;
  cv.width = w * dpr;
  cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = TOKENS.bg;
  ctx.fillRect(0, 0, w, h);
  if (agent.bg?.complete && agent.bg.naturalWidth) {
    ctx.globalAlpha = 0.35;
    ctx.drawImage(agent.bg, 0, 0, w, h);
    ctx.globalAlpha = 1;
  }
  for (const o of sketch.objects) {
    const b = o.start_box;
    const color = o.absent ? ABSENT_COLOR : COLORS[o.label] || "#fff";
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    if (o.absent) {
      ctx.setLineDash([4, 3]);
      ctx.fillStyle = hexA(ABSENT_COLOR, 0.15);
      ctx.fillRect(b.x * w, b.y * h, b.w * w, b.h * h);
    }
    ctx.strokeRect(b.x * w, b.y * h, b.w * w, b.h * h);
    ctx.restore();
    const pts = o.path && o.path.length >= 2 ? o.path : o.end_box ? [center(o.start_box), center(o.end_box)] : null;
    if (!o.absent && pts && pathLength(pts) > 0.01) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pts[0][0] * w, pts[0][1] * h);
      for (const [x, y] of pts.slice(1)) ctx.lineTo(x * w, y * h);
      ctx.stroke();
      const [ax, ay] = pts[pts.length - 2], [bx, by] = pts[pts.length - 1];
      const ang = Math.atan2((by - ay) * h, (bx - ax) * w);
      ctx.beginPath();
      ctx.moveTo(bx * w, by * h);
      ctx.lineTo(bx * w - 7 * Math.cos(ang - 0.45), by * h - 7 * Math.sin(ang - 0.45));
      ctx.lineTo(bx * w - 7 * Math.cos(ang + 0.45), by * h - 7 * Math.sin(ang + 0.45));
      ctx.fill();
      ctx.restore();
    }
  }
}

function miniFigure(sketch, caption) {
  const fig = document.createElement("div");
  fig.className = "mini";
  const cv = document.createElement("canvas");
  drawMini(cv, sketch);
  fig.appendChild(cv);
  if (caption) {
    const span = document.createElement("span");
    span.textContent = caption;
    fig.appendChild(span);
  }
  return fig;
}

// ---------------------------------------------------------------------------
// Trace timeline
// ---------------------------------------------------------------------------
function traceItem(kind, html) {
  const li = document.createElement("li");
  li.className = `k-${kind}`;
  li.innerHTML = html;
  $("#trace").appendChild(li);
  return li;
}

function thumbHtml(frameUrl, clipUrl) {
  if (frameUrl) return `<img src="${escapeHtml(frameUrl)}" alt="" loading="lazy">`;
  if (clipUrl) return `<video src="${escapeHtml(clipUrl)}#t=1" muted preload="metadata"></video>`;
  return "";
}

function segLabel(segmentId) {
  const m = /^(.*)_(\d+)$/.exec(segmentId);
  return m ? `${shortCamera(m[1])} · ${clock(Number(m[2]))}` : segmentId;
}

function renderEvent(ev) {
  switch (ev.event) {
    case "start":
      traceItem("start", `<div class="k-title">Goal: ${escapeHtml(ev.goal)}</div>
        <div class="k-sub">Planned by ${escapeHtml(ev.llm)} · at most ${ev.budget} AI video checks${ev.replayed ? " · saved run, replayed" : ""}</div>`);
      break;
    case "plan": {
      const li = traceItem("plan", `<div class="k-title">Plan: ${ev.sketches.length} searches</div>
        <div class="k-sub">${escapeHtml(ev.summary || "")}</div><div class="mini-row"></div>`);
      const row = $(".mini-row", li);
      for (const s of ev.sketches) row.appendChild(miniFigure(s.sketch, s.title));
      break;
    }
    case "sketch": {
      const li = traceItem("sketch", `<div class="k-title">${ev.refined ? "Improved search" : "Search"}: ${escapeHtml(ev.title)}</div>
        <div class="k-sub">${escapeHtml(ev.why || "")}</div><div class="mini-row"></div>`);
      $(".mini-row", li).appendChild(miniFigure(ev.sketch));
      break;
    }
    case "results": {
      const thumbs = ev.results.slice(0, 3).map((r) =>
        `<figure>${thumbHtml(r.frame_url, r.clip_url)}<figcaption>${escapeHtml(segLabel(r.segment_id))} · ${r.score.toFixed(2)}</figcaption></figure>`).join("");
      traceItem("results", `<div class="k-sub">${ev.count} matches in ${Math.round(ev.elapsed_ms)} ms. The AI checks the best 3.</div><div class="thumbs">${thumbs}</div>`);
      break;
    }
    case "verify": {
      const cls = `v-${ev.verdict.toLowerCase()}`;
      const img = thumbHtml(ev.frame_url, ev.clip_url);
      traceItem("verify", `<div class="k-verify-row">${img}<span class="${cls} verdict-tag">${icon(VERDICT_ICON[ev.verdict] || "dot")}${VERDICT_UI[ev.verdict]?.text || "Skipped"}</span>
        <span><strong>${escapeHtml(segLabel(ev.segment_id))}</strong>: ${escapeHtml(ev.verdict === "SKIPPED" ? "Not checked: this sweep had used all its AI checks." : ev.reason)}${ev.cached ? ' <span class="muted">(saved answer)</span>' : ""}</span></div>`);
      break;
    }
    case "refine": {
      if (!ev.ok) {
        traceItem("refine", `<div class="k-title">Could not improve “${escapeHtml(agent.titles[ev.id] || ev.id)}”</div><div class="k-sub">${escapeHtml(ev.reason)}</div>`);
        break;
      }
      const li = traceItem("refine", `<div class="k-title">Only ${Math.round(ev.confirm_rate * 100)}% confirmed, so the agent rewrote it as “${escapeHtml(ev.title)}”</div>
        <div class="k-sub">${escapeHtml(ev.why || "")}</div><div class="mini-row"></div>`);
      $(".mini-row", li).appendChild(miniFigure(ev.sketch));
      break;
    }
    case "report":
      agent.markdown = ev.markdown;
      $("#report").innerHTML = renderMarkdown(ev.markdown);
      $("#report").hidden = false;
      $("#copyReport").disabled = false;
      traceItem("report", `<div class="k-title">Report: ${ev.stats.confirmed} confirmed of ${ev.stats.clips_checked} clips checked</div>
        <div class="k-sub">${ev.stats.new_video_calls} AI video checks used${ev.stats.skipped_for_budget ? `, ${ev.stats.skipped_for_budget} clip${ev.stats.skipped_for_budget === 1 ? "" : "s"} left unchecked` : ""}</div>`);
      break;
    case "done":
      traceItem("done", `<div class="k-sub">Done in ${ev.seconds} s${ev.replayed ? " (saved run, replayed)" : ""}.</div>`);
      break;
    case "error":
      traceItem("error", `<div class="k-title">The sweep stopped</div><div class="k-sub">${escapeHtml(ev.message)}</div>`);
      break;
    default:
      break;
  }
}

// ---------------------------------------------------------------------------
// Markdown (just what the report uses: headings, tables, images, links, bold/italic, lists)
// ---------------------------------------------------------------------------
function safeUrl(url) {
  return /^\s*(javascript|data|vbscript):/i.test(url) ? "#" : url;
}

function mdInline(text) {
  let s = escapeHtml(text);
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, src) => `<img alt="${alt}" src="${safeUrl(src)}" loading="lazy">`);
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, href) => `<a href="${safeUrl(href)}" target="_blank" rel="noopener">${t}</a>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  return s;
}

function renderMarkdown(md) {
  const lines = md.split("\n");
  const out = [];
  let para = [], list = [], table = [];
  const flush = () => {
    if (para.length) out.push(`<p>${para.join("")}</p>`);
    if (list.length) out.push(`<ul>${list.map((l) => `<li>${l}</li>`).join("")}</ul>`);
    if (table.length) {
      const rows = table.filter((r) => !/^\|[\s|:-]+\|$/.test(r));
      const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => mdInline(c.trim()));
      const [head, ...body] = rows;
      out.push(`<table><thead><tr>${cells(head).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
        body.map((r) => `<tr>${cells(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody></table>");
    }
    para = []; list = []; table = [];
  };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    const heading = /^(#{1,3}) (.*)$/.exec(line);
    if (heading) { flush(); out.push(`<h${heading[1].length}>${mdInline(heading[2])}</h${heading[1].length}>`); continue; }
    if (line.startsWith("|")) { if (para.length || list.length) flush(); table.push(line); continue; }
    if (line.startsWith("- ")) { if (para.length || table.length) flush(); list.push(mdInline(line.slice(2))); continue; }
    if (!line) { flush(); continue; }
    if (table.length || list.length) flush();
    para.push(mdInline(line) + (raw.endsWith("  ") ? "<br>" : " "));
  }
  flush();
  return out.join("\n");
}

// ---------------------------------------------------------------------------
// Sweep screen: choices, progress stages, findings
// ---------------------------------------------------------------------------
function renderSweeps(recorded = []) {
  $("#sweepOptions").innerHTML = SWEEPS.map((s) => `
    <button type="button" class="sweep-option" role="radio" aria-checked="false" data-id="${s.id}">
      <span class="sweep-icon">${icon(s.icon)}</span>
      <span class="sweep-text"><b>${escapeHtml(s.title)}</b><small>${escapeHtml(s.sub)}</small></span>
      ${recorded.includes(s.goal) ? `<span class="saved-tag" title="Demo mode replays a saved run of this sweep">Saved run</span>` : ""}
    </button>`).join("");
  selectSweep(SWEEPS[0].id);
}

function selectSweep(id) {
  for (const b of document.querySelectorAll(".sweep-option")) b.setAttribute("aria-checked", String(b.dataset.id === id));
  agent.selected = id;
}

function chosenGoal() {
  const typed = $("#agentGoal").value.trim();
  if (typed) return typed;
  return SWEEPS.find((s) => s.id === agent.selected)?.goal || "";
}

function setStage(name, status, note = null) {
  const li = $(`#stages li[data-stage="${name}"]`);
  li.className = status;
  const small = $("small", li);
  if (!small.dataset.default) small.dataset.default = small.textContent;
  small.textContent = note ?? small.dataset.default;
}

function resetStages() {
  for (const li of document.querySelectorAll("#stages li")) setStage(li.dataset.stage, "");
}

function setNow(text) {
  agent.now = text;
  const secs = agent.running ? ` · ${((performance.now() - agent.started) / 1000).toFixed(0)} s` : "";
  $("#agentStatus").textContent = text ? text + secs : "";
}

function hitRank(h) {
  return agent.lastReport ? agent.lastReport.confirmed.indexOf(h) + 1 : 0;
}

/** A confirmed moment. While the sweep runs only the verify event is known (segment + reason); the report then
 *  brings the full clip details, the "Add to case" button and the "Open" link. */
function findingHtml(h, full) {
  const where = full ? whereLabel(h) : segLabel(h.segment_id);
  const rank = full ? hitRank(h) : 0;
  return `<article class="finding" data-seg="${escapeHtml(h.segment_id)}">
    <div class="finding-media">${h.frame_url ? `<img src="${escapeHtml(h.frame_url)}" alt="" loading="lazy">` : ""}
      <video muted loop playsinline preload="none"></video>
      <span class="stamp yes">${icon("check")}Confirmed</span>
      ${rank ? `<span class="rank">${rank}</span>` : ""}
    </div>
    <div class="finding-body">
      <div class="finding-where">${escapeHtml(where)}</div>
      <p class="finding-reason" title="${escapeHtml(h.reason)}">${escapeHtml(h.reason)}</p>
      ${h.sketch ? `<p class="finding-from">Found by: ${escapeHtml(h.sketch)}</p>` : ""}
      ${full ? `<div class="finding-actions">
        <button class="case-add" data-key="${escapeHtml(caseKey(h))}"></button>
        <button class="ghost-btn open-search" title="Open this moment in Find a moment to look for similar ones">${icon("search")}Open</button>
      </div>` : ""}
    </div></article>`;
}

function addLiveFinding(ev) {
  if (agent.live.some((h) => h.segment_id === ev.segment_id)) return;
  const h = { segment_id: ev.segment_id, reason: ev.reason, frame_url: ev.frame_url, clip_url: ev.clip_url, window: [0, 0],
              sketch: agent.titles[ev.id] || "" };
  agent.live.push(h);
  const grid = $("#findingGrid");
  if (agent.live.length === 1) grid.innerHTML = "";
  grid.insertAdjacentHTML("beforeend", findingHtml(h, false));
  $("#findingsTitle").textContent = `${agent.live.length} confirmed so far`;
}

function renderFindings(rep) {
  const grid = $("#findingGrid");
  const n = rep.confirmed.length;
  $("#findingsTitle").textContent = n ? `${n} confirmed moment${n === 1 ? "" : "s"}` : "No confirmed moments";
  $("#findingsSub").textContent = `${rep.stats.clips_checked} clips checked by the video AI · ${rep.stats.sketch_runs} searches · ${Math.round(rep.stats.seconds)} s`;
  grid.innerHTML = n ? rep.confirmed.map((h) => findingHtml(h, true)).join("")
    : `<div class="state sweep-empty">${icon("shield-check")}<p><b>Nothing confirmed.</b><br>The video AI rejected every clip it checked.
      The list below shows what it looked at and why.</p></div>`;
  const rej = $("#rejected");
  rej.hidden = !rep.rejected.length;
  $("summary", rej).textContent = `Not confirmed (${rep.rejected.length})`;
  $("#rejectedList").innerHTML = rep.rejected.map((h) => {
    const ui = VERDICT_UI[h.verdict];
    return `<li>${h.frame_url ? `<img src="${escapeHtml(h.frame_url)}" alt="" loading="lazy">` : ""}
      <span><b>${escapeHtml(whereLabel(h))}</b> <span class="verdict-tag ${ui?.cls || ""}">${ui ? icon(ui.icon) + ui.text : "Skipped"}</span><br>
      <span class="muted">${escapeHtml(h.reason)}</span></span></li>`;
  }).join("");
  $("#caseAll").disabled = !n;
  $("#sweepReport").disabled = !(n || rep.rejected.length);
  syncCaseButtons();
}

function hitBySeg(seg) {
  return agent.lastReport?.confirmed.find((h) => h.segment_id === seg) || agent.live.find((h) => h.segment_id === seg);
}

/** The last sweep's clips, for the report: confirmed first. */
function sweepItems() {
  const rep = agent.lastReport;
  if (!rep) return [];
  return [...rep.confirmed, ...rep.rejected].map((h) => ({ r: h, v: { verdict: h.verdict, reason: h.reason }, from: h.sketch }));
}

function openInSearch(h) {
  showView("search");
  enterStudio();
  moreLikeThis(h);
}

/** Progress and findings from the agent's events (the raw log goes to the drawer via renderEvent). */
function sweepEvent(ev) {
  switch (ev.event) {
    case "start":
      setStage("plan", "active", "Working out what to search for…");
      setNow("Planning the searches");
      break;
    case "plan":
      agent.summary = ev.summary || "";
      agent.planned = ev.sketches.length;
      for (const s of ev.sketches) agent.titles[s.id] = s.title;
      setStage("plan", "done", `${ev.sketches.length} searches planned`);
      setStage("search", "active", `0 of ${ev.sketches.length} run`);
      break;
    case "sketch":
      agent.titles[ev.id] = ev.title;
      agent.searches += 1;
      setStage("search", "active", `${agent.searches} run${ev.refined ? " (one improved)" : ""}`);
      setNow(`${ev.refined ? "Trying a better version" : "Searching"}: “${ev.title}”`);
      break;
    case "results":
      setNow(`“${agent.titles[ev.id] || ev.id}”: ${ev.count} matches, the AI is checking the best 3`);
      break;
    case "verify":
      if (ev.verdict !== "SKIPPED") agent.checked += 1;
      setStage("check", "active", `${agent.checked} clips checked, ${agent.live.length + (ev.verdict === "YES" ? 1 : 0)} confirmed`);
      if (ev.verdict === "YES") addLiveFinding(ev);
      break;
    case "refine":
      if (ev.ok) setNow(`Only ${Math.round(ev.confirm_rate * 100)}% confirmed, so the agent is rewriting “${agent.titles[ev.id] || ev.id}”`);
      break;
    case "report":
      agent.lastReport = ev;
      setStage("search", "done", `${ev.stats.sketch_runs} searches run`);
      setStage("check", "done", `${ev.stats.clips_checked} clips checked, ${ev.stats.confirmed} confirmed`);
      setStage("report", "done", "Ready below");
      renderFindings(ev);
      break;
    case "done":
      agent.running = false;
      setNow(`Finished in ${ev.seconds} s${ev.replayed ? " (saved run, replayed)" : ""}.`);
      break;
    case "error":
      for (const li of document.querySelectorAll("#stages li.active")) li.className = "error";
      setNow(`The sweep stopped: ${ev.message}`);
      break;
    default:
      break;
  }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------
function setRunButton(running) {
  const btn = $("#agentRun");
  btn.classList.toggle("stop", running);
  btn.innerHTML = running ? `${icon("square")}Stop` : `${icon("play")}Start sweep`;
}

async function runAgent(e) {
  e?.preventDefault();
  if (agent.running && agent.abort) { // the button reads "Stop" while a sweep runs
    agent.abort.abort();
    return;
  }
  const goal = chosenGoal();
  if (goal.length < 3) return toast("Pick a sweep, or describe what to look for.");
  const controller = new AbortController();
  Object.assign(agent, { abort: controller, running: true, goal, summary: "", titles: {}, planned: 0, searches: 0,
                         checked: 0, live: [], lastReport: null, markdown: "" });
  setRunButton(true);
  resetStages();
  $("#trace").innerHTML = "";
  $("#report").hidden = true;
  $("#copyReport").disabled = true;
  $("#traceBtn").disabled = false;
  $("#caseAll").disabled = true;
  $("#sweepReport").disabled = true;
  $("#rejected").hidden = true;
  $("#findingsTitle").textContent = "Findings";
  $("#findingsSub").textContent = `Goal: ${goal}`;
  $("#findingGrid").innerHTML = `<div class="state sweep-empty"><div class="spinner"></div>
    <p>Confirmed moments appear here as soon as the video AI confirms them.</p></div>`;
  agent.started = performance.now();
  clearInterval(agent.timer);
  agent.timer = setInterval(() => setNow(agent.now), 1000);
  let gotReport = false;
  try {
    const res = await fetch("api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal, demo: demoMode() }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        const data = /^data: (.*)$/m.exec(chunk)?.[1];
        if (!data) continue;
        const ev = JSON.parse(data);
        if (ev.event === "report") gotReport = true;
        renderEvent(ev);
        sweepEvent(ev);
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      for (const li of document.querySelectorAll("#stages li.active")) li.className = "error";
      agent.running = false;
      setNow("Stopped. Anything already confirmed stays below.");
    } else {
      renderEvent({ event: "error", message: err.message });
      sweepEvent({ event: "error", message: err.message });
    }
  } finally {
    clearInterval(agent.timer);
    if (agent.abort === controller) {
      agent.abort = null;
      agent.running = false;
      setRunButton(false);
      if (!gotReport && !agent.live.length) {
        $("#findingGrid").innerHTML = `<div class="state sweep-empty">${icon("circle-slash")}<p>No findings: the sweep did not finish.</p></div>`;
      }
    }
  }
}

async function initAgent() {
  initViews();
  renderSweeps();
  $("#agentForm").addEventListener("submit", runAgent);
  $("#sweepOptions").addEventListener("click", (e) => {
    const b = e.target.closest(".sweep-option");
    if (!b) return;
    selectSweep(b.dataset.id);
    $("#agentGoal").value = "";
  });
  $("#agentGoal").addEventListener("input", (e) => selectSweep(e.target.value.trim() ? null : SWEEPS[0].id));
  $("#traceBtn").addEventListener("click", () => $("#traceDrawer").showModal());
  $("#sweepReport").addEventListener("click", () => openReport("sweep"));
  $("#caseAll").addEventListener("click", () => {
    for (const h of agent.lastReport?.confirmed || []) addToCase(h, h, `Safety sweep: ${h.sketch}`);
  });
  const grid = $("#findingGrid");
  grid.addEventListener("click", (e) => {
    const card = e.target.closest(".finding");
    const h = card && hitBySeg(card.dataset.seg);
    if (!h) return;
    if (e.target.closest(".case-add")) toggleCase(h, h, `Safety sweep: ${h.sketch}`);
    else if (e.target.closest(".open-search")) openInSearch(h);
  });
  bindPreview(grid, (m) => hitBySeg(m.closest(".finding")?.dataset.seg));
  $("#copyReport").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(agent.markdown);
      toast("Report copied as Markdown.");
    } catch {
      toast("Couldn't copy. Your browser blocked the clipboard.", true);
    }
  });
  try {
    const info = await api("api/demo");
    SWEEPS[0].goal = info.preset_goal || FALLBACK_GOAL;
    renderSweeps(info.recordings || []);
  } catch { /* the built-in sweeps still work live */ }
  const loadBg = () => {
    const cam = state.cameras.find((c) => c.camera_id === "warehouse_cam1") || state.cameras[0];
    if (!cam || agent.bg) return;
    agent.bg = new Image();
    agent.bg.src = cam.background_url;
  };
  loadBg();
  if (!agent.bg) setTimeout(loadBg, 1500); // cameras load asynchronously in app.js
}

initAgent();
