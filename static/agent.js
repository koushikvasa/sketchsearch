"use strict";
// Agent tab, view switching and demo mode. Loaded after app.js and shares its helpers ($, api, COLORS, ...).

const agent = { running: false, markdown: "", abort: null, bg: null, timer: null, started: 0 };
const VERDICT_ICON = { YES: "check", NO: "x", UNSURE: "circle-help", SKIPPED: "skip-forward" };

function demoMode() {
  return $("#demoMode").checked;
}

// ---------------------------------------------------------------------------
// Views + demo toggle
// ---------------------------------------------------------------------------
function showView(view) {
  $("#searchView").hidden = view !== "search";
  $("#agentView").hidden = view !== "agent";
  for (const t of document.querySelectorAll(".tab")) {
    const on = t.dataset.view === view;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", String(on));
  }
  if (view === "search") window.dispatchEvent(new Event("resize")); // canvas may have been hidden
}

function initViews() {
  for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => showView(t.dataset.view));
  const box = $("#demoMode");
  try { box.checked = localStorage.getItem("sketchsearch.demo") === "1"; } catch { /* storage blocked */ }
  box.addEventListener("change", () => {
    try { localStorage.setItem("sketchsearch.demo", box.checked ? "1" : "0"); } catch { /* storage blocked */ }
    toast(box.checked
      ? "Demo mode: examples auto-run the (cached) AI check and the agent replays its recorded run."
      : "Demo mode off: everything runs live.");
  });
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
  li.scrollIntoView({ block: "nearest", behavior: "smooth" });
  return li;
}

function thumbHtml(frameUrl, clipUrl) {
  if (frameUrl) return `<img src="${escapeHtml(frameUrl)}" alt="" loading="lazy">`;
  if (clipUrl) return `<video src="${escapeHtml(clipUrl)}#t=1" muted preload="metadata"></video>`;
  return "";
}

function segLabel(segmentId) {
  const m = /^(.*)_(\d+)$/.exec(segmentId);
  return m ? `${cameraName(m[1])} ${m[2]} s` : segmentId;
}

function renderEvent(ev) {
  switch (ev.event) {
    case "start":
      traceItem("start", `<div class="k-title">Goal: ${escapeHtml(ev.goal)}</div>
        <div class="k-sub">Planner ${escapeHtml(ev.llm)} · up to ${ev.budget} new video-model checks${ev.replayed ? " · replaying recorded run" : ""}</div>`);
      break;
    case "plan": {
      const li = traceItem("plan", `<div class="k-title">Plan: ${ev.sketches.length} sketches</div>
        <div class="k-sub">${escapeHtml(ev.summary || "")}</div><div class="mini-row"></div>`);
      const row = $(".mini-row", li);
      for (const s of ev.sketches) row.appendChild(miniFigure(s.sketch, `${s.id}: ${s.title}`));
      break;
    }
    case "sketch": {
      const li = traceItem("sketch", `<div class="k-title">${ev.refined ? "↻ Re-run" : "▶ Search"} ${escapeHtml(ev.id)}: ${escapeHtml(ev.title)}</div>
        <div class="k-sub">${escapeHtml(ev.why || "")}</div><div class="mini-row"></div>`);
      $(".mini-row", li).appendChild(miniFigure(ev.sketch));
      break;
    }
    case "results": {
      const thumbs = ev.results.slice(0, 3).map((r) =>
        `<figure>${thumbHtml(r.frame_url, r.clip_url)}<figcaption>${escapeHtml(segLabel(r.segment_id))} · ${r.score.toFixed(2)}</figcaption></figure>`).join("");
      traceItem("results", `<div class="k-sub">${ev.count} matches in ${ev.elapsed_ms} ms; checking the top 3</div><div class="thumbs">${thumbs}</div>`);
      break;
    }
    case "verify": {
      const cls = `v-${ev.verdict.toLowerCase()}`;
      const img = thumbHtml(ev.frame_url, ev.clip_url);
      traceItem("verify", `<div class="k-verify-row">${img}<span class="${cls} verdict-tag">${icon(VERDICT_ICON[ev.verdict] || "dot")}${ev.verdict}</span>
        <span><strong>${escapeHtml(segLabel(ev.segment_id))}</strong>: ${escapeHtml(ev.reason)}${ev.cached ? ' <span class="muted">(cached)</span>' : ""}</span></div>`);
      break;
    }
    case "refine": {
      if (!ev.ok) {
        traceItem("refine", `<div class="k-title">↻ Could not refine ${escapeHtml(ev.id)}</div><div class="k-sub">${escapeHtml(ev.reason)}</div>`);
        break;
      }
      const li = traceItem("refine", `<div class="k-title">↻ Refine ${escapeHtml(ev.id)}: only ${Math.round(ev.confirm_rate * 100)}% confirmed → “${escapeHtml(ev.title)}”</div>
        <div class="k-sub">${escapeHtml(ev.why || "")}</div><div class="mini-row"></div>`);
      $(".mini-row", li).appendChild(miniFigure(ev.sketch));
      break;
    }
    case "report":
      agent.markdown = ev.markdown;
      $("#report").innerHTML = renderMarkdown(ev.markdown);
      $("#copyReport").disabled = false;
      traceItem("report", `<div class="k-title">Report: ${ev.stats.confirmed} confirmed of ${ev.stats.clips_checked} clips checked</div>
        <div class="k-sub">${ev.stats.new_video_calls} new video-model calls${ev.stats.skipped_for_budget ? `, ${ev.stats.skipped_for_budget} skipped (budget)` : ""}</div>`);
      $("#agentStatus").textContent = `${ev.stats.confirmed} confirmed · ${ev.stats.sketch_runs} sketch runs · ${ev.stats.new_video_calls} new video checks`;
      break;
    case "done":
      traceItem("done", `<div class="k-sub">Done in ${ev.seconds} s${ev.replayed ? " (recorded run, replayed)" : ""}.</div>`);
      break;
    case "error":
      traceItem("error", `<div class="k-title">Agent failed</div><div class="k-sub">${escapeHtml(ev.message)}</div>`);
      $("#agentStatus").textContent = "Failed";
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
// Run
// ---------------------------------------------------------------------------
async function runAgent(e) {
  e?.preventDefault();
  const goal = $("#agentGoal").value.trim();
  if (goal.length < 3) return toast("Give the agent a goal first, or use the preset.");
  if (agent.abort) agent.abort.abort();
  const controller = new AbortController();
  agent.abort = controller;
  agent.running = true;
  $("#agentRun").disabled = true;
  $("#agentRun").textContent = "Running…";
  $("#trace").innerHTML = "";
  $("#report").innerHTML = `<div class="state"><div class="spinner"></div>The agent is working: watch the trace.</div>`;
  $("#copyReport").disabled = true;
  agent.started = performance.now();
  clearInterval(agent.timer);
  agent.timer = setInterval(() => {
    $("#agentStatus").textContent = `Running… ${((performance.now() - agent.started) / 1000).toFixed(0)} s`;
  }, 500);
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
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") renderEvent({ event: "error", message: err.message });
  } finally {
    clearInterval(agent.timer);
    if (agent.abort === controller) {
      agent.abort = null;
      agent.running = false;
      $("#agentRun").disabled = false;
      $("#agentRun").textContent = "Run agent";
      if (!gotReport) $("#report").innerHTML = `<p class="muted">No report: the run did not finish.</p>`;
    }
  }
}

async function initAgent() {
  initViews();
  $("#agentForm").addEventListener("submit", runAgent);
  $("#copyReport").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(agent.markdown);
      toast("Report markdown copied.");
    } catch {
      toast("Couldn't copy. Your browser blocked the clipboard.", true);
    }
  });
  try {
    const info = await api("api/demo");
    const btn = $("#agentPreset");
    btn.innerHTML = `${icon("radar")}${escapeHtml(info.preset_goal)}`;
    btn.addEventListener("click", () => {
      $("#agentGoal").value = info.preset_goal;
      runAgent();
    });
  } catch {
    $("#agentPreset").hidden = true;
  }
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
