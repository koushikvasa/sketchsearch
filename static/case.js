"use strict";
// Case: the clips an investigator saves as evidence, from searches and sweeps. Loaded after app.js (shares $, icon,
// whereLabel, escapeHtml, toast). Kept in this browser only (localStorage), so a reload does not lose the case.

const CASE_KEY = "sketchsearch.case";
const CASE_MAX = 50;
const caseItems = []; // {key, r: result fields, v: {verdict, reason} | null, from: what found it, added}

function caseKey(r) {
  return `${r.segment_id}@${r.window?.[0] ?? 0}`;
}

function slimResult(r) {
  const keys = ["segment_id", "camera_id", "start", "end", "window", "score", "explanation", "clip_url", "frame_url"];
  return Object.fromEntries(keys.map((k) => [k, r[k] ?? null]));
}

function loadCase() {
  try {
    const saved = JSON.parse(localStorage.getItem(CASE_KEY) || "[]");
    if (Array.isArray(saved)) caseItems.push(...saved.filter((i) => i?.r?.segment_id).slice(0, CASE_MAX));
  } catch { /* storage blocked or corrupt: start empty */ }
}

function saveCase() {
  try { localStorage.setItem(CASE_KEY, JSON.stringify(caseItems)); } catch { /* storage blocked */ }
}

function inCase(r) {
  return caseItems.some((i) => i.key === caseKey(r));
}

function addToCase(r, v, from) {
  if (inCase(r)) return;
  if (caseItems.length >= CASE_MAX) return toast(`A case holds up to ${CASE_MAX} clips. Remove some first.`, true);
  caseItems.push({ key: caseKey(r), r: slimResult(r), v: v ? { verdict: v.verdict, reason: v.reason } : null,
                   from: from || "", added: Date.now() });
  caseChanged();
  toast(`Added ${whereLabel(r)} to the case (${caseItems.length} clip${caseItems.length === 1 ? "" : "s"}).`);
}

function removeFromCase(key) {
  const i = caseItems.findIndex((c) => c.key === key);
  if (i < 0) return;
  caseItems.splice(i, 1);
  caseChanged();
}

function toggleCase(r, v, from) {
  inCase(r) ? removeFromCase(caseKey(r)) : addToCase(r, v, from);
}

/** A verdict that lands after the clip was saved is copied into the case. */
function updateCaseVerdict(r, v) {
  const item = caseItems.find((i) => i.key === caseKey(r));
  if (!item || !v) return;
  item.v = { verdict: v.verdict, reason: v.reason };
  saveCase();
  renderCaseList();
}

function caseChanged() {
  saveCase();
  const n = caseItems.length;
  $("#caseCount").textContent = n;
  $("#caseBtn").classList.toggle("has", n > 0);
  $("#caseSub").textContent = n ? `${n} clip${n === 1 ? "" : "s"} · ${caseItems.filter((i) => i.v?.verdict === "YES").length} confirmed by AI` : "";
  $("#caseReport").disabled = !n;
  $("#caseClear").disabled = !n;
  renderCaseList();
  syncCaseButtons();
}

/** Every "Add to case" button on the page shows whether its clip is already in the case. */
function syncCaseButtons() {
  for (const btn of document.querySelectorAll(".case-add[data-key]")) {
    const on = caseItems.some((i) => i.key === btn.dataset.key);
    btn.setAttribute("aria-pressed", String(on));
    btn.innerHTML = `${icon(on ? "folder-check" : "folder-plus")}<span>${on ? "In case" : "Add to case"}</span>`;
  }
}

function renderCaseList() {
  const list = $("#caseList");
  if (!caseItems.length) {
    list.innerHTML = `<li class="case-empty"><i data-lucide="folder-open"></i><p><b>No clips yet.</b><br>
      Press <b>Add to case</b> on a match or a sweep finding to keep it as evidence. The case report lists them all.</p></li>`;
    return;
  }
  list.innerHTML = caseItems.map((i) => {
    const ui = i.v ? VERDICT_UI[i.v.verdict] : null;
    return `<li class="case-item" data-key="${escapeHtml(i.key)}">
      <div class="case-media">${i.r.frame_url ? `<img src="${escapeHtml(i.r.frame_url)}" alt="">` : ""}
        <video muted loop playsinline preload="none"></video></div>
      <div class="case-body">
        <div class="case-where">${escapeHtml(whereLabel(i.r))}
          ${ui ? `<span class="verdict-tag ${ui.cls}">${icon(ui.icon)}${ui.text}</span>` : `<span class="verdict-tag">Not checked</span>`}</div>
        ${i.v?.reason ? `<p class="case-reason">${escapeHtml(i.v.reason)}</p>` : ""}
        ${i.from ? `<p class="case-from">Found by: ${escapeHtml(i.from)}</p>` : ""}
      </div>
      <button class="icon-btn case-remove" title="Remove from case" aria-label="Remove from case"><i data-lucide="x"></i></button>
    </li>`;
  }).join("");
}

/** Hover or tap a thumbnail to play its matched seconds. */
function bindPreview(container, itemOf) {
  const media = (e) => e.target.closest?.(".case-media, .finding-media");
  const start = (m) => {
    const it = itemOf(m);
    const v = $("video", m);
    if (!it || !v) return;
    if (!v.getAttribute("src")) v.src = it.clip_url;
    try { v.currentTime = it.window?.[0] || 0; } catch { /* metadata not loaded yet */ }
    m.classList.add("playing");
    v.play().catch(() => m.classList.remove("playing"));
  };
  const stop = (m) => { const v = $("video", m); if (v) v.pause(); m.classList.remove("playing"); };
  container.addEventListener("pointerover", (e) => { const m = media(e); if (m && e.pointerType === "mouse") start(m); });
  container.addEventListener("pointerout", (e) => { const m = media(e); if (m && !m.contains(e.relatedTarget)) stop(m); });
  container.addEventListener("click", (e) => { const m = media(e); if (m) (m.classList.contains("playing") ? stop(m) : start(m)); });
}

function initCase() {
  loadCase();
  const drawer = $("#caseDrawer");
  $("#caseBtn").addEventListener("click", () => { renderCaseList(); drawer.showModal(); });
  $("#caseList").addEventListener("click", (e) => {
    const rm = e.target.closest(".case-remove");
    if (rm) removeFromCase(rm.closest(".case-item").dataset.key);
  });
  bindPreview($("#caseList"), (m) => caseItems.find((i) => i.key === m.closest(".case-item")?.dataset.key)?.r);
  $("#caseReport").addEventListener("click", () => openReport("case"));
  // Two-step clear, so one stray click can't wipe the case.
  const clear = $("#caseClear");
  clear.addEventListener("click", () => {
    if (clear.dataset.armed) {
      caseItems.splice(0);
      caseChanged();
      delete clear.dataset.armed;
      clear.innerHTML = `${icon("trash-2")}Clear case`;
      return;
    }
    clear.dataset.armed = "1";
    clear.innerHTML = `${icon("trash-2")}Press again to clear`;
    setTimeout(() => { delete clear.dataset.armed; clear.innerHTML = `${icon("trash-2")}Clear case`; }, 3000);
  });
  for (const dlg of document.querySelectorAll("dialog.drawer")) {
    dlg.addEventListener("click", (e) => { if (e.target === dlg || e.target.closest("[data-close]")) dlg.close(); });
  }
  caseChanged();
}

initCase();
