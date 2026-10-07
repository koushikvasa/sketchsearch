"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const LABELS = ["person", "forklift", "robot", "transporter"];
const COLORS = { person: "#4fc3f7", forklift: "#ffb74d", robot: "#ce93d8", transporter: "#81c784" };
const ABSENT_COLOR = "#ff6b6b";
const DEFAULT_SIZE = { person: [0.03, 0.13], forklift: [0.05, 0.1], robot: [0.05, 0.05], transporter: [0.07, 0.04] };
const ABSENT_SIZE = [0.25, 0.25];
const COMPONENTS = ["relations", "position", "motion", "size", "keyframe"];
const COMPONENT_NAMES = { relations: "relations", position: "position", motion: "motion", size: "size", keyframe: "end layout" };
const TOP_N = 12;
const VERIFY_N = 10;
const FULL = { x: 0, y: 0, w: 1, h: 1 };
const ZOOM_PAD = 0.3; // padding around the action, as a fraction of its size
const ZOOM_MIN = 0.22; // never zoom in more than ~4.5x
const ZOOM_SKIP = 0.8; // action already fills the frame: don't zoom
const DRAW_STAGGER_MS = 150;
const KF_HELP = {
  start: "Draw where things are at the start.",
  end: "Drag each box to where it ends up.",
};

const $ = (sel, el = document) => el.querySelector(sel);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const center = (b) => [b.x + b.w / 2, b.y + b.h / 2];
const round4 = (v) => Math.round(v * 1e4) / 1e4;

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function clampBox(b) {
  const w = clamp(b.w, 0.005, 1), h = clamp(b.h, 0.005, 1);
  return { x: round4(clamp(b.x, 0, 1 - w)), y: round4(clamp(b.y, 0, 1 - h)), w: round4(w), h: round4(h) };
}

function boxAround([cx, cy], w, h) {
  return clampBox({ x: cx - w / 2, y: cy - h / 2, w, h });
}

function pathLength(pts) {
  let d = 0;
  for (let i = 1; i < pts.length; i++) d += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return d;
}

/** Resample a polyline to n points evenly spaced by arc length. */
function resample(pts, n) {
  const total = pathLength(pts);
  if (pts.length < 2 || total < 1e-6) return [pts[0], pts[pts.length - 1]];
  const out = [pts[0]];
  let seg = 0, acc = 0;
  for (let k = 1; k < n - 1; k++) {
    const target = (total * k) / (n - 1);
    while (seg < pts.length - 2 && acc + Math.hypot(pts[seg + 1][0] - pts[seg][0], pts[seg + 1][1] - pts[seg][1]) < target) {
      acc += Math.hypot(pts[seg + 1][0] - pts[seg][0], pts[seg + 1][1] - pts[seg][1]);
      seg++;
    }
    const len = Math.hypot(pts[seg + 1][0] - pts[seg][0], pts[seg + 1][1] - pts[seg][1]) || 1;
    const f = clamp((target - acc) / len, 0, 1);
    out.push([pts[seg][0] + f * (pts[seg + 1][0] - pts[seg][0]), pts[seg][1] + f * (pts[seg + 1][1] - pts[seg][1])]);
  }
  out.push(pts[pts.length - 1]);
  return out.map(([x, y]) => [round4(x), round4(y)]);
}

/** Point at fraction u (0..1) along a polyline, by arc length. */
function pointAlong(pts, u) {
  if (pts.length === 1) return pts[0];
  const total = pathLength(pts);
  if (total < 1e-9) return pts[0];
  let target = u * total;
  for (let i = 1; i < pts.length; i++) {
    const len = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    if (target <= len || i === pts.length - 1) {
      const f = len ? clamp(target / len, 0, 1) : 0;
      return [pts[i - 1][0] + f * (pts[i][0] - pts[i - 1][0]), pts[i - 1][1] + f * (pts[i][1] - pts[i - 1][1])];
    }
    target -= len;
  }
  return pts[pts.length - 1];
}

/** The motion an object was drawn with: its path, else start -> end, else just where it starts. */
function drawnPath(o) {
  if (o.path && o.path.length >= 2) return o.path;
  if (o.end) return [center(o.start), center(o.end)];
  return [center(o.start)];
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  objects: [], // {id, label, absent, start:{x,y,w,h}, end:{...}|null, path:[[x,y]...]|null}
  tab: "start",
  tool: "box",
  label: "person",
  selectedId: null,
  cameras: [],
  cameraId: null,
  bgUrl: null,
  text: null, // sentence a Words -> Sketch came from
  history: [],
};

let canvas = null;
let W = 0, H = 0;
let rendering = false;
let drag = null;
let pendingSnapshot = null;
let idCounter = 0;
let toastTimer = null;

// ---------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------
function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 7000 : 4000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) msg += ` — ${typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)}`;
    } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return res.json();
}

function newId(label) {
  let id;
  do { id = `${label}${++idCounter}`; } while (state.objects.some((o) => o.id === id));
  return id;
}

function findObject(id) {
  return state.objects.find((o) => o.id === id);
}

function cameraName(id) {
  return state.cameras.find((c) => c.camera_id === id)?.name ?? id;
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------
function snapshot() {
  state.history.push(JSON.stringify(state.objects));
  if (state.history.length > 100) state.history.shift();
}

function undo() {
  const prev = state.history.pop();
  if (prev === undefined) return toast("Nothing to undo.");
  state.objects = JSON.parse(prev);
  if (!findObject(state.selectedId)) state.selectedId = null;
  render();
}

function deleteSelected() {
  if (!state.selectedId) return toast("Select a box first.");
  snapshot();
  state.objects = state.objects.filter((o) => o.id !== state.selectedId);
  state.selectedId = null;
  render();
}

function clearAll() {
  if (!state.objects.length) return;
  snapshot();
  state.objects = [];
  state.selectedId = null;
  render();
}

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------
function initCanvas() {
  // Fabric 5.3 sets the invalid textBaseline "alphabetical" (browsers want "alphabetic") and the
  // console fills with warnings; translate it at the canvas context.
  const baseline = Object.getOwnPropertyDescriptor(CanvasRenderingContext2D.prototype, "textBaseline");
  Object.defineProperty(CanvasRenderingContext2D.prototype, "textBaseline", {
    ...baseline,
    set(v) { baseline.set.call(this, v === "alphabetical" ? "alphabetic" : v); },
  });
  canvas = new fabric.Canvas("sketch", {
    selection: false, preserveObjectStacking: true, uniformScaling: false, stopContextMenu: true,
  });
  Object.assign(fabric.Object.prototype, {
    transparentCorners: false, cornerColor: "#ffffff", cornerStrokeColor: "#0e1015",
    cornerSize: 11, borderColor: "#7c9cff", borderScaleFactor: 2, padding: 0,
  });
  canvas.on("mouse:down", onMouseDown);
  canvas.on("mouse:move", onMouseMove);
  canvas.on("mouse:up", onMouseUp);
  canvas.on("before:transform", () => { pendingSnapshot = JSON.stringify(state.objects); });
  canvas.on("object:modified", onModified);
  const onSelect = (e) => {
    if (rendering) return;
    const id = e.selected?.[0]?.data?.id;
    if (id) { state.selectedId = id; updateUi(); }
  };
  canvas.on("selection:created", onSelect);
  canvas.on("selection:updated", onSelect);
  canvas.on("selection:cleared", () => { if (!rendering) { state.selectedId = null; updateUi(); } });
  new ResizeObserver(resizeCanvas).observe($("#canvasWrap"));
  resizeCanvas();
}

function resizeCanvas() {
  const r = $("#canvasWrap").getBoundingClientRect();
  if (!r.width || (Math.round(r.width) === W && Math.round(r.height) === H)) return;
  W = Math.round(r.width);
  H = Math.round(r.height);
  canvas.setDimensions({ width: W, height: H });
  setBackground();
  render();
}

function setBackground() {
  if (!canvas) return;
  const url = $("#showFrame").checked ? state.bgUrl : null;
  if (!url) {
    canvas.setBackgroundImage(null, canvas.renderAll.bind(canvas));
    return;
  }
  fabric.Image.fromURL(url, (img) => {
    if (!img || !img.width) return toast("Couldn't load the camera frame.", true);
    img.set({ opacity: 0.42, originX: "left", originY: "top", scaleX: W / img.width, scaleY: H / img.height });
    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
  });
}

function norm(pointer) {
  return { x: clamp(pointer.x / W, 0, 1), y: clamp(pointer.y / H, 0, 1) };
}

function px(b) {
  return { left: b.x * W, top: b.y * H, width: b.w * W, height: b.h * H };
}

function rectFor(b, color, opts = {}) {
  const r = new fabric.Rect({
    ...px(b),
    fill: opts.absent ? "rgba(255,107,107,0.14)" : hexA(color, opts.ghost ? 0.04 : 0.18),
    stroke: color,
    strokeWidth: opts.ghost ? 1.5 : 2.5,
    strokeDashArray: opts.absent || opts.ghost || opts.placeholder ? [7, 5] : null,
    strokeUniform: true,
    opacity: opts.ghost ? 0.5 : 1,
    selectable: !!opts.interactive,
    evented: !!opts.interactive,
    hasRotatingPoint: false,
    lockRotation: true,
    objectCaching: false,
    hoverCursor: "move",
    data: { id: opts.id, kind: opts.kind, interactive: !!opts.interactive },
  });
  r.setControlsVisibility({ mtr: false });
  return r;
}

function tagFor(text, b, color) {
  return new fabric.Text(text, {
    left: b.x * W, top: Math.max(0, b.y * H - 21), fontSize: 15, fontWeight: "600",
    fontFamily: "Segoe UI, system-ui, sans-serif", fill: color, backgroundColor: "rgba(10,12,16,0.72)",
    selectable: false, evented: false,
  });
}

function addArrow(pts, color, dashed) {
  if (pts.length < 2 || pathLength(pts) < 0.005) return;
  const P = pts.map(([x, y]) => ({ x: x * W, y: y * H }));
  canvas.add(new fabric.Polyline(P, {
    fill: "", stroke: color, strokeWidth: 3, strokeDashArray: dashed ? [6, 6] : null, strokeLineJoin: "round",
    strokeLineCap: "round", selectable: false, evented: false, objectCaching: false, opacity: 0.95,
  }));
  let a = P[P.length - 2];
  const b = P[P.length - 1];
  for (let i = P.length - 2; i >= 0 && Math.hypot(b.x - P[i].x, b.y - P[i].y) < 6; i--) a = P[i];
  const angle = (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI + 90;
  canvas.add(new fabric.Triangle({
    left: b.x, top: b.y, originX: "center", originY: "center", width: 15, height: 17, fill: color, angle,
    selectable: false, evented: false,
  }));
}

function drawObject(o) {
  const color = o.absent ? ABSENT_COLOR : COLORS[o.label];
  const name = o.absent ? `🚫 no ${o.label}` : o.label;
  if (!o.absent) addArrow(drawnPath(o), color, !o.path);
  if (state.tab === "start") {
    if (o.end && !o.absent) canvas.add(rectFor(o.end, color, { ghost: true }));
    canvas.add(rectFor(o.start, color, { absent: o.absent, interactive: true, id: o.id, kind: "start" }));
    canvas.add(tagFor(name, o.start, color));
  } else {
    canvas.add(rectFor(o.start, color, { ghost: true, absent: o.absent }));
    if (!o.absent) {
      const b = o.end || o.start;
      canvas.add(rectFor(b, color, { interactive: true, id: o.id, kind: "end", placeholder: !o.end }));
      canvas.add(tagFor(o.end ? `${o.label} · end` : `${o.label} · drag to set end`, b, color));
    }
  }
}

function render() {
  if (!canvas || !W) return;
  rendering = true;
  canvas.discardActiveObject();
  canvas.remove(...canvas.getObjects());
  state.objects.forEach(drawObject);
  canvas.skipTargetFind = state.tool === "path";
  if (state.selectedId && state.tool !== "path") {
    const f = canvas.getObjects().find((x) => x.data?.interactive && x.data.id === state.selectedId);
    if (f) canvas.setActiveObject(f);
  }
  canvas.requestRenderAll();
  rendering = false;
  updateUi();
}

/** The object under (or nearest to) a normalized point, for starting a path. */
function objectAt(p) {
  const present = state.objects.filter((o) => !o.absent);
  const pad = 0.02;
  const hits = present.filter((o) => {
    const b = state.tab === "end" && o.end ? o.end : o.start;
    return p.x >= b.x - pad && p.x <= b.x + b.w + pad && p.y >= b.y - pad && p.y <= b.y + b.h + pad;
  });
  if (hits.length) return hits.sort((a, b) => a.start.w * a.start.h - b.start.w * b.start.h)[0];
  let best = null, bestD = 0.12;
  for (const o of present) {
    const [cx, cy] = center(o.start);
    const d = Math.hypot(cx - p.x, cy - p.y);
    if (d < bestD) { best = o; bestD = d; }
  }
  return best;
}

function onMouseDown(opt) {
  if (opt.e.button === 2) return;
  const p = norm(canvas.getPointer(opt.e));
  if (state.tool === "path") {
    const o = objectAt(p);
    if (!o) return toast("Start the path on an object: press on its box and drag.");
    drag = { mode: "path", id: o.id, points: [center(o.start)], temp: null };
    state.selectedId = o.id;
    return;
  }
  if (opt.target) return; // moving / resizing an existing box
  if (state.tab !== "start") return toast("Switch to Start to add objects. In End, drag boxes to where they finish.");
  const temp = new fabric.Rect({
    left: p.x * W, top: p.y * H, width: 1, height: 1, fill: "rgba(255,255,255,0.08)", stroke: "#ffffff",
    strokeDashArray: [4, 4], strokeWidth: 1.5, selectable: false, evented: false,
  });
  canvas.add(temp);
  drag = { mode: "box", start: p, temp };
}

function onMouseMove(opt) {
  if (!drag) return;
  const p = norm(canvas.getPointer(opt.e));
  if (drag.mode === "box") {
    const s = drag.start;
    drag.temp.set({
      left: Math.min(s.x, p.x) * W, top: Math.min(s.y, p.y) * H,
      width: Math.abs(p.x - s.x) * W, height: Math.abs(p.y - s.y) * H,
    });
  } else {
    const last = drag.points[drag.points.length - 1];
    if (Math.hypot(p.x - last[0], p.y - last[1]) < 0.006) return;
    drag.points.push([p.x, p.y]);
    if (drag.temp) canvas.remove(drag.temp);
    drag.temp = new fabric.Polyline(drag.points.map(([x, y]) => ({ x: x * W, y: y * H })), {
      fill: "", stroke: COLORS[findObject(drag.id).label], strokeWidth: 3, strokeLineCap: "round",
      strokeLineJoin: "round", selectable: false, evented: false, objectCaching: false,
    });
    canvas.add(drag.temp);
  }
  canvas.requestRenderAll();
}

function onMouseUp(opt) {
  if (!drag) return;
  const d = drag;
  drag = null;
  if (d.temp) canvas.remove(d.temp);
  if (d.mode === "box") {
    const p = norm(canvas.getPointer(opt.e));
    let x = Math.min(d.start.x, p.x), y = Math.min(d.start.y, p.y);
    let w = Math.abs(p.x - d.start.x), h = Math.abs(p.y - d.start.y);
    const absent = state.tool === "absent";
    if (w * W < 10 || h * H < 10) { // a click: drop a default-size box centred on it
      [w, h] = absent ? ABSENT_SIZE : DEFAULT_SIZE[state.label];
      x = d.start.x - w / 2;
      y = d.start.y - h / 2;
    }
    snapshot();
    const o = { id: newId(state.label), label: state.label, absent, start: clampBox({ x, y, w, h }), end: null, path: null };
    state.objects.push(o);
    state.selectedId = o.id;
    render();
    return;
  }
  const pts = d.points.length > 1 ? resample(d.points, Math.min(10, Math.max(3, d.points.length))) : d.points;
  if (pts.length < 2 || pathLength(pts) < 0.02) {
    render();
    return toast("Path too short. Drag further from the object.");
  }
  snapshot();
  const o = findObject(d.id);
  o.path = pts;
  o.end = boxAround(pts[pts.length - 1], o.start.w, o.start.h);
  render();
}

function onModified(e) {
  const f = e.target;
  const o = f?.data?.id && findObject(f.data.id);
  if (!o) return;
  if (pendingSnapshot) {
    state.history.push(pendingSnapshot);
    pendingSnapshot = null;
  }
  const b = clampBox({ x: f.left / W, y: f.top / H, w: (f.width * f.scaleX) / W, h: (f.height * f.scaleY) / H });
  if (f.data.kind === "start") {
    // Moving an object moves its whole drawn motion with it.
    const [dx, dy] = [center(b)[0] - center(o.start)[0], center(b)[1] - center(o.start)[1]];
    if (o.path) o.path = o.path.map(([x, y]) => [round4(clamp(x + dx, 0, 1)), round4(clamp(y + dy, 0, 1))]);
    if (o.end) o.end = clampBox({ ...o.end, x: o.end.x + dx, y: o.end.y + dy });
    o.start = b;
  } else {
    // Dragging the End box bends the path so it still finishes on the box.
    const prev = center(o.end || o.start);
    const [dx, dy] = [center(b)[0] - prev[0], center(b)[1] - prev[1]];
    if (o.path) {
      const n = o.path.length - 1;
      o.path = o.path.map(([x, y], i) => [round4(clamp(x + (dx * i) / n, 0, 1)), round4(clamp(y + (dy * i) / n, 0, 1))]);
    }
    o.end = b;
  }
  state.selectedId = o.id;
  render();
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function buildPalette() {
  const pal = $("#palette");
  for (const label of LABELS) {
    const btn = document.createElement("button");
    btn.className = "label-btn";
    btn.dataset.label = label;
    btn.title = `Next box is a ${label}`;
    btn.innerHTML = `<span class="chip" style="background:${COLORS[label]}"></span>${label}`;
    btn.addEventListener("click", () => setLabel(label));
    pal.appendChild(btn);
  }
}

function setLabel(label) {
  state.label = label;
  updateUi();
}

function setTool(tool) {
  state.tool = tool;
  render();
  if (tool === "path") toast("Path: press on an object and drag the way it moves.");
  if (tool === "absent") toast(`Absent: draw a zone where no ${state.label} may be. Pick the label first.`);
}

function setTab(tab) {
  state.tab = tab;
  render();
}

function updateUi() {
  for (const b of document.querySelectorAll(".label-btn")) b.classList.toggle("active", b.dataset.label === state.label);
  for (const b of document.querySelectorAll(".tool")) b.classList.toggle("active", b.dataset.tool === state.tool);
  for (const b of document.querySelectorAll(".kf")) b.classList.toggle("active", b.dataset.tab === state.tab);
  $("#kfHelp").textContent = KF_HELP[state.tab];
  $("#canvasEmpty").hidden = state.objects.length > 0;
  $("#hint").hidden = state.objects.filter((o) => !o.absent).length !== 1;
  $("#search").disabled = state.objects.length === 0;
  $("#delete").disabled = !state.selectedId;
  $("#undo").disabled = state.history.length === 0;
  $("#clear").disabled = state.objects.length === 0;
}

function selectCamera(cameraId, bgUrl = null) {
  const cam = state.cameras.find((c) => c.camera_id === cameraId);
  if (!cam) return;
  state.cameraId = cameraId;
  state.bgUrl = bgUrl || cam.background_url;
  $("#camera").value = cameraId;
  setBackground();
}

function loadSketch(sketch, { cameraId = null, bgUrl = null, text = null } = {}) {
  snapshot();
  state.text = text;
  state.objects = sketch.objects.map((o) => ({
    id: o.id, label: o.label, absent: !!o.absent,
    start: o.start_box, end: o.absent ? null : o.end_box || null, path: o.absent ? null : o.path || null,
  }));
  state.selectedId = null;
  state.tab = "start";
  if (cameraId) selectCamera(cameraId, bgUrl);
  render();
}

function toSketch() {
  return {
    objects: state.objects.map((o) => ({
      id: o.id, label: o.label, absent: o.absent, start_box: o.start,
      end_box: o.absent ? null : o.end, path: o.absent ? null : o.path,
    })),
    camera_ids: $("#onlyCamera").checked && state.cameraId ? [state.cameraId] : null,
    text: state.text,
  };
}

// ---------------------------------------------------------------------------
// Search + results
// ---------------------------------------------------------------------------
let searchSeq = 0;
let lastSketch = null;
let verifyAbort = null;
let cards = [];
let animating = false;

function showState(kind, message = "") {
  stopAllVideos();
  resetVerify();
  cards = [];
  const box = $("#results");
  if (kind === "loading") {
    box.innerHTML = `<div class="state"><div class="spinner"></div>Searching every 1.5 s window…</div>`;
    $("#status").textContent = "Searching…";
  } else if (kind === "empty") {
    box.innerHTML = `<div class="state">No segment has all of these objects at once.<br>
      Try fewer objects, another label, or untick “Only this camera”.</div>`;
    $("#status").textContent = "0 matches";
  } else if (kind === "error") {
    box.innerHTML = `<div class="state error">Search failed: ${escapeHtml(message)}<br><button id="retry">Try again</button></div>`;
    $("#status").textContent = "Error";
    $("#retry").addEventListener("click", runSearch);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

async function runSearch() {
  if (!state.objects.length) return toast("Draw something first, or pick a demo preset.");
  const seq = ++searchSeq;
  const sketch = toSketch();
  showState("loading");
  $("#search").disabled = true;
  try {
    const res = await api("api/search", { method: "POST", body: JSON.stringify({ sketch, top_n: TOP_N }) });
    if (seq !== searchSeq) return;
    renderResults(res, sketch);
  } catch (err) {
    if (seq === searchSeq) showState("error", err.message);
  } finally {
    if (seq === searchSeq) updateUi();
  }
}

function renderResults(res, sketch) {
  resetVerify();
  lastSketch = sketch;
  if (!res.results.length) return showState("empty");
  stopAllVideos();
  const box = $("#results");
  box.innerHTML = "";
  const boosted = res.weights && res.weights.motion >= 0.25;
  $("#status").textContent = `Top ${res.results.length} · ${res.elapsed_ms} ms` +
    (boosted ? " · motion weighted up (you drew clear movement)" : "");
  const objects = sketch.objects.map((o) => ({
    ...o, start: o.start_box, end: o.end_box, path: o.path,
  }));
  cards = res.results.map((r, i) => buildCard(r, i + 1, objects));
  cards.forEach((c) => box.appendChild(c.el));
  $("#verifyBtn").disabled = false;
  for (const card of cards) {
    api(`api/segments/${encodeURIComponent(card.result.segment_id)}`)
      .then((seg) => { card.segment = seg; updateZoom(card); })
      .catch(() => { /* overlay just shows the sketch */ });
    drawOverlay(card, card.posterTime);
  }
}

function buildCard(r, rank, objects) {
  const el = $("#cardTpl").content.firstElementChild.cloneNode(true);
  const video = $("video", el);
  if (r.frame_url) {
    video.src = r.clip_url;
    video.poster = r.frame_url;
  } else {
    video.src = `${r.clip_url}#t=${(r.window[0] + r.window[1]) / 2}`; // no keyframe image: show a frame of the clip
  }
  $(".rank", el).textContent = `#${rank}`;
  $(".seg", el).innerHTML = `${escapeHtml(cameraName(r.camera_id))} <small>${r.start}–${r.end} s · best ${r.window[0]}–${r.window[1]} s</small>`;
  $(".score", el).textContent = r.score.toFixed(2);
  $(".explain", el).textContent = r.explanation || "";

  const abs = $(".absence", el);
  if (r.absence_ok !== null && r.absence_ok !== undefined) {
    abs.hidden = false;
    abs.className = `badge absence ${r.absence_ok ? "ok" : "bad"}`;
    abs.textContent = r.absence_ok ? "✅ nobody in the zone" : "⚠️ zone violated";
  }

  const bars = $(".bars", el);
  for (const name of COMPONENTS) {
    if (!(name in r.components)) continue;
    const v = r.components[name];
    bars.insertAdjacentHTML("beforeend",
      `<span class="name">${COMPONENT_NAMES[name]}</span>
       <span class="track"><span class="fill" style="width:${(v * 100).toFixed(0)}%"></span></span>
       <span class="val">${v.toFixed(2)}</span>`);
  }

  const duration = r.end - r.start;
  const card = {
    el, video, overlay: $(".overlay", el), result: r, objects, segment: null,
    pinned: false, playing: false, started: false, duration, posterTime: duration / 2,
    zoom: FULL, zoomOn: true, verdict: null,
  };
  updateZoom(card);
  const zoomBtn = $(".zoom-btn", el);
  zoomBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    card.zoomOn = !card.zoomOn;
    applyZoom(card);
  });
  const vbadge = $(".badge.verify", el);
  vbadge.addEventListener("click", () => {
    const reason = $(".verify-reason", el);
    if (card.verdict) reason.hidden = !reason.hidden;
  });
  const win = $(".timeline .win", el);
  win.style.left = `${(r.window[0] / duration) * 100}%`;
  win.style.width = `${((r.window[1] - r.window[0]) / duration) * 100}%`;

  const media = $(".media", el);
  media.addEventListener("mouseenter", () => playCard(card));
  media.addEventListener("mouseleave", () => { if (!card.pinned) pauseCard(card); });
  media.addEventListener("click", () => {
    card.pinned = !card.pinned;
    card.pinned ? playCard(card) : pauseCard(card);
  });
  video.addEventListener("seeked", () => { if (!card.playing) drawOverlay(card, video.currentTime); });
  $(".more", el).addEventListener("click", () => moreLikeThis(r));
  return card;
}

function playCard(card) {
  if (!card.started) {
    card.started = true;
    try { card.video.currentTime = card.result.window[0]; } catch { /* metadata not loaded yet */ }
  }
  card.playing = true;
  card.el.classList.add("playing");
  card.video.play().catch(() => { card.playing = false; card.el.classList.remove("playing"); });
  if (!animating) {
    animating = true;
    requestAnimationFrame(tick);
  }
}

function pauseCard(card) {
  card.video.pause();
  card.playing = false;
  card.el.classList.remove("playing");
  drawOverlay(card, card.started ? card.video.currentTime : card.posterTime);
}

function stopAllVideos() {
  for (const c of cards) {
    c.video.pause();
    c.video.removeAttribute("src");
    c.video.load();
  }
}

function tick() {
  let any = false;
  for (const c of cards) {
    if (c.playing) {
      any = true;
      drawOverlay(c, c.video.currentTime);
    }
  }
  animating = any;
  if (any) requestAnimationFrame(tick);
}

/** A real track's box at time t (linear interpolation between samples), or null outside the track. */
function trackBoxAt(track, t) {
  const pts = track.points;
  if (!pts.length || t < pts[0].t - 0.1 || t > pts[pts.length - 1].t + 0.1) return null;
  if (t <= pts[0].t) return pts[0].box;
  for (let i = 1; i < pts.length; i++) {
    if (t <= pts[i].t) {
      const a = pts[i - 1], b = pts[i];
      const f = (t - a.t) / (b.t - a.t || 1);
      return { x: a.box.x + f * (b.box.x - a.box.x), y: a.box.y + f * (b.box.y - a.box.y),
               w: a.box.w + f * (b.box.w - a.box.w), h: a.box.h + f * (b.box.h - a.box.h) };
    }
  }
  return pts[pts.length - 1].box;
}

function drawOverlay(card, t) {
  const cv = card.overlay;
  const rect = cv.getBoundingClientRect();
  if (!rect.width) return;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(rect.width * dpr)) {
    cv.width = Math.round(rect.width * dpr);
    cv.height = Math.round(rect.height * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const r = card.result;
  const [t0, t1] = r.window;
  const u = clamp((t - t0) / (t1 - t0 || 1), 0, 1);
  const z = card.zoomOn ? card.zoom : FULL;
  const X = (x) => ((x - z.x) / z.w) * w, Y = (y) => ((y - z.y) / z.h) * h;
  const SX = (v) => (v / z.w) * w, SY = (v) => (v / z.h) * h;
  const tracks = Object.fromEntries((card.segment?.tracks || []).map((tr) => [tr.track_id, tr]));

  // Absent zones
  for (const o of card.objects.filter((o) => o.absent)) {
    const b = o.start;
    ctx.save();
    ctx.setLineDash([7, 5]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = ABSENT_COLOR;
    ctx.fillStyle = r.absence_ok === false ? "rgba(255,107,107,0.22)" : "rgba(255,107,107,0.08)";
    ctx.fillRect(X(b.x), Y(b.y), SX(b.w), SY(b.h));
    ctx.strokeRect(X(b.x), Y(b.y), SX(b.w), SY(b.h));
    label(ctx, `no ${o.label}`, X(b.x) + 3, Y(b.y + b.h) - 4, ABSENT_COLOR);
    ctx.restore();
  }

  for (const o of card.objects.filter((o) => !o.absent)) {
    const color = COLORS[o.label] || "#ffffff";
    const pts = drawnPath(o);
    const end = o.end || o.start;

    // Ghost replay: the whole drawn path faintly, the part already travelled brighter.
    if (pts.length >= 2 && pathLength(pts) > 0.005) {
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = hexA(color, 0.22);
      ctx.lineWidth = 7;
      polyline(ctx, pts.map(([x, y]) => [X(x), Y(y)]));
      const trail = [];
      for (let k = 0; k <= 24; k++) trail.push(pointAlong(pts, (u * k) / 24));
      ctx.strokeStyle = hexA(color, 0.55);
      ctx.lineWidth = 4;
      polyline(ctx, trail.map(([x, y]) => [X(x), Y(y)]));
      ctx.restore();
    }

    // Sketch box (dashed) moving along the drawn path during the matched window.
    const [gx, gy] = pointAlong(pts, u);
    const gw = o.start.w + u * (end.w - o.start.w), gh = o.start.h + u * (end.h - o.start.h);
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = hexA(color, 0.95);
    ctx.fillStyle = hexA(color, 0.1);
    ctx.fillRect(X(gx - gw / 2), Y(gy - gh / 2), SX(gw), SY(gh));
    ctx.strokeRect(X(gx - gw / 2), Y(gy - gh / 2), SX(gw), SY(gh));
    ctx.restore();
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.fillStyle = hexA(color, 0.9);
    ctx.beginPath();
    ctx.arc(X(gx), Y(gy), 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Real track (solid) for the matched object, plus a line pairing it with the sketch box.
    const track = tracks[r.assignment[o.id]];
    const real = track && trackBoxAt(track, t);
    if (real) {
      const [rx, ry] = center(real);
      ctx.save();
      ctx.strokeStyle = hexA(color, 0.75);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(X(gx), Y(gy));
      ctx.lineTo(X(rx), Y(ry));
      ctx.stroke();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.strokeRect(X(real.x), Y(real.y), SX(real.w), SY(real.h));
      label(ctx, o.label, X(real.x), Y(real.y) - 4, color);
      ctx.restore();
    }
  }

  const head = $(".timeline .head", card.el);
  head.style.left = `${clamp(t / card.duration, 0, 1) * 100}%`;
}

function polyline(ctx, P) {
  ctx.beginPath();
  ctx.moveTo(P[0][0], P[0][1]);
  for (let i = 1; i < P.length; i++) ctx.lineTo(P[i][0], P[i][1]);
  ctx.stroke();
}

function label(ctx, text, x, y, color) {
  ctx.save();
  ctx.font = "600 12px Segoe UI, system-ui, sans-serif";
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(10,12,16,0.75)";
  ctx.fillRect(x - 2, y - 12, tw + 6, 15);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 1, y);
  ctx.restore();
}

async function moreLikeThis(r) {
  try {
    const res = await api(`api/segments/${encodeURIComponent(r.segment_id)}/as-sketch?t0=${r.window[0]}&max_objects=4`);
    if (!res.sketch.objects.length) return toast("No tracks to copy in that window.");
    const counts = {};
    const objects = res.sketch.objects.map((o) => {
      counts[o.label] = (counts[o.label] || 0) + 1;
      return { ...o, id: `${o.label}${counts[o.label]}` };
    });
    loadSketch({ objects }, { cameraId: r.camera_id, bgUrl: r.frame_url });
    window.scrollTo({ top: 0, behavior: "smooth" });
    toast(`Loaded ${objects.length} objects from ${cameraName(r.camera_id)} at ${r.start + res.window[0]}–${r.start + res.window[1]} s. Edit them, then Search.`);
  } catch (err) {
    toast(`Couldn't load that segment: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// Zoom to action
// ---------------------------------------------------------------------------
/** Crop (normalized, same w and h so the 16:9 aspect holds) around the matched tracks + absent zones. */
function computeZoom(card) {
  const r = card.result;
  const [t0, t1] = r.window;
  const boxes = [];
  const tracks = Object.fromEntries((card.segment?.tracks || []).map((tr) => [tr.track_id, tr]));
  for (const o of card.objects) {
    if (o.absent) { boxes.push(o.start); continue; }
    const tr = tracks[r.assignment[o.id]];
    const pts = tr ? tr.points.filter((p) => p.t >= t0 - 1e-6 && p.t <= t1 + 1e-6) : [];
    if (pts.length) pts.forEach((p) => boxes.push(p.box));
    else { boxes.push(o.start); if (o.end) boxes.push(o.end); } // segment not loaded yet: use the sketch
  }
  if (!boxes.length) return FULL;
  let x0 = Math.min(...boxes.map((b) => b.x)), y0 = Math.min(...boxes.map((b) => b.y));
  let x1 = Math.max(...boxes.map((b) => b.x + b.w)), y1 = Math.max(...boxes.map((b) => b.y + b.h));
  const pw = (x1 - x0) * ZOOM_PAD, ph = (y1 - y0) * ZOOM_PAD;
  x0 -= pw; x1 += pw; y0 -= ph; y1 += ph;
  const side = clamp(Math.max(x1 - x0, y1 - y0), ZOOM_MIN, 1);
  if (side >= ZOOM_SKIP) return FULL;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return { x: clamp(cx - side / 2, 0, 1 - side), y: clamp(cy - side / 2, 0, 1 - side), w: side, h: side };
}

function updateZoom(card) {
  card.zoom = computeZoom(card);
  applyZoom(card);
}

function applyZoom(card) {
  const z = card.zoomOn ? card.zoom : FULL;
  card.video.style.transform = z === FULL ? "" : `scale(${1 / z.w}) translate(${-z.x * 100}%, ${-z.y * 100}%)`;
  const btn = $(".zoom-btn", card.el);
  btn.hidden = card.zoom === FULL;
  btn.classList.toggle("on", card.zoomOn);
  btn.textContent = card.zoomOn ? `🔍 ${(1 / card.zoom.w).toFixed(1)}×` : "🔍 off";
  drawOverlay(card, card.started ? card.video.currentTime : card.posterTime);
}

// ---------------------------------------------------------------------------
// AI verification (server-sent events)
// ---------------------------------------------------------------------------
const VERDICT_UI = {
  YES: { cls: "yes", text: "✅ AI: yes" },
  NO: { cls: "no", text: "❌ AI: no" },
  UNSURE: { cls: "unsure", text: "⚠️ AI: unsure" },
};

function resetVerify() {
  if (verifyAbort) verifyAbort.abort();
  verifyAbort = null;
  const btn = $("#verifyBtn");
  btn.disabled = true;
  btn.classList.remove("busy");
  btn.textContent = `🤖 AI check top ${VERIFY_N}`;
  $("#verifySummary").hidden = true;
}

function setVerdict(card, v) {
  card.verdict = v;
  const ui = VERDICT_UI[v.verdict] || VERDICT_UI.UNSURE;
  const badge = $(".badge.verify", card.el);
  badge.hidden = false;
  badge.className = `badge verify ${ui.cls} landed`;
  badge.textContent = ui.text;
  badge.title = `${v.reason}\n(${v.model || "video model"}${v.cached ? ", cached" : ""}; click to show or hide)`;
  const reason = $(".verify-reason", card.el);
  reason.textContent = v.reason;
  reason.hidden = v.verdict === "YES"; // the honest "no" / "unsure" reasons show right away
  card.el.classList.toggle("rejected", v.verdict === "NO");
}

function updateVerifySummary(counts, done, total) {
  const el = $("#verifySummary");
  el.hidden = false;
  const extra = [counts.NO && `${counts.NO} ❌`, counts.UNSURE && `${counts.UNSURE} ⚠️`].filter(Boolean).join(", ");
  el.textContent = done < total
    ? `Verified ${counts.YES} of ${total} · checking ${done}/${total}…`
    : `Verified ${counts.YES} of ${total}${extra ? ` (${extra})` : ""}`;
}

async function runVerify() {
  if (!lastSketch || !cards.length) return;
  const targets = cards.slice(0, VERIFY_N);
  const btn = $("#verifyBtn");
  if (verifyAbort) verifyAbort.abort();
  const controller = new AbortController();
  verifyAbort = controller;
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "🤖 Checking…";
  for (const c of targets) {
    c.verdict = null;
    c.el.classList.remove("rejected");
    const b = $(".badge.verify", c.el);
    b.hidden = false;
    b.className = "badge verify checking";
    b.textContent = "⏳ AI checking…";
    b.title = "";
    $(".verify-reason", c.el).hidden = true;
  }
  const counts = { YES: 0, NO: 0, UNSURE: 0 };
  let done = 0;
  updateVerifySummary(counts, 0, targets.length);
  try {
    const res = await fetch("api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sketch: lastSketch,
        items: targets.map((c) => ({ segment_id: c.result.segment_id, window: c.result.window })),
      }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        const event = /^event: (.*)$/m.exec(chunk)?.[1];
        const data = /^data: (.*)$/m.exec(chunk)?.[1];
        if (event !== "verdict" || !data) continue;
        const payload = JSON.parse(data);
        const card = targets.find((c) => c.result.segment_id === payload.segment_id);
        if (card) setVerdict(card, payload);
        counts[payload.verdict] = (counts[payload.verdict] || 0) + 1;
        done += 1;
        updateVerifySummary(counts, done, targets.length);
      }
    }
    updateVerifySummary(counts, targets.length, targets.length);
  } catch (err) {
    if (err.name === "AbortError") return;
    toast(`AI check failed: ${err.message}`, true);
    for (const c of targets) if (!c.verdict) $(".badge.verify", c.el).hidden = true;
  } finally {
    if (verifyAbort === controller) {
      verifyAbort = null;
      btn.disabled = false;
      btn.classList.remove("busy");
      btn.textContent = `🤖 AI check top ${VERIFY_N}`;
    }
  }
}

// ---------------------------------------------------------------------------
// ✨ Words -> Sketch and 📄 Diagram -> Sketch
// ---------------------------------------------------------------------------
function toStateObject(o) {
  return {
    id: o.id, label: o.label, absent: !!o.absent,
    start: o.start_box, end: o.absent ? null : o.end_box || null, path: o.absent ? null : o.path || null,
  };
}

async function animateSketch(objects) {
  snapshot();
  state.objects = [];
  state.selectedId = null;
  state.tab = "start";
  render();
  for (const o of objects) {
    await new Promise((resolve) => setTimeout(resolve, DRAW_STAGGER_MS));
    state.objects.push(toStateObject(o));
    render();
  }
}

async function sketchFromText(e) {
  e.preventDefault();
  const text = $("#sketchText").value.trim();
  if (text.length < 2) return toast("Describe the moment first, e.g. “two people walking toward each other”.");
  const btn = $("#sketchIt");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "✨ Sketching…";
  try {
    const res = await api("api/text-to-sketch", { method: "POST", body: JSON.stringify({ text }) });
    await animateSketch(res.sketch.objects);
    state.text = text;
    toast(`✨ Sketched by ${res.model} (${res.provider}) in ${(res.elapsed_ms / 1000).toFixed(1)} s. Searching…`);
    await runSearch();
  } catch (err) {
    toast(`Couldn't sketch that: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.textContent = "✨ Sketch it";
  }
}

async function uploadDiagram() {
  const input = $("#diagramFile");
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  const btn = $("#diagramBtn");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "📄 Reading diagram…";
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("api/diagram-to-sketch", { method: "POST", body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
    loadSketch(body.sketch);
    state.bgUrl = URL.createObjectURL(file); // show the diagram faintly behind the boxes
    $("#showFrame").checked = true;
    setBackground();
    toast(`📄 ${body.sketch.objects.length} objects read from the diagram by ${body.model}. ` +
          "Your diagram is the faint background: adjust the boxes, then Search.");
  } catch (err) {
    toast(`Couldn't read the diagram: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.textContent = "📄 Upload diagram";
  }
}


// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function bindControls() {
  for (const b of document.querySelectorAll(".tool")) b.addEventListener("click", () => setTool(b.dataset.tool));
  for (const b of document.querySelectorAll(".kf")) b.addEventListener("click", () => setTab(b.dataset.tab));
  $("#undo").addEventListener("click", undo);
  $("#delete").addEventListener("click", deleteSelected);
  $("#clear").addEventListener("click", clearAll);
  $("#search").addEventListener("click", runSearch);
  $("#verifyBtn").addEventListener("click", runVerify);
  $("#textForm").addEventListener("submit", sketchFromText);
  $("#diagramBtn").addEventListener("click", () => $("#diagramFile").click());
  $("#diagramFile").addEventListener("change", uploadDiagram);
  $("#showFrame").addEventListener("change", setBackground);
  $("#camera").addEventListener("change", (e) => selectCamera(e.target.value));
  document.addEventListener("keydown", (e) => {
    const el = document.activeElement;
    if (el && (el.tagName === "TEXTAREA" || el.tagName === "SELECT" || (el.tagName === "INPUT" && el.type === "text"))) return;
    const key = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && key === "z") { e.preventDefault(); undo(); }
    else if ((e.ctrlKey || e.metaKey) && key === "enter") { e.preventDefault(); runSearch(); }
    else if (e.key === "Delete" || e.key === "Backspace") { if (state.selectedId) { e.preventDefault(); deleteSelected(); } }
    else if (!e.ctrlKey && !e.metaKey && !e.altKey) {
      if (key === "b") setTool("box");
      else if (key === "p") setTool("path");
      else if (key === "n") setTool("absent");
    }
  });
}

async function init() {
  buildPalette();
  bindControls();
  if (!window.fabric) {
    toast("Couldn't load Fabric.js from cdnjs. Check the network and reload.", true);
    return;
  }
  initCanvas();
  updateUi();
  try {
    const [cams, presets] = await Promise.all([api("api/cameras"), api("api/presets")]);
    state.cameras = cams;
    $("#camera").innerHTML = cams.map((c) => `<option value="${c.camera_id}">${escapeHtml(c.name)}</option>`).join("");
    selectCamera((cams.find((c) => c.camera_id === "warehouse_cam1") || cams[0]).camera_id);
    const holder = $("#presets");
    for (const p of presets) {
      const b = document.createElement("button");
      b.textContent = p.title;
      b.title = p.description;
      b.addEventListener("click", () => {
        loadSketch(p.sketch, { cameraId: p.camera_id || state.cameraId });
        // Demo mode: verdicts are pre-cached by scripts/warm_demo.py, so check right away.
        runSearch().then(() => { if ($("#demoMode").checked && cards.length) runVerify(); });
      });
      holder.appendChild(b);
    }
  } catch (err) {
    showState("error", `Can't reach the server (${err.message}).`);
  }
}

init();
