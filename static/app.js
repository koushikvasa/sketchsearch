"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const LABELS = ["person", "forklift", "robot", "transporter"];
// Colours come from the CSS tokens on :root (static/style.css). Your sketch is amber (--accent), real tracks are
// --info, empty areas are --danger. Labels are told apart by icon and tag, not colour.
const TOKENS = (() => {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(`--${name}`).trim();
  return Object.fromEntries(["bg", "surface", "surface-2", "border", "text", "muted", "accent", "success", "danger", "info"]
    .map((n) => [n, get(n)]));
})();
const COLORS = { person: TOKENS.accent, forklift: TOKENS.accent, robot: TOKENS.accent, transporter: TOKENS.accent };
const ABSENT_COLOR = TOKENS.danger;
const LABEL_ICONS = { person: "user", forklift: "forklift", robot: "bot", transporter: "shopping-cart" };
const LABEL_NAMES = { person: "Person", forklift: "Forklift", robot: "Robot", transporter: "Cart" };
const NOUNS = { person: "person", forklift: "forklift", robot: "robot", transporter: "cart" };
const PLURALS = { person: "people", forklift: "forklifts", robot: "robots", transporter: "carts" };
const MOVE_WARN = 0.25; // a drawn move longer than this is far more than 1.5 s of motion
const TOOLBAR = [
  { tool: "box", label: "person", text: "Person", icon: "user" },
  { tool: "box", label: "forklift", text: "Forklift", icon: "forklift" },
  { tool: "box", label: "robot", text: "Robot", icon: "bot" },
  { tool: "box", label: "transporter", text: "Cart", icon: "shopping-cart" },
  { tool: "path", text: "Arrow", icon: "move-up-right", title: "Press on a box and drag the way it moves" },
  { tool: "absent", label: "person", text: "Empty area", icon: "ban", title: "Mark an area where no one may be" },
];
const DEFAULT_SIZE = { person: [0.03, 0.13], forklift: [0.05, 0.1], robot: [0.05, 0.05], transporter: [0.07, 0.04] };
const ABSENT_SIZE = [0.25, 0.25];
const COMPONENTS = ["relations", "position", "motion", "size", "keyframe"];
const COMPONENT_NAMES = { relations: "layout", position: "position", motion: "movement", size: "size", keyframe: "end spot" };
// Plain words for "why it matched": [good, partly, poor].
const WHY_WORDS = {
  relations: ["same layout", "similar layout", "different layout"],
  motion: ["same movement", "similar movement", "different movement"],
  position: ["same spot", "slightly different position", "different position"],
  keyframe: ["ends in the same place", "ends nearby", "ends somewhere else"],
  size: ["same distance from the camera", "slightly nearer or farther", "different distance from the camera"],
};
const WHY_ORDER = ["relations", "motion", "position", "keyframe", "size"];
const TOP_N = 12;
const VERIFY_N = 10;
const AUTO_CHECK_N = 3; // the AI watches the top few right away; "Check more" does the rest
const AUTO_SEARCH_MS = 600; // re-search this long after the last sketch edit
const CHIP_W = 22; // px, a numbered result chip on the camera lanes
const FULL = { x: 0, y: 0, w: 1, h: 1 };
const ZOOM_PAD = 0.3; // padding around the action, as a fraction of its size
const ZOOM_MIN = 0.22; // never zoom in more than ~4.5x
const ZOOM_SKIP = 0.8; // action already fills the frame: don't zoom
const DRAW_STAGGER_MS = 150;
// What each kind of example means to a safety lead (scripts/make_presets.py picks the kinds).
const RISK_TAGS = {
  approach: { text: "Near-miss", icon: "triangle-alert" },
  meet: { text: "Aisle traffic", icon: "arrow-left-right" },
  absent: { text: "Lone worker", icon: "user" },
  group: { text: "Crowding", icon: "users" },
  cross: { text: "Aisle crossing", icon: "move-horizontal" },
};
const KF_HELP = {
  start: "Draw where things are at the start.",
  end: "Drag each box to where it ends up.",
};

const $ = (sel, el = document) => el.querySelector(sel);
const icon = (name) => `<i data-lucide="${name}" aria-hidden="true"></i>`;

// Lucide (from the CDN) swaps every <i data-lucide> for an SVG; watch the DOM so icons in new markup render too.
(() => {
  if (!window.lucide) return; // offline: buttons keep their text labels
  let queued = false;
  const run = () => { queued = false; window.lucide.createIcons(); };
  new MutationObserver(() => { if (!queued) { queued = true; requestAnimationFrame(run); } })
    .observe(document.documentElement, { childList: true, subtree: true });
  run();
})();
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
  checked: false, // AI check finished for the current results
  fromPreset: false, // the next search comes from an example chip
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
    fill: opts.absent ? hexA(ABSENT_COLOR, 0.14) : hexA(color, opts.ghost ? 0.04 : 0.16),
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
    fontFamily: "IBM Plex Sans, Segoe UI, sans-serif", fill: color, backgroundColor: hexA(TOKENS.bg, 0.78),
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
  const name = o.absent ? "empty area" : NOUNS[o.label] || o.label;
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
      canvas.add(tagFor(o.end ? `${NOUNS[o.label]} · end` : `${NOUNS[o.label]} · drag to set end`, b, color));
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
    left: p.x * W, top: p.y * H, width: 1, height: 1, fill: hexA(TOKENS.text, 0.08), stroke: TOKENS.text,
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
  for (const t of TOOLBAR) {
    const btn = document.createElement("button");
    btn.className = "tb-btn";
    btn.dataset.tool = t.tool;
    if (t.label) btn.dataset.label = t.label;
    btn.title = t.title || `Drag on the picture to add a ${t.text.toLowerCase()}`;
    btn.innerHTML = `${icon(t.icon)}${t.text}`;
    btn.addEventListener("click", () => setTool(t.tool, t.label));
    pal.appendChild(btn);
  }
}

function setLabel(label) {
  setTool("box", label);
}

const toolTips = new Set();
function setTool(tool, label = null) {
  state.tool = tool;
  if (label) state.label = label;
  render();
  if (toolTips.has(tool)) return; // explain each tool once
  toolTips.add(tool);
  if (tool === "path") toast("Arrow: press on a box and drag the way it moves. Keep it short: about 1.5 seconds of walking.");
  if (tool === "absent") toast("Empty area: drag a rectangle where nobody should be, e.g. around the forklift.");
}

function setTab(tab) {
  state.tab = tab;
  render();
}

function updateUi() {
  for (const b of document.querySelectorAll(".tb-btn")) {
    b.classList.toggle("active", b.dataset.tool === state.tool && (state.tool !== "box" || b.dataset.label === state.label));
  }
  for (const b of document.querySelectorAll(".kf")) b.classList.toggle("active", b.dataset.tab === state.tab);
  $("#caption").textContent = describeSketch(state.objects);
  $("#caption").title = $("#caption").textContent; // the caption shows at most four lines
  $("#moveWarn").hidden = !state.objects.some((o) => !o.absent && moveLength(o) > MOVE_WARN);
  setProgress();
  $("#kfHelp").textContent = KF_HELP[state.tab];
  $("#canvasEmpty").hidden = state.objects.length > 0;
  $("#hint").hidden = state.objects.filter((o) => !o.absent).length !== 1;
  $("#delete").disabled = !state.selectedId;
  $("#undo").disabled = state.history.length === 0;
  $("#clear").disabled = state.objects.length === 0;
  scheduleAutoSearch();
}

// Auto-search: any change to the sketch (or the camera filter) re-runs the search shortly after.
let autoTimer = null;
let searchedKey = null;
let holdAuto = false; // while a sketch is drawn in step by step, the caller searches once at the end

function sketchKey() {
  return JSON.stringify([state.objects.map((o) => [o.label, o.absent, o.start, o.end, o.path]), $("#scope").value]);
}

function scheduleAutoSearch() {
  clearTimeout(autoTimer);
  if (holdAuto || !document.body.classList.contains("mode-studio") || !state.objects.length) return;
  if (sketchKey() === searchedKey) return;
  setAutoNote(true);
  autoTimer = setTimeout(() => { if (!holdAuto && sketchKey() !== searchedKey) runSearch(); }, AUTO_SEARCH_MS);
}

function setAutoNote(busy) {
  const note = $("#autoNote");
  note.classList.toggle("busy", busy);
  $("span", note).textContent = busy ? "Updating…" : "Live";
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
    camera_ids: $("#scope").value ? [$("#scope").value] : null,
    text: state.text,
  };
}

// ---------------------------------------------------------------------------
// Search + results
// ---------------------------------------------------------------------------
let searchSeq = 0;
let lastSketch = null;
let lastResponse = null;
let verifyAbort = null;
let cards = [];
let animating = false;

function showState(kind, message = "") {
  stopAllVideos();
  resetVerify();
  cards = [];
  $("#stage").classList.remove("updating");
  placeChips();
  const box = $("#results");
  if (kind === "loading") {
    box.innerHTML = `<div class="state"><div class="spinner"></div>Sliding your sketch over every clip…</div>`;
    $("#status").textContent = "Searching…";
  } else if (kind === "empty") {
    box.innerHTML = `<div class="state">No clip shows all of these things at the same time.<br>
      Try removing a shape, or search in “All cameras”.</div>`;
    $("#status").textContent = "No matches";
  } else if (kind === "error") {
    box.innerHTML = `<div class="state error">Something went wrong: ${escapeHtml(message)}<br><button id="retry">Try again</button></div>`;
    $("#status").textContent = "";
    $("#retry").addEventListener("click", runSearch);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

async function runSearch() {
  if (!state.objects.length) return toast("Draw something first, or pick an example.");
  clearTimeout(autoTimer);
  searchedKey = sketchKey();
  const seq = ++searchSeq;
  const sketch = toSketch();
  const fromPreset = state.fromPreset;
  state.fromPreset = false;
  if (cards.length) { // keep the current match playing, dimmed, until the new results land
    $("#stage").classList.add("updating");
    $("#status").textContent = "Updating…";
  } else {
    showState("loading");
  }
  setAutoNote(true);
  try {
    const res = await api("api/search", { method: "POST", body: JSON.stringify({ sketch, top_n: TOP_N }) });
    if (seq !== searchSeq) return;
    renderResults(res, sketch, fromPreset);
  } catch (err) {
    if (seq === searchSeq) showState("error", err.message);
  } finally {
    if (seq === searchSeq) {
      setAutoNote(false);
      updateUi();
    }
  }
}

function renderResults(res, sketch, fromPreset = false) {
  lastSketch = sketch;
  lastResponse = res;
  if (!res.results.length) {
    showState("empty");
    return renderLanes(res);
  }
  stopAllVideos();
  resetVerify();
  $("#stage").classList.remove("updating");
  $("#results").innerHTML = `<div class="featured" id="featured"></div>`;
  const c = res.counts || {};
  $("#status").textContent = c.searched
    ? `${(c.great || 0) + (c.good || 0)} great or good in ${c.searched} clips · ${(res.elapsed_ms / 1000).toFixed(2)} s`
    : `${res.results.length} moments in ${(res.elapsed_ms / 1000).toFixed(2)} s`;
  const objects = sketch.objects.map((o) => ({
    ...o, start: o.start_box, end: o.end_box, path: o.path,
  }));
  cards = res.results.map((r, i) => buildCard(r, i + 1, objects, res.results.length));
  $("#reportBtn").disabled = false;
  $("#copyLinkBtn").disabled = false;
  renderLanes(res);
  featureCard(cards[0]);
  syncCaseButtons();
  updateShareHash();
  state.checked = false;
  setProgress();
  for (const card of cards) {
    api(`api/segments/${encodeURIComponent(card.result.segment_id)}`)
      .then((seg) => { card.segment = seg; updateZoom(card); })
      .catch(() => { /* overlay just shows the sketch */ });
  }
  // Demo-mode examples have saved verdicts (scripts/warm_demo.py), so check all of them; anything else costs live calls.
  runVerify($("#demoMode").checked && fromPreset ? VERIFY_N : AUTO_CHECK_N);
}

function buildCard(r, rank, objects, total) {
  // importNode, not cloneNode: off-page cards must resolve their relative clip URLs against this page.
  const el = document.importNode($("#cardTpl").content.firstElementChild, true);
  const video = $("video", el);
  if (r.frame_url) {
    video.src = r.clip_url;
    video.poster = r.frame_url;
  } else {
    video.src = `${r.clip_url}#t=${(r.window[0] + r.window[1]) / 2}`; // no keyframe image: show a frame of the clip
  }
  const m = matchLabel(r.score);
  const match = $(".match", el);
  match.innerHTML = `${m.text.split(" ")[0]}<span class="word"> match</span>`;
  match.className = `match ${m.cls}`;
  $(".where", el).textContent = whereLabel(r);
  $(".rank", el).textContent = `${rank} of ${total}`;
  $(".score", el).textContent = r.score.toFixed(2);
  $(".explain", el).textContent = r.explanation || "";
  $(".explain", el).title = r.explanation || ""; // clamped to three lines beside the player
  $(".why", el).innerHTML = whyMatched(r).map((w) =>
    `<li class="${w.cls}"><span class="mark" aria-hidden="true">${w.mark}</span>${escapeHtml(w.text)}</li>`).join("");

  const abs = $(".absence", el);
  if (r.absence_ok !== null && r.absence_ok !== undefined) {
    abs.hidden = false;
    abs.className = `badge absence ${r.absence_ok ? "ok" : "bad"}`;
    abs.textContent = r.absence_ok ? "Empty area: nobody there" : "Empty area: someone is there";
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
    zoom: FULL, zoomOn: true, verdict: null, featured: false, rank,
  };
  updateZoom(card);
  const zoomBtn = $(".zoom-btn", el);
  zoomBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    card.zoomOn = !card.zoomOn;
    applyZoom(card);
  });
  const win = $(".timeline .win", el);
  win.style.left = `${(r.window[0] / duration) * 100}%`;
  win.style.width = `${((r.window[1] - r.window[0]) / duration) * 100}%`;

  // Click (or tap) the video to pause or play; swipe left or right on a touch screen for the next match.
  const media = $(".media", el);
  let touchX = null, swiped = false;
  media.addEventListener("touchstart", (e) => { touchX = e.touches[0].clientX; }, { passive: true });
  media.addEventListener("touchend", (e) => {
    const dx = touchX === null ? 0 : e.changedTouches[0].clientX - touchX;
    touchX = null;
    if (Math.abs(dx) > 48) { swiped = true; stepFeatured(dx < 0 ? 1 : -1); }
  }, { passive: true });
  media.addEventListener("click", () => {
    if (swiped) { swiped = false; return; }
    togglePlay(card);
  });
  for (const [sel, d] of [[".nav.prev", -1], [".nav.next", 1]]) {
    $(sel, el).addEventListener("click", (e) => { e.stopPropagation(); stepFeatured(d); });
  }
  const numbers = $(".numbers-btn", el);
  numbers.addEventListener("click", () => {
    const on = el.classList.toggle("show-numbers");
    numbers.setAttribute("aria-expanded", String(on));
    numbers.textContent = on ? "Show reasons" : "Show numbers";
  });
  video.addEventListener("seeked", () => { if (!card.playing) drawOverlay(card, video.currentTime); });
  $(".more", el).addEventListener("click", () => moreLikeThis(r));
  const caseBtn = $(".case-add", el);
  caseBtn.dataset.key = caseKey(r);
  caseBtn.addEventListener("click", () => toggleCase(r, card.verdict, describeSketch(objects)));
  return card;
}

function togglePlay(card) {
  card.pinned = !card.playing;
  card.playing ? pauseCard(card) : playCard(card);
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
    ctx.fillStyle = hexA(ABSENT_COLOR, r.absence_ok === false ? 0.22 : 0.08);
    ctx.fillRect(X(b.x), Y(b.y), SX(b.w), SY(b.h));
    ctx.strokeRect(X(b.x), Y(b.y), SX(b.w), SY(b.h));
    label(ctx, `no ${o.label}`, X(b.x) + 3, Y(b.y + b.h) - 4, ABSENT_COLOR);
    ctx.restore();
  }

  for (const o of card.objects.filter((o) => !o.absent)) {
    const color = TOKENS.accent; // the sketch
    const realColor = TOKENS.info; // what the camera actually saw
    const pts = drawnPath(o);
    const end = o.end || o.start;

    // Ghost replay: the whole drawn path faintly, the part already travelled brighter.
    if (pts.length >= 2 && pathLength(pts) > 0.005) {
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = hexA(color, 0.18);
      ctx.lineWidth = 7;
      polyline(ctx, pts.map(([x, y]) => [X(x), Y(y)]));
      const trail = [];
      for (let k = 0; k <= 24; k++) trail.push(pointAlong(pts, (u * k) / 24));
      ctx.strokeStyle = hexA(color, 0.4); // ghost path: amber at 40%
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
    ctx.strokeStyle = color;
    ctx.fillStyle = hexA(color, 0.08);
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
      ctx.strokeStyle = hexA(TOKENS.text, 0.45);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(X(gx), Y(gy));
      ctx.lineTo(X(rx), Y(ry));
      ctx.stroke();
      ctx.lineWidth = 2;
      ctx.strokeStyle = realColor;
      ctx.strokeRect(X(real.x), Y(real.y), SX(real.w), SY(real.h));
      label(ctx, NOUNS[o.label] || o.label, X(real.x), Y(real.y) - 4, realColor);
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
  ctx.font = "600 12px 'IBM Plex Sans', 'Segoe UI', sans-serif";
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = hexA(TOKENS.bg, 0.8);
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
    toast(`Copied this moment into the sketch (${whereLabel(r)}). Change anything and the results follow.`);
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
  btn.innerHTML = `${icon("zoom-in")}${card.zoomOn ? `${(1 / card.zoom.w).toFixed(1)}×` : "off"}`;
  drawOverlay(card, card.started ? card.video.currentTime : card.posterTime);
}

// ---------------------------------------------------------------------------
// AI verification (server-sent events)
// ---------------------------------------------------------------------------
const VERDICT_UI = {
  YES: { cls: "yes", text: "Confirmed", icon: "check" },
  NO: { cls: "no", text: "Not a match", icon: "x" },
  UNSURE: { cls: "unsure", text: "Not sure", icon: "circle-help" },
};

function resetVerify() {
  if (verifyAbort) verifyAbort.abort();
  verifyAbort = null;
  $("#verifySummary").hidden = true;
  verifyLabel();
}

/** "Check with AI" → "Check 7 more" → "Check again", depending on what the AI has already watched. */
function verifyLabel() {
  const btn = $("#verifyBtn");
  const busy = !!verifyAbort;
  const left = cards.slice(0, VERIFY_N).filter((c) => !c.verdict).length;
  const any = cards.some((c) => c.verdict);
  const text = busy ? "AI is checking…" : !any ? "Check with AI" : left ? `Check ${left} more` : "Check again";
  btn.innerHTML = `${icon(busy ? "loader" : any && !left ? "refresh-cw" : "scan-search")}<span class="label">${text}</span>`;
  btn.disabled = busy || !cards.length;
  btn.classList.toggle("busy", busy);
}

function checkMore() {
  if (!cards.length || verifyAbort) return;
  runVerify(VERIFY_N, cards.slice(0, VERIFY_N).every((c) => c.verdict));
}

function setVerdict(card, v) {
  card.verdict = v;
  const ui = VERDICT_UI[v.verdict] || VERDICT_UI.UNSURE;
  const badge = $(".badge.verify", card.el);
  badge.hidden = false;
  badge.className = `badge verify ${ui.cls} landed`;
  badge.innerHTML = `${icon(ui.icon)}${ui.text}`;
  badge.title = `${v.model || "video model"}${v.cached ? " (cached)" : ""}`;
  $(".verify-reason", card.el).textContent = v.reason;
  const stamp = $(".stamp", card.el);
  stamp.hidden = false;
  stamp.className = `stamp ${ui.cls}`;
  stamp.innerHTML = `${icon(ui.icon)}${ui.text}`;
  stamp.title = v.reason;
  // Cached verdicts arrive all at once: reveal them in rank order instead of in one flash.
  stamp.style.animationDelay = v.cached ? `${Math.min(card.rank - 1, 10) * 70}ms` : "0ms";
  stamp.style.animationName = "none";
  void stamp.offsetWidth; // restart the reveal
  stamp.style.animationName = "";
  card.el.classList.toggle("rejected", v.verdict === "NO");
  updateMarks();
  updateCaseVerdict(card.result, v);
}

function updateVerifySummary() {
  const el = $("#verifySummary");
  const done = cards.filter((c) => c.verdict);
  const pending = cards.filter((c) => c.checking).length;
  el.hidden = !done.length && !pending;
  const n = (v) => done.filter((c) => c.verdict.verdict === v).length;
  if (pending) {
    el.innerHTML = `${icon("loader")}AI is watching ${pending} clip${pending === 1 ? "" : "s"}…` +
      (done.length ? ` <span class="muted small">${n("YES")} confirmed so far</span>` : "");
  } else if (done.length) {
    const extra = [n("NO") && `${n("NO")} rejected`, n("UNSURE") && `${n("UNSURE")} unsure`].filter(Boolean).join(", ");
    el.innerHTML = `AI confirmed <b>${n("YES")}</b> of ${done.length}` + (extra ? ` <span class="muted small">(${extra})</span>` : "");
    state.checked = true;
    setProgress();
  }
}

/** Ask the video AI about the top `n` matches (only the unchecked ones unless `again`), streaming verdicts in. */
async function runVerify(n = VERIFY_N, again = false) {
  if (!lastSketch || !cards.length) return;
  const targets = cards.slice(0, n).filter((c) => again || !c.verdict);
  if (!targets.length) return;
  if (verifyAbort) verifyAbort.abort();
  const controller = new AbortController();
  verifyAbort = controller;
  verifyLabel();
  for (const c of targets) {
    c.verdict = null;
    c.checking = true;
    c.el.classList.remove("rejected");
    const b = $(".badge.verify", c.el);
    b.hidden = false;
    b.className = "badge verify checking";
    b.innerHTML = `${icon("loader")}Watching…`;
    b.title = "";
    $(".verify-reason", c.el).textContent = "The video AI is watching this clip.";
    const st = $(".stamp", c.el);
    st.hidden = false;
    st.className = "stamp checking";
    st.style.animationDelay = "0ms";
    st.innerHTML = `${icon("loader")}Checking`;
  }
  updateMarks();
  updateVerifySummary();
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
        if (!card) continue;
        card.checking = false;
        setVerdict(card, payload);
        updateVerifySummary();
      }
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    toast(`AI check failed: ${err.message}`, true);
  } finally {
    if (verifyAbort === controller) {
      verifyAbort = null;
      for (const c of targets) {
        c.checking = false;
        if (c.verdict) continue;
        $(".stamp", c.el).hidden = true;
        const b = $(".badge.verify", c.el);
        b.className = "badge verify";
        b.textContent = "Not checked yet";
        $(".verify-reason", c.el).textContent = "Press C or “Check with AI” to have the video AI watch it.";
      }
      verifyLabel();
      updateMarks();
      updateVerifySummary();
    }
  }
}

// ---------------------------------------------------------------------------
// Words -> Sketch and Diagram -> Sketch
// ---------------------------------------------------------------------------
function toStateObject(o) {
  return {
    id: o.id, label: o.label, absent: !!o.absent,
    start: o.start_box, end: o.absent ? null : o.end_box || null, path: o.absent ? null : o.path || null,
  };
}

async function animateSketch(objects) {
  holdAuto = true; // the caller searches once the whole sketch is in
  try {
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
  } finally {
    holdAuto = false;
  }
}

async function sketchFromText(e) {
  e.preventDefault();
  const text = $("#sketchText").value.trim();
  if (text.length < 2) return toast("Describe the moment first, e.g. “two people walking toward each other”.");
  const preset = presetForText(text);
  if (preset) return runPreset(preset);
  enterStudio();
  $("#caption").textContent = "Reading your description…";
  const btn = $("#sketchIt");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Sketching…";
  try {
    const res = await api("api/text-to-sketch", { method: "POST", body: JSON.stringify({ text }) });
    await animateSketch(res.sketch.objects);
    state.text = text;
    toast("Here's the sketch. Searching every clip now…");
    await runSearch();
  } catch (err) {
    toast(`Couldn't sketch that: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.textContent = "Find it";
  }
}

async function uploadDiagram() {
  const input = $("#diagramFile");
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  enterStudio();
  const btn = $("#diagramBtn");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.innerHTML = `${icon("loader")}<span class="label">Reading your drawing…</span>`;
  $("#caption").textContent = "Reading your drawing…";
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
    toast(`I found ${body.sketch.objects.length} things in your drawing (shown faintly behind). ` +
          "Fix anything that's off and the results follow.");
  } catch (err) {
    toast(`Couldn't read the diagram: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.innerHTML = `${icon("upload")}<span class="label">Upload a drawing</span>`;
  }
}


// ---------------------------------------------------------------------------
// Plain-language helpers (caption, match labels, why it matched)
// ---------------------------------------------------------------------------
function moveLength(o) {
  const pts = drawnPath(o);
  return Math.hypot(pts[pts.length - 1][0] - pts[0][0], pts[pts.length - 1][1] - pts[0][1]);
}

function placeWords([x, y]) {
  if (y < 0.25) return "at the far end";
  if (y > 0.65) return "near the camera";
  return x < 0.4 ? "on the left" : x > 0.6 ? "on the right" : "in the middle";
}

function moveWords(o) {
  const pts = drawnPath(o);
  const [dx, dy] = [pts[pts.length - 1][0] - pts[0][0], pts[pts.length - 1][1] - pts[0][1]];
  const len = Math.hypot(dx, dy);
  if (len < 0.03) return null;
  const parts = [];
  if (Math.abs(dx) >= 0.4 * len) parts.push(dx > 0 ? "right" : "left");
  if (Math.abs(dy) >= 0.4 * len) parts.push(dy < 0 ? "away from the camera" : "toward the camera");
  return `${o.label === "person" ? "walking" : "moving"} ${parts.join(" and ")}`;
}

/** "A person on the left walking right, toward a forklift; nobody inside the empty area." */
function describeSketch(objects) {
  const present = objects.filter((o) => !o.absent);
  if (!present.length && !objects.length) return "Pick a shape below, then drag on the picture.";
  const counts = {};
  present.forEach((o) => { counts[o.label] = (counts[o.label] || 0) + 1; });
  const end = (o) => { const p = drawnPath(o); return p[p.length - 1]; };
  const clauses = [];
  const used = new Set();
  // A still group of the same label reads better as one phrase.
  for (const [lbl, n] of Object.entries(counts)) {
    const group = present.filter((o) => o.label === lbl && !moveWords(o));
    if (n >= 3 && group.length === n) {
      const cs = group.map((o) => center(o.start));
      const mid = [cs.reduce((a, c) => a + c[0], 0) / n, cs.reduce((a, c) => a + c[1], 0) / n];
      clauses.push(`a group of ${n} ${PLURALS[lbl]} together ${placeWords(mid)}`);
      group.forEach((o) => used.add(o.id));
    }
  }
  for (const o of present) {
    if (used.has(o.id)) continue;
    let text = `a ${NOUNS[o.label]} ${placeWords(center(o.start))}`;
    const mv = moveWords(o);
    const drawnMove = !!(o.end || (o.path && o.path.length >= 2));
    if (!mv) {
      if (drawnMove) text += o.label === "person" ? ", standing still" : ", not moving";
    } else {
      text += ` ${mv}`;
      for (const other of present) {
        if (other === o) continue;
        const before = Math.hypot(center(o.start)[0] - center(other.start)[0], center(o.start)[1] - center(other.start)[1]);
        const after = Math.hypot(end(o)[0] - end(other)[0], end(o)[1] - end(other)[1]);
        const name = counts[other.label] > 1 ? `another ${NOUNS[other.label]}` : `the ${NOUNS[other.label]}`;
        if (after < before - 0.03) { text += `, toward ${name}`; break; }
        if (after > before + 0.03) { text += `, away from ${name}`; break; }
      }
    }
    clauses.push(text);
  }
  for (const z of objects.filter((o) => o.absent)) {
    const anchor = present.find((o) => {
      const [cx, cy] = center(o.start);
      return cx >= z.start.x && cx <= z.start.x + z.start.w && cy >= z.start.y && cy <= z.start.y + z.start.h;
    });
    const who = z.label === "person" ? "nobody" : `no ${NOUNS[z.label]}`;
    clauses.push(anchor ? `${who} near the ${NOUNS[anchor.label]}` : `${who} inside the empty area`);
  }
  const text = clauses.join("; ");
  return text ? text[0].toUpperCase() + text.slice(1) + "." : "Pick a shape below, then drag on the picture.";
}

function matchLabel(score) {
  if (score >= 0.85) return { text: "Great match", cls: "great" };
  if (score >= 0.7) return { text: "Good match", cls: "good" };
  return { text: "Weak match", cls: "weak" };
}

function shortCamera(cameraId) {
  return cameraName(cameraId).replace(/^warehouse cam\s*/i, "Camera ");
}

function clock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function whereLabel(r) {
  return `${shortCamera(r.camera_id)} · ${clock(r.start + r.window[0])}`;
}

function whyMatched(r) {
  const out = [];
  for (const name of WHY_ORDER) {
    if (!(name in r.components)) continue;
    const v = r.components[name];
    const [good, partly, poor] = WHY_WORDS[name];
    if (v >= 0.85) out.push({ cls: "yes", mark: icon("check"), text: good });
    else if (v >= 0.6) out.push({ cls: "some", mark: icon("minus"), text: partly });
    else out.push({ cls: "no", mark: icon("x"), text: poor });
  }
  if (r.absence_ok === true) out.push({ cls: "yes", mark: icon("check"), text: "nobody in the empty area" });
  if (r.absence_ok === false) out.push({ cls: "no", mark: icon("x"), text: "someone is in the empty area" });
  return out.slice(0, 5);
}

// ---------------------------------------------------------------------------
// Landing -> studio, featured result, progress
// ---------------------------------------------------------------------------
function enterStudio() {
  if (document.body.classList.contains("mode-studio")) return;
  $("#topSearch").append($("#textForm"), $(".hero-actions")); // the search bar moves up into the header
  document.body.classList.replace("mode-landing", "mode-studio");
  setProgress();
}

function goHome() {
  const current = cards.find((c) => c.featured);
  if (current) pauseCard(current);
  $("#examples").before($("#textForm"), $(".hero-actions"));
  document.body.classList.replace("mode-studio", "mode-landing");
  $("#sketchText").focus();
  setProgress();
}

function startDrawing() {
  enterStudio();
  if (state.objects.length) {
    snapshot();
    state.objects = [];
    state.selectedId = null;
    state.text = null;
    $("#sketchText").value = "";
  }
  setTool("box", "person");
  toast("Choose Person, Forklift, Robot or Cart, then drag a box on the picture. Add an Arrow to show movement.");
}

/** Put a match on the big player and start it; the lanes mark it as the current one. */
function featureCard(card) {
  const current = cards.find((c) => c.featured);
  if (!card || (current === card && card.el.isConnected)) return;
  if (current) {
    pauseCard(current);
    current.pinned = false;
    current.featured = false;
    current.el.remove();
  }
  card.featured = true;
  card.pinned = true;
  $("#featured").appendChild(card.el);
  requestAnimationFrame(() => {
    applyZoom(card);
    if (card.featured && document.body.classList.contains("mode-studio")) playCard(card);
  });
  updateMarks();
}

function stepFeatured(d) {
  const i = cards.findIndex((c) => c.featured);
  if (i >= 0) featureCard(cards[(i + d + cards.length) % cards.length]);
}

function setProgress() {
  const studio = document.body.classList.contains("mode-studio");
  const done = {
    describe: studio,
    sketch: studio && state.objects.length > 0,
    find: cards.length > 0,
    check: state.checked,
  };
  let activeSet = false;
  for (const li of document.querySelectorAll("#steps li")) {
    const isDone = done[li.dataset.step];
    li.classList.toggle("done", isDone);
    const active = !isDone && !activeSet;
    li.classList.toggle("active", active);
    if (active) activeSet = true;
  }
}

// ---------------------------------------------------------------------------
// Examples, voice, quick guide
// ---------------------------------------------------------------------------
function presetForText(text) {
  const t = text.trim().toLowerCase();
  return (state.presets || []).find((p) => (p.chip || "").toLowerCase() === t);
}

async function runPreset(p) {
  enterStudio();
  $("#sketchText").value = p.chip || p.title;
  if (p.camera_id) selectCamera(p.camera_id);
  await animateSketch(p.sketch.objects);
  state.text = null;
  state.fromPreset = true;
  await runSearch();
}

function initMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = $("#micBtn");
  if (!SR) return; // stays hidden
  btn.hidden = false;
  let rec = null;
  btn.addEventListener("click", () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = "en-US";
    rec.interimResults = true;
    let finalText = "";
    rec.onresult = (e) => {
      const said = Array.from(e.results).map((r) => r[0].transcript).join(" ");
      $("#sketchText").value = said;
      if (e.results[e.results.length - 1].isFinal) finalText = said;
    };
    rec.onerror = (e) => toast(`Couldn't hear that (${e.error}). You can type instead.`, true);
    rec.onend = () => {
      btn.classList.remove("listening");
      rec = null;
      if (finalText.trim()) $("#textForm").requestSubmit();
    };
    btn.classList.add("listening");
    rec.start();
  });
}

function initGuide() {
  const dlg = $("#guide");
  const slides = [...document.querySelectorAll("#slides .slide")];
  const dots = $("#guideDots");
  dots.innerHTML = slides.map(() => "<span></span>").join("");
  let i = 0;
  const show = (n) => {
    i = clamp(n, 0, slides.length - 1);
    slides.forEach((s, k) => s.classList.toggle("active", k === i));
    [...dots.children].forEach((d, k) => d.classList.toggle("on", k === i));
    $("#guidePrev").disabled = i === 0;
    $("#guideNext").textContent = i === slides.length - 1 ? "Got it" : "Next";
  };
  $("#helpBtn").addEventListener("click", () => { show(0); dlg.showModal(); });
  $("#guideClose").addEventListener("click", () => dlg.close());
  $("#guidePrev").addEventListener("click", () => show(i - 1));
  $("#guideNext").addEventListener("click", () => (i === slides.length - 1 ? dlg.close() : show(i + 1)));
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") show(i + 1);
    if (e.key === "ArrowLeft") show(i - 1);
  });
}


// ---------------------------------------------------------------------------
// When it happened: one lane per camera on a shared time axis
// ---------------------------------------------------------------------------
// Faint ticks = every great or good match (how often), numbered chips = the top results (click to play).
const startOf = (r) => r.start + r.window[0];

function laneDuration(res) {
  const cams = state.cameras.map((c) => c.duration || 0);
  const ends = (res?.results || []).map((r) => r.end);
  return Math.max(res?.how_often?.duration || 0, ...cams, ...ends, 60);
}

function renderLanes(res) {
  const h = res?.how_often;
  $("#insight").textContent = h ? h.insight : "Every match across all cameras shows up on this timeline.";
  const dur = laneDuration(res);
  const pct = (t) => `${(clamp(t / dur, 0, 1) * 100).toFixed(2)}%`;
  const scope = $("#scope").value;
  const counts = Object.fromEntries((h?.per_camera || []).map((c) => [c.camera_id, c.great + c.good]));
  const lanes = state.cameras.map((cam) => {
    const ticks = (h?.marks || []).filter((m) => m.camera_id === cam.camera_id)
      .map((m) => `<i class="tick${m.great ? " great" : ""}" style="left:${pct(m.t)}"></i>`).join("");
    const off = scope && scope !== cam.camera_id ? " off" : "";
    const n = counts[cam.camera_id] ?? 0;
    return `<div class="lane${off}" data-camera="${escapeHtml(cam.camera_id)}">
      <span class="lane-name">${escapeHtml(shortCamera(cam.camera_id))}<b title="${n} great or good matches">${n}</b></span>
      <div class="track">${ticks}</div></div>`;
  }).join("");
  const minutes = [];
  for (let m = 0; m * 60 <= dur; m++) minutes.push(`<span style="left:${pct(m * 60)}">${m}:00</span>`);
  $("#lanesBody").innerHTML = `${lanes}<div class="axis"><span class="lane-name"></span><div class="track">${minutes.join("")}</div></div>`;
  placeChips();
}

/** Numbered chips at each top result's start; chips that would overlap are nudged apart (the thin bar under each
 *  chip stays at the true time). */
function placeChips() {
  hideTip();
  for (const el of document.querySelectorAll("#lanesBody .chip-mark, #lanesBody .win-mark")) el.remove();
  if (!cards.length) return;
  const dur = laneDuration(lastResponse);
  for (const lane of document.querySelectorAll("#lanesBody .lane")) {
    const track = $(".track", lane);
    const w = track.clientWidth;
    if (!w) continue;
    const mine = cards.filter((c) => c.result.camera_id === lane.dataset.camera)
      .sort((a, b) => startOf(a.result) - startOf(b.result));
    const xs = [];
    let right = -Infinity;
    for (const c of mine) {
      const x = Math.max(clamp((startOf(c.result) / dur) * w - CHIP_W / 2, 0, w - CHIP_W), right + 2);
      xs.push(x);
      right = x + CHIP_W;
    }
    let left = w + 2;
    for (let i = xs.length - 1; i >= 0; i--) { xs[i] = Math.min(xs[i], left - 2 - CHIP_W); left = xs[i]; }
    if (xs[0] < 0) xs.forEach((_, i) => { xs[i] = mine.length > 1 ? (i * (w - CHIP_W)) / (mine.length - 1) : 0; }); // too many to fit: spread evenly
    mine.forEach((c, i) => {
      const r = c.result;
      const x0 = (startOf(r) / dur) * w, x1 = ((r.start + r.window[1]) / dur) * w;
      track.insertAdjacentHTML("beforeend",
        `<i class="win-mark" style="left:${x0.toFixed(1)}px;width:${Math.max(3, x1 - x0).toFixed(1)}px"></i>` +
        `<button class="chip-mark ${matchLabel(r.score).cls}" data-rank="${c.rank}" style="left:${xs[i].toFixed(1)}px">${c.rank}</button>`);
    });
  }
  updateMarks();
}

function updateMarks() {
  for (const b of document.querySelectorAll("#lanesBody .chip-mark")) {
    const c = cards[b.dataset.rank - 1];
    if (!c) continue;
    const v = c.verdict?.verdict;
    b.classList.toggle("current", c.featured);
    b.classList.toggle("yes", v === "YES");
    b.classList.toggle("no", v === "NO");
    b.classList.toggle("unsure", v === "UNSURE");
    b.classList.toggle("checking", !!c.checking);
    b.setAttribute("aria-current", c.featured ? "true" : "false");
    b.setAttribute("aria-label", `Match ${c.rank}: ${matchLabel(c.result.score).text}, ${whereLabel(c.result)}` +
      (v ? `, AI: ${VERDICT_UI[v]?.text || v}` : ""));
  }
}

function showTip(btn) {
  const c = cards[btn.dataset.rank - 1];
  if (!c) return;
  const r = c.result, v = c.verdict;
  const tip = $("#laneTip");
  const ai = v ? `${VERDICT_UI[v.verdict]?.text || v.verdict}: ${v.reason}` : c.checking ? "The AI is watching it now." : "Not checked by AI yet.";
  tip.innerHTML = `${r.frame_url ? `<img src="${escapeHtml(r.frame_url)}" alt="">` : ""}
    <div><b>${c.rank}. ${matchLabel(r.score).text}</b> · ${escapeHtml(whereLabel(r))}
    <p class="${v ? v.verdict.toLowerCase() : "muted"}">${escapeHtml(ai)}</p></div>`;
  tip.hidden = false;
  const panel = $("#lanes").getBoundingClientRect(), b = btn.getBoundingClientRect();
  tip.style.left = `${clamp(b.left + b.width / 2 - panel.left - tip.offsetWidth / 2, 8, panel.width - tip.offsetWidth - 8)}px`;
  tip.style.bottom = `${panel.bottom - b.top + 8}px`;
}

function hideTip() {
  $("#laneTip").hidden = true;
}

function initLanes() {
  const body = $("#lanesBody");
  const chip = (e) => e.target.closest?.(".chip-mark");
  body.addEventListener("click", (e) => { const b = chip(e); if (b) featureCard(cards[b.dataset.rank - 1]); });
  body.addEventListener("pointerover", (e) => { const b = chip(e); if (b && e.pointerType === "mouse") showTip(b); });
  body.addEventListener("pointerout", (e) => { if (chip(e)) hideTip(); });
  body.addEventListener("focusin", (e) => { const b = chip(e); if (b) showTip(b); });
  body.addEventListener("focusout", hideTip);
  let frame = 0;
  new ResizeObserver(() => { cancelAnimationFrame(frame); frame = requestAnimationFrame(placeChips); }).observe(body);
}

// ---------------------------------------------------------------------------
// Share link: the description + sketch + scope live in the URL hash
// ---------------------------------------------------------------------------
function b64urlEncode(text) {
  const bytes = new TextEncoder().encode(text);
  let bin = "";
  bytes.forEach((b) => { bin += String.fromCharCode(b); });
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s) {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
  return new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0)));
}

function shareState() {
  const r = (b) => (b ? [b.x, b.y, b.w, b.h].map(round4) : undefined);
  return {
    v: 1,
    t: $("#sketchText").value.trim() || undefined,
    c: $("#scope").value || undefined,
    o: state.objects.map((o) => ({
      i: o.id, l: o.label, a: o.absent ? 1 : undefined, s: r(o.start), e: r(o.end),
      p: o.path ? o.path.map(([x, y]) => [round4(x), round4(y)]) : undefined,
    })),
  };
}

function shareUrl() {
  const url = new URL(window.location.href);
  url.hash = `s=${b64urlEncode(JSON.stringify(shareState()))}`;
  return url.toString();
}

function updateShareHash() {
  if (!state.objects.length) return;
  history.replaceState(null, "", shareUrl());
}

function restoreFromHash() {
  const m = /^#s=([A-Za-z0-9_-]+)/.exec(window.location.hash);
  if (!m) return false;
  let data;
  try {
    data = JSON.parse(b64urlDecode(m[1]));
  } catch {
    toast("That link looks broken, so here's a fresh start.", true);
    return false;
  }
  const box = (a) => (a ? { x: a[0], y: a[1], w: a[2], h: a[3] } : null);
  enterStudio();
  $("#sketchText").value = data.t || "";
  if (data.c && [...$("#scope").options].some((o) => o.value === data.c)) {
    $("#scope").value = data.c;
    selectCamera(data.c);
  }
  loadSketch({
    objects: (data.o || []).map((o) => ({
      id: o.i, label: o.l, absent: !!o.a, start_box: box(o.s), end_box: box(o.e), path: o.p || null,
    })),
  });
  toast("Opened a shared search.");
  runSearch();
  return true;
}

async function copyLink() {
  const url = shareUrl();
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    toast("Link copied. Anyone who opens it sees this sketch and its results.");
  } catch {
    window.prompt("Copy this link:", url);
  }
}

// ---------------------------------------------------------------------------
// Incident report (printable page, "Save as PDF" via the browser's print)
// ---------------------------------------------------------------------------
function sketchImage() {
  rendering = true;
  canvas.discardActiveObject();
  const bg = canvas.backgroundColor;
  canvas.backgroundColor = "#11141a"; // the faint camera picture needs the dark ground it is drawn on
  canvas.renderAll();
  try {
    return canvas.toDataURL({ format: "png", multiplier: 1 });
  } catch {
    return ""; // a cross-origin background would taint the canvas; the report still works without it
  } finally {
    canvas.backgroundColor = bg;
    canvas.renderAll();
    rendering = false;
  }
}

function absUrl(u) {
  return u ? new URL(u, document.baseURI).href : "";
}

/** A printable report in a new window. kind: "search" (current results), "sweep" (last safety sweep) or "case"
 *  (the clips saved as evidence). */
async function openReport(kind = "search") {
  let items; // [{r: result, v: {verdict, reason} | null, from}]
  if (kind === "case") {
    items = caseItems.map((i) => ({ r: i.r, v: i.v, from: i.from }));
  } else if (kind === "sweep") {
    items = sweepItems();
  } else {
    if (!lastResponse || !cards.length) return toast("Run a search first, then create the report.");
    const verified = cards.filter((c) => c.verdict);
    items = (verified.length ? verified : cards).slice(0, 6).map((c) => ({ r: c.result, v: c.verdict, from: "" }));
  }
  if (!items.length) return toast("Nothing to report yet.");
  const w = window.open("", "_blank"); // open first, inside the click, so pop-up blockers allow it
  if (!w) return toast("Your browser blocked the report window. Allow pop-ups for this page.", true);
  w.document.write("<p style='font-family:sans-serif;padding:2rem'>Preparing the report…</p>");
  if (!state.health) {
    try { state.health = await api("api/health"); } catch { /* settings just show "-" */ }
  }
  const health = state.health || {};
  const checked = items.filter((i) => i.v && i.v.verdict !== "SKIPPED");
  const confirmed = checked.filter((i) => i.v.verdict === "YES").length;
  const cams = new Set(items.map((i) => i.r.camera_id)).size;

  const title = { search: "Search report", sweep: "Safety sweep report", case: "Case report" }[kind];
  let lookedFor, counts;
  if (kind === "search") {
    const src = sketchImage();
    lookedFor = `<div class="sketch">${src ? `<img src="${src}" alt="The sketch">` : ""}
      <p class="caption">${escapeHtml(describeSketch(state.objects))}</p></div>
      <p>${escapeHtml(lastResponse.how_often?.insight || "")}</p>`;
    counts = [[lastResponse.counts.searched, "clips searched"], [lastResponse.counts.great + lastResponse.counts.good, "great or good matches"],
      [checked.length ? confirmed : "-", checked.length ? `confirmed by AI (of ${checked.length} checked)` : "confirmed by AI (not checked yet)"]];
  } else if (kind === "sweep") {
    const rep = agent.lastReport;
    lookedFor = `<p class="caption">${escapeHtml(agent.goal || "")}</p><p class="muted">${escapeHtml(agent.summary || "")}</p>`;
    counts = [[rep.stats.sketch_runs, "searches run by the agent"], [rep.stats.clips_checked, "clips checked by AI"],
      [rep.stats.confirmed, "confirmed moments"]];
  } else {
    const froms = [...new Set(items.map((i) => i.from).filter(Boolean))];
    lookedFor = `<p class="caption">${items.length} clip${items.length === 1 ? "" : "s"} saved as evidence</p>` +
      (froms.length ? `<p class="muted">Found with: ${froms.map(escapeHtml).join(" · ")}</p>` : "");
    counts = [[items.length, "clips in the case"], [cams, `camera${cams === 1 ? "" : "s"}`],
      [checked.length ? confirmed : "-", checked.length ? `confirmed by AI (of ${checked.length} checked)` : "confirmed by AI (not checked yet)"]];
  }
  const settings = [
    ...(kind === "search" ? [["Description", $("#sketchText").value.trim() || "(drawn by hand)"],
      ["Searched in", $("#scope").selectedOptions[0]?.textContent || "All cameras"]] : []),
    ["Footage", health.index ? `${health.index.segments} clips of 4 s from ${health.index.cameras} cameras (${health.index.source})` : "-"],
    ["AI checker", (health.verify?.models || [])[0] || "-"],
    ["Mode", $("#demoMode").checked ? "Demo mode (pre-computed AI checks)" : "Live"],
  ];
  const clip = ({ r, v, from }) => {
    const m = matchLabel(r.score);
    const verdict = v && v.verdict !== "SKIPPED" ? `${VERDICT_UI[v.verdict]?.text || v.verdict}: ${escapeHtml(v.reason)}` : "Not checked by AI";
    const img = r.frame_url ? `<img src="${escapeHtml(absUrl(r.frame_url))}" alt="">` : "";
    return `<article class="clip">${img}<div>
      <h3>${escapeHtml(whereLabel(r))} <span class="lvl ${m.cls}">${m.text}</span></h3>
      <p class="verdict ${v ? v.verdict.toLowerCase() : "none"}">${verdict}</p>
      ${from ? `<p class="muted">Found by: ${escapeHtml(from)}</p>` : ""}
      <p class="muted">${escapeHtml(r.explanation || "")}</p>
      <p class="muted small">Clip <a href="${escapeHtml(absUrl(r.clip_url))}">${escapeHtml(r.segment_id)}</a>,
        matched seconds ${r.window[0]}-${r.window[1]} · score ${r.score.toFixed(2)}</p></div></article>`;
  };
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SketchSearch ${title.toLowerCase()}</title>
<style>
  :root { --ink: #1b1d22; --paper: #ffffff; --page: #f4f2ec; --line: #e3ded2; --soft: #666b75; --accent: #b37708;
    --accent-fill: #c2800e; --ok: #1f7a45; --bad: #b23a2c; --warn: #8a5a00;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); }
  body { margin: 0; background: var(--page); }
  .page { max-width: 860px; margin: 24px auto; background: var(--paper); padding: 36px 44px; box-shadow: 0 6px 30px color-mix(in srgb, var(--ink) 12%, transparent); }
  header { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 3px solid var(--accent-fill); padding-bottom: 12px; }
  h1 { font-family: Archivo, "Segoe UI", sans-serif; margin: 0; font-size: 26px; }
  h1 span { color: var(--accent); } h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .1em; color: var(--soft); margin: 26px 0 10px; }
  .meta { text-align: right; font-size: 13px; color: var(--soft); }
  .sketch { display: grid; grid-template-columns: 300px 1fr; gap: 20px; align-items: center; }
  .sketch img { width: 300px; border-radius: 6px; border: 1px solid var(--line); }
  .caption { font-size: 18px; font-weight: 600; margin: 0 0 6px; }
  .counts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .count { border: 1px solid var(--line); border-radius: 8px; padding: 12px; } .count b { display: block; font-size: 28px; font-family: Archivo, sans-serif; }
  .clip { display: grid; grid-template-columns: 220px 1fr; gap: 16px; padding: 12px 0; border-top: 1px solid var(--line); break-inside: avoid; }
  .clip img { width: 220px; border-radius: 6px; } .clip h3 { margin: 0 0 6px; font-size: 16px; }
  .lvl { font-size: 12px; border-radius: 999px; padding: 2px 8px; margin-left: 6px; border: 1px solid var(--accent-fill); color: var(--warn); }
  .lvl.great { background: var(--accent-fill); color: var(--paper); }
  .verdict { margin: 0 0 6px; font-weight: 600; } .verdict.yes { color: var(--ok); } .verdict.no { color: var(--bad); } .verdict.unsure { color: var(--warn); }
  .muted { color: var(--soft); margin: 0 0 4px; } .small { font-size: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; } td { padding: 4px 8px; border-bottom: 1px solid var(--line); } td:first-child { color: var(--soft); width: 140px; }
  .toolbar { max-width: 860px; margin: 16px auto 0; display: flex; gap: 10px; justify-content: flex-end; }
  .toolbar button { font: inherit; padding: 8px 16px; border-radius: 8px; border: 1px solid var(--accent-fill); background: var(--accent-fill); color: var(--paper); cursor: pointer; }
  .toolbar button.ghost { background: var(--paper); color: var(--warn); }
  footer { margin-top: 24px; font-size: 12px; color: var(--soft); }
  @media print { body { background: var(--paper); } .toolbar { display: none; } .page { box-shadow: none; margin: 0; max-width: none; padding: 0; } }
</style></head><body>
<div class="toolbar"><button class="ghost" onclick="window.close()">Close</button><button onclick="window.print()">Save as PDF</button></div>
<div class="page">
  <header><div><h1>Sketch<span>Search</span> ${escapeHtml(title.toLowerCase())}</h1><div class="muted">Warehouse camera footage, checked by a video AI</div></div>
    <div class="meta">${escapeHtml(new Date().toLocaleString())}</div></header>
  <h2>${kind === "case" ? "What's in this case" : "What we looked for"}</h2>
  ${lookedFor}
  <h2>Summary</h2>
  <div class="counts">${counts.map(([n, label]) => `<div class="count"><b>${escapeHtml(n)}</b>${escapeHtml(label)}</div>`).join("")}</div>
  <h2>${kind === "search" ? (checked.length ? "Clips checked by AI" : "Top matches") : "Clips"}</h2>
  ${items.map(clip).join("")}
  <h2>Settings</h2>
  <table>${settings.map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(v)}</td></tr>`).join("")}</table>
  <footer>Generated by SketchSearch. Matches are geometric similarity to a sketch; AI verdicts come from a video model
    and can be wrong. Review the clips before acting on them.</footer>
</div></body></html>`;
  w.document.open();
  w.document.write(html);
  w.document.close();
}


// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function bindControls() {
  for (const b of document.querySelectorAll(".kf")) b.addEventListener("click", () => setTab(b.dataset.tab));
  $("#undo").addEventListener("click", undo);
  $("#delete").addEventListener("click", deleteSelected);
  $("#clear").addEventListener("click", clearAll);
  $("#verifyBtn").addEventListener("click", checkMore);
  $("#textForm").addEventListener("submit", sketchFromText);
  $("#diagramBtn").addEventListener("click", () => $("#diagramFile").click());
  $("#diagramFile").addEventListener("change", uploadDiagram);
  $("#showFrame").addEventListener("change", setBackground);
  $("#camera").addEventListener("change", (e) => selectCamera(e.target.value));
  $("#scope").addEventListener("change", (e) => {
    if (e.target.value) selectCamera(e.target.value);
    if (state.objects.length && cards.length) runSearch();
  });
  $("#drawBtn").addEventListener("click", startDrawing);
  $("#reportBtn").addEventListener("click", () => openReport("search"));
  $("#copyLinkBtn").addEventListener("click", copyLink);
  $("#homeLink").addEventListener("click", goHome);
  initMic();
  initGuide();
  initLanes();
  initPresenterMenu();
  document.addEventListener("keydown", (e) => {
    const el = document.activeElement;
    if ($("#guide").open) return; // the guide has its own arrow keys
    if (el && (el.tagName === "TEXTAREA" || el.tagName === "SELECT" || (el.tagName === "INPUT" && el.type === "text"))) return;
    const key = e.key.toLowerCase();
    const inStudio = document.body.classList.contains("mode-studio") && !$("#searchView").hidden;
    const featured = cards.find((c) => c.featured);
    if ((e.ctrlKey || e.metaKey) && key === "z") { e.preventDefault(); undo(); }
    else if ((e.ctrlKey || e.metaKey) && key === "enter") { e.preventDefault(); runSearch(); }
    else if (e.key === "Delete" || e.key === "Backspace") { if (state.selectedId) { e.preventDefault(); deleteSelected(); } }
    else if (e.ctrlKey || e.metaKey || e.altKey) return;
    else if (e.key === "?") $("#helpBtn").click();
    else if (key === "d") setDemo(!$("#demoMode").checked, true);
    else if (!inStudio) return;
    else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      if (featured) { e.preventDefault(); stepFeatured(e.key === "ArrowRight" ? 1 : -1); }
    } else if (e.key === " ") {
      // Space on a focused button presses it; anywhere else it plays or pauses the match.
      if (featured && !(el && el.matches("button, summary, input, a"))) { e.preventDefault(); togglePlay(featured); }
    }
    else if (key === "c") checkMore();
    else if (key === "b") setTool("box");
    else if (key === "p" || key === "a") setTool("path");
    else if (key === "n" || key === "e") setTool("absent", "person");
  });
}

// Presenter options live behind the ⋮ menu so the main screen stays clean.
function setDemo(on, announce = false) {
  $("#demoMode").checked = on;
  $("#demoBadge").hidden = !on;
  document.body.classList.toggle("demo-on", on);
  try { localStorage.setItem("sketchsearch.demo", on ? "1" : "0"); } catch { /* storage blocked */ }
  if (announce) {
    toast(on ? "Demo mode on: AI checks and the agent replay saved results, so nothing waits."
      : "Demo mode off: everything runs live.");
  }
}

function initPresenterMenu() {
  const menu = $("#presenterMenu");
  $("#demoMode").addEventListener("change", (e) => setDemo(e.target.checked, true));
  document.addEventListener("click", (e) => { if (menu.open && !menu.contains(e.target)) menu.open = false; });
  menu.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { menu.open = false; $("summary", menu).focus(); }
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
    const minutes = Math.round(cams.reduce((t, c) => t + (c.duration || 0), 0) / 60);
    $("#footage").innerHTML = `${icon("cctv")}Searching ${cams.length} warehouse camera${cams.length === 1 ? "" : "s"}` +
      (minutes ? `, ${minutes} minutes of footage` : "");
    $("#camera").innerHTML = cams.map((c) => `<option value="${c.camera_id}">${escapeHtml(c.name)}</option>`).join("");
    $("#scope").insertAdjacentHTML("beforeend",
      cams.map((c) => `<option value="${c.camera_id}">${escapeHtml(shortCamera(c.camera_id))}</option>`).join(""));
    selectCamera((cams.find((c) => c.camera_id === "warehouse_cam1") || cams[0]).camera_id);
    state.presets = presets;
    const holder = $("#examples");
    for (const p of presets) {
      const b = document.createElement("button");
      b.className = "chip";
      const risk = RISK_TAGS[p.kind || p.id];
      b.innerHTML = `${risk ? `<span class="risk">${icon(risk.icon)}${risk.text}</span>` : ""}<span>${escapeHtml(p.chip || p.title)}</span>`;
      b.title = p.description;
      b.addEventListener("click", () => runPreset(p));
      holder.appendChild(b);
    }
    api("api/health").then((h) => { state.health = h; }).catch(() => {});
    restoreFromHash();
  } catch (err) {
    showState("error", `Can't reach the server (${err.message}).`);
  }
}

init();
