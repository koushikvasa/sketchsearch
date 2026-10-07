# SketchSearch — Project Plan & Build Spec (v2)

> **"Words get you close. A sketch gets you exact. AI proves it."**
> Type or draw the moment you're looking for — SketchSearch finds it in hours of video, an AI watches each hit to
> verify it, and explains why it matched.

**Event:** VAST Builders Challenge — Real-Time Video Agents Hack NYC, **Fri Oct 9, 8:30 AM–6:30 PM**.
**Strategy:** build and demo everything now on **free stand-ins**, behind adapters. Friday morning = swap to the
organizers' stack (VastDB / Cosmos / W&B / CoreWeave K8s), then polish.
This file is the single source of truth for Claude Code. Build phase by phase; verify, summarize, commit after each.

---

## 1. Positioning (why this wins)

- **Prior art exists for basic sketch search** (VideoQ 1990s, VISIONE/vitrivr at Video Browser Showdown, SketchQL
  VLDB 2024, BriefCam path filters). We do **not** pitch "search by drawing" as the invention.
- **Our novelty (not found in prior art or in the 45 SF projects):**
  1. **AI verification** — a video model watches every top hit: ✅ confirmed / ⚠️ unsure / ❌ rejected + reason.
  2. **Words ↔ sketch** — type a sentence, the LLM draws the sketch; text search narrows, the sketch makes it exact.
  3. **"Nobody here" boxes** — search for what is *absent* (forklift in aisle, **no** person nearby).
  4. **Diagram → search** — upload a hand-drawn accident/incident diagram; it becomes a sketch query.
  5. **Agent mode** — "find dangerous moments": the agent writes its own sketches, runs, verifies, refines, reports.
- **Story / user:** the **incident investigator** (workplace safety, insurance, traffic engineering, AV teams).
- **SF lesson we exploit:** finalist LongTail had 22 text hits → 1 confirmed because captions never state distances
  or motion. Geometry from YOLO tracks is exactly what text search misses.

## 2. Free stand-ins now → organizers' stack Friday

| Role | Now (free) | Friday (organizers) | Swap point |
|---|---|---|---|
| Video archive | 15–20 Pexels clips in `data/videos/` | Team VSS archive | `SOURCE=vast` |
| Object tracks | YOLO11 + ByteTrack on laptop | YOLO11 detections in VastDB (or `$YOLO_URL`) | `VastSource.candidates()` |
| Text search (prefilter) | CLIP keyframe embeddings (`open_clip`, CPU) | `search` skill (Cosmos Embed) | `text_search()` |
| Video verification / Q&A | Gemini Flash (free tier) | Cosmos3-Reason via `agent-qa` / `$COSMOS3_REASON_URL` | `ask()` |
| LLM (text→sketch, agent) | Groq `openai/gpt-oss-120b` | W&B Inference | `LLM_PROVIDER=wandb` |
| Diagram → sketch (vision) | Gemini Flash | Gemini or W&B vision model if available | `VISION_PROVIDER` |
| Tracing | W&B Weave (free account) | Same | — |
| Hosting | localhost (optional Railway) | CoreWeave K8s at `/app` via `deploy-app-no-registry` | `ROOT_PATH` |

**Clip guidance (owner):** prefer **fixed cameras** (intersections, parking lots, warehouse/CCTV-style), 10–40 s each.
Dashcam clips are fine for text search but positions shift with camera motion — keep them as a minority.

## 3. Feature scope

**Must-have (demo core)**
1. Sketch canvas: labeled boxes (person, car, truck, bus, bicycle, motorcycle, forklift→local alias "truck"),
   motion arrow/path per object, **"nobody here" (absent) box**, Start/End keyframe tabs.
2. **✨ Words → Sketch**: sentence → LLM → sketch JSON, animated onto canvas.
3. **Hybrid search**: optional text prefilter (top 200 by text) → geometric sketch matching → ranked results.
4. **Results**: clip player with drawn boxes (dashed) vs real boxes (solid) overlaid and time-synced,
   score breakdown bars, one-line explanation.
5. **AI verification** of top 10 with live streaming badges + confirmation rate ("7 of 10 confirmed").
6. **More like this**: load a result's real boxes onto the canvas as an editable sketch.

**Should-have (differentiators)**
7. **Diagram upload → sketch** (photo of hand-drawn diagram).
8. **Agent mode** with visible trace (plan → sketches → results → verify → refine → report).
9. **Ghost replay**: animate the drawn path as a translucent ghost over the playing clip.
10. Weave tracing of LLM calls, matcher runs, verifications.

**Stretch (only if ahead)**
11. "How often does this happen?" mini chart (matches by hour / camera).
12. 👍/👎 feedback that re-weights scoring, before/after precision logged to W&B.

## 4. Architecture

```
static/ (index.html, app.js, style.css, Fabric.js via cdnjs — no build step)
   │ JSON
FastAPI app/
 ├─ models.py        Box, Track, Segment, Sketch (normalized 0–1 coords)
 ├─ sources/         VideoSource protocol → LocalSource (now) | VastSource (Friday stub)
 ├─ matcher/         geometric scoring (pure Python, tested)
 ├─ llm/             OpenAI-compatible client (groq|wandb) + Weave
 ├─ vision/          diagram→sketch, verification (gemini now; cosmos Friday via source.ask)
 ├─ agent/           text→sketch, explanations, agent mode (SSE)
 └─ main.py          routes, static, ROOT_PATH
scripts/build_index.py   segment + track + keyframes + CLIP embeddings → data/index.json (+ .npy)
```
Single process, relative URLs everywhere (served under `/app` on Friday). Python 3.12, `uv`, FastAPI, pydantic v2,
numpy, scipy, ultralytics, open_clip_torch, google-genai, openai, weave, python-dotenv.

## 5. Data model

```python
class Box(BaseModel):        x: float; y: float; w: float; h: float           # normalized, top-left
class TrackPoint(BaseModel): t: float; box: Box                               # t = seconds in segment
class Track(BaseModel):      label: str; track_id: str; points: list[TrackPoint]
class Segment(BaseModel):
    segment_id: str; video_id: str; camera_id: str; start: float; end: float
    caption: str | None = None; tracks: list[Track]; keyframe_only: bool = False
class SketchObject(BaseModel):
    id: str; label: str; start_box: Box
    end_box: Box | None = None
    path: list[tuple[float, float]] | None = None
    absent: bool = False                     # "nobody here" box: label must NOT appear in this region
class Sketch(BaseModel):
    objects: list[SketchObject]; text: str | None = None; camera_ids: list[str] | None = None
```

## 6. VideoSource adapter

```python
class VideoSource(Protocol):
    def list_cameras(self) -> list[dict]: ...
    def candidates(self, labels: set[str], camera_ids: list[str] | None, segment_ids: list[str] | None) -> list[Segment]: ...
    def get_segment(self, segment_id: str) -> Segment: ...
    def text_search(self, query: str, k: int = 200) -> list[tuple[str, float]]: ...   # (segment_id, score)
    def ask(self, segment_id: str, question: str) -> str: ...
    def clip_url(self, segment_id: str) -> str: ...
    def frame_url(self, segment_id: str, t: float = 0.5) -> str: ...
```
- **LocalSource**: `build_index.py` splits videos into 4 s segments (ffmpeg), runs `YOLO("yolo11n.pt").track(persist=True)`
  at ~5 fps, normalizes boxes, drops tracks < 3 points, saves mid-frame JPG, computes CLIP ViT-B/32 image embedding per
  keyframe; `text_search` = CLIP text embedding cosine. `ask` = Gemini with the segment mp4 (cached on disk; retry with
  backoff on 429). Static mount serves clips/frames.
- **VastSource**: stub raising `NotImplementedError("Friday")` with docstrings: detections via `vastdb-read`, clips via
  `videos`, `text_search` via `search`, `ask` via `agent-qa` or `$COSMOS3_REASON_URL`. Fallbacks:
  per-segment boxes only → `keyframe_only=True` (motion checked in verification); or call `$YOLO_URL` on frames of the
  candidate subset and cache tracks.

## 7. Matcher

1. **Filter**: count-aware presence of every non-absent label. Restrict to text-search hits when `sketch.text` set.
2. **Window**: slide a 1.5 s window over each segment; score each window, keep the best.
3. **Assignment**: `scipy.optimize.linear_sum_assignment` between sketch objects and same-label tracks.
4. **Components (0–1)**: `relations` (pairwise left/right, above/below, near/far agreement — weight .35),
   `position` (Gaussian center distance, σ≈0.15 — .25), `motion` (16-point resampled path: direction cosine + mean
   distance; "converging" bonus — .25), `size` (sqrt area ratio — .10), `keyframe` (End layout — .05). Renormalize over
   available components.
5. **Absent boxes**: if any track of that label overlaps the absent region (IoU > 0.1 or center inside) during the
   window → multiply score by 0.2. Report "absence satisfied: yes/no".
6. **Output**: `{segment_id, score, components, assignment, window, absence_ok}`; 2,000 segments < 1 s.
7. **Tests** (synthetic): exact match ≈1; mirrored layout low relations; reversed motion low motion; count-aware filter;
   absent box penalizes; keyframe_only still scores.

## 8. AI features

- **LLM client**: OpenAI-compatible; `groq` now, `wandb` Friday (`https://api.inference.wandb.ai/v1`, `WANDB_API_KEY`,
  `WANDB_MODEL`). All calls `@weave.op` (no-op without key).
- **Words → Sketch**: system prompt defines canvas coords, labels, absent boxes, paths; strict JSON validated by pydantic,
  1 retry; 4 few-shot examples (approach, cut-in, no-spotter absence, crossing).
- **Explain**: one sentence per result from assignment + components (no video).
- **Verification**: question built from sketch + explanation ("Does a forklift move toward a person standing on the left?
  Is there no other person nearby? Answer YES/NO/UNSURE and one short reason."). Concurrency 3, cached, SSE stream.
- **Diagram → Sketch**: image upload → vision model returns the same sketch JSON (objects, arrows→paths); show it on
  canvas for the user to adjust before searching.
- **Agent mode**: goal → plan 3–5 sketches (JSON) → match → verify top 5 each → if confirm rate < 40% refine that sketch
  once → merge/dedupe → markdown report with confirmed clips. SSE events: `plan, sketch, results, verify, refine, report`.

## 9. API

`GET /` · `GET /api/health` · `GET /api/cameras` · `POST /api/search {sketch, top_n}` ·
`POST /api/verify {segment_ids, sketch}` (SSE) · `POST /api/text-to-sketch {text}` ·
`POST /api/diagram-to-sketch` (multipart image) · `GET /api/segments/{id}/as-sketch` · `POST /api/agent {goal}` (SSE) ·
static clips/frames.

## 10. UI

- Header: name + tagline, camera multi-select, tabs **Search | Agent**.
- Left panel: canvas (16:9, optional faint camera frame background), palette, path tool, 🚫 absent tool,
  Start/End tabs, undo/clear, text box + "✨ Sketch it", 📄 "Upload diagram", **Search**.
- Right panel: results grid → video card with synced overlay + ghost replay, score bars, explanation, verdict badge +
  reason, "More like this". Header line: "Verified 7 of 10".
- Agent tab: goal box, live trace timeline, final report.
- Dark theme, large type for projector; loading/empty/error states; Delete key + Ctrl+Z.

## 11. Config (`.env.example`)

```
SOURCE=local
LLM_PROVIDER=groq
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
WANDB_API_KEY=
WANDB_MODEL=
WEAVE_PROJECT=sketchsearch
GEMINI_API_KEY=
GEMINI_MODEL=
VISION_PROVIDER=gemini
ROOT_PATH=
```

## 12. Timeline

| When | Phase | Done when |
|---|---|---|
| **Wed (today)** | 1 Data + LocalSource + CLIP | `build_index.py` runs on owner's clips; stats printed; clips served |
| **Wed** | 2 Matcher + search API + tests | tests green; curl search returns sensible ranks |
| **Thu AM** | 3 UI core (canvas, results, overlays, ghost, More like this) | full draw→search→play loop in browser |
| **Thu PM** | 4 AI layer (words→sketch, verify, diagram→sketch, Weave) | badges stream; sentence draws a sketch |
| **Thu eve** | 5 Agent mode + FRIDAY.md + VastSource stub + polish | rehearsed demo; backup video recorded |
| **Fri 9–10:30** | Swap | VastSource implemented; W&B LLM; running on real footage |
| **Fri 10:30–4** | Tune + build stretch | weights/labels tuned on their data; deploy at `/app` |
| **Fri 4–5** | Ship | final demo video, `help me submit our project` |

## 13. Friday playbook (also write as FRIDAY.md)

1. VM: `git clone <repo>`; copy `.env`; `SOURCE=vast`, `LLM_PROVIDER=wandb`.
2. Ask Cursor: "Using vastdb-read, show raw YOLO detections for 3 segments each from warehouse, I-24 and SF streets:
   per-frame or per-segment boxes, coordinate format, label names (forklift?), and how to get clip URLs."
3. Implement `VastSource` from its docstrings; run tests + one known-good sketch per pack.
4. Pick demo packs where tracks are richest (likely Warehouse C and I-24 A); tune weights.
5. Optional: re-ingest a few segments with a prompt describing distances/motion to strengthen verification.
6. Deploy with `deploy-app-no-registry` at `/app`; check relative URLs.
7. Record backup demo; `help me submit our project`.

## 14. Demo script (2 minutes)

1. **Problem (15 s):** "Hours of warehouse and street video. Text search finds 'forklift near person' — but SF teams
   found most hits were wrong: captions don't capture *where* things are or *how* they move."
2. **Words → Sketch (20 s):** type "worker on the left, forklift coming at them" → boxes animate in.
3. **Search (20 s):** results with real boxes overlaid, ghost path replaying.
4. **Verify (20 s):** badges stream in — "Verified 7 of 10" — click a ❌ to show the honest reason.
5. **Absence (15 s):** add 🚫 person box beside the forklift → "forklift in aisle with no spotter".
6. **Diagram (15 s):** upload a napkin accident sketch → becomes a search → finds the real moment.
7. **Agent (10 s):** "find dangerous moments" → trace runs → report.
8. **Close (5 s):** "Sketch search has existed in labs for 25 years. We made it verified, agentic and
   language-aware — running on VAST, Cosmos and W&B."

## 15. Risks & mitigations

| Risk | Mitigation |
|---|---|
| VastDB stores per-segment boxes only | `keyframe_only` mode; motion via verification; or `$YOLO_URL` on candidates |
| Gemini free-tier rate limits | cache, concurrency 3, backoff; pre-verify demo queries |
| No "forklift" label in COCO | label alias map; check real labels Friday |
| Pre-built code rules unclear | ask organizers on tokens& Discord; worst case rebuild fast from this spec |
| Demo-day network/latency | recorded backup video; cached demo queries |
