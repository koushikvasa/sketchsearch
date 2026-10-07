import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from app.config import (
    FRAMES_DIR, GEMINI_FALLBACK_MODEL, GEMINI_MODEL, INDEX_FILES, LLM_PROVIDER, ROOT, ROOT_PATH, SEGMENTS_DIR,
    SOURCE, TEXT_PREFILTER, VERIFY_CONCURRENCY, VERIFY_TIMEOUT_S, VIDEOS_DIR,
)
from app.models import Segment, Sketch
from app.presets import PRESETS
from app.search import get_matcher, run_search
from app.sketches import sketch_from_segment
from app.sketch_json import SketchParseError
from app.sources import get_source
from app.tracing import init_tracing

STATIC_DIR = ROOT / "static"
log = logging.getLogger("uvicorn.error")
STARTUP: dict[str, str | None] = {"error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if init_tracing():
        log.info("Weave tracing on")
    t = time.perf_counter()
    try:
        matcher = get_matcher()  # loads the index and precomputes window features once
        log.info("matcher ready: %d segments, %d windows in %.1fs",
                 len(matcher.segments), matcher.n_windows, time.perf_counter() - t)
    except Exception as e:  # keep serving (health shows why) rather than crash-looping the pod
        STARTUP["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        log.exception("warm-up failed; /api/health reports it")
    yield


class EnsureRootPath:
    """Starlette 1.x expects scope["path"] to include root_path (ASGI spec), but the /app ingress strips
    the prefix (rewrite-target /$2). Put it back when it is missing, so routes and static mounts resolve
    the same whether a request arrives as /api/... (behind the ingress) or /app/api/... (direct)."""

    def __init__(self, app, prefix: str):
        self.app, self.prefix = app, prefix.rstrip("/")

    async def __call__(self, scope, receive, send):
        if self.prefix and scope["type"] in ("http", "websocket"):
            path = scope["path"]
            if not (path == self.prefix or path.startswith(self.prefix + "/")):
                full = self.prefix + path
                scope = {**scope, "path": full, "raw_path": full.encode(), "root_path": self.prefix}
        await self.app(scope, receive, send)


app = FastAPI(title="SketchSearch", root_path=ROOT_PATH, lifespan=lifespan)
app.add_middleware(EnsureRootPath, prefix=ROOT_PATH)


class SearchRequest(BaseModel):
    sketch: Sketch
    top_n: int = Field(20, ge=1, le=500)


class VerifyItem(BaseModel):
    segment_id: str
    window: list[float] | None = None


class VerifyRequest(BaseModel):
    sketch: Sketch
    items: list[VerifyItem] = Field(..., min_length=1, max_length=20)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=500)


class AgentRequest(BaseModel):
    goal: str = Field(..., min_length=3, max_length=300)
    demo: bool = False  # replay a recorded run when one exists (scripts/warm_demo.py)


IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _segment(segment_id: str) -> Segment:
    try:
        return get_source().get_segment(segment_id)
    except KeyError:
        raise HTTPException(404, f"unknown segment {segment_id!r}")


@app.get("/", include_in_schema=False)
def index():
    """The UI. Under a path prefix (ROOT_PATH=/app behind the ingress) a <base href> makes every relative
    URL resolve below the prefix, even when the page is opened as /app without a trailing slash."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    if ROOT_PATH:
        html = html.replace("<head>", f'<head>\n  <base href="{ROOT_PATH.rstrip("/")}/">', 1)
    return HTMLResponse(html)


@app.get("/health", include_in_schema=False)
def liveness():
    """Readiness probe target for deploy-app-no-registry (httpGet /health)."""
    return {"ok": STARTUP["error"] is None, "error": STARTUP["error"]}


@app.get("/api/health")
def health():
    from app import tracing
    from app.demo import recordings
    from app.llm import provider_config

    source = get_source()
    segments = getattr(source, "segments", {})
    index = INDEX_FILES.get(getattr(source, "index_source", ""), None)
    try:
        llm_model = provider_config(LLM_PROVIDER)["model"]
    except Exception as e:  # missing key: report, don't fail the health check
        llm_model = f"unavailable ({e})"
    if STARTUP["error"]:
        return {"ok": False, "source": SOURCE, "root_path": ROOT_PATH or "/", "error": STARTUP["error"]}
    return {
        "ok": True,
        "source": SOURCE,
        "root_path": ROOT_PATH or "/",
        "index": {
            "source": getattr(source, "index_source", None),
            "segments": len(segments),
            "tracks": sum(len(s.tracks) for s in segments.values()),
            "cameras": len(source.list_cameras()),
            "file_mb": round(index.stat().st_size / 1e6, 1) if index and index.exists() else None,
        },
        "llm": {"provider": LLM_PROVIDER, "model": llm_model},
        "verify": {"models": [m for m in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL) if m],
                   "timeout_s": VERIFY_TIMEOUT_S, "concurrency": VERIFY_CONCURRENCY},
        "text_prefilter": TEXT_PREFILTER,
        "weave_tracing": tracing.is_ready(),
        "demo_recordings": recordings(),
    }


@app.get("/api/cameras")
def cameras():
    """Cameras plus a background keyframe each: the emptiest segment's mid frame (fewest tracks)."""
    source = get_source()
    out = []
    for cam in source.list_cameras():
        segs = source.candidates(set(), camera_ids=[cam["camera_id"]])
        emptiest = min(segs, key=lambda s: (len(s.tracks), s.start)) if segs else None
        out.append({**cam, "background_url": source.frame_url(emptiest.segment_id) if emptiest else None})
    return out


@app.get("/api/presets")
def presets():
    return [{"id": key, **p} for key, p in PRESETS.items()]


@app.post("/api/search")
def search(req: SearchRequest):
    return run_search(req.sketch, req.top_n)


@app.post("/api/verify")
async def verify(req: VerifyRequest):
    """Server-sent events: one `verdict` event per segment as it completes, then `done`."""
    from app.verify import verify_stream

    for item in req.items:
        _segment(item.segment_id)

    async def events():
        counts = {"YES": 0, "NO": 0, "UNSURE": 0}
        yield _sse("start", {"total": len(req.items), "concurrency": VERIFY_CONCURRENCY})
        async for out in verify_stream(req.sketch, [i.model_dump() for i in req.items]):
            counts[out["verdict"]] += 1
            yield _sse("verdict", out)
        yield _sse("done", {"counts": counts, "total": len(req.items)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/agent")
async def agent(req: AgentRequest):
    """Server-sent events: start, plan, sketch, results, verify, refine, report, done (or error)."""
    from app import demo
    from app.agent.runner import agent_run

    goal = req.goal.strip()

    async def events():
        if req.demo and demo.has_recording(goal):
            async for ev in demo.replay(goal):
                yield _sse(ev["event"], ev)
            return
        recorded = []
        try:
            async for ev in agent_run(goal):
                recorded.append(ev)
                yield _sse(ev["event"], ev)
        except Exception as e:  # surface failures in the trace instead of a dead stream
            log.exception("agent run failed")
            yield _sse("error", {"event": "error", "message": f"{type(e).__name__}: {str(e)[:300]}"})
            return
        if req.demo:
            demo.save_recording(goal, recorded)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/demo")
def demo_status():
    from app.agent.runner import PRESET_GOAL
    from app.demo import recordings

    return {"preset_goal": PRESET_GOAL, "recordings": recordings()}


@app.post("/api/text-to-sketch")
def text_to_sketch(req: TextRequest):
    from app.agent.words_to_sketch import words_to_sketch

    try:
        return words_to_sketch(req.text.strip())
    except SketchParseError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {type(e).__name__}: {str(e)[:200]}")


@app.post("/api/diagram-to-sketch")
async def diagram(file: UploadFile = File(...)):
    from app.vision.diagram import diagram_to_sketch

    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(415, f"upload a PNG, JPG or WebP image (got {file.content_type})")
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image is larger than 8 MB")
    try:
        return await asyncio.to_thread(diagram_to_sketch, data, file.content_type)
    except SketchParseError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"vision call failed: {type(e).__name__}: {str(e)[:200]}")


@app.get("/api/vast/clip/{segment_id}", include_in_schema=False)
def vast_clip(segment_id: str, request: Request):
    """Proxy a VAST segment clip (Range-capable) so the backend JWT never reaches the browser."""
    source = get_source()
    if not hasattr(source, "stream_clip"):
        raise HTTPException(404, "clip proxy is only used with SOURCE=vast")
    _segment(segment_id)
    upstream = source.stream_clip(segment_id, request.headers.get("range"))
    keep = ("content-type", "content-length", "content-range", "accept-ranges")
    headers = {k: v for k, v in upstream.headers.items() if k.lower() in keep}
    return StreamingResponse(upstream.iter_bytes(), status_code=upstream.status_code, headers=headers,
                             background=BackgroundTask(upstream.close))


@app.get("/api/segments/{segment_id}")
def segment(segment_id: str):
    seg, source = _segment(segment_id), get_source()
    return {**seg.model_dump(), "clip_url": source.clip_url(segment_id), "frame_url": source.frame_url(segment_id)}


@app.get("/api/segments/{segment_id}/as-sketch")
def as_sketch(segment_id: str, t0: float | None = None, max_objects: int = 6):
    sketch, window = sketch_from_segment(_segment(segment_id), t0=t0, max_objects=max_objects)
    return {"sketch": sketch, "window": window}


app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
app.mount("/clips", StaticFiles(directory=SEGMENTS_DIR, check_dir=False), name="clips")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR, check_dir=False), name="frames")
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR, check_dir=False), name="videos")
