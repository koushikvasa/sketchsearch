import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import (
    FRAMES_DIR, GEMINI_FALLBACK_MODEL, GEMINI_MODEL, LLM_PROVIDER, ROOT, ROOT_PATH, SEGMENTS_DIR, SOURCE,
    VERIFY_CONCURRENCY, VIDEOS_DIR,
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    if init_tracing():
        log.info("Weave tracing on")
    t = time.perf_counter()
    matcher = get_matcher()  # loads the index and precomputes window features once
    log.info("matcher ready: %d segments, %d windows in %.1fs",
             len(matcher.segments), matcher.n_windows, time.perf_counter() - t)
    yield


app = FastAPI(title="SketchSearch", root_path=ROOT_PATH, lifespan=lifespan)


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


IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _segment(segment_id: str) -> Segment:
    try:
        return get_source().get_segment(segment_id)
    except KeyError:
        raise HTTPException(404, f"unknown segment {segment_id!r}")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    source = get_source()
    return {
        "ok": True,
        "source": SOURCE,
        "index_source": getattr(source, "index_source", None),
        "segments": len(getattr(source, "segments", {})),
        "llm_provider": LLM_PROVIDER,
        "vision_models": [m for m in (GEMINI_MODEL, GEMINI_FALLBACK_MODEL) if m],
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
