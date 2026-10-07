import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import FRAMES_DIR, ROOT, ROOT_PATH, SEGMENTS_DIR, SOURCE, VIDEOS_DIR
from app.models import Segment, Sketch
from app.presets import PRESETS
from app.search import get_matcher, run_search
from app.sketches import sketch_from_segment
from app.sources import get_source

STATIC_DIR = ROOT / "static"
log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    t = time.perf_counter()
    matcher = get_matcher()  # loads the index and precomputes window features once
    log.info("matcher ready: %d segments, %d windows in %.1fs",
             len(matcher.segments), matcher.n_windows, time.perf_counter() - t)
    yield


app = FastAPI(title="SketchSearch", root_path=ROOT_PATH, lifespan=lifespan)


class SearchRequest(BaseModel):
    sketch: Sketch
    top_n: int = Field(20, ge=1, le=500)


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
