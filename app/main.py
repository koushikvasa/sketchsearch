from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import FRAMES_DIR, ROOT_PATH, SEGMENTS_DIR, SOURCE, VIDEOS_DIR
from app.models import Segment, Sketch
from app.search import run_search
from app.sketches import sketch_from_segment
from app.sources import get_source

app = FastAPI(title="SketchSearch", root_path=ROOT_PATH)


class SearchRequest(BaseModel):
    sketch: Sketch
    top_n: int = Field(20, ge=1, le=500)


def _segment(segment_id: str) -> Segment:
    try:
        return get_source().get_segment(segment_id)
    except KeyError:
        raise HTTPException(404, f"unknown segment {segment_id!r}")


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
    return get_source().list_cameras()


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


app.mount("/clips", StaticFiles(directory=SEGMENTS_DIR, check_dir=False), name="clips")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR, check_dir=False), name="frames")
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR, check_dir=False), name="videos")
