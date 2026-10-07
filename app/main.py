from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import FRAMES_DIR, ROOT_PATH, SEGMENTS_DIR, SOURCE, VIDEOS_DIR
from app.sources import get_source

app = FastAPI(title="SketchSearch", root_path=ROOT_PATH)


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


app.mount("/clips", StaticFiles(directory=SEGMENTS_DIR, check_dir=False), name="clips")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR, check_dir=False), name="frames")
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR, check_dir=False), name="videos")
