import time
from functools import lru_cache

from app.config import TEXT_PREFILTER
from app.explain import explain
from app.matcher import Matcher, effective_weights
from app.models import Sketch
from app.sources import get_source


@lru_cache(maxsize=1)
def get_matcher() -> Matcher:
    source = get_source()
    return Matcher(source.candidates(set()), getattr(source, "label_aliases", {}))


def run_search(sketch: Sketch, top_n: int = 20, weights: dict[str, float] | None = None) -> dict:
    t = time.perf_counter()
    source = get_source()
    segment_ids = None
    if TEXT_PREFILTER and sketch.text:
        segment_ids = [sid for sid, _ in source.text_search(sketch.text, k=200)]
    results = get_matcher().search(sketch, weights=weights, segment_ids=segment_ids, top_n=top_n)
    for r in results:
        seg = source.get_segment(r["segment_id"])
        r.update(camera_id=seg.camera_id, start=seg.start, end=seg.end,
                 clip_url=source.clip_url(seg.segment_id), frame_url=source.frame_url(seg.segment_id),
                 explanation=explain(sketch, r, seg))
    return {
        "results": results,
        "weights": {k: round(v, 3) for k, v in effective_weights(sketch, weights).items()},
        "text_prefilter": segment_ids is not None,
        "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
    }
