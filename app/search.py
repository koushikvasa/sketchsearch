import re
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


def _caption_fallback(source, sketch: Sketch) -> tuple[Sketch, set[str] | None]:
    """Labels the detector can't see (source.caption_labels, e.g. forklift on COCO YOLO): keep only
    segments whose captions mention them and match the rest of the sketch geometrically."""
    labels = getattr(source, "caption_labels", set()) & {o.label for o in sketch.objects if not o.absent}
    if not labels:
        return sketch, None
    tagged: set[str] | None = None
    for label in labels:
        hits = source.caption_tagged(label)
        tagged = hits if tagged is None else tagged & hits
    rest = [o for o in sketch.objects if o.label not in labels or o.absent]
    return sketch.model_copy(update={"objects": rest}), tagged


# Match levels shown in the UI (static/app.js matchLabel uses the same numbers).
GREAT, GOOD = 0.85, 0.70


def short_camera(name: str) -> str:
    return re.sub(r"(?i)^warehouse cam\s*", "Camera ", name)


def how_often(source, ranked: list[dict]) -> dict:
    """Great/Good matches per camera and per minute of video, plus a one-line insight."""
    cams = {c["camera_id"]: c for c in source.list_cameras()}
    good = [r for r in ranked if r["score"] >= GOOD]
    per_camera = []
    for cam_id in sorted(cams):
        hits = [r for r in good if source.get_segment(r["segment_id"]).camera_id == cam_id]
        per_camera.append({"camera_id": cam_id, "name": short_camera(cams[cam_id].get("name") or cam_id),
                           "great": sum(r["score"] >= GREAT for r in hits), "good": sum(r["score"] < GREAT for r in hits)})
    minutes = {}
    for r in good:
        seg = source.get_segment(r["segment_id"])
        m = int((seg.start + r["window"][0]) // 60)
        minutes.setdefault(m, {"great": 0, "good": 0})["great" if r["score"] >= GREAT else "good"] += 1
    durations = [c.get("duration") or 0 for c in cams.values()]
    last = max([int(max(durations) // 60) if durations and max(durations) else 0, *minutes.keys(), 0])
    per_minute = [{"minute": m, **minutes.get(m, {"great": 0, "good": 0})} for m in range(last + 1)]
    return {"per_camera": per_camera, "per_minute": per_minute, "total": len(good),
            "insight": insight(per_camera, per_minute)}


def insight(per_camera: list[dict], per_minute: list[dict]) -> str:
    total = sum(c["great"] + c["good"] for c in per_camera)
    if not total:
        return "No great or good matches anywhere, so this moment looks rare in this footage."
    top = max(per_camera, key=lambda c: c["great"] + c["good"])
    share = (top["great"] + top["good"]) / total
    counts = [m["great"] + m["good"] for m in per_minute]
    span = 2 if len(counts) > 2 else 1
    start = max(range(len(counts) - span + 1), key=lambda i: sum(counts[i:i + span]))
    when = f"between minutes {per_minute[start]['minute']} and {per_minute[start]['minute'] + span}"
    if share >= 0.5:
        return f"Most matches on {top['name']}, {when}."
    return f"Spread across cameras (most on {top['name']}); busiest {when}."


def run_search(sketch: Sketch, top_n: int = 20, weights: dict[str, float] | None = None) -> dict:
    t = time.perf_counter()
    source = get_source()
    segment_ids = None
    if TEXT_PREFILTER and sketch.text:
        segment_ids = [sid for sid, _ in source.text_search(sketch.text, k=200)]
    match_sketch, tagged = _caption_fallback(source, sketch)
    if tagged is not None:
        segment_ids = list(tagged if segment_ids is None else set(segment_ids) & tagged)
    matcher = get_matcher()
    in_scope, candidates = matcher.scope_counts(match_sketch, segment_ids)
    ranked = matcher.search(match_sketch, weights=weights, segment_ids=segment_ids)  # all, for "how often"
    results = ranked[:top_n]
    for r in results:
        seg = source.get_segment(r["segment_id"])
        r.update(camera_id=seg.camera_id, start=seg.start, end=seg.end,
                 clip_url=source.clip_url(seg.segment_id), frame_url=source.frame_url(seg.segment_id),
                 explanation=explain(sketch, r, seg))
    return {
        "results": results,
        "counts": {"searched": in_scope, "candidates": candidates, "matches": len(ranked),
                   "great": sum(r["score"] >= GREAT for r in ranked), "good": sum(GOOD <= r["score"] < GREAT for r in ranked)},
        "how_often": how_often(source, ranked),
        "weights": {k: round(v, 3) for k, v in effective_weights(sketch, weights).items()},
        "text_prefilter": segment_ids is not None,
        "elapsed_ms": round((time.perf_counter() - t) * 1000, 1),
    }
