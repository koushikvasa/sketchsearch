"""One-line, local (no LLM) explanation of a search result, from the real matched tracks."""

from app.models import Segment, Sketch
from app.spatial import NOUNS, aisle_position, center, direction, names, relative_motion

WEAK = 0.6
COMPONENT_WORDS = {"relations": "layout", "position": "position", "motion": "motion", "size": "size",
                   "keyframe": "end layout"}


def _ends(segment: Segment, track_id: str, t0: float, t1: float):
    track = next((t for t in segment.tracks if t.track_id == track_id), None)
    if track is None:
        return None
    pts = [p for p in track.points if t0 - 1e-6 <= p.t <= t1 + 1e-6] or track.points
    return center(pts[0].box), center(pts[-1].box), (pts[0].box.h + pts[-1].box.h) / 2


def explain(sketch: Sketch, result: dict, segment: Segment) -> str:
    t0, t1 = result["window"]
    present = [o for o in sketch.objects if not o.absent]
    who = names(present)
    real = {o.id: _ends(segment, result["assignment"].get(o.id, ""), t0, t1) for o in present}
    real = {k: v for k, v in real.items() if v}

    clauses = []
    for o in present[:3]:
        if o.id not in real:
            continue
        c0, c1, height = real[o.id]
        name = who[o.id].removeprefix("the ")
        where = aisle_position(c0)
        move = direction((c1[0] - c0[0], c1[1] - c0[1]), aisle=True, height=height)
        if move is None:
            verb = "stands" if o.label == "person" else "stays put"
            clauses.append(f"{name} {where} {verb}")
            continue
        verb = "walks" if o.label == "person" else "moves"
        target = ""
        for other in present:
            if other.id != o.id and other.id in real:
                rel = relative_motion(c0, c1, *real[other.id][:2])
                if rel:
                    target = f" {rel} {who[other.id]}"
                    break
        clauses.append(f"{name} {where} {verb} {move}{target}")

    for a in (o for o in sketch.objects if o.absent):
        noun = NOUNS.get(a.label, a.label)
        if result.get("absence_ok") is True:
            clauses.append(f"no {noun} in the marked zone")
        elif result.get("absence_ok") is False:
            clauses.append(f"but a {noun} is inside the marked zone")

    weak = [(v, k) for k, v in result.get("components", {}).items() if v < WEAK]
    if weak:
        v, k = min(weak)
        clauses.append(f"{COMPONENT_WORDS.get(k, k)} is the weakest match ({v:.2f})")
    if not clauses:
        return "Matches the sketch."
    text = "; ".join(clauses)
    return text[0].upper() + text[1:] + "."
