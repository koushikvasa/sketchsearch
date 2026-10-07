"""Turn a segment's real tracks into an editable sketch ("More like this", eval queries)."""

import numpy as np

from app.matcher.core import WINDOW, WINDOW_STEP
from app.models import Box, Segment, Sketch, SketchObject, Track, TrackPoint

_EPS = 1e-6


def window_starts(seg: Segment) -> list[float]:
    dur = seg.end - seg.start
    if dur <= WINDOW + _EPS:
        return [0.0]
    return [float(t) for t in np.arange(0.0, dur - WINDOW + _EPS, WINDOW_STEP)]


def window_tracks(seg: Segment, t0: float, t1: float) -> list[tuple[Track, list[TrackPoint]]]:
    """Tracks with at least 2 points inside [t0, t1], largest first."""
    out = []
    for tr in seg.tracks:
        pts = [p for p in tr.points if t0 - _EPS <= p.t <= t1 + _EPS]
        if len(pts) >= 2:
            out.append((tr, pts))
    return sorted(out, key=lambda tp: -tp[1][0].box.w * tp[1][0].box.h)


def default_window(seg: Segment) -> tuple[float, float]:
    """Window with the most tracks; ties go to the one closest to the segment's middle."""
    dur = seg.end - seg.start
    mid = (dur - WINDOW) / 2
    t0 = max(window_starts(seg), key=lambda t: (len(window_tracks(seg, t, t + WINDOW)), -abs(t - mid)))
    return t0, min(t0 + WINDOW, dur)


def track_to_object(obj_id: str, label: str, pts: list[TrackPoint]) -> SketchObject:
    def r(b: Box) -> Box:
        return Box(x=round(b.x, 4), y=round(b.y, 4), w=round(b.w, 4), h=round(b.h, 4))

    return SketchObject(
        id=obj_id, label=label, start_box=r(pts[0].box), end_box=r(pts[-1].box),
        path=[(round(p.box.x + p.box.w / 2, 4), round(p.box.y + p.box.h / 2, 4)) for p in pts],
    )


def sketch_from_segment(seg: Segment, t0: float | None = None, max_objects: int = 6) -> tuple[Sketch, list[float]]:
    if t0 is None:
        t0, t1 = default_window(seg)
    else:
        t1 = min(t0 + WINDOW, seg.end - seg.start)
    objects = [track_to_object(f"o{i + 1}", tr.label, pts)
               for i, (tr, pts) in enumerate(window_tracks(seg, t0, t1)[:max_objects])]
    return Sketch(objects=objects, camera_ids=None), [t0, t1]
