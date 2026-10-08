"""Find 4 strong example sketches for whatever index is connected, write data/presets_<index>.json and warm the
AI-check cache, so the landing page's example chips always work on the data at hand.

  uv run python -m scripts.make_presets              # current SOURCE / INDEX_SOURCE
  uv run python -m scripts.make_presets --no-warm    # skip the video-model checks

How: scan every 1.5 s window of every segment for real moments of four kinds (in order of preference):
  approach  something walks toward something else (2 objects, clear motion; mixed labels preferred)
  meet      two of the same kind walk toward each other
  absent    an object with nobody near it (an absent zone)
  group     3+ of the same kind close together            (fallback: "cross", one walker crossing)
Each candidate becomes a sketch built from the real tracks. A candidate is accepted only if searching for it
returns at least 5 Good matches (score >= 0.70), so the chip never lands on an empty page.
Only labels that exist in the index are used.
"""

import argparse
import asyncio
import json
import math
from datetime import datetime, timezone

import numpy as np

from app.matcher import Matcher
from app.models import Box, Segment, Sketch, SketchObject
from app.presets import presets_path
from app.search import GOOD
from app.sketches import WINDOW, window_starts, window_tracks
from app.spatial import aisle_position

MIN_GOOD = 5
MIN_AREA = 0.0005  # skip specks (a far person at the end of an aisle is ~0.0008)
MOVE = 0.05  # "clear motion" in 1.5 s
ZONE = 0.15  # half-size of the absent zone around the anchor
KIND_ORDER = ["approach", "meet", "absent", "group", "cross"]
NOUN = {"person": "person", "transporter": "cart"}
PLURAL = {"person": "people"}


def noun(label: str) -> str:
    return NOUN.get(label, label)


def plural(label: str) -> str:
    return PLURAL.get(label, noun(label) + "s")


def _feat(tr, pts) -> dict:
    c = np.array([[p.box.x + p.box.w / 2, p.box.y + p.box.h / 2] for p in pts])
    return {"track": tr, "pts": pts, "label": tr.label, "c0": c[0], "c1": c[-1], "d": c[-1] - c[0],
            "area": pts[0].box.w * pts[0].box.h, "centers": c}


def _dist(a, b) -> float:
    return float(np.hypot(*(a - b)))


def _obj(oid: str, f: dict, motion: bool = True) -> SketchObject:
    pts = f["pts"]
    idx = sorted(set(np.linspace(0, len(pts) - 1, min(len(pts), 5)).round().astype(int)))
    r = lambda b: Box(x=round(b.x, 4), y=round(b.y, 4), w=round(b.w, 4), h=round(b.h, 4))  # noqa: E731
    if not motion:
        return SketchObject(id=oid, label=f["label"], start_box=r(pts[0].box))
    return SketchObject(id=oid, label=f["label"], start_box=r(pts[0].box), end_box=r(pts[-1].box),
                        path=[(round(float(f["centers"][i][0]), 4), round(float(f["centers"][i][1]), 4)) for i in idx])


def candidates(segments: list[Segment]) -> dict[str, list[dict]]:
    """Every real moment that fits a kind, with a quality score."""
    out: dict[str, list[dict]] = {k: [] for k in KIND_ORDER}
    for seg in segments:
        for t0 in window_starts(seg):
            feats = [_feat(tr, pts) for tr, pts in window_tracks(seg, t0, t0 + WINDOW)]
            vis = [f for f in feats if f["area"] >= MIN_AREA]
            persons = [f for f in feats if f["label"] == "person"]
            for a in vis:
                moving = np.hypot(*a["d"]) >= MOVE
                for b in vis:
                    if a is b:
                        continue
                    closer = _dist(a["c0"], b["c0"]) - _dist(a["c1"], b["c1"])
                    if moving and closer >= 0.04:
                        mixed = a["label"] != b["label"]
                        out["approach"].append({"seg": seg, "t0": t0, "kind": "approach", "feats": [a, b],
                                                "mixed": mixed, "score": closer + 0.05 * (b["label"] != "person")})
                    if (a["label"] == b["label"] and id(a) < id(b) and moving and np.hypot(*b["d"]) >= MOVE
                            and closer >= 0.06 and a["d"] @ (b["c0"] - a["c0"]) > 0 and b["d"] @ (a["c0"] - b["c0"]) > 0):
                        out["meet"].append({"seg": seg, "t0": t0, "kind": "meet", "feats": [a, b], "score": closer})
                if moving and abs(a["d"][0]) >= 0.08:
                    out["cross"].append({"seg": seg, "t0": t0, "kind": "cross", "feats": [a], "score": abs(a["d"][0])})
                # absent: no other person anywhere inside the zone during the window
                x0, y0 = a["c0"] - ZONE
                x1, y1 = a["c0"] + ZONE
                intruders = [p for p in persons if p is not a and any(
                    x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in p["centers"])]
                if not intruders and persons:  # persons exist in the scene, just not near the anchor
                    out["absent"].append({"seg": seg, "t0": t0, "kind": "absent", "feats": [a],
                                          "score": a["area"] * 10 + 0.2 * (a["label"] != "person")})
            seen_groups = set()
            for f in vis:  # tight clusters of 3+ of one label (others may be elsewhere in the frame)
                cluster = [g for g in vis if g["label"] == f["label"] and _dist(g["c0"], f["c0"]) <= ZONE * 0.8]
                key = frozenset(id(g) for g in cluster)
                if len(cluster) < 3 or key in seen_groups:
                    continue
                seen_groups.add(key)
                spread = max(_dist(p["c0"], q["c0"]) for p in cluster for q in cluster)
                if spread <= ZONE:
                    out["group"].append({"seg": seg, "t0": t0, "kind": "group", "feats": cluster[:4],
                                         "score": len(cluster) - spread})
    return out


def to_preset(c: dict) -> dict:
    f = c["feats"]
    kind = c["kind"]
    if kind == "approach":
        a, b = f
        verb = "walks" if a["label"] == "person" else "moves"
        who = "Worker" if a["label"] == "person" else noun(a["label"]).capitalize()
        chip = f"{who} {verb} toward {noun(b['label'])}" if a["label"] != b["label"] else f"{who} {verb} toward another {noun(b['label'])}"
        objects = [_obj(f"{noun(a['label'])}", a), _obj(f"{noun(b['label'])}2", b)]
    elif kind == "meet":
        chip = "Two people meet in an aisle" if f[0]["label"] == "person" else f"Two {plural(f[0]['label'])} meet"
        objects = [_obj(f"{noun(x['label'])}{i}", x) for i, x in enumerate(f, 1)]
    elif kind == "absent":
        a = f[0]
        cx, cy = a["c0"]
        zx0, zy0, zx1, zy1 = max(0.0, cx - ZONE), max(0.0, cy - ZONE), min(1.0, cx + ZONE), min(1.0, cy + ZONE)
        zone = SketchObject(id="nobody", label="person", absent=True,
                            start_box=Box(x=round(zx0, 4), y=round(zy0, 4), w=round(zx1 - zx0, 4), h=round(zy1 - zy0, 4)))
        anchor = _obj(noun(a["label"]), a, motion=np.hypot(*a["d"]) >= MOVE)
        chip = f"{noun(a['label']).capitalize()} with nobody nearby" if a["label"] != "person" else "Person alone, nobody nearby"
        objects = [anchor, zone]
    elif kind == "group":
        chip = f"Group of {len(f)} {plural(f[0]['label'])}"
        objects = [_obj(f"{noun(x['label'])}{i}", x, motion=False) for i, x in enumerate(f, 1)]
    else:  # cross
        a = f[0]
        chip = f"{'Worker' if a['label'] == 'person' else noun(a['label']).capitalize()} crosses the aisle"
        objects = [_obj(noun(a["label"]), a)]
    where = aisle_position(tuple(f[0]["c0"]))
    return {"id": kind, "kind": kind, "chip": chip, "title": chip,
            "description": f"{chip} ({where}), found in {c['seg'].camera_id} at {c['seg'].start + c['t0']:.0f} s",
            "camera_id": c["seg"].camera_id, "sketch": Sketch(objects=objects)}


def find_presets(segments: list[Segment], matcher: Matcher, n: int = 4, min_good: int = MIN_GOOD,
                 tries: int = 40, log=print) -> list[dict]:
    found, used_segments = [], set()
    pool = candidates(segments)
    for kind in KIND_ORDER:
        if len(found) >= n or (kind == "cross" and len(found) >= n):
            break
        # Mixed labels first (a worker and a forklift beats two workers), then by quality.
        ranked = sorted(pool[kind], key=lambda c: (not c.get("mixed", False), -c["score"]))
        tried = 0
        for c in ranked:
            if c["seg"].segment_id in used_segments:
                continue
            tried += 1
            if tried > tries:
                break
            preset = to_preset(c)
            results = matcher.search(preset["sketch"])
            good = sum(r["score"] >= GOOD for r in results)
            if good >= min_good:
                preset["validated"] = {"good_matches": good, "top_score": results[0]["score"],
                                       "source_segment": c["seg"].segment_id, "window": [c["t0"], c["t0"] + WINDOW]}
                found.append(preset)
                used_segments.add(c["seg"].segment_id)
                log(f"  {kind:9s} -> {preset['chip']!r}: {good} good matches (from {c['seg'].segment_id})")
                break
        else:
            log(f"  {kind:9s} -> no candidate with >= {min_good} good matches")
    return found[:n]


def write_presets(index: str, presets: list[dict]) -> None:
    data = {"index": index, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": "scripts/make_presets.py",
            "presets": [{**p, "sketch": p["sketch"].model_dump(exclude_none=True, exclude={"text", "camera_ids"})}
                        for p in presets]}
    presets_path(index).write_text(json.dumps(data, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-warm", action="store_true", help="don't pre-run the AI checks")
    ap.add_argument("--min-good", type=int, default=MIN_GOOD)
    args = ap.parse_args()

    from app.search import get_matcher
    from app.sources import get_source
    from app.tracing import init_tracing

    source = get_source()
    index = getattr(source, "index_source", "default")
    matcher = get_matcher()
    labels = sorted({t.label for s in matcher.segments for t in s.tracks})
    print(f"index {index!r}: {len(matcher.segments)} segments, labels {labels}")
    presets = find_presets(matcher.segments, matcher, min_good=args.min_good)
    if not presets:
        raise SystemExit("no example sketches found; is the index empty?")
    write_presets(index, presets)
    print(f"wrote {presets_path(index)} ({len(presets)} presets)")
    if not args.no_warm:
        from scripts.warm_demo import warm_presets

        init_tracing()
        print("Warming the AI-check cache (top 10 per preset):")
        asyncio.run(warm_presets({p["id"]: p for p in presets}))


if __name__ == "__main__":
    main()
