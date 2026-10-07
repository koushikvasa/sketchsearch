"""Parse and normalize model-written sketch JSON (words -> sketch, diagram -> sketch)."""

import json
import re

from pydantic import ValidationError

from app.models import Box, Sketch, SketchObject

LABELS = ("person", "forklift", "robot", "transporter")
MAX_OBJECTS = 8
LABEL_SYNONYMS = {
    "people": "person", "worker": "person", "man": "person", "woman": "person", "pedestrian": "person",
    "human": "person", "fork lift": "forklift", "fork-lift": "forklift", "mobile robot": "robot",
    "amr": "robot", "agv": "robot", "novacarter": "robot", "cart": "transporter", "trolley": "transporter",
}

SKETCH_FORMAT = """Return ONE JSON object and nothing else:
{"objects": [
  {"id": "short-name", "label": "person|forklift|robot|transporter",
   "start_box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
   "end_box": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0} or null,
   "path": [[x, y], [x, y], ...] or null,
   "absent": false}
]}
Coordinates are fractions of the frame: x from 0 (left edge) to 1 (right edge), y from 0 (top edge) to 1
(bottom edge); a box is its top-left corner (x, y) plus width w and height h.
- start_box: where the object is at the start; end_box: where it is about 1.5 seconds later (null if it
  does not move); path: the object's centre from start to end, 2-6 points (null if it does not move).
- absent: true marks a "nobody here" zone: start_box is the zone that must NOT contain that label
  (end_box and path must be null). Use it for "nobody near", "no one in", "without a spotter", etc.
- Use only these labels: person, forklift, robot, transporter. At most 6 objects."""


class SketchParseError(ValueError):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise SketchParseError("no JSON object found")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise SketchParseError(f"invalid JSON: {e}") from None
    if isinstance(data.get("sketch"), dict):
        data = data["sketch"]
    if not isinstance(data.get("objects"), list):
        raise SketchParseError('expected {"objects": [...]}')
    return data


def _box(raw, what: str) -> Box:
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        raw = dict(zip("xywh", raw))
    if not isinstance(raw, dict):
        raise SketchParseError(f"{what} must be an object with x, y, w, h")
    try:
        x, y, w, h = (float(raw[k]) for k in "xywh")
    except (KeyError, TypeError, ValueError):
        raise SketchParseError(f"{what} needs numeric x, y, w, h") from None
    if max(abs(x), abs(y), abs(w), abs(h)) > 1.5:  # looks like pixels or percent
        raise SketchParseError(f"{what} must use 0-1 fractions of the frame, got {raw}")
    w, h = min(max(w, 0.005), 1.0), min(max(h, 0.005), 1.0)
    x, y = min(max(x, 0.0), 1.0 - w), min(max(y, 0.0), 1.0 - h)
    return Box(x=round(x, 4), y=round(y, 4), w=round(w, 4), h=round(h, 4))


def _center(b: Box) -> tuple[float, float]:
    return round(b.x + b.w / 2, 4), round(b.y + b.h / 2, 4)


def normalize_objects(raw_objects: list) -> Sketch:
    if not raw_objects:
        raise SketchParseError("the sketch has no objects")
    objects, seen = [], set()
    for i, o in enumerate(raw_objects[:MAX_OBJECTS]):
        if not isinstance(o, dict):
            raise SketchParseError(f"object {i} is not a JSON object")
        label = str(o.get("label", "")).strip().lower()
        label = LABEL_SYNONYMS.get(label, label)
        if label not in LABELS:
            raise SketchParseError(f"object {i} has label {o.get('label')!r}; use one of {', '.join(LABELS)}")
        absent = bool(o.get("absent", False))
        start = _box(o.get("start_box"), f"object {i} start_box")
        end = _box(o["end_box"], f"object {i} end_box") if o.get("end_box") and not absent else None
        path = None
        if o.get("path") and not absent:
            try:
                path = [(round(min(max(float(p[0]), 0), 1), 4), round(min(max(float(p[1]), 0), 1), 4))
                        for p in o["path"]]
            except (TypeError, ValueError, IndexError):
                raise SketchParseError(f"object {i} path must be a list of [x, y] pairs") from None
            if len(path) < 2:
                path = None
        # Keep start/end/path consistent so the canvas and matcher agree.
        if path and end is None:
            end = Box(x=round(min(max(path[-1][0] - start.w / 2, 0), 1 - start.w), 4),
                      y=round(min(max(path[-1][1] - start.h / 2, 0), 1 - start.h), 4), w=start.w, h=start.h)
        if end is not None and path is None:
            path = [_center(start), _center(end)]
        oid = re.sub(r"[^A-Za-z0-9_-]", "", str(o.get("id") or "")) or f"{label}{i + 1}"
        while oid in seen:
            oid = f"{oid}_{i + 1}"
        seen.add(oid)
        objects.append(SketchObject(id=oid, label=label, start_box=start, end_box=end, path=path, absent=absent))
    if all(o.absent for o in objects):
        raise SketchParseError("the sketch needs at least one object that is present (absent zones alone)")
    try:
        return Sketch(objects=objects)
    except ValidationError as e:
        raise SketchParseError(str(e)) from None


def parse_sketch(text: str) -> Sketch:
    return normalize_objects(_extract_json(text)["objects"])
