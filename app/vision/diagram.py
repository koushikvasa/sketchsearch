"""📄 Diagram -> Sketch: a photo/scan of a hand-drawn incident diagram becomes sketch JSON (Gemini vision)."""

import io
import time

import weave
from PIL import Image

from app.sketch_json import SKETCH_FORMAT, SketchParseError, _extract_json, normalize_objects
from app.vision.gemini import VisionTimeout, generate_json, image_part

DIAGRAM_TIMEOUT_S = 30  # per attempt; a hung request is retried once (flash-lite latency varies a lot)

PROMPT = f"""This image is a hand-drawn diagram of a moment in a warehouse, drawn like an incident sketch.
Turn it into a search sketch of where things are in the picture.

How people draw these diagrams:
- a stick figure, a circle with a body, or a box/circle labelled "person", "worker", "P" = a person
- a rectangle labelled "forklift" / "FL" (or a drawing of one) = a forklift; "robot" = robot; "cart" = transporter
- an arrow starting at an object = the way that object moves: make it that object's path (follow the arrow,
  2-6 points) and put its end_box where the arrow ends
- a big X, a crossed-out area, a dashed zone, or text such as "nobody", "no one here", "no spotter" = an
  absent zone: absent=true, label = what must not be there (person unless the text says otherwise).
  start_box = the WHOLE marked area: if an X or note marks a circled / outlined region, the zone is the
  full circle or outline (it may surround another object, e.g. a forklift), not just the X itself
- ignore titles, legends, scribbles and any text that is not a label

Use the diagram's own layout. Every coordinate is a FRACTION of the image size between 0 and 1
(left edge x=0, right edge x=1, top edge y=0, bottom edge y=1), never pixels.
Box sizes should match the drawn shapes.

{SKETCH_FORMAT}"""


def _to_fractions(data: dict, width: int, height: int) -> list:
    """Models sometimes mix pixel values into fractional coordinates: scale anything > 1.5 by the image size."""
    def fx(v):
        return v / width if isinstance(v, (int, float)) and v > 1.5 else v

    def fy(v):
        return v / height if isinstance(v, (int, float)) and v > 1.5 else v

    objects = data.get("objects", [])
    for o in objects:
        if not isinstance(o, dict):
            continue
        for key in ("start_box", "end_box"):
            b = o.get(key)
            if isinstance(b, dict):
                o[key] = {k: (fx(v) if k in "xw" else fy(v)) for k, v in b.items()}
        if isinstance(o.get("path"), list):
            o["path"] = [[fx(p[0]), fy(p[1])] if isinstance(p, (list, tuple)) and len(p) == 2 else p
                         for p in o["path"]]
    return objects


@weave.op(name="diagram_to_sketch")
def diagram_to_sketch(image: bytes, mime_type: str) -> dict:
    """Returns {"sketch", "model", "attempts", "elapsed_ms"}. One retry on an invalid sketch or a timeout."""
    t = time.perf_counter()
    width, height = Image.open(io.BytesIO(image)).size
    contents = [image_part(image, mime_type), PROMPT]
    error = None
    for attempt in (1, 2):
        try:
            raw, model = generate_json(contents, timeout_s=DIAGRAM_TIMEOUT_S)
        except VisionTimeout:
            if attempt == 2:
                raise
            continue
        try:
            sketch = normalize_objects(_to_fractions(_extract_json(raw), width, height))
            return {"sketch": sketch, "model": model, "attempts": attempt,
                    "elapsed_ms": round((time.perf_counter() - t) * 1000)}
        except SketchParseError as e:
            error = e
            contents = [image_part(image, mime_type), PROMPT,
                        f"Your previous answer was invalid ({e}). Reply with only the corrected JSON object."]
    raise SketchParseError(f"vision model returned an invalid sketch twice: {error}")
