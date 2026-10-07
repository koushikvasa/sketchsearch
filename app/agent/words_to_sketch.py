"""✨ Words -> Sketch: a sentence becomes sketch JSON via the text LLM (Groq now, W&B Inference Friday)."""

import json
import time

import weave

from app.config import LLM_PROVIDER
from app.llm import chat_json, provider_config
from app.models import Box, Sketch, SketchObject
from app.presets import FORKLIFT, PERSON, moving
from app.sketch_json import SKETCH_FORMAT, SketchParseError, parse_sketch

SCENE = """The cameras are fixed and look down a warehouse aisle. The far end of the aisle is the top-centre of the
frame (y about 0.05-0.2); the area near the camera is the bottom (y about 0.6-0.95); shelving fills the
left and right edges. Things far away are small: a far person is about 0.02 wide x 0.09 tall, a near
person about 0.05 x 0.22. A forklift is usually at the far end, about 0.04 x 0.10. A robot (small mobile
robot) is about 0.04 x 0.04; a transporter (low cart) about 0.06 x 0.04.
A sketch covers about 1.5 seconds: a walking person moves about 0.05-0.15 in that time. "Toward" /
"approaching" means the path heads at the other object's centre; "away" means the opposite."""

SYSTEM = f"""You turn a description of a moment in warehouse CCTV into a search sketch.

{SCENE}
Only draw what the description mentions. If it mentions no position, pick a natural one in the aisle.

{SKETCH_FORMAT}"""


def _ex(text: str, objects: list[SketchObject]) -> list[dict]:
    answer = json.dumps(Sketch(objects=objects).model_dump(exclude_none=True, exclude={"text", "camera_ids"}))
    return [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]


def _absent(id: str, label: str, x, y, w, h) -> SketchObject:
    return SketchObject(id=id, label=label, start_box=Box(x=x, y=y, w=w, h=h), absent=True)


FEW_SHOT = [
    *_ex("two people walking toward each other in the aisle", [
        moving("upper", "person", (0.46, 0.30), (0.47, 0.42), PERSON),
        moving("lower", "person", (0.50, 0.64), (0.49, 0.52), (0.04, 0.2)),
    ]),
    *_ex("a worker walks up to the forklift", [
        moving("forklift", "forklift", (0.45, 0.10), (0.45, 0.10), FORKLIFT).model_copy(update={"end_box": None, "path": None}),
        moving("worker", "person", (0.50, 0.26), (0.48, 0.17), PERSON),
    ]),
    *_ex("forklift at the end of the aisle with nobody near it", [
        moving("forklift", "forklift", (0.45, 0.10), (0.45, 0.10), FORKLIFT).model_copy(update={"end_box": None, "path": None}),
        _absent("nobody", "person", 0.30, 0.0, 0.30, 0.30),
    ]),
    *_ex("a group of three people standing together on the right", [
        moving(f"p{i}", "person", c, c, (0.035, 0.16)).model_copy(update={"end_box": None, "path": None})
        for i, c in enumerate([(0.62, 0.52), (0.67, 0.54), (0.645, 0.60)], 1)
    ]),
    *_ex("a person crossing from left to right near the camera", [
        moving("walker", "person", (0.30, 0.72), (0.44, 0.73), (0.05, 0.22)),
    ]),
]


@weave.op(name="words_to_sketch")
def words_to_sketch(text: str, provider: str = LLM_PROVIDER) -> dict:
    """Returns {"sketch", "provider", "model", "attempts", "elapsed_ms"}; raises SketchParseError after 1 retry."""
    t = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM}, *FEW_SHOT, {"role": "user", "content": text}]
    error = None
    for attempt in (1, 2):
        raw = chat_json(messages, provider=provider)
        try:
            sketch = parse_sketch(raw)
            return {
                "sketch": sketch.model_copy(update={"text": text}),
                "provider": provider, "model": provider_config(provider)["model"], "attempts": attempt,
                "elapsed_ms": round((time.perf_counter() - t) * 1000),
            }
        except SketchParseError as e:
            error = e
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That sketch is invalid: {e}. Reply with only the corrected JSON object."}]
    raise SketchParseError(f"model returned an invalid sketch twice: {error}")
