"""Demo sketches: shown as preset buttons in the UI and printed by scripts/example_searches.py."""

import json
from pathlib import Path

from app.config import DATA_DIR
from app.models import Box, Sketch, SketchObject

PERSON = (0.025, 0.12)
FORKLIFT = (0.035, 0.095)


def box(cx, cy, size) -> Box:
    return Box(x=round(cx - size[0] / 2, 4), y=round(cy - size[1] / 2, 4), w=size[0], h=size[1])


def moving(id, label, start, end, size) -> SketchObject:
    return SketchObject(id=id, label=label, start_box=box(*start, size), end_box=box(*end, size), path=[start, end])


def two_people(converging: bool = True) -> Sketch:
    """Two people in the aisle walking toward (or, same start layout, away from) each other."""
    upper, lower = ((0.46, 0.30), (0.47, 0.42)), ((0.50, 0.64), (0.49, 0.52))
    if not converging:
        upper, lower = (upper[0], (0.45, 0.18)), (lower[0], (0.51, 0.76))
    return Sketch(objects=[
        moving("upper", "person", *upper, PERSON),
        moving("lower", "person", *lower, (0.04, 0.2)),
    ])


# Ordered as the landing page's example chips.
PRESETS: dict[str, dict] = {
    "forklift_approach": {
        "chip": "Worker walks toward forklift",
        "title": "Person approaches forklift",
        "description": "A worker walks up to the forklift at the end of the aisle",
        "sketch": Sketch(objects=[
            moving("forklift", "forklift", (0.45, 0.10), (0.45, 0.10), FORKLIFT),
            moving("worker", "person", (0.50, 0.26), (0.48, 0.17), PERSON),
        ]),
        "camera_id": "warehouse_cam1",
    },
    "converging": {
        "chip": "Two people meet in an aisle",
        "title": "Two people converging",
        "description": "Two people in the aisle walking toward each other",
        "sketch": two_people(converging=True),
    },
    "forklift_alone": {
        "chip": "Forklift with nobody nearby",
        "title": "Forklift, nobody nearby",
        "description": "Forklift with no person inside the zone around it",
        "sketch": Sketch(objects=[
            moving("forklift", "forklift", (0.45, 0.10), (0.45, 0.10), FORKLIFT),
            SketchObject(id="nobody", label="person", start_box=Box(x=0.30, y=0.0, w=0.30, h=0.30), absent=True),
        ]),
        "camera_id": "warehouse_cam1",
    },
    "group": {
        "chip": "Group of 3 people",
        "title": "Group of three people",
        "description": "Three people standing close together in the aisle",
        "sketch": Sketch(objects=[
            SketchObject(id=f"p{i}", label="person", start_box=box(cx, cy, (0.035, 0.16)))
            for i, (cx, cy) in enumerate([(0.47, 0.45), (0.52, 0.47), (0.495, 0.53)], 1)
        ]),
    },
}


def presets_path(index: str) -> Path:
    return DATA_DIR / f"presets_{index}.json"


def load_presets(index: str) -> dict[str, dict]:
    """Example chips for an index: data/presets_<index>.json from scripts/make_presets.py, else the hand-made
    PRESETS above (drawn for the Warehouse_022 ground-truth index)."""
    path = presets_path(index)
    if not path.exists():
        return PRESETS
    data = json.loads(path.read_text())
    return {p["id"]: {**p, "sketch": Sketch.model_validate(p["sketch"])} for p in data["presets"]}
