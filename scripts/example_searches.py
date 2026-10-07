"""Three hand-drawn example searches on the GT index, with top-5 results and score breakdowns.

  uv run python -m scripts.example_searches
"""

from app.models import Box, Sketch, SketchObject
from app.search import run_search

PERSON = (0.025, 0.12)
FORKLIFT = (0.035, 0.095)


def box(cx, cy, size):
    return Box(x=cx - size[0] / 2, y=cy - size[1] / 2, w=size[0], h=size[1])


def moving(id, label, start, end, size):
    return SketchObject(id=id, label=label, start_box=box(*start, size), end_box=box(*end, size), path=[start, end])


EXAMPLES = {
    "Two people walking toward each other in the aisle": Sketch(objects=[
        moving("upper", "person", (0.46, 0.30), (0.47, 0.38), PERSON),
        moving("lower", "person", (0.50, 0.62), (0.49, 0.52), (0.04, 0.2)),
    ]),
    "Person walking up to the forklift at the end of the aisle": Sketch(objects=[
        moving("forklift", "forklift", (0.45, 0.10), (0.45, 0.10), FORKLIFT),
        moving("worker", "person", (0.50, 0.26), (0.48, 0.17), PERSON),
    ]),
    "Forklift with nobody within reach (absent person zone)": Sketch(objects=[
        moving("forklift", "forklift", (0.45, 0.10), (0.45, 0.10), FORKLIFT),
        SketchObject(id="nobody", label="person", start_box=Box(x=0.30, y=0.0, w=0.30, h=0.30), absent=True),
    ]),
}


def main():
    for title, sketch in EXAMPLES.items():
        res = run_search(sketch, top_n=5)
        print(f"\n=== {title} ===")
        for o in sketch.objects:
            b = o.start_box
            what = "ABSENT " if o.absent else ""
            move = ""
            if o.end_box:
                e = o.end_box
                move = f" -> ({e.x + e.w / 2:.2f},{e.y + e.h / 2:.2f})"
            print(f"  sketch: {what}{o.label} '{o.id}' at ({b.x + b.w / 2:.2f},{b.y + b.h / 2:.2f}){move}")
        print(f"  {len(res['results'])} shown, {res['elapsed_ms']} ms")
        for i, r in enumerate(res["results"], 1):
            comps = "  ".join(f"{k[:3]} {v:.2f}" for k, v in r["components"].items())
            absent = "" if r["absence_ok"] is None else f"  absence_ok={r['absence_ok']}"
            print(f"  {i}. {r['segment_id']:20s} score {r['score']:.3f}  [{comps}]  "
                  f"t={r['window'][0]:.1f}-{r['window'][1]:.1f}s{absent}  {r['assignment']}")


if __name__ == "__main__":
    main()
