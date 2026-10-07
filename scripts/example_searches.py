"""Three hand-drawn example searches on the GT index, with top-5 results and score breakdowns.

  uv run python -m scripts.example_searches
"""

from app.presets import PRESETS
from app.search import run_search


def main():
    for preset in PRESETS.values():
        sketch = preset["sketch"]
        res = run_search(sketch, top_n=5)
        print(f"\n=== {preset['title']}: {preset['description']} ===")
        print("  weights: " + ", ".join(f"{k} {v:.2f}" for k, v in res["weights"].items()))
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
