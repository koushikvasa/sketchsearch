"""Matcher eval: human-like noisy sketches built from real GT segments must retrieve their source segment.

Samples 100 GT segments (70 person-only, every person+forklift segment, the rest robot/transporter),
builds a 1-3 object sketch from each segment's real tracks in one 1.5 s window (like "More like this"),
adds noise (box centers +-0.05, sizes +-20%, paths simplified to 3-5 points), searches the full index
and reports Recall@1/5/10 + MRR overall and per category, latency, and ablations. Results go to
data/eval_report.json and W&B Weave (WEAVE_PROJECT).

Configs: "plan" (PLAN_WEIGHTS, plan section 7) vs "tuned" (DEFAULT_WEIGHTS), ablations on the tuned
weights, and plan vs tuned again on a held-out query set (--holdout-seed) to check the tuning generalizes.

  uv run python -m scripts.eval_matcher
  uv run python -m scripts.eval_matcher --weights '{"relations": 0.2, "position": 0.35}'  # try a variant
"""

import argparse
import json
import os
import time
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from app.config import DATA_DIR, WEAVE_PROJECT
from app.matcher import DEFAULT_WEIGHTS, PLAN_WEIGHTS, Matcher
from app.models import Box, Segment, Sketch, SketchObject, TrackPoint
from app.sketches import window_starts, window_tracks
from app.sources.local import LocalSource

REPORT = DATA_DIR / "eval_report.json"
CATEGORY_SIZES = {"person_only": 70, "person_forklift": None, "vehicle": None}  # None: all / the rest
VEHICLES = {"robot", "transporter"}
JITTER = 0.05
SIZE_NOISE = 0.20
PATH_POINTS = (3, 5)
WINDOW = 1.5


# ---------- sampling ----------

def categorize(seg: Segment) -> str | None:
    labels = {t.label for t in seg.tracks}
    if {"person", "forklift"} <= labels:
        return "person_forklift"
    if labels & VEHICLES:
        return "vehicle"
    if labels == {"person"}:
        return "person_only"
    return None


def sample_segments(segments: list[Segment], n: int, rng: np.random.Generator) -> list[tuple[Segment, str]]:
    by_cat = defaultdict(list)
    for s in segments:
        if (c := categorize(s)) and _windows_for(s, c):
            by_cat[c].append(s)
    picked = [(s, "person_forklift") for s in by_cat["person_forklift"]]
    n_person = min(CATEGORY_SIZES["person_only"], len(by_cat["person_only"]))
    n_vehicle = min(max(0, n - n_person - len(picked)), len(by_cat["vehicle"]))
    for cat, k in (("person_only", n_person), ("vehicle", n_vehicle)):
        idx = rng.choice(len(by_cat[cat]), size=k, replace=False)
        picked += [(by_cat[cat][i], cat) for i in sorted(idx)]
    return picked


def _windows_for(seg: Segment, category: str) -> list[float]:
    """Window starts where the category's defining objects are all visible (>= 2 points)."""
    need = {"person_only": [{"person"}], "person_forklift": [{"forklift"}, {"person"}], "vehicle": [VEHICLES]}[category]
    out = []
    for t0 in window_starts(seg):
        labels = {tr.label for tr, _ in window_tracks(seg, t0, t0 + WINDOW)}
        if all(labels & group for group in need):
            out.append(t0)
    return out


# ---------- noisy sketch ----------

def _clamp_box(cx, cy, w, h) -> Box:
    w, h = min(w, 1.0), min(h, 1.0)
    x = min(max(cx - w / 2, 0.0), 1.0 - w)
    y = min(max(cy - h / 2, 0.0), 1.0 - h)
    return Box(x=round(x, 4), y=round(y, 4), w=round(w, 4), h=round(h, 4))


def noisy_object(obj_id: str, label: str, pts: list[TrackPoint], rng: np.random.Generator) -> SketchObject:
    c = np.array([[p.box.x + p.box.w / 2, p.box.y + p.box.h / 2] for p in pts])
    j_start, j_end = rng.uniform(-JITTER, JITTER, 2), rng.uniform(-JITTER, JITTER, 2)
    sx, sy = rng.uniform(1 - SIZE_NOISE, 1 + SIZE_NOISE, 2)
    m = int(rng.integers(PATH_POINTS[0], PATH_POINTS[1] + 1))
    keep = np.unique(np.linspace(0, len(c) - 1, m).round().astype(int))
    frac = keep / max(len(c) - 1, 1)
    path = c[keep] + j_start + frac[:, None] * (j_end - j_start)  # path ends sit on the jittered boxes
    b0, b1 = pts[0].box, pts[-1].box
    return SketchObject(
        id=obj_id, label=label,
        start_box=_clamp_box(*path[0], b0.w * sx, b0.h * sy),
        end_box=_clamp_box(*path[-1], b1.w * sx, b1.h * sy),
        path=[(round(float(x), 4), round(float(y), 4)) for x, y in np.clip(path, 0, 1)],
    )


def build_query(seg: Segment, category: str, rng: np.random.Generator) -> dict:
    t0 = float(rng.choice(_windows_for(seg, category)))
    tracks = window_tracks(seg, t0, t0 + WINDOW)
    persons = [tp for tp in tracks if tp[0].label == "person"]
    if category == "person_only":
        k = min(int(rng.integers(1, 4)), len(persons))
        chosen = [persons[i] for i in sorted(rng.choice(len(persons), size=k, replace=False))]
    else:
        anchor_labels = {"forklift"} if category == "person_forklift" else VEHICLES
        anchors = [tp for tp in tracks if tp[0].label in anchor_labels]
        anchor = anchors[int(rng.integers(len(anchors)))]
        lo = 1 if category == "person_forklift" else 0  # forklift queries always show a person too
        k_p = min(int(rng.integers(lo, 3)), len(persons))
        ac = anchor[1][0].box
        near = sorted(persons, key=lambda tp: np.hypot(tp[1][0].box.x - ac.x, tp[1][0].box.y - ac.y))
        chosen = [anchor] + near[:k_p]
    objects = [noisy_object(f"o{i + 1}", tr.label, pts, rng) for i, (tr, pts) in enumerate(chosen)]
    return {
        "segment_id": seg.segment_id, "camera_id": seg.camera_id, "start": seg.start, "category": category,
        "window": [t0, t0 + WINDOW], "labels": [o.label for o in objects],
        "sketch": Sketch(objects=objects),
    }


# ---------- metrics ----------

def rank_of(results: list[dict], query: dict, segments: dict[str, Segment]) -> tuple[int | None, int | None]:
    """(strict rank of the source segment, lenient rank counting adjacent segments of the same camera)."""
    strict = lenient = None
    for i, r in enumerate(results, 1):
        if strict is None and r["segment_id"] == query["segment_id"]:
            strict = i
        s = segments[r["segment_id"]]
        if lenient is None and s.camera_id == query["camera_id"] and abs(s.start - query["start"]) <= 4 + 1e-6:
            lenient = i
        if strict is not None:
            break
    return strict, lenient


def metrics(ranks: list[int | None], lenient: list[int | None]) -> dict:
    n = len(ranks)
    def recall(k, rs):
        return round(sum(r is not None and r <= k for r in rs) / n, 3) if n else None
    return {
        "n": n,
        "recall@1": recall(1, ranks), "recall@5": recall(5, ranks), "recall@10": recall(10, ranks),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3) if n else None,
        "lenient_recall@5": recall(5, lenient),
    }


def run_config(name: str, weights: dict, matcher: Matcher, queries: list[dict], segments: dict) -> dict:
    ranks, lenient, lat = [], [], []
    for q in queries:
        t = time.perf_counter()
        results = matcher.search(q["sketch"], weights=weights)
        lat.append((time.perf_counter() - t) * 1000)
        r, l = rank_of(results, q, segments)
        ranks.append(r)
        lenient.append(l)
        q.setdefault("ranks", {})[name] = r
    by_cat = {}
    for cat in sorted({q["category"] for q in queries}):
        idx = [i for i, q in enumerate(queries) if q["category"] == cat]
        by_cat[cat] = metrics([ranks[i] for i in idx], [lenient[i] for i in idx])
    by_k = {}
    for k in sorted({len(q["labels"]) for q in queries}):
        idx = [i for i, q in enumerate(queries) if len(q["labels"]) == k]
        by_k[f"{k}_objects"] = metrics([ranks[i] for i in idx], [lenient[i] for i in idx])
    return {
        "weights": weights, "overall": metrics(ranks, lenient), "by_category": by_cat, "by_object_count": by_k,
        "latency_ms": {"mean": round(float(np.mean(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1)},
    }


def latency_at_scale(segments: list[Segment], queries: list[dict], copies: int = 7) -> dict:
    big = [s.model_copy(update={"segment_id": f"{s.segment_id}#{i}"}) for i in range(copies) for s in segments]
    m = Matcher(big)
    lat = []
    for q in queries[::10]:
        t = time.perf_counter()
        m.search(q["sketch"])
        lat.append((time.perf_counter() - t) * 1000)
    return {"segments": len(big), "queries": len(lat), "mean_ms": round(float(np.mean(lat)), 1),
            "max_ms": round(float(np.max(lat)), 1)}


def ablate(weights: dict, drop: str) -> dict:
    return {**weights, drop: 0.0}


def print_table(configs: dict):
    print(f"\n{'config':16s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'lenR@5':>7s} {'ms':>6s}")
    for name, c in configs.items():
        o = c["overall"]
        print(f"{name:16s} {o['recall@1']:6.3f} {o['recall@5']:6.3f} {o['recall@10']:6.3f} {o['mrr']:6.3f} "
              f"{o['lenient_recall@5']:7.3f} {c['latency_ms']['mean']:6.1f}")
    for name in configs:
        if name.startswith("no_") or name == "plan":
            continue
        print(f"\n{name} by category / object count:")
        for group in ("by_category", "by_object_count"):
            for key, m in configs[name][group].items():
                print(f"  {key:16s} n={m['n']:3d}  R@1 {m['recall@1']:.3f}  R@5 {m['recall@5']:.3f}  "
                      f"R@10 {m['recall@10']:.3f}  MRR {m['mrr']:.3f}")


def make_queries(segments: list[Segment], n: int, seed: int) -> list[dict]:
    picked = sample_segments(segments, n, np.random.default_rng(seed))
    return [build_query(s, c, np.random.default_rng([seed, zlib.crc32(s.segment_id.encode())])) for s, c in picked]


def describe(queries: list[dict]) -> dict:
    return {
        "categories": dict(sorted(Counter(q["category"] for q in queries).items())),
        "objects_per_sketch": dict(sorted(Counter(len(q["labels"]) for q in queries).items())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-seed", type=int, default=7)
    ap.add_argument("--weights", type=json.loads, default=None, help="override tuned weights (JSON)")
    ap.add_argument("--no-weave", action="store_true")
    args = ap.parse_args()

    source = LocalSource("gt")
    segments = source.candidates(set())
    seg_by_id = {s.segment_id: s for s in segments}
    queries = make_queries(segments, args.n, args.seed)
    holdout = make_queries(segments, args.n, args.holdout_seed)
    print(f"{len(queries)} queries (seed {args.seed}): {describe(queries)}")

    t = time.perf_counter()
    matcher = Matcher(segments, source.label_aliases)
    prep_s = round(time.perf_counter() - t, 2)

    tuned = {**DEFAULT_WEIGHTS, **(args.weights or {})}
    plan_runs = {"plan": (PLAN_WEIGHTS, queries), "tuned": (tuned, queries)}
    plan_runs |= {f"no_{c}": (ablate(tuned, c), queries) for c in ("relations", "motion", "position")}
    plan_runs |= {"holdout_plan": (PLAN_WEIGHTS, holdout), "holdout_tuned": (tuned, holdout)}

    use_weave = not args.no_weave and bool(os.getenv("WANDB_API_KEY"))
    run = lambda name, weights, qs: run_config(name, weights, matcher, qs, seg_by_id)  # noqa: E731
    if use_weave:
        import weave

        weave.init(WEAVE_PROJECT)
        run = weave.op(name="matcher_eval_config")(run)
    configs = {name: run(name, w, qs) for name, (w, qs) in plan_runs.items()}

    scale = latency_at_scale(segments, queries)
    report = {
        "meta": {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "index": "gt", "segments": len(segments), "windows": matcher.n_windows, "prepare_s": prep_s,
            "n_queries": len(queries), "seed": args.seed, "holdout_seed": args.holdout_seed,
            "queries": describe(queries), "holdout_queries": describe(holdout),
            "noise": {"center_jitter": JITTER, "size_noise": SIZE_NOISE, "path_points": list(PATH_POINTS),
                      "objects": [1, 3], "window_s": WINDOW},
            "target": "the query's own segment (strict); lenient also accepts the adjacent 4 s segments",
            "tuning": "one grid search over weights on the seed-42 queries; holdout_* rows use unseen queries",
        },
        "configs": configs,
        "latency_at_scale": scale,
        "queries": [{k: v for k, v in q.items() if k != "sketch"} | {"sketch": q["sketch"].model_dump()}
                    for q in queries],
    }
    REPORT.write_text(json.dumps(report, indent=1))
    print_table(configs)
    print(f"\nlatency: {configs['tuned']['latency_ms']} on {len(segments)} segments; "
          f"{scale['mean_ms']} ms mean / {scale['max_ms']} ms max on {scale['segments']} segments")
    print(f"wrote {REPORT.relative_to(DATA_DIR.parent)}")
    if use_weave:
        weave.publish({k: v for k, v in report.items() if k != "queries"}, name="matcher-eval-report")
        weave.publish(weave.Dataset(name="matcher-eval-queries", rows=[
            {"segment_id": q["segment_id"], "category": q["category"], "labels": q["labels"],
             "window": q["window"], "ranks": q["ranks"]} for q in queries]))
        print(f"logged to Weave project {WEAVE_PROJECT!r}")


if __name__ == "__main__":
    main()
