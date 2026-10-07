"""Tests against the real GT index (data/index.json); skipped when it hasn't been built."""

import pytest

from app.config import INDEX_FILES

pytestmark = pytest.mark.skipif(not INDEX_FILES["gt"].exists(), reason="run scripts/build_index.py first")


@pytest.fixture(scope="module")
def source():
    from app.sources.local import LocalSource

    return LocalSource("gt")


@pytest.fixture(scope="module")
def matcher(source):
    from app.matcher import Matcher

    return Matcher(source.candidates(set()))


def test_forklift_with_no_person_nearby(source, matcher):
    """Real case: forklift in cam1 with an absent-person zone of +-0.2 around it.

    warehouse_cam1_108: the nearest person stays >= 0.27 from the forklift -> absence satisfied.
    warehouse_cam1_280: a person walks 0.04-0.08 from the forklift -> absence violated.
    """
    from app.models import Box, Sketch, SketchObject
    from app.sketches import sketch_from_segment

    sketch, _ = sketch_from_segment(source.get_segment("warehouse_cam1_108"), t0=0.0)
    fk = next(o for o in sketch.objects if o.label == "forklift")
    cx, cy = fk.start_box.x + fk.start_box.w / 2, fk.start_box.y + fk.start_box.h / 2
    x0, y0, x1, y1 = max(0.0, cx - 0.2), max(0.0, cy - 0.2), min(1.0, cx + 0.2), min(1.0, cy + 0.2)
    zone = Box(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
    query = Sketch(objects=[fk, SketchObject(id="nobody", label="person", start_box=zone, absent=True)])

    results = matcher.search(query)
    by_id = {r["segment_id"]: r for r in results}
    assert results[0]["segment_id"] == "warehouse_cam1_108"
    assert by_id["warehouse_cam1_108"]["absence_ok"] is True
    assert by_id["warehouse_cam1_280"]["absence_ok"] is False
    assert len(results) == 16  # every forklift segment is a candidate


def test_api_search_and_as_sketch_round_trip():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.get("/api/segments/warehouse_cam0_40/as-sketch", params={"max_objects": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sketch"]["objects"]) == 3 and len(body["window"]) == 2

    resp = client.post("/api/search", json={"sketch": body["sketch"], "top_n": 5})
    assert resp.status_code == 200
    top = resp.json()["results"][0]
    assert top["segment_id"] == "warehouse_cam0_40"
    assert top["clip_url"] == "clips/warehouse_cam0_40.mp4"
    assert set(top) >= {"score", "components", "assignment", "window", "absence_ok", "camera_id"}

    assert client.get("/api/segments/nope/as-sketch").status_code == 404


def _pair_distance_change(source, result) -> float:
    """How much the two matched real tracks' distance changes over the result window."""
    seg = source.get_segment(result["segment_id"])
    t0, t1 = result["window"]
    tracks = {t.track_id: t for t in seg.tracks}
    ends = []
    for track_id in result["assignment"].values():
        pts = [p for p in tracks[track_id].points if t0 - 1e-6 <= p.t <= t1 + 1e-6]
        ends.append([(p.box.x + p.box.w / 2, p.box.y + p.box.h / 2) for p in (pts[0], pts[-1])])
    (a0, a1), (b0, b1) = ends
    dist = lambda p, q: ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5  # noqa: E731
    return dist(a1, b1) - dist(a0, b0)


def test_converging_vs_diverging_rank_different_real_segments(source, matcher):
    """Same start layout, arrows reversed: the adaptive motion weight must change what comes first."""
    from app.matcher import has_clear_motion
    from app.presets import two_people

    converging, diverging = two_people(converging=True), two_people(converging=False)
    assert [o.start_box for o in converging.objects] == [o.start_box for o in diverging.objects]
    assert has_clear_motion(converging) and has_clear_motion(diverging)

    top_c = matcher.search(converging, top_n=1)[0]
    top_d = matcher.search(diverging, top_n=1)[0]
    assert top_c["segment_id"] != top_d["segment_id"]
    assert _pair_distance_change(source, top_c) < 0  # the matched people really get closer
    assert _pair_distance_change(source, top_d) > 0  # ... and really move apart
