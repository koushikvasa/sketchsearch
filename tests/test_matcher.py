"""Synthetic matcher tests (plan section 7.7) plus absent-box cases."""

import numpy as np
import pytest

from app.matcher import Matcher
from app.models import Box, Segment, Sketch, SketchObject, Track, TrackPoint
from app.sketches import sketch_from_segment

SIZE = (0.05, 0.15)  # person-ish box


def track(track_id, label, start, end, size=SIZE, t_end=4.0, fps=5):
    """Linear track from start center to end center over the whole segment."""
    ts = np.arange(0, t_end + 1e-9, 1 / fps)
    pts = []
    for t in ts:
        a = t / t_end
        cx, cy = start[0] + a * (end[0] - start[0]), start[1] + a * (end[1] - start[1])
        pts.append(TrackPoint(t=round(float(t), 3), box=Box(x=cx - size[0] / 2, y=cy - size[1] / 2, w=size[0], h=size[1])))
    return Track(label=label, track_id=track_id, points=pts)


def segment(segment_id, tracks, camera_id="cam0", keyframe_only=False):
    return Segment(segment_id=segment_id, video_id=camera_id, camera_id=camera_id, start=0, end=4,
                   tracks=tracks, keyframe_only=keyframe_only)


def box_at(center, size=SIZE):
    return Box(x=center[0] - size[0] / 2, y=center[1] - size[1] / 2, w=size[0], h=size[1])


def obj(id, label, start, end=None, size=SIZE, absent=False):
    return SketchObject(id=id, label=label, start_box=box_at(start, size),
                        end_box=box_at(end, size) if end else None,
                        path=[start, end] if end else None, absent=absent)


def by_id(results):
    return {r["segment_id"]: r for r in results}


# Two people: one walks left->right on the left side, one stands on the right.
WALKER = dict(start=(0.2, 0.5), end=(0.4, 0.5))
STANDER = dict(start=(0.75, 0.5), end=(0.75, 0.5))


@pytest.fixture
def scene():
    return segment("scene", [track("a", "person", **WALKER), track("b", "person", **STANDER)])


def test_exact_match_scores_about_one(scene):
    distractors = [
        segment("other1", [track("c", "person", (0.8, 0.2), (0.6, 0.3)), track("d", "person", (0.1, 0.9), (0.1, 0.9))]),
        segment("other2", [track("e", "person", (0.5, 0.5), (0.5, 0.8)), track("f", "person", (0.3, 0.1), (0.9, 0.1))]),
    ]
    sketch, _ = sketch_from_segment(scene)
    results = Matcher([scene, *distractors]).search(sketch)
    assert results[0]["segment_id"] == "scene"
    assert results[0]["score"] > 0.99
    assert results[1]["score"] < 0.9


def test_mirrored_layout_has_low_relations():
    # Distinct labels, so the assignment can't simply swap identities to fit a mirrored drawing.
    seg = segment("s", [track("f", "forklift", (0.2, 0.4), (0.2, 0.4), size=(0.1, 0.1)),
                        track("p", "person", (0.6, 0.6), (0.6, 0.6))])
    exact = Sketch(objects=[obj("o1", "forklift", (0.2, 0.4), size=(0.1, 0.1)), obj("o2", "person", (0.6, 0.6))])
    mirrored = Sketch(objects=[obj("o1", "forklift", (0.8, 0.6), size=(0.1, 0.1)), obj("o2", "person", (0.4, 0.4))])
    m = Matcher([seg])
    assert m.search(exact)[0]["components"]["relations"] > 0.95
    assert m.search(mirrored)[0]["components"]["relations"] < 0.4


def test_reversed_motion_has_low_motion(scene):
    # The walker covers 0.075 per 1.5 s window.
    same = Sketch(objects=[obj("o1", "person", (0.2, 0.5), (0.275, 0.5))])
    reversed_ = Sketch(objects=[obj("o1", "person", (0.2, 0.5), (0.125, 0.5))])  # same start, walks the other way
    m = Matcher([scene])
    assert m.search(same)[0]["components"]["motion"] > 0.95
    assert m.search(reversed_)[0]["components"]["motion"] < 0.4


def test_count_aware_filter():
    one = segment("one", [track("a", "person", (0.5, 0.5), (0.5, 0.5))])
    two = segment("two", [track("a", "person", (0.3, 0.5), (0.3, 0.5)), track("b", "person", (0.7, 0.5), (0.7, 0.5))])
    m = Matcher([one, two])
    sketch2 = Sketch(objects=[obj("o1", "person", (0.3, 0.5)), obj("o2", "person", (0.7, 0.5))])
    assert [r["segment_id"] for r in m.search(sketch2)] == ["two"]
    sketch1 = Sketch(objects=[obj("o1", "person", (0.5, 0.5))])
    assert {r["segment_id"] for r in m.search(sketch1)} == {"one", "two"}
    assert m.search(Sketch(objects=[obj("o1", "forklift", (0.5, 0.5))])) == []


def test_hungarian_assigns_the_right_tracks():
    seg = segment("s", [
        track("left", "person", (0.2, 0.5), (0.2, 0.5)),
        track("mid", "person", (0.5, 0.5), (0.5, 0.5)),
        track("right", "person", (0.8, 0.5), (0.8, 0.5)),
    ])
    sketch = Sketch(objects=[obj("A", "person", (0.82, 0.5)), obj("B", "person", (0.18, 0.5))])
    assert Matcher([seg]).search(sketch)[0]["assignment"] == {"A": "right", "B": "left"}


def test_converging_beats_diverging():
    converge = segment("converge", [track("a", "person", (0.3, 0.5), (0.45, 0.5)), track("b", "person", (0.7, 0.5), (0.55, 0.5))])
    diverge = segment("diverge", [track("a", "person", (0.3, 0.5), (0.15, 0.5)), track("b", "person", (0.7, 0.5), (0.85, 0.5))])
    # The sketch only shows where they start plus "coming together"; the exact paths are rough.
    sketch = Sketch(objects=[obj("o1", "person", (0.3, 0.5), (0.4, 0.52)), obj("o2", "person", (0.7, 0.5), (0.6, 0.48))])
    results = by_id(Matcher([converge, diverge]).search(sketch))
    assert results["converge"]["score"] > results["diverge"]["score"]
    assert results["converge"]["components"]["motion"] > results["diverge"]["components"]["motion"] + 0.3


def test_camera_and_segment_filters(scene):
    other = segment("cam1_scene", scene.tracks, camera_id="cam1")
    m = Matcher([scene, other])
    sketch, _ = sketch_from_segment(scene)
    assert [r["segment_id"] for r in m.search(sketch, camera_ids=["cam1"])] == ["cam1_scene"]
    assert [r["segment_id"] for r in m.search(sketch, segment_ids=["scene"])] == ["scene"]


def test_label_aliases():
    seg = segment("s", [track("t", "truck", (0.5, 0.3), (0.5, 0.3), size=(0.1, 0.1))])
    sketch = Sketch(objects=[obj("o1", "forklift", (0.5, 0.3), size=(0.1, 0.1))])
    assert Matcher([seg], aliases={"forklift": ["truck"]}).search(sketch)[0]["assignment"] == {"o1": "t"}
    assert Matcher([seg], aliases={"forklift": []}).search(sketch) == []


def test_keyframe_only_segments_still_score():
    def still(track_id, center):
        return Track(label="person", track_id=track_id, points=[TrackPoint(t=2.0, box=box_at(center))])

    kf = segment("kf", [still("a", (0.3, 0.5)), still("b", (0.7, 0.5))], keyframe_only=True)
    kf_far = segment("kf_far", [still("a", (0.1, 0.1)), still("b", (0.9, 0.9))], keyframe_only=True)
    # Drawn with motion; keyframe-only segments can't check motion, so it is left out and renormalized.
    sketch = Sketch(objects=[obj("o1", "person", (0.3, 0.5), (0.4, 0.5)), obj("o2", "person", (0.7, 0.5))])
    results = Matcher([kf, kf_far]).search(sketch)
    assert [r["segment_id"] for r in results] == ["kf", "kf_far"]
    assert results[0]["score"] > 0.95
    assert set(results[0]["components"]) == {"position", "size", "relations"}


# ---------- absent ("nobody here") boxes ----------

FORKLIFT = dict(start=(0.5, 0.3), end=(0.5, 0.5), size=(0.1, 0.1))
NEAR_ZONE = Box(x=0.3, y=0.2, w=0.4, h=0.4)


def absent_person(zone=NEAR_ZONE):
    return SketchObject(id="nobody", label="person", start_box=zone, absent=True)


def forklift_sketch(*extra):
    return Sketch(objects=[obj("fk", "forklift", FORKLIFT["start"], FORKLIFT["end"], size=FORKLIFT["size"]), *extra])


def test_absent_box_penalizes_a_person_in_the_zone():
    alone = segment("alone", [track("f", "forklift", **FORKLIFT), track("p", "person", (0.9, 0.8), (0.9, 0.8))])
    spotter = segment("spotter", [track("f", "forklift", **FORKLIFT), track("p", "person", (0.4, 0.45), (0.4, 0.45))])
    m = Matcher([alone, spotter])
    plain = by_id(m.search(forklift_sketch()))
    withabs = by_id(m.search(forklift_sketch(absent_person())))
    assert withabs["alone"]["absence_ok"] is True
    assert withabs["alone"]["score"] == pytest.approx(plain["alone"]["score"])
    assert withabs["spotter"]["absence_ok"] is False
    assert withabs["spotter"]["score"] == pytest.approx(0.2 * plain["spotter"]["score"], abs=1e-3)
    assert plain["alone"]["absence_ok"] is None


def test_absent_box_picks_the_moment_before_a_person_walks_in():
    # The person enters the zone (x <= 0.7) at t = 2 s; only windows ending before that satisfy the sketch.
    walk_in = segment("walk_in", [track("f", "forklift", **FORKLIFT), track("p", "person", (0.95, 0.4), (0.45, 0.4))])
    result = Matcher([walk_in]).search(forklift_sketch(absent_person()))[0]
    assert result["absence_ok"] is True
    assert result["window"][1] < 2.0
    # A person inside the zone for the whole segment can't be avoided.
    stays = segment("stays", [track("f", "forklift", **FORKLIFT), track("p", "person", (0.6, 0.4), (0.62, 0.4))])
    assert Matcher([stays]).search(forklift_sketch(absent_person()))[0]["absence_ok"] is False


def test_absent_box_ignores_the_drawn_object_itself():
    seg = segment("s", [track("p", "person", (0.5, 0.5), (0.5, 0.5))])
    sketch = Sketch(objects=[obj("o1", "person", (0.5, 0.5)), absent_person(Box(x=0.3, y=0.3, w=0.4, h=0.4))])
    assert Matcher([seg]).search(sketch)[0]["absence_ok"] is True


def test_absent_box_only_concerns_its_label():
    seg = segment("s", [track("f", "forklift", **FORKLIFT), track("r", "robot", (0.45, 0.4), (0.45, 0.4))])
    assert Matcher([seg]).search(forklift_sketch(absent_person()))[0]["absence_ok"] is True


def test_absent_only_sketch():
    empty = segment("empty", [track("p", "person", (0.9, 0.9), (0.9, 0.9))])
    busy = segment("busy", [track("p", "person", (0.5, 0.4), (0.5, 0.4))])
    results = Matcher([empty, busy]).search(Sketch(objects=[absent_person()]))
    assert [(r["segment_id"], r["absence_ok"]) for r in results] == [("empty", True), ("busy", False)]
