"""scripts/make_presets.py on a synthetic index that contains each kind of moment."""

from app.matcher import Matcher
from app.search import GOOD
from scripts.make_presets import find_presets
from tests.test_matcher import segment, track


def scene(k: int):
    dx = 0.004 * k  # 8 near-copies, so every preset has >= 5 good matches
    return segment(f"cam0_{4 * k}", [
        track("a", "person", (0.30 + dx, 0.50), (0.42 + dx, 0.50)),  # walks toward the forklift
        track("f", "forklift", (0.62 + dx, 0.50), (0.62 + dx, 0.50), size=(0.1, 0.1)),
        track("b", "person", (0.80 + dx, 0.20), (0.80 + dx, 0.20)),  # a group of three
        track("c", "person", (0.85 + dx, 0.22), (0.85 + dx, 0.22)),
        track("d", "person", (0.82 + dx, 0.27), (0.82 + dx, 0.27)),
        track("g", "person", (0.15 + dx, 0.20), (0.40 + dx, 0.20)),  # two people meeting
        track("h", "person", (0.65 + dx, 0.05), (0.42 + dx, 0.05)),
    ])


def test_finds_one_preset_per_kind():
    segs = [scene(k) for k in range(8)]
    matcher = Matcher(segs)
    presets = find_presets(segs, matcher, log=lambda *a: None)
    by_kind = {p["kind"]: p for p in presets}
    assert list(by_kind) == ["approach", "meet", "absent", "group"]
    assert by_kind["approach"]["chip"] == "Worker walks toward forklift"  # mixed labels preferred
    assert by_kind["absent"]["chip"] == "Forklift with nobody nearby"
    assert [o.absent for o in by_kind["absent"]["sketch"].objects] == [False, True]
    assert by_kind["group"]["chip"] == "Group of 3 people"
    assert all(o.end_box is None for o in by_kind["group"]["sketch"].objects)  # no invented motion
    for p in presets:
        assert p["validated"]["good_matches"] >= 5
        good = sum(r["score"] >= GOOD for r in matcher.search(p["sketch"]))
        assert good == p["validated"]["good_matches"]


def test_rejects_candidates_without_enough_matches():
    segs = [scene(0)]  # a single scene: nothing can have 5 good matches
    assert find_presets(segs, Matcher(segs), log=lambda *a: None) == []


def test_presets_file_round_trip(tmp_path, monkeypatch):
    import app.presets as presets_mod
    from scripts.make_presets import write_presets

    monkeypatch.setattr(presets_mod, "DATA_DIR", tmp_path)
    segs = [scene(k) for k in range(8)]
    found = find_presets(segs, Matcher(segs), log=lambda *a: None)
    write_presets("synthetic", found)
    loaded = presets_mod.load_presets("synthetic")
    assert list(loaded) == [p["id"] for p in found]
    assert loaded["approach"]["sketch"] == found[0]["sketch"]
    assert presets_mod.load_presets("missing") is presets_mod.PRESETS  # hand-made fallback
