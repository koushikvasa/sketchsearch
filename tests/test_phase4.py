"""Offline tests for the AI layer: sketch JSON parsing, verification questions/verdicts/cache, explanations."""

import json

import pytest

from app.models import Box, Sketch, SketchObject
from app.presets import PRESETS
from app.sketch_json import SketchParseError, parse_sketch
from tests.test_matcher import segment, track


def test_parse_sketch_normalizes_model_output():
    raw = """```json
    {"objects": [
      {"id": "w 1", "label": "Worker", "start_box": {"x": 0.1, "y": 0.5, "w": 0.05, "h": 0.2},
       "path": [[0.125, 0.6], [0.3, 0.6]]},
      {"id": "zone", "label": "person", "absent": true, "start_box": [0.6, 0.0, 0.3, 0.3],
       "path": [[0, 0], [1, 1]]},
      {"id": "fl", "label": "fork lift", "start_box": {"x": 0.98, "y": 0.1, "w": 0.1, "h": 0.1},
       "end_box": {"x": 0.5, "y": 0.1, "w": 0.1, "h": 0.1}}
    ]}
    ```"""
    sketch = parse_sketch(raw)
    walker, zone, fork = sketch.objects
    assert (walker.id, walker.label) == ("w1", "person")
    assert walker.end_box is not None and walker.end_box.x == pytest.approx(0.3 - 0.025)  # end follows the path
    assert zone.absent and zone.path is None and zone.end_box is None
    assert fork.label == "forklift" and fork.start_box.x == pytest.approx(0.9)  # clamped into the frame
    assert fork.path == [(0.95, 0.15), (0.55, 0.15)]  # end box without path gets a straight path


@pytest.mark.parametrize("raw, message", [
    ("not json", "no JSON object"),
    ('{"objects": []}', "no objects"),
    ('{"objects": [{"label": "car", "start_box": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}]}', "use one of"),
    ('{"objects": [{"label": "person", "start_box": {"x": 120, "y": 40, "w": 30, "h": 80}}]}', "fractions"),
    ('{"objects": [{"label": "person", "absent": true, "start_box": {"x": 0, "y": 0, "w": 0.3, "h": 0.3}}]}',
     "at least one object"),
])
def test_parse_sketch_rejects_bad_output(raw, message):
    with pytest.raises(SketchParseError, match=message):
        parse_sketch(raw)


def test_diagram_pixel_coordinates_become_fractions():
    from app.vision.diagram import _to_fractions

    objs = _to_fractions({"objects": [{"label": "person", "start_box": {"x": 0.2, "y": 540, "w": 64, "h": 0.3},
                                       "path": [[640, 360], [0.9, 0.5]]}]}, 1280, 720)
    assert objs[0]["start_box"] == {"x": 0.2, "y": 0.75, "w": 0.05, "h": 0.3}
    assert objs[0]["path"] == [[0.5, 0.5], [0.9, 0.5]]


def test_verification_question_uses_plain_spatial_words():
    from app.verify import build_question

    q = build_question(PRESETS["forklift_approach"]["sketch"], [1.0, 2.5])
    assert "Focus on 1.0-2.5 s" in q
    assert "a forklift in the top-centre of the frame, staying roughly in place" in q
    assert "moving left and up the frame (away from the camera) toward the forklift" in q
    assert '"verdict": "YES" | "NO" | "UNSURE"' in q

    q = build_question(PRESETS["forklift_alone"]["sketch"])
    assert "no person visible anywhere inside the image region spanning 30%-60% of the frame width" in q
    assert "(the area around the forklift)" in q

    q = build_question(PRESETS["converging"]["sketch"])
    assert "person A" in q and "toward person B" in q and "toward person A" in q


@pytest.mark.parametrize("text, verdict", [
    ('{"verdict": "yes", "reason": "Clear."}', "YES"),
    ('```json\n{"verdict": "NO", "reason": "Nobody moves."}\n```', "NO"),
    ('{"verdict": "MAYBE", "reason": "?"}', "UNSURE"),
    ("I think so", "UNSURE"),
])
def test_parse_verdict(text, verdict):
    from app.verify import parse_verdict

    assert parse_verdict(text)["verdict"] == verdict


def test_verify_segment_caches_and_turns_timeouts_into_unsure(tmp_path, monkeypatch):
    import app.verify as verify
    from app.vision.gemini import VisionTimeout

    calls = []

    class FakeSource:
        mode = "ok"

        def ask(self, segment_id, question):
            calls.append(segment_id)
            if self.mode == "timeout":
                raise VisionTimeout("slow")
            return json.dumps({"verdict": "NO", "reason": "The person walks away.", "_model": "fake-model"})

    fake = FakeSource()
    monkeypatch.setattr(verify, "VERIFY_CACHE_DIR", tmp_path)
    monkeypatch.setattr(verify, "get_source", lambda: fake)
    first = verify.verify_segment("seg_1", "Q?")
    second = verify.verify_segment("seg_1", "Q?")
    assert first["verdict"] == "NO" and first["model"] == "fake-model" and not first["cached"]
    assert second["cached"] and calls == ["seg_1"]

    fake.mode = "timeout"
    out = verify.verify_segment("seg_2", "Q?")
    assert out["verdict"] == "UNSURE" and f"{verify.VERIFY_TIMEOUT_S:g} s" in out["reason"]
    assert not list(tmp_path.glob("seg_2__*"))  # timeouts are not cached


def test_explanation_describes_the_real_tracks():
    from app.explain import explain

    seg = segment("s", [
        track("w", "person", (0.2, 0.5), (0.6, 0.5)),  # walks right toward the forklift
        track("f", "forklift", (0.7, 0.5), (0.7, 0.5), size=(0.1, 0.1)),
        track("p", "person", (0.95, 0.9), (0.95, 0.9)),
    ])
    sketch = Sketch(objects=[
        SketchObject(id="worker", label="person", start_box=Box(x=0.2, y=0.45, w=0.05, h=0.15)),
        SketchObject(id="fl", label="forklift", start_box=Box(x=0.65, y=0.45, w=0.1, h=0.1)),
        SketchObject(id="zone", label="person", start_box=Box(x=0.55, y=0.3, w=0.3, h=0.4), absent=True),
    ])
    result = {"window": [0.0, 1.5], "assignment": {"worker": "w", "fl": "f"}, "absence_ok": True,
              "components": {"position": 0.9, "size": 0.45}}
    text = explain(sketch, result, seg)
    assert text == ("Person on the left walks right toward the forklift; forklift on the right stays put; "
                    "no person in the marked zone; size is the weakest match (0.45).")
