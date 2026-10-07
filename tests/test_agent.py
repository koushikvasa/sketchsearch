"""Agent run with the LLM, matcher and video model mocked: event order, refinement, budget, report."""

import asyncio

import pytest

import app.agent.runner as runner
from app.models import Box, Sketch, SketchObject


def _sketch(label: str) -> Sketch:
    return Sketch(objects=[SketchObject(id="o1", label=label, start_box=Box(x=0.4, y=0.1, w=0.05, h=0.1))])


def _result(seg: str, score: float) -> dict:
    cam, start = seg.rsplit("_", 1)
    return {"segment_id": seg, "camera_id": cam, "start": float(start), "end": float(start) + 4, "score": score,
            "window": [0.0, 1.5], "explanation": f"{seg} explained.", "clip_url": f"clips/{seg}.mp4",
            "frame_url": f"frames/{seg}.jpg", "absence_ok": None, "components": {"position": score}}


@pytest.fixture
def mocked(monkeypatch):
    plan = {"summary": "Look for people near forklifts.", "sketches": [
        {"id": "s1", "title": "Person near forklift", "why": "proximity", "sketch": _sketch("person")},
        {"id": "s2", "title": "Forklift alone", "why": "no spotter", "sketch": _sketch("forklift")},
    ]}
    results = {
        "person": [_result("cam1_100", 0.9), _result("cam1_104", 0.8), _result("cam1_200", 0.7)],
        "forklift": [_result("cam1_100", 0.85), _result("cam2_8", 0.6), _result("cam2_40", 0.5)],
        "robot": [_result("cam3_12", 0.7), _result("cam3_16", 0.65), _result("cam3_60", 0.6)],
    }
    verdicts = {"cam1_100": "YES", "cam1_104": "YES", "cam1_200": "NO",  # s1: 2/3 confirmed
                "cam2_8": "NO", "cam2_40": "NO",  # s2: 0-1/3 -> refine
                "cam3_12": "YES", "cam3_16": "YES", "cam3_60": "UNSURE"}
    refined = []

    def fake_refine(goal, item, vs):
        refined.append((item["id"], [v["verdict"] for v in vs]))
        return {"id": f"{item['id']}r", "title": "Robot instead", "why": "changed label", "sketch": _sketch("robot"),
                "refines": item["id"]}

    async def fake_verify_stream(sketch, items):
        for it in items:
            yield {"segment_id": it["segment_id"], "verdict": verdicts[it["segment_id"]],
                   "reason": f"{it['segment_id']} checked", "cached": False, "model": "fake"}

    class FakeSource:
        def list_cameras(self):
            return [{"camera_id": "cam1", "name": "Cam 1"}]

    monkeypatch.setattr(runner, "plan_sketches", lambda goal: plan)
    monkeypatch.setattr(runner, "refine_sketch", fake_refine)
    monkeypatch.setattr(runner, "run_search", lambda sketch, top_n: {
        "results": results[sketch.objects[0].label], "elapsed_ms": 5})
    monkeypatch.setattr(runner, "verify_stream", fake_verify_stream)
    monkeypatch.setattr(runner, "is_cached", lambda seg, q: seg == "cam1_100")  # one free cached verdict
    monkeypatch.setattr(runner, "get_source", lambda: FakeSource())
    monkeypatch.setattr(runner, "provider_config", lambda p: {"model": "fake-llm"})
    return refined


def _run(goal="find risky moments"):
    async def collect():
        return [ev async for ev in runner._agent_run(goal)]
    return asyncio.run(collect())


def test_agent_events_refinement_and_report(mocked, monkeypatch):
    monkeypatch.setattr(runner, "VERIFY_BUDGET", 6)
    events = _run()
    kinds = [e["event"] for e in events]
    assert kinds[:2] == ["start", "plan"] and kinds[-2:] == ["report", "done"]
    assert kinds.count("sketch") == 3 and kinds.count("refine") == 1  # s1, s2, refined s2r
    assert mocked == [("s2", ["YES", "NO", "NO"])]  # s2 confirmed 1/3 < 40% -> refined once

    verifies = [e for e in events if e["event"] == "verify"]
    # Budget 6 new calls: s1 uses 2 (cam1_100 is cached), s2 uses 2 (cam1_100 cached again), s2r gets 2 + 1 skipped.
    assert sum(v["verdict"] == "SKIPPED" for v in verifies) == 1
    report = events[-2]
    assert report["stats"]["new_video_calls"] == 6 and report["stats"]["skipped_for_budget"] == 1
    # cam1_100 and cam1_104 start 4 s apart: separate moments; dedupe keeps one row per segment.
    confirmed = [h["segment_id"] for h in report["confirmed"]]
    assert confirmed == ["cam1_100", "cam1_104", "cam3_12", "cam3_16"]
    md = report["markdown"]
    assert "# Agent report: find risky moments" in md and "![cam1_100](frames/cam1_100.jpg)" in md
    assert "| Forklift alone | 0.85 | 1/3 | refined below |" in md and "[open clip](clips/cam3_12.mp4)" in md


def test_same_moment_on_adjacent_segments_is_merged():
    hits = [{**_result("cam1_100", 0.9), "window": [2.5, 4.0]}, {**_result("cam1_104", 0.8), "window": [0.0, 1.5]}]
    runs = [{"item": {"title": "t"}, "results": hits, "rate": 1.0, "refined": False,
             "verdicts": [{"segment_id": h["segment_id"], "verdict": "YES", "reason": "r", "result": h} for h in hits]}]

    class FakeSource:
        def list_cameras(self):
            return []

    runner_get_source = runner.get_source
    runner.get_source = lambda: FakeSource()
    try:
        report = runner.build_report("g", "s", runs, runner._Budget(12), 1.0)
    finally:
        runner.get_source = runner_get_source
    assert [h["segment_id"] for h in report["confirmed"]] == ["cam1_100"]  # 102.5 s vs 104.0 s: same moment
