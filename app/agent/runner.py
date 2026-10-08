"""Agent mode: goal -> planned sketches -> match -> verify -> refine weak sketches once -> report.

Streams events (plan, sketch, results, verify, refine, report, done). New video-model calls are capped
at VERIFY_BUDGET per run; cached verdicts are free. The whole run is one Weave trace (agent_run) with
the LLM, matcher and verification calls nested under it.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator

import weave

from app.agent.words_to_sketch import SCENE
from app.config import LLM_PROVIDER
from app.jsonfix import loads_lenient
from app.llm import chat_json, provider_config
from app.search import run_search
from app.sketch_json import SKETCH_FORMAT, SketchParseError, normalize_objects
from app.sources import get_source
from app.verify import build_question, is_cached, verify_stream

VERIFY_BUDGET = 12  # new (uncached) video-model calls per run
VERIFY_TOP = 3
RESULTS_TOP = 10
REFINE_BELOW = 0.4  # confirm rate that triggers one refinement
MAX_SKETCHES = 4
PRESET_GOAL = "Find risky moments between people and the forklift."

PLAN_SYSTEM = f"""You are a video-search agent for warehouse safety CCTV. Given a goal, plan 3 distinct
search sketches (4 only if the goal clearly needs it) that would find matching moments. Each sketch is
a layout of boxes (and paths for movement) on the camera frame; a geometric matcher finds video moments
with that layout and a video model then verifies the top hits.

{SCENE}

Make the sketches genuinely different (different relations, motions, or an absent "nobody here" zone),
each with 1-3 present objects. A spotter or helper is a person, so "no spotter" is an absent person zone. Prefer layouts that are likely in this footage: forklifts sit at the far
end of the aisle (top-centre); people walk along the aisle.

Return ONE JSON object:
{{"summary": "<one sentence: how you will search>",
  "sketches": [{{"title": "<3-6 words>", "why": "<one sentence: why this is risky / relevant>",
                 "sketch": {{"objects": [...]}}}}]}}
Each "sketch" follows this format:
{SKETCH_FORMAT}"""

REFINE_SYSTEM = f"""You improve one search sketch for warehouse CCTV. A video model checked the top matches of
the sketch and mostly did not confirm them; its reasons are given. Change the sketch so the matcher
finds moments that really fit the intent (adjust positions, sizes, motion, or add/remove an absent zone),
keeping the same intent.

{SCENE}

Return ONE JSON object: {{"title": "...", "why": "<what you changed and why>", "sketch": {{"objects": [...]}}}}
The "sketch" follows this format:
{SKETCH_FORMAT}"""


def _parse(raw: str) -> dict:
    try:
        data = loads_lenient(raw)
    except ValueError as e:
        raise SketchParseError(str(e)) from None
    if not isinstance(data, dict):
        raise SketchParseError("expected a JSON object")
    return data


@weave.op(name="agent_plan")
def plan_sketches(goal: str) -> dict:
    messages = [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": f"Goal: {goal}"}]
    error = None
    for _ in range(2):
        raw = chat_json(messages)
        try:
            data = _parse(raw)
            planned = []
            for i, item in enumerate((data.get("sketches") or [])[:MAX_SKETCHES]):
                sketch = normalize_objects((item.get("sketch") or {}).get("objects") or [])
                planned.append({"id": f"s{i + 1}", "title": str(item.get("title") or f"Sketch {i + 1}")[:60],
                                "why": str(item.get("why") or "")[:240], "sketch": sketch})
            if not planned:
                raise SketchParseError("the plan has no sketches")
            return {"summary": str(data.get("summary") or "")[:300], "sketches": planned}
        except SketchParseError as e:
            error = e
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That plan is invalid: {e}. Reply with only the corrected JSON."}]
    raise SketchParseError(f"the planner returned an invalid plan twice: {error}")


@weave.op(name="agent_refine")
def refine_sketch(goal: str, item: dict, verdicts: list[dict]) -> dict:
    notes = "\n".join(f"- {v['segment_id']}: {v['verdict']} - {v['reason']}" for v in verdicts)
    user = (f"Goal: {goal}\nSketch title: {item['title']}\nWhy: {item['why']}\n"
            f"Sketch: {json.dumps(item['sketch'].model_dump(exclude_none=True, exclude={'text', 'camera_ids'}))}\n"
            f"Video-model verdicts on its top matches:\n{notes}")
    messages = [{"role": "system", "content": REFINE_SYSTEM}, {"role": "user", "content": user}]
    error = None
    for _ in range(2):
        raw = chat_json(messages)
        try:
            data = _parse(raw)
            sketch = normalize_objects((data.get("sketch") or {}).get("objects") or [])
            return {"id": f"{item['id']}r", "title": str(data.get("title") or item["title"])[:60],
                    "why": str(data.get("why") or "")[:240], "sketch": sketch, "refines": item["id"]}
        except SketchParseError as e:
            error = e
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That sketch is invalid: {e}. Reply with only the corrected JSON."}]
    raise SketchParseError(f"refinement failed twice: {error}")


def _result_view(r: dict) -> dict:
    keys = ("segment_id", "camera_id", "start", "end", "score", "window", "explanation", "clip_url", "frame_url",
            "absence_ok", "components")
    return {k: r.get(k) for k in keys}


class _Budget:
    def __init__(self, n: int):
        self.left, self.used, self.skipped = n, 0, 0


async def _verify_top(sketch, results: list[dict], budget: _Budget) -> tuple[list[dict], list[dict]]:
    """Verify up to VERIFY_TOP results, spending budget only on uncached calls. Returns (to_verify, skipped)."""
    todo, skipped = [], []
    for r in results[:VERIFY_TOP]:
        q = build_question(sketch, r["window"])
        if is_cached(r["segment_id"], q):
            todo.append(r)
        elif budget.left > 0:
            budget.left -= 1
            budget.used += 1
            todo.append(r)
        else:
            budget.skipped += 1
            skipped.append(r)
    return todo, skipped


def _ev(kind: str, **data) -> dict:
    return {"event": kind, "t": round(time.time(), 3), **data}


async def _agent_run(goal: str) -> AsyncIterator[dict]:
    t0 = time.perf_counter()
    budget = _Budget(VERIFY_BUDGET)
    model = provider_config(LLM_PROVIDER)["model"]
    yield _ev("start", goal=goal, budget=VERIFY_BUDGET, llm=f"{model} ({LLM_PROVIDER})")

    plan = await asyncio.to_thread(plan_sketches, goal)
    yield _ev("plan", summary=plan["summary"], sketches=[
        {"id": s["id"], "title": s["title"], "why": s["why"], "sketch": s["sketch"].model_dump()} for s in plan["sketches"]])

    runs: list[dict] = []  # one per (refined) sketch

    async def run_sketch(item: dict, refined: bool = False):
        yield _ev("sketch", id=item["id"], title=item["title"], why=item["why"], refined=refined,
                  refines=item.get("refines"), sketch=item["sketch"].model_dump())
        res = await asyncio.to_thread(run_search, item["sketch"], RESULTS_TOP)
        results = res["results"]
        yield _ev("results", id=item["id"], count=len(results), elapsed_ms=res["elapsed_ms"],
                  results=[_result_view(r) for r in results[:5]])
        todo, skipped = await _verify_top(item["sketch"], results, budget)
        verdicts = []
        if todo:
            by_id = {r["segment_id"]: r for r in todo}
            async for v in verify_stream(item["sketch"], [{"segment_id": r["segment_id"], "window": r["window"]}
                                                         for r in todo]):
                v["result"] = _result_view(by_id[v["segment_id"]])
                verdicts.append(v)
                yield _ev("verify", id=item["id"], segment_id=v["segment_id"], verdict=v["verdict"],
                          reason=v["reason"], cached=v.get("cached", False), model=v.get("model"),
                          frame_url=v["result"]["frame_url"], clip_url=v["result"]["clip_url"])
        for r in skipped:
            yield _ev("verify", id=item["id"], segment_id=r["segment_id"], verdict="SKIPPED",
                      reason="Verification budget for this run is used up.", cached=False, model=None,
                      frame_url=r.get("frame_url"), clip_url=r.get("clip_url"))
        yes = sum(v["verdict"] == "YES" for v in verdicts)
        rate = yes / len(verdicts) if verdicts else None
        runs.append({"item": item, "results": results, "verdicts": verdicts, "rate": rate, "refined": refined})

    for item in plan["sketches"]:
        async for ev in run_sketch(item):
            yield ev
        last = runs[-1]
        if last["rate"] is not None and last["rate"] < REFINE_BELOW:
            try:
                better = await asyncio.to_thread(refine_sketch, goal, item, last["verdicts"])
            except Exception as e:  # keep going with the other sketches
                yield _ev("refine", id=item["id"], ok=False, reason=f"Refinement failed: {e}")
                continue
            yield _ev("refine", id=item["id"], ok=True, new_id=better["id"], title=better["title"], why=better["why"],
                      confirm_rate=last["rate"], sketch=better["sketch"].model_dump())
            async for ev in run_sketch(better, refined=True):
                yield ev

    report = build_report(goal, plan["summary"], runs, budget, time.perf_counter() - t0)
    yield _ev("report", **report)
    yield _ev("done", seconds=round(time.perf_counter() - t0, 1))


_traced_run = weave.op(name="agent_run")(_agent_run)


def agent_run(goal: str) -> AsyncIterator[dict]:
    """The agent's event stream; one Weave trace per run when tracing is on (Weave refuses to wrap
    async generators before weave.init)."""
    from app import tracing

    return _traced_run(goal) if tracing.is_ready() else _agent_run(goal)


VERDICT_RANK = {"YES": 0, "UNSURE": 1, "NO": 2}
SAME_MOMENT_S = 3.0  # hits on one camera starting this close together are the same moment


def _hit_key(h: dict) -> tuple:
    return VERDICT_RANK.get(h["verdict"], 3), -h["score"]


def _abs_start(h: dict) -> float:
    return h["start"] + h["window"][0]


def build_report(goal: str, summary: str, runs: list[dict], budget: _Budget, seconds: float) -> dict:
    """Merge every verified hit, dedupe by segment (best verdict, then score), rank, render markdown."""
    best: dict[str, dict] = {}
    for run in runs:
        for v in run["verdicts"]:
            hit = {**v["result"], "verdict": v["verdict"], "reason": v["reason"], "sketch": run["item"]["title"]}
            cur = best.get(hit["segment_id"])
            if cur is None or _hit_key(hit) < _hit_key(cur):
                best[hit["segment_id"]] = hit
    ranked = []  # best hit first; drop later hits of the same moment (adjacent segments overlap in time)
    for h in sorted(best.values(), key=_hit_key):
        if not any(k["camera_id"] == h["camera_id"] and abs(_abs_start(k) - _abs_start(h)) < SAME_MOMENT_S
                   for k in ranked):
            ranked.append(h)
    confirmed = [h for h in ranked if h["verdict"] == "YES"]
    get_cam = {c["camera_id"]: c.get("name", c["camera_id"]) for c in get_source().list_cameras()}

    lines = [f"# Agent report: {goal}", "", f"*{summary}*" if summary else "", "",
             f"**{len(confirmed)} confirmed moment{'s' if len(confirmed) != 1 else ''}** from {len(runs)} sketch runs, "
             f"{len(best)} distinct clips checked; {budget.used} new video-model calls "
             f"(budget {VERIFY_BUDGET}), {seconds:.0f} s.", "", "## Sketches", "",
             "| Sketch | Top score | Confirmed | Note |", "|---|---|---|---|"]
    for run in runs:
        item, vs = run["item"], run["verdicts"]
        top = f"{run['results'][0]['score']:.2f}" if run["results"] else "-"
        conf = f"{sum(v['verdict'] == 'YES' for v in vs)}/{len(vs)}" if vs else "not checked"
        note = "refined" if run["refined"] else ("refined below" if run["rate"] is not None and run["rate"] < REFINE_BELOW else "")
        lines.append(f"| {item['title']} | {top} | {conf} | {note} |")
    lines += ["", "## Confirmed moments", ""]
    if not confirmed:
        lines.append("None of the checked clips were confirmed by the video model.")
    for h in confirmed:
        cam = get_cam.get(h["camera_id"], h["camera_id"])
        t = f"{h['start'] + h['window'][0]:.1f}-{h['start'] + h['window'][1]:.1f} s"
        lines += [f"### {cam}, {t} (score {h['score']:.2f}, from \"{h['sketch']}\")", ""]
        if h.get("frame_url"):
            lines += [f"![{h['segment_id']}]({h['frame_url']})", ""]
        lines += [
                  f"✅ {h['reason']}", "", f"{h['explanation']}  ", f"[open clip]({h['clip_url']})", ""]
    rejected = [h for h in ranked if h["verdict"] != "YES"]
    if rejected:
        lines += ["## Not confirmed", ""] + [
            f"- {get_cam.get(h['camera_id'], h['camera_id'])} {h['start']:.0f}-{h['end']:.0f} s: "
            f"{h['verdict']} - {h['reason']}" for h in rejected]
    return {"markdown": "\n".join(lines).strip() + "\n", "confirmed": confirmed, "rejected": rejected,
            "stats": {"sketch_runs": len(runs), "clips_checked": len(best), "confirmed": len(confirmed),
                      "new_video_calls": budget.used, "skipped_for_budget": budget.skipped,
                      "seconds": round(seconds, 1)}}
