"""Warm everything the live demo shows, so Demo mode has zero waiting.

- the 3 search presets: search + AI-check the top 10 (verdicts land in data/verify_cache/)
- the agent preset goal: one full run recorded to data/demo_cache/ (replayed in Demo mode)

  uv run python -m scripts.warm_demo            # skips what is already warm
  uv run python -m scripts.warm_demo --force    # re-record the agent run
"""

import argparse
import asyncio
import time
from collections import Counter

from app.agent.runner import PRESET_GOAL, agent_run
from app.demo import has_recording, save_recording
from app.presets import load_presets
from app.sources import get_source
from app.search import run_search
from app.tracing import init_tracing
from app.verify import verify_stream

VERIFY_N = 10


async def warm_presets(presets: dict | None = None) -> None:
    """AI-check the top 10 of every example chip (the current index's presets by default)."""
    if presets is None:
        presets = load_presets(getattr(get_source(), "index_source", "default"))
    for key, preset in presets.items():
        t = time.perf_counter()
        results = run_search(preset["sketch"], top_n=12)["results"][:VERIFY_N]
        counts, fresh = Counter(), 0
        async for v in verify_stream(preset["sketch"], [{"segment_id": r["segment_id"], "window": r["window"]}
                                                        for r in results]):
            counts[v["verdict"]] += 1
            fresh += not v.get("cached")
        print(f"  {preset['title']:28s} YES {counts['YES']:2d}  NO {counts['NO']:2d}  UNSURE {counts['UNSURE']:2d}"
              f"   ({fresh} new video-model calls, {time.perf_counter() - t:.0f} s)")
        if counts["UNSURE"]:
            print("    note: UNSURE verdicts from timeouts/errors are not cached; rerun to retry them")


async def warm_agent(force: bool) -> None:
    if has_recording(PRESET_GOAL) and not force:
        print(f"  agent run for {PRESET_GOAL!r} already recorded (use --force to re-record)")
        return
    t = time.perf_counter()
    events = [ev async for ev in agent_run(PRESET_GOAL)]
    report = next(ev for ev in events if ev["event"] == "report")
    save_recording(PRESET_GOAL, events)
    s = report["stats"]
    print(f"  agent: {s['confirmed']} confirmed of {s['clips_checked']} clips, {s['sketch_runs']} sketch runs, "
          f"{s['new_video_calls']} new video-model calls, {time.perf_counter() - t:.0f} s -> recorded")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-record the agent run")
    args = ap.parse_args()
    init_tracing()
    print("Search presets (top 10 AI-checked):")
    asyncio.run(warm_presets())
    print("Agent preset:")
    asyncio.run(warm_agent(args.force))
    print("Done. Turn on 🎬 Demo mode in the UI.")


if __name__ == "__main__":
    main()
