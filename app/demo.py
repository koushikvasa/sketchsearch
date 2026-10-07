"""Demo mode: agent runs recorded by scripts/warm_demo.py are replayed instantly (verifications are already
in the verify cache), so a live demo never waits on the LLM or the video model."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator

from app.config import DATA_DIR

DEMO_DIR = DATA_DIR / "demo_cache"
REPLAY_MAX_GAP_S = 0.35  # keep the trace's rhythm, minus the waiting


def _path(goal: str):
    return DEMO_DIR / f"agent_{hashlib.sha1(goal.strip().lower().encode()).hexdigest()[:12]}.json"


def has_recording(goal: str) -> bool:
    return _path(goal).exists()


def save_recording(goal: str, events: list[dict]) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    _path(goal).write_text(json.dumps({"goal": goal, "events": events}))


async def replay(goal: str) -> AsyncIterator[dict]:
    events = json.loads(_path(goal).read_text())["events"]
    prev = None
    for ev in events:
        if prev is not None:
            await asyncio.sleep(min(max(ev["t"] - prev, 0.0), REPLAY_MAX_GAP_S))
        prev = ev["t"]
        yield {**ev, "replayed": True}


def recordings() -> list[str]:
    if not DEMO_DIR.exists():
        return []
    return sorted(json.loads(p.read_text())["goal"] for p in DEMO_DIR.glob("agent_*.json"))
