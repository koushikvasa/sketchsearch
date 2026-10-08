"""AI verification: a video model watches a result clip and answers whether it shows the sketch.

The question is written from the sketch in plain spatial words. Calls go through source.ask (Gemini
now, Cosmos on Friday), run 3 at a time, are cached on disk by (segment, model, question) and are
traced in Weave. Timeouts and errors become UNSURE so the stream always completes.
"""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator

import weave

from app.config import GEMINI_MODEL, VERIFY_CACHE_DIR, VERIFY_CONCURRENCY, VERIFY_TIMEOUT_S
from app.models import Sketch
from app.sources import get_source
from app.spatial import NOUNS, center, direction, dist, names, region, relative_motion, zone_anchor
from app.vision.gemini import MAX_ATTEMPTS, VisionTimeout

VERDICTS = ("YES", "NO", "UNSURE")


def _article(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


def build_question(sketch: Sketch, window: list[float] | None = None) -> str:
    present = [o for o in sketch.objects if not o.absent]
    who = names(present)
    starts = {o.id: center(o.start_box) for o in present}
    ends = {}
    # Only describe motion the user drew: a box without an arrow or End box means "any motion",
    # and the matcher treats it that way, so the question must not claim it stays in place.
    drawn = {o.id: o.end_box is not None or bool(o.path and len(o.path) >= 2) for o in present}
    for o in present:
        if o.end_box is not None:
            ends[o.id] = center(o.end_box)
        elif o.path and len(o.path) >= 2:
            ends[o.id] = tuple(o.path[-1])
        else:
            ends[o.id] = starts[o.id]

    lines = []
    for o in present:
        noun = NOUNS.get(o.label, o.label)
        label = f"{_article(noun)} {noun}" if who[o.id].startswith("the ") else who[o.id]
        s, e = starts[o.id], ends[o.id]
        move = direction((e[0] - s[0], e[1] - s[1]))
        text = f"{label} in the {region(s)} of the frame"
        if not drawn[o.id]:
            pass  # motion not specified
        elif move is None:
            text += ", staying roughly in place"
        else:
            text += f", moving {move}"
            for other in present:
                if other.id != o.id:
                    rel = relative_motion(s, e, starts[other.id], ends[other.id])
                    if rel:
                        text += f" {rel} {who[other.id]}"
                        break
        lines.append(text)

    still = all(direction((ends[o.id][0] - starts[o.id][0], ends[o.id][1] - starts[o.id][1])) is None for o in present)
    if len(present) == 2 and still:
        a, b = present
        d = dist(starts[a.id], starts[b.id])
        lines.append(f"{who[a.id]} and {who[b.id]} are {'close to each other' if d < 0.15 else 'apart'}")
    elif len(present) >= 3 and still:
        spread = max(dist(starts[a.id], starts[b.id]) for a in present for b in present)
        if spread < 0.2:
            lines.append(f"these {len(present)} are close together, as a group")

    for z in (o for o in sketch.objects if o.absent):
        noun = NOUNS.get(z.label, z.label)
        anchor = zone_anchor(z.start_box, list(starts.items()))
        other = "other " if any(o.label == z.label for o in present) else ""
        # Absent zones are image areas, not physical distances: at the far end of an aisle a small patch
        # of the frame spans metres, and "near the forklift" alone gets judged in metres.
        b = z.start_box
        extent = (f"the image region spanning {b.x * 100:.0f}%-{(b.x + b.w) * 100:.0f}% of the frame width "
                  f"and {b.y * 100:.0f}%-{(b.y + b.h) * 100:.0f}% of the frame height")
        around = f" (the area around {who[anchor]})" if anchor else f" ({region(center(b))} of the frame)"
        lines.append(f"no {other}{noun} visible anywhere inside {extent}{around}, even far away in depth")

    focus = f" Focus on {window[0]:.1f}-{window[1]:.1f} s of the clip." if window else ""
    bullets = "\n".join(f"- {line}" for line in lines)
    return (
        f"This is a 4-second clip from a fixed warehouse camera looking down an aisle.{focus}\n"
        f"Does the clip show all of the following? Positions are approximate.\n{bullets}\n"
        'Answer with JSON only: {"verdict": "YES" | "NO" | "UNSURE", "reason": "<one short sentence>"}. '
        "YES if the clip clearly shows all of it, NO if any part clearly does not happen, "
        "UNSURE if you cannot tell."
    )


def parse_verdict(text: str) -> dict:
    try:
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1])
        verdict = str(data.get("verdict", "")).strip().upper()
        reason = str(data.get("reason", "")).strip()
        model = data.get("_model")
    except (ValueError, AttributeError):
        verdict, reason, model = "", text.strip(), None
    if verdict not in VERDICTS:
        return {"verdict": "UNSURE", "reason": f"Unclear answer: {reason[:160] or 'empty'}", "model": model}
    return {"verdict": verdict, "reason": reason[:300] or "No reason given.", "model": model}


def _cache_path(segment_id: str, question: str):
    key = hashlib.sha1(f"{GEMINI_MODEL}\n{question}".encode()).hexdigest()[:16]
    return VERIFY_CACHE_DIR / f"{segment_id}__{key}.json"


def is_cached(segment_id: str, question: str) -> bool:
    return _cache_path(segment_id, question).exists()


OS_RETRIES = 2  # transient local file errors, e.g. antivirus briefly locking a clip


def _read_cache(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (OSError, ValueError):  # locked or half-written: just verify again
        return None


@weave.op(name="verify_segment")
def verify_segment(segment_id: str, question: str) -> dict:
    path = _cache_path(segment_id, question)
    if (hit := _read_cache(path)) is not None:
        return {**hit, "cached": True}
    t = time.perf_counter()
    source = get_source()
    cacheable = False
    for attempt in range(OS_RETRIES + 1):
        try:
            out = parse_verdict(source.ask(segment_id, question))
            cacheable = True
        except VisionTimeout:  # a TimeoutError (an OSError), so it must come first: no retry
            out = {"verdict": "UNSURE", "reason": f"The video model did not answer within {VERIFY_TIMEOUT_S:g} s."}
        except OSError as e:
            # Seen on Windows: Norton's file monitor (\\.\nllMonFltProxy\...) briefly denying access to a clip
            # the browser is streaming at the same time. It clears within a moment, so try again.
            if attempt < OS_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            out = {"verdict": "UNSURE", "reason": f"A local file or network access was blocked ({type(e).__name__}: "
                                                  f"{str(e)[:120]}). Run the check again."}
        except Exception as e:  # rate limits after all retries, API errors, ...
            out = {"verdict": "UNSURE", "reason": f"Verification failed: {type(e).__name__}: {str(e)[:160]}"}
        break
    out.update(segment_id=segment_id, elapsed_s=round(time.perf_counter() - t, 1))
    if cacheable:
        try:
            VERIFY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**out, "question": question}))
        except OSError:  # a blocked cache write must not cost us the verdict
            pass
    return {**out, "cached": False}


async def verify_stream(sketch: Sketch, items: list[dict]) -> AsyncIterator[dict]:
    """Yield one verdict per item ({segment_id, window}) as soon as it is ready."""
    sem = asyncio.Semaphore(VERIFY_CONCURRENCY)
    # Worst case per item: every attempt hits the timeout plus backoff sleeps.
    guard = VERIFY_TIMEOUT_S * MAX_ATTEMPTS + 30

    async def one(item: dict) -> dict:
        question = build_question(sketch, item.get("window"))
        async with sem:
            try:
                out = await asyncio.wait_for(asyncio.to_thread(verify_segment, item["segment_id"], question), guard)
            except asyncio.TimeoutError:
                out = {"segment_id": item["segment_id"], "verdict": "UNSURE",
                       "reason": "Verification took too long.", "cached": False}
        return {**out, "question": question}

    for fut in asyncio.as_completed([asyncio.create_task(one(i)) for i in items]):
        yield await fut
