"""Gemini calls for verification (video) and diagram -> sketch (image).

Each attempt has a VERIFY_TIMEOUT_S timeout (raised as VisionTimeout). 429 / 5xx are retried with
exponential backoff; after two failed attempts the call moves to GEMINI_FALLBACK_MODEL if one is set.
"""

import random
import re
import threading
import time
from pathlib import Path

import httpx

from app.config import GEMINI_API_KEY, GEMINI_FALLBACK_MODEL, GEMINI_MODEL, VERIFY_TIMEOUT_S

RETRY_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
VIDEO_FPS = 4  # Gemini samples 1 fps by default; 4 s clips need more frames to judge motion


class VisionTimeout(TimeoutError):
    pass


_lock = threading.Lock()
_shared = None


def _client():
    """One shared client. Built under a lock: concurrent verifications otherwise each build one, and the
    losers get garbage-collected mid-request ("client has been closed")."""
    global _shared
    with _lock:
        if _shared is None:
            from google import genai
            from google.genai import types

            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY is not set")
            _shared = genai.Client(api_key=GEMINI_API_KEY,
                                   http_options=types.HttpOptions(timeout=VERIFY_TIMEOUT_S * 1000))
        return _shared


_exhausted: dict[str, float] = {}  # model -> time its quota should be back
QUOTA_COOLDOWN_S = 600


def models() -> list[str]:
    chain = [GEMINI_MODEL] + ([GEMINI_FALLBACK_MODEL] if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL else [])
    available = [m for m in chain if _exhausted.get(m, 0) < time.time()]
    return available or chain


def _quota_exhausted(e) -> bool:
    """A 429 for a used-up quota ("retry in 6h"), as opposed to a short burst limit worth waiting out."""
    msg = str(e)
    return e.code == 429 and ("PerDay" in msg or re.search(r"retry in \d+h", msg) is not None)


def generate_json(contents: list, timeout_s: float = VERIFY_TIMEOUT_S) -> tuple[str, str]:
    """Run one JSON-mode generation; returns (text, model used). timeout_s applies per attempt."""
    from google.genai import errors, types

    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1,
                                         http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    delay, last = 1.0, None
    for attempt in range(MAX_ATTEMPTS):
        chain = models()
        model = chain[1] if attempt >= 2 and len(chain) > 1 else chain[0]
        try:
            resp = _client().models.generate_content(model=model, contents=contents, config=config)
            return resp.text or "", model
        except errors.APIError as e:
            if e.code not in RETRY_CODES:
                raise
            last = e
            if _quota_exhausted(e):  # no point backing off for hours: move to the next model now
                _exhausted[model] = time.time() + QUOTA_COOLDOWN_S
                continue
        except (httpx.TimeoutException, TimeoutError) as e:
            raise VisionTimeout(f"{model} did not answer within {timeout_s:g} s") from e
        time.sleep(delay + random.uniform(0, 0.5))
        delay *= 2
    raise last


def video_part(path: Path):
    from google.genai import types

    return types.Part(inline_data=types.Blob(data=path.read_bytes(), mime_type="video/mp4"),
                      video_metadata=types.VideoMetadata(fps=VIDEO_FPS))


def image_part(data: bytes, mime_type: str):
    from google.genai import types

    return types.Part.from_bytes(data=data, mime_type=mime_type)


def ask_video(path: Path, question: str) -> tuple[str, str]:
    """Inline video bytes (no file upload) + question -> (raw JSON text, model)."""
    return generate_json([video_part(path), question])


def ask_video_bytes(data: bytes, question: str) -> tuple[str, str]:
    from google.genai import types

    part = types.Part(inline_data=types.Blob(data=data, mime_type="video/mp4"),
                      video_metadata=types.VideoMetadata(fps=VIDEO_FPS))
    return generate_json([part, question])
