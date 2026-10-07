from functools import lru_cache

from app.config import SOURCE
from app.sources.base import VideoSource


@lru_cache(maxsize=1)
def get_source() -> VideoSource:
    if SOURCE == "local":
        from app.sources.local import LocalSource

        return LocalSource()
    raise NotImplementedError(f"SOURCE={SOURCE!r} (VastSource arrives Friday)")
