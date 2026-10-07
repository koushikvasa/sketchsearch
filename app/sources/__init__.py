from functools import lru_cache

from app.config import SOURCE
from app.sources.base import VideoSource


@lru_cache(maxsize=1)
def get_source() -> VideoSource:
    if SOURCE == "local":
        from app.sources.local import LocalSource

        return LocalSource()
    if SOURCE == "vast":
        from app.sources.vast import VastSource

        return VastSource()
    raise ValueError(f"SOURCE must be 'local' or 'vast', got {SOURCE!r}")
