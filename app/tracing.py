"""W&B Weave tracing. Functions decorated with @weave.op are traced once init_tracing() has run;
without a WANDB_API_KEY (or with WEAVE_DISABLED=1) they run untraced."""

import logging
import os

from app.config import WEAVE_PROJECT

log = logging.getLogger("uvicorn.error")
_ready = False


def init_tracing() -> bool:
    global _ready
    if _ready:
        return True
    if not os.getenv("WANDB_API_KEY") or os.getenv("WEAVE_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    import time

    import weave

    for attempt in range(3):
        try:
            weave.init(WEAVE_PROJECT)
            _ready = True
            break
        except OSError as e:
            # Seen on Windows: antivirus (Norton's \\.\nllMonFltProxy\...) briefly denying a file open.
            if attempt < 2:
                log.info("Weave init blocked (%s); retrying", e)
                time.sleep(1.0 * (attempt + 1))
                continue
            log.warning("Weave tracing disabled: %s", e)
        except Exception as e:  # tracing must never take the app down
            log.warning("Weave tracing disabled: %s", e)
            break
    return _ready


def is_ready() -> bool:
    return _ready
