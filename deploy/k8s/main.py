"""Entry point for deploy-app-no-registry (python:3.12-slim, code mounted flat from a ConfigMap at /code).

A ConfigMap built with --from-file=<dir> only carries top-level files, so the app ships as one zip
(app/ + static/) next to this file. We unpack it and start uvicorn on $PORT. The ingress strips /app,
so ROOT_PATH=/app only affects generated links (<base href>, /docs).
"""

import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = Path(os.getenv("SKETCHSEARCH_HOME", "/tmp/sketchsearch"))


def unpack() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(HERE / "sketchsearch.zip") as z:
        z.extractall(TARGET)
    sys.path.insert(0, str(TARGET))
    os.chdir(TARGET)


if __name__ == "__main__":
    unpack()
    os.environ.setdefault("ROOT_PATH", "/app")
    os.environ.setdefault("SOURCE", "vast")
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")),
                root_path=os.environ["ROOT_PATH"], proxy_headers=True, forwarded_allow_ips="*")
