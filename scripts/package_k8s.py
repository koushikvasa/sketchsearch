"""Build the flat folder deploy-app-no-registry mounts as a ConfigMap: main.py + requirements.txt + one zip.

  uv run python -m scripts.package_k8s          -> deploy/k8s/build/
  kubectl -n "$NS" create configmap sketchsearch-code --from-file=deploy/k8s/build \\
      --dry-run=client -o yaml | kubectl apply -f -

ConfigMaps are limited to ~1 MiB, so only code and static files go in (no data/, no models).
"""

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "deploy" / "k8s"
OUT = SRC / "build"
INCLUDE = ("app", "static")
LIMIT = 1_000_000  # ConfigMap size limit (~1 MiB)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copy(SRC / "main.py", OUT / "main.py")
    shutil.copy(SRC / "requirements.txt", OUT / "requirements.txt")
    with zipfile.ZipFile(OUT / "sketchsearch.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for top in INCLUDE:
            for path in sorted((ROOT / top).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    z.write(path, path.relative_to(ROOT).as_posix())
    total = sum(p.stat().st_size for p in OUT.iterdir())
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:22s} {p.stat().st_size / 1024:7.1f} KiB")
    print(f"total {total / 1024:.1f} KiB ({'OK' if total < LIMIT else 'TOO BIG'} for a ConfigMap)")
    if total >= LIMIT:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
