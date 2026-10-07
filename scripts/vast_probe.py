"""Friday morning: probe the real VAST stack and answer every FRIDAY-VERIFY question in one run.

  uv run --extra vast python -m scripts.vast_probe            # on the hackathon VM (env from /config)
  uv run --extra vast python -m scripts.vast_probe --camera sdg_warehouse_cam-2

Prints a PASS/FAIL line per step and writes raw samples to data/vast_probe.json (no secrets).
"""

import argparse
import json
import os
import time
import traceback

import httpx

from app.config import DATA_DIR
from app.sources.vast import VastConfig, VastSource, load_rows_vastdb, parse_detections, row_to_segment

OUT = DATA_DIR / "vast_probe.json"
ENV = ["INGRESS_URL", "VSS_URL", "USERNAME", "PASSWORD", "S3_ENDPOINT", "VDB_ENDPOINT", "ACCESS_KEY",
       "SECRET_KEY", "VASTDB_BUCKET", "VDB_SCHEMA", "VDB_COLLECTION", "COSMOS3_REASON_URL", "COSMOS3_REASON_MODEL",
       "YOLO_URL", "GPU_BEARER_TOKEN", "WANDB_API_KEY", "WANDB_TEAM", "WANDB_PROJECT", "GEMINI_API_KEY"]
report: dict = {}


def step(name: str):
    def wrap(fn):
        def run(*a, **kw):
            t = time.perf_counter()
            try:
                out = fn(*a, **kw)
                report[name] = {"ok": True, "result": out}
                print(f"PASS {name} ({time.perf_counter() - t:.1f}s)")
                return out
            except Exception as e:  # keep probing the other pieces
                report[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
                print(f"FAIL {name}: {type(e).__name__}: {str(e)[:200]}")
                return None
        return run
    return wrap


def sample(v, n=600):
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", help="camera_id to sample, e.g. sdg_warehouse_cam-2 (Pack C)")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--snapshot", help="also write all rows (no vectors) to this .json.gz for VAST_ROWS_FILE")
    args = ap.parse_args()

    print("env (set / missing, values never printed):")
    for k in ENV:
        print(f"  {k:22s} {'set' if os.getenv(k) else '-'}")

    cfg = VastConfig()
    if args.camera:
        cfg.camera_ids = [args.camera]
    src = VastSource(cfg, rows_loader=lambda c: [])

    @step("login")
    def login():
        src.login()
        return src.request("GET", "/auth/me").json()

    @step("dashboard labels (objects[])")
    def labels():
        stats = src.request("GET", "/dashboard/stats", params={"scope": "all"}).json()
        return {"objects": stats.get("objects", [])[:40], "overview": stats.get("overview")}

    @step("metadata schema")
    def schema():
        return src.request("GET", "/metadata/schema").json()

    @step("camera_id values")
    def cameras():
        return src.request("GET", "/metadata/values", params={"field": "camera_id", "limit": 50}).json()

    @step("vastdb rows")
    def rows():
        rs = load_rows_vastdb(cfg, limit=args.rows)
        out = []
        for r in rs:
            seg = row_to_segment(r, parse_detections)
            out.append({
                "columns": {k: type(v).__name__ for k, v in r.items()},
                "source": r.get("source"), "camera_id": r.get("camera_id"),
                "timing": {k: r[k] for k in r if any(w in k for w in ("start", "end", "time", "segment", "dur"))},
                "perception_json": sample(r.get("perception_json"), 1500),
                "parsed": None if seg is None else {
                    "start": seg.start, "end": seg.end, "keyframe_only": seg.keyframe_only,
                    "tracks": [(t.label, len(t.points)) for t in seg.tracks]},
            })
        return out

    @step("search 'forklift near a person'")
    def search():
        data = src.request("POST", "/search", json={"query": "forklift near a person", "top_k": 5,
                                                    "llm_top_n": 0, "min_similarity": 0.1}).json()
        return {"keys": list(data), "first_result": sample((data.get("results") or [{}])[0], 1500),
                "first_chunk": sample((data.get("chunk_results") or [{}])[0], 800)}

    first_source = None
    res = search()
    if res:
        try:
            first_source = json.loads(res["first_result"]).get("source")
        except ValueError:
            pass

    @step("detections sidecar")
    def sidecar():
        if not first_source:
            raise RuntimeError("no segment source from search")
        r = src.request("GET", "/videos/detections", params={"source": first_source})
        raw = r.json()
        tracks, kf = parse_detections(raw)
        return {"sample": sample(raw, 1500), "parsed_tracks": [(t.label, len(t.points)) for t in tracks],
                "keyframe_only": kf}

    @step("clip stream")
    def stream():
        if not first_source:
            raise RuntimeError("no segment source from search")
        r = src.request("GET", "/videos/stream", params={"source": first_source, "token": src.token()},
                        headers={"Range": "bytes=0-1023"})
        return {"status": r.status_code, "headers": {k: v for k, v in r.headers.items()
                                                     if k.lower() in ("content-type", "content-range", "accept-ranges")}}

    @step("cosmos3-reason models")
    def cosmos_models():
        headers = {"Authorization": f"Bearer {cfg.gpu_token}"} if cfg.gpu_token else {}
        return httpx.get(f"{cfg.cosmos_url}/v1/models", headers=headers, timeout=20).json()

    @step("cosmos3-reason ask on one clip")
    def cosmos_ask():
        if not first_source:
            raise RuntimeError("no segment source from search")
        from app.sources.vast import segment_key

        sid = segment_key(first_source)
        src.sources[sid] = first_source
        t = time.perf_counter()
        answer = src._ask_cosmos(src.clip_bytes(sid), 'Is there a forklift? Answer JSON {"verdict": "YES"|"NO", "reason": "..."}')
        return {"answer": answer, "seconds": round(time.perf_counter() - t, 1)}

    @step("yolo openapi")
    def yolo():
        url = os.getenv("YOLO_URL", "").rstrip("/")
        spec = httpx.get(f"{url}/openapi.json", timeout=20).json()
        return {"paths": list(spec.get("paths", {})), "schemas": list(spec.get("components", {}).get("schemas", {}))}

    @step("w&b inference")
    def wandb():
        from app.llm import chat_json

        return chat_json([{"role": "user", "content": 'Reply with JSON {"ok": true}'}], provider="wandb")

    @step("rows snapshot")
    def snapshot():
        import gzip

        rs = load_rows_vastdb(cfg)
        with gzip.open(args.snapshot, "wt", encoding="utf-8") as f:
            json.dump(rs, f, default=str)
        return {"rows": len(rs), "path": args.snapshot}

    steps = [login, labels, schema, cameras, rows, sidecar, stream, cosmos_models, cosmos_ask, yolo, wandb]
    if args.snapshot:
        steps.append(snapshot)
    for fn in steps:
        fn()
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {OUT} - paste the FAIL lines and the samples into Cursor/Claude to adapt the parser.")


if __name__ == "__main__":
    main()
