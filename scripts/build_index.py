"""Build SketchSearch's local indexes from data/videos/warehouse_cam<N>.mp4.

Steps (comma-separated with --steps; default: all, in this order):
  segments  4 s mp4 clips (stream copy) -> data/segments/<camera>_<start>.mp4
  decode    one pass per video: mid-frame JPGs -> data/frames/, YOLO11 + ByteTrack @ 5 fps -> data/yolo_raw/
  gt        ground_truth.json -> data/index.json
  yolo      data/yolo_raw/ -> data/index_yolo.json
  clip      CLIP ViT-B/32 keyframe embeddings -> data/clip_embeddings.npy + data/clip_ids.json
  report    YOLO vs GT label calibration -> data/alias_report.json
  stats     print stats for both indexes

  uv run python -m scripts.build_index
  uv run python -m scripts.build_index --steps gt,stats
"""

import argparse
import json
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app.config import (
    ALIAS_REPORT,
    CLIP_EMBEDDINGS,
    CLIP_IDS,
    FFMPEG,
    FRAMES_DIR,
    GT_FILE,
    INDEX_FILES,
    SEGMENTS_DIR,
    VIDEOS_DIR,
    YOLO_RAW_DIR,
)
from app.models import Box, Segment, Track, TrackPoint

SEGMENT_SECONDS = 4
SAMPLE_FPS = 5
MIN_AREA = 0.0004  # boxes smaller than this fraction of the frame are dropped
MIN_POINTS = 3  # tracks with fewer points in a segment are dropped
FRAME_SIZE = (960, 540)  # saved keyframe JPG size
GT_LABELS = {"Person": "person", "Forklift": "forklift", "Transporter": "transporter", "NovaCarter": "robot"}
STEPS = ["segments", "decode", "gt", "yolo", "clip", "report", "stats"]


# ---------- cameras & segment grid ----------

def load_cameras() -> list[dict]:
    cams = []
    for p in sorted(VIDEOS_DIR.glob("warehouse_cam*.mp4")):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        n = int(p.stem.removeprefix("warehouse_cam"))
        seg_frames = round(SEGMENT_SECONDS * fps)
        cams.append({
            "camera_id": p.stem,
            "video_id": p.stem,
            "name": f"Warehouse cam {n}",
            "gt_camera": f"Camera_{n:04d}",
            "fps": fps,
            "frames": frames,
            "duration": round(frames / fps, 3),
            "width": w,
            "height": h,
            "video_url": f"videos/{p.name}",
            "seg_frames": seg_frames,
            "n_segments": frames // seg_frames,  # trailing partial segment is dropped
            "step": round(fps / SAMPLE_FPS),
        })
    if not cams:
        raise SystemExit(f"no videos in {VIDEOS_DIR}")
    return cams


def segment_id(camera_id: str, k: int) -> str:
    return f"{camera_id}_{k * SEGMENT_SECONDS}"


def public_camera(cam: dict) -> dict:
    return {k: cam[k] for k in ("camera_id", "video_id", "name", "fps", "duration", "width", "height", "video_url")}


# ---------- step: segments ----------

def step_segments(cams):
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        with tempfile.TemporaryDirectory(dir=SEGMENTS_DIR) as tmp:
            subprocess.run([
                FFMPEG, "-v", "error", "-y", "-i", str(VIDEOS_DIR / f"{cam['video_id']}.mp4"),
                "-map", "0:v:0", "-c", "copy", "-an", "-f", "segment",
                "-segment_time", str(SEGMENT_SECONDS), "-reset_timestamps", "1",
                "-segment_format_options", "movflags=+faststart",
                str(Path(tmp) / "%04d.mp4"),
            ], check=True)
            for k in range(cam["n_segments"]):
                (Path(tmp) / f"{k:04d}.mp4").replace(SEGMENTS_DIR / f"{segment_id(cam['camera_id'], k)}.mp4")
        print(f"[segments] {cam['camera_id']}: {cam['n_segments']} clips")


# ---------- step: decode (keyframes + YOLO) ----------

def step_decode(cams, yolo: bool, model_name: str, imgsz: int, conf: float):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        sf, step = cam["seg_frames"], cam["step"]
        mids = {k * sf + sf // 2: segment_id(cam["camera_id"], k) for k in range(cam["n_segments"])}
        last = cam["n_segments"] * sf
        model = None
        if yolo:
            from ultralytics import YOLO

            model = YOLO(model_name)  # fresh model per video -> fresh ByteTrack state
        dets: dict[str, list] = {}
        cap = cv2.VideoCapture(str(VIDEOS_DIR / f"{cam['video_id']}.mp4"))
        t0 = time.time()
        for f in range(last):
            if not cap.grab():
                break
            run_yolo = model is not None and f % step == 0
            if not (run_yolo or f in mids):
                continue
            _, frame = cap.retrieve()
            if f in mids:
                small = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(FRAMES_DIR / f"{mids[f]}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if run_yolo:
                r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, imgsz=imgsz, verbose=False)[0]
                b = r.boxes
                if b.id is not None:
                    dets[str(f)] = [
                        [int(i), r.names[int(c)], round(float(s), 3), *(round(float(v), 5) for v in xyxyn)]
                        for i, c, s, xyxyn in zip(b.id, b.cls, b.conf, b.xyxyn)
                    ]
            if f and f % 900 == 0:
                print(f"[decode] {cam['camera_id']}: frame {f}/{last} ({time.time() - t0:.0f}s)", flush=True)
        cap.release()
        if model is not None:
            (YOLO_RAW_DIR / f"{cam['camera_id']}.json").write_text(json.dumps({
                "camera_id": cam["camera_id"], "model": model_name, "imgsz": imgsz, "conf": conf,
                "tracker": "bytetrack", "step": step,
                "columns": ["track_id", "label", "conf", "x1", "y1", "x2", "y2"],
                "detections": dets,
            }))
        print(f"[decode] {cam['camera_id']}: done in {time.time() - t0:.0f}s", flush=True)


# ---------- track building (shared by gt + yolo) ----------

def norm_box(x1, y1, x2, y2) -> Box:
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    return Box(x=round(x1, 4), y=round(y1, 4), w=round(x2 - x1, 4), h=round(y2 - y1, 4))


def build_segments(cam: dict, observations) -> list[Segment]:
    """observations: iterable of (frame, track_key, label, Box) at sampled frames."""
    sf, fps = cam["seg_frames"], cam["fps"]
    per: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f, key, label, box in observations:
        k = f // sf
        if k < cam["n_segments"] and box.w * box.h >= MIN_AREA:
            per[k][key].append((f, label, box))
    segments = []
    for k in range(cam["n_segments"]):
        start = k * SEGMENT_SECONDS
        tracks = []
        for key, pts in sorted(per[k].items()):
            if len(pts) < MIN_POINTS:
                continue
            pts.sort(key=lambda p: p[0])
            label = Counter(p[1] for p in pts).most_common(1)[0][0]
            tracks.append(Track(
                label=label, track_id=key,
                points=[TrackPoint(t=round(f / fps - start, 3), box=b) for f, _, b in pts],
            ))
        segments.append(Segment(
            segment_id=segment_id(cam["camera_id"], k), video_id=cam["video_id"], camera_id=cam["camera_id"],
            start=start, end=start + SEGMENT_SECONDS, tracks=tracks,
        ))
    return segments


def write_index(path: Path, source: str, cams: list[dict], segments: list[Segment], extra: dict):
    meta = {
        "source": source, "segment_seconds": SEGMENT_SECONDS, "sample_fps": SAMPLE_FPS,
        "min_area": MIN_AREA, "min_points": MIN_POINTS,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra,
    }
    data = {"meta": meta, "cameras": [public_camera(c) for c in cams],
            "segments": [s.model_dump(exclude_none=True) for s in segments]}
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"[{source}] wrote {path.name}: {len(segments)} segments, {path.stat().st_size / 1e6:.1f} MB")


# ---------- step: gt ----------

def load_gt() -> dict:
    return json.loads(GT_FILE.read_text())


def gt_observations(gt: dict, cam: dict):
    w, h = cam["width"], cam["height"]
    for fs, objs in gt.items():
        f = int(fs)
        if f % cam["step"]:
            continue
        for o in objs:
            b = o["2d bounding box visible"].get(cam["gt_camera"])
            if b:
                yield f, f"gt{o['object id']}", GT_LABELS.get(o["object type"], o["object type"].lower()), \
                    norm_box(b[0] / w, b[1] / h, b[2] / w, b[3] / h)


def step_gt(cams, gt):
    segs = [s for cam in cams for s in build_segments(cam, gt_observations(gt, cam))]
    write_index(INDEX_FILES["gt"], "gt", cams, segs, {"gt_file": str(GT_FILE.relative_to(GT_FILE.parents[2]))})


# ---------- step: yolo ----------

def load_yolo_raw(cam: dict) -> dict:
    p = YOLO_RAW_DIR / f"{cam['camera_id']}.json"
    if not p.exists():
        raise SystemExit(f"{p} missing; run the decode step with YOLO first")
    return json.loads(p.read_text())


def yolo_observations(raw: dict):
    for fs, rows in raw["detections"].items():
        for tid, label, _conf, x1, y1, x2, y2 in rows:
            yield int(fs), f"y{tid}", label, norm_box(x1, y1, x2, y2)


def step_yolo(cams):
    segs, raw0 = [], None
    for cam in cams:
        raw = load_yolo_raw(cam)
        raw0 = raw0 or raw
        segs += build_segments(cam, yolo_observations(raw))
    write_index(INDEX_FILES["yolo"], "yolo", cams, segs,
                {k: raw0[k] for k in ("model", "imgsz", "conf", "tracker")})


# ---------- step: clip ----------

def step_clip(cams):
    from app.clip_embed import MODEL, PRETRAINED, embed_images
    import numpy as np

    ids = [segment_id(c["camera_id"], k) for c in cams for k in range(c["n_segments"])]
    t0 = time.time()
    emb = embed_images([FRAMES_DIR / f"{i}.jpg" for i in ids]).astype(np.float32)
    np.save(CLIP_EMBEDDINGS, emb)
    CLIP_IDS.write_text(json.dumps({"model": MODEL, "pretrained": PRETRAINED, "ids": ids}))
    print(f"[clip] {emb.shape[0]} keyframes x {emb.shape[1]} dims in {time.time() - t0:.0f}s")


# ---------- step: report ----------

def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gts: list, preds: list, thr: float) -> dict[int, int]:
    """One-to-one matching by descending IoU; returns {gt_index: pred_index}."""
    pairs = sorted(((iou(g[1], p[1]), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(preds)), reverse=True)
    out, used = {}, set()
    for v, gi, pi in pairs:
        if v <= thr:
            break
        if gi not in out and pi not in used:
            out[gi] = pi
            used.add(pi)
    return out


def suggest_aliases(per_class: dict) -> dict[str, list[str]]:
    """Sketch label -> COCO labels that YOLO uses for it (>=10% of matches and >=5 boxes)."""
    out = {}
    for label, r in per_class.items():
        n = r["matched@0.5"]
        out[label] = [c for c, cnt in r["yolo_labels@0.5"].items() if n and cnt >= 5 and cnt / n >= 0.10]
    return out


def step_report(cams, gt):
    LARGE = 0.005  # ~100x100 px on 1080p
    stats = defaultdict(lambda: {"gt_boxes": 0, "gt_boxes_large": 0, "matched@0.5": 0, "matched@0.3": 0,
                                 "matched_large@0.5": 0, "yolo_labels@0.5": Counter()})
    unmatched_pred = Counter()
    meta = None
    for cam in cams:
        raw = load_yolo_raw(cam)
        meta = meta or {k: raw[k] for k in ("model", "imgsz", "conf", "tracker")}
        gt_by_frame = defaultdict(list)
        for f, _key, label, b in gt_observations(gt, cam):
            if b.w * b.h >= MIN_AREA:
                gt_by_frame[f].append((label, (b.x, b.y, b.x + b.w, b.y + b.h), b.w * b.h))
        for f in range(0, cam["n_segments"] * cam["seg_frames"], cam["step"]):
            gts = gt_by_frame.get(f, [])
            preds = [(r[1], tuple(r[3:7])) for r in raw["detections"].get(str(f), [])]
            m5, m3 = greedy_match(gts, preds, 0.5), greedy_match(gts, preds, 0.3)
            for gi, (label, _box, area) in enumerate(gts):
                s = stats[label]
                s["gt_boxes"] += 1
                s["gt_boxes_large"] += area >= LARGE
                s["matched@0.3"] += gi in m3
                if gi in m5:
                    s["matched@0.5"] += 1
                    s["matched_large@0.5"] += area >= LARGE
                    s["yolo_labels@0.5"][preds[m5[gi]][0]] += 1
            matched = set(m5.values())
            unmatched_pred.update(p[0] for pi, p in enumerate(preds) if pi not in matched)
    per_class = {}
    for label, s in sorted(stats.items()):
        n = s["gt_boxes"]
        per_class[label] = {
            "gt_boxes": n,
            "matched@0.5": s["matched@0.5"],
            "recall@0.5": round(s["matched@0.5"] / n, 3) if n else None,
            "recall@0.3": round(s["matched@0.3"] / n, 3) if n else None,
            "gt_boxes_large": s["gt_boxes_large"],
            "recall_large@0.5": round(s["matched_large@0.5"] / s["gt_boxes_large"], 3) if s["gt_boxes_large"] else None,
            "yolo_labels@0.5": dict(s["yolo_labels@0.5"].most_common()),
        }
    report = {
        "yolo": meta, "iou_threshold": 0.5, "sample_fps": SAMPLE_FPS, "min_gt_area": MIN_AREA,
        "large_area": LARGE, "per_class": per_class,
        "unmatched_yolo_labels": dict(unmatched_pred.most_common()),
        "suggested_label_aliases": suggest_aliases(per_class),
    }
    ALIAS_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {ALIAS_REPORT.name}")
    print_report(report)


def print_report(r: dict):
    print(f"\n=== Alias report (YOLO {r['yolo']['model']} imgsz={r['yolo']['imgsz']} vs GT, IoU > 0.5) ===")
    print(f"{'GT class':12s} {'GT boxes':>9s} {'recall@.5':>9s} {'@.3':>6s} {'large@.5':>9s}  YOLO labels on matches")
    for label, c in r["per_class"].items():
        labels = ", ".join(f"{k} {v}" for k, v in list(c["yolo_labels@0.5"].items())[:4]) or "-"
        large = "-" if c["recall_large@0.5"] is None else f"{c['recall_large@0.5']:.2f}"
        print(f"{label:12s} {c['gt_boxes']:9d} {c['recall@0.5']:9.2f} {c['recall@0.3']:6.2f} {large:>9s}  {labels}")
    top = ", ".join(f"{k} {v}" for k, v in list(r["unmatched_yolo_labels"].items())[:6])
    print(f"unmatched YOLO boxes: {top or '-'}")
    print(f"suggested LABEL_ALIASES = {json.dumps(r['suggested_label_aliases'])}")


# ---------- step: stats ----------

def print_index_stats(name: str, path: Path, aliases: dict[str, list[str]] | None = None):
    if not path.exists():
        print(f"\n=== {name}: {path.name} missing ===")
        return
    data = json.loads(path.read_text())
    segs = [Segment.model_validate(s) for s in data["segments"]]
    n_tracks, lengths, durations = Counter(), defaultdict(list), defaultdict(list)
    for s in segs:
        for t in s.tracks:
            n_tracks[t.label] += 1
            lengths[t.label].append(len(t.points))
            durations[t.label].append(t.points[-1].t - t.points[0].t)
    print(f"\n=== {name}: {path.name} ===")
    print(f"segments: {len(segs)} ({len(data['cameras'])} cameras), "
          f"with any track: {sum(bool(s.tracks) for s in segs)}, tracks: {sum(n_tracks.values())}")
    print(f"{'label':14s} {'tracks':>7s} {'avg pts':>8s} {'avg sec':>8s} {'segments':>9s}")
    for label, n in n_tracks.most_common():
        in_segs = sum(any(t.label == label for t in s.tracks) for s in segs)
        print(f"{label:14s} {n:7d} {sum(lengths[label]) / n:8.1f} {sum(durations[label]) / n:8.2f} {in_segs:9d}")

    def both(person_labels, forklift_labels):
        return [s for s in segs if {t.label for t in s.tracks} & person_labels
                and {t.label for t in s.tracks} & forklift_labels]

    hits = both({"person"}, {"forklift"})
    print(f"segments with person AND forklift: {len(hits)}"
          + (f" ({', '.join(s.segment_id for s in hits[:8])}{' ...' if len(hits) > 8 else ''})" if hits else ""))
    if aliases and aliases.get("forklift"):
        fl = set(aliases["forklift"])
        hits = both(set(aliases.get("person") or ["person"]), fl)
        print(f"  ... with suggested aliases forklift->{sorted(fl)}: {len(hits)}")


def step_stats():
    aliases = json.loads(ALIAS_REPORT.read_text())["suggested_label_aliases"] if ALIAS_REPORT.exists() else None
    print_index_stats("GT index", INDEX_FILES["gt"])
    print_index_stats("YOLO index", INDEX_FILES["yolo"], aliases)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--no-yolo", action="store_true", help="decode step extracts keyframes only")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    args = ap.parse_args()
    steps = args.steps.split(",")
    if bad := set(steps) - set(STEPS):
        raise SystemExit(f"unknown steps: {sorted(bad)}")

    cams = load_cameras()
    print(f"{len(cams)} cameras: {', '.join(c['camera_id'] for c in cams)}; "
          f"{sum(c['n_segments'] for c in cams)} segments of {SEGMENT_SECONDS}s")
    gt = load_gt() if {"gt", "report"} & set(steps) else None
    if "segments" in steps:
        step_segments(cams)
    if "decode" in steps:
        step_decode(cams, not args.no_yolo, args.model, args.imgsz, args.conf)
    if "gt" in steps:
        step_gt(cams, gt)
    if "yolo" in steps:
        step_yolo(cams)
    if "clip" in steps:
        step_clip(cams)
    if "report" in steps:
        step_report(cams, gt)
    if "stats" in steps:
        step_stats()


if __name__ == "__main__":
    main()
