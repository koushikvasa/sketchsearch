"""VastSource: the organizers' stack (VSS backend + VastDB + Cosmos3-Reason), behind the VideoSource protocol.

Written from the vast-builders-challenge skill docs before we could touch the real system. Everything the
docs leave open is marked "# FRIDAY-VERIFY:". Run the checklist in FRIDAY.md to confirm them.

Pieces:
- VSS backend at $INGRESS_URL (or $VSS_URL inside the deployed pod), JWT from POST /api/v1/auth/login
  (re-login once on 401): text search, clip streaming, per-segment detection sidecars, captions.
- VastDB table vss-collection (one row per segment): the segment list and per-segment perception.
- Cosmos3-Reason at $COSMOS3_REASON_URL (OpenAI-compatible): ask() on one segment, clip sent as base64.
- Detection parsing is pluggable (DetectionParser), since the box format isn't documented.

Segment ids: VAST identifies a segment by its S3 URI ("source"). URIs contain slashes, so we expose a
short stable id "v_<sha1(source)[:16]>" and keep the id -> source map.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import httpx

from app.models import Box, Segment, Track, TrackPoint

log = logging.getLogger("uvicorn.error")

API = "/api/v1"
VECTOR_COLUMNS = ("vectors", "vectors_visual")  # never select these: the SDK can crash on vector types
SEGMENT_SECONDS_GUESS = 5.0  # FRIDAY-VERIFY: segment length is not documented ("short fixed-length clips")
FRAME_SIZE_GUESS = (1920, 1080)  # FRIDAY-VERIFY: used only when boxes are pixels and no size is given
MIN_POINTS = 1


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


@dataclass
class VastConfig:
    """Env var names from config.example; the deployed pod only gets VSS_URL / VSS_USERNAME / VSS_PASSWORD.
    VSS_* win: USERNAME is also a built-in variable on Windows (and some shells)."""

    ingress_url: str = field(default_factory=lambda: _env("VSS_URL", "INGRESS_URL").rstrip("/"))
    username: str = field(default_factory=lambda: _env("VSS_USERNAME", "USERNAME"))
    password: str = field(default_factory=lambda: _env("VSS_PASSWORD", "PASSWORD"))
    vdb_endpoint: str = field(default_factory=lambda: _env("VDB_ENDPOINT", "S3_ENDPOINT"))
    access_key: str = field(default_factory=lambda: _env("VAST_ACCESS_KEY", "ACCESS_KEY"))
    secret_key: str = field(default_factory=lambda: _env("VAST_SECRET_KEY", "SECRET_KEY"))
    bucket: str = field(default_factory=lambda: _env("VASTDB_BUCKET"))
    schema: str = field(default_factory=lambda: _env("VDB_SCHEMA", default="vss-schema"))
    collection: str = field(default_factory=lambda: _env("VDB_COLLECTION", default="vss-collection"))
    cosmos_url: str = field(default_factory=lambda: _env("COSMOS3_REASON_URL").rstrip("/"))
    cosmos_model: str = field(default_factory=lambda: _env("COSMOS3_REASON_MODEL", default="nvidia/cosmos3-reason"))
    gpu_token: str = field(default_factory=lambda: _env("GPU_BEARER_TOKEN"))
    # Restrict to one pack, e.g. "sdg_warehouse_cam-2" (Pack C). Comma-separated; empty = everything.
    camera_ids: list[str] = field(default_factory=lambda: [c for c in _env("VAST_CAMERA_IDS").split(",") if c])
    # Where per-frame boxes come from: "perception" (VastDB perception_json) or "sidecar" (GET /videos/detections).
    detections: str = field(default_factory=lambda: _env("VAST_DETECTIONS", default="perception"))
    # Labels YOLO doesn't detect (e.g. forklift: not a COCO class) found through caption search instead.
    caption_labels: set[str] = field(default_factory=lambda: {
        c.strip().lower() for c in _env("VAST_CAPTION_LABELS").split(",") if c.strip()})


def segment_key(source: str) -> str:
    return "v_" + hashlib.sha1(source.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Detection parsing (pluggable)
# ---------------------------------------------------------------------------
# A parser turns one segment's raw detection JSON into (tracks, keyframe_only).
DetectionParser = Callable[[object], tuple[list[Track], bool]]

LABEL_KEYS = ("label", "class", "class_name", "name", "category")
BOX_KEYS = ("bbox", "box", "xyxy", "xywh", "bbox_xyxy")
FRAME_LIST_KEYS = ("frames", "detections_per_frame", "per_frame")
DET_LIST_KEYS = ("detections", "boxes", "objects")
TIME_KEYS = ("t", "time", "timestamp", "ts", "sec", "seconds")


def _first(d: dict, keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_box(det: dict, size: tuple[float, float]) -> Box | None:
    """Box from a detection dict. FRIDAY-VERIFY: xyxy vs xywh and pixel vs normalized are guesses:
    keys named "xywh" (or x/y/w/h fields) are xywh, any other 4-list is xyxy; values > 1.5 are pixels."""
    raw, fmt = None, "xyxy"
    for k in BOX_KEYS:
        if k in det:
            raw, fmt = det[k], ("xywh" if k == "xywh" else "xyxy")
            break
    if raw is None and all(k in det for k in ("x", "y", "w", "h")):
        raw, fmt = [det["x"], det["y"], det["w"], det["h"]], "xywh"
    if isinstance(raw, dict):
        if all(k in raw for k in ("x1", "y1", "x2", "y2")):
            raw, fmt = [raw["x1"], raw["y1"], raw["x2"], raw["y2"]], "xyxy"
        elif all(k in raw for k in ("x", "y", "w", "h")):
            raw, fmt = [raw["x"], raw["y"], raw["w"], raw["h"]], "xywh"
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    a, b, c, d = (float(v) for v in raw)
    if fmt == "xywh":
        c, d = a + c, b + d
    if max(a, b, c, d) > 1.5:  # pixels
        a, c = a / size[0], c / size[0]
        b, d = b / size[1], d / size[1]
    a, b, c, d = (min(max(v, 0.0), 1.0) for v in (a, b, c, d))
    if c <= a or d <= b:
        return None
    return Box(x=round(a, 4), y=round(b, 4), w=round(c - a, 4), h=round(d - b, 4))


def _iou(p: Box, q: Box) -> float:
    ix = max(0.0, min(p.x + p.w, q.x + q.w) - max(p.x, q.x))
    iy = max(0.0, min(p.y + p.h, q.y + q.h) - max(p.y, q.y))
    inter = ix * iy
    union = p.w * p.h + q.w * q.h - inter
    return inter / union if union > 0 else 0.0


def _center_dist(p: Box, q: Box) -> float:
    return ((p.x + p.w / 2 - q.x - q.w / 2) ** 2 + (p.y + p.h / 2 - q.y - q.h / 2) ** 2) ** 0.5


def link_tracks(frames: list[tuple[float, list[tuple[str, Box, str | None]]]], min_iou: float = 0.3,
                gate: float = 1.5) -> list[Track]:
    """Frames of (t, [(label, box, track_id|None)]) -> tracks. Uses detector track ids when present,
    otherwise greedy same-label linking frame to frame: best IoU, or (sparse sampling: boxes that moved
    further than their own size) the nearest centre within `gate` box heights."""
    tracks: dict[str, Track] = {}
    open_tracks: dict[str, Box] = {}  # track id -> last box (unlabelled linking)
    next_id = 0
    for t, dets in sorted(frames, key=lambda f: f[0]):
        used: set[str] = set()
        for label, box, tid in dets:
            if tid is None:
                best, best_iou = None, min_iou
                for k, last in open_tracks.items():
                    if k not in used and tracks[k].label == label and (v := _iou(last, box)) > best_iou:
                        best, best_iou = k, v
                if best is None:
                    best_d = gate * max(box.h, box.w)
                    for k, last in open_tracks.items():
                        if k not in used and tracks[k].label == label and (d := _center_dist(last, box)) < best_d:
                            best, best_d = k, d
                if best is None:
                    best = f"a{next_id}"
                    next_id += 1
                tid = best
            tid = str(tid)
            if tid not in tracks:
                tracks[tid] = Track(label=label, track_id=tid, points=[])
            tracks[tid].points.append(TrackPoint(t=round(t, 3), box=box))
            open_tracks[tid] = box
            used.add(tid)
    return [tr for tr in tracks.values() if len(tr.points) >= MIN_POINTS]


def parse_detections(raw: object, segment_seconds: float = SEGMENT_SECONDS_GUESS) -> tuple[list[Track], bool]:
    """Default parser. FRIDAY-VERIFY: accepts the shapes the skill docs hint at:
    - {"frames": [{"t"|"timestamp"|"frame_index": .., "detections": [{"label", "bbox", "confidence", "track_id"?}]}]}
    - [{"label", "bbox", ...}, ...] or {"detections": [...]}  -> one keyframe (keyframe_only)
    - {"object_classes": [...], "object_counts": {...}} only -> no boxes (returns [], True)
    Frame indices without times are spread evenly over the segment."""
    if isinstance(raw, (bytes, str)):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return [], True
    if not raw:
        return [], True
    size = FRAME_SIZE_GUESS
    if isinstance(raw, dict):
        w, h = _first(raw, ("width", "frame_width", "img_w")), _first(raw, ("height", "frame_height", "img_h"))
        if w and h:
            size = (float(w), float(h))

    def dets_of(container) -> list[tuple[str, Box, str | None]]:
        items = container if isinstance(container, list) else (_first(container, DET_LIST_KEYS) or [])
        out = []
        for det in items:
            if not isinstance(det, dict):
                continue
            label = _first(det, LABEL_KEYS)
            box = _to_box(det, size)
            if label is None or box is None:
                continue
            tid = _first(det, ("track_id", "id", "tracker_id"))
            out.append((str(label).lower(), box, None if tid is None else str(tid)))
        return out

    frame_list = _first(raw, FRAME_LIST_KEYS) if isinstance(raw, dict) else None
    if isinstance(frame_list, list) and frame_list:
        frames = []
        n = len(frame_list)
        for i, fr in enumerate(frame_list):
            if not isinstance(fr, dict):
                continue
            t = _first(fr, TIME_KEYS)
            if t is None:
                idx = _first(fr, ("frame_index", "frame", "index"))
                t = (float(idx if idx is not None else i) / max(n - 1, 1)) * segment_seconds
            frames.append((float(t), dets_of(fr)))
        tracks = link_tracks(frames)
        keyframe_only = len({t for t, _ in frames}) < 2
        return tracks, keyframe_only
    dets = dets_of(raw)
    if dets:
        return link_tracks([(segment_seconds / 2, dets)]), True
    return [], True


# ---------------------------------------------------------------------------
# VastDB rows
# ---------------------------------------------------------------------------
def load_rows_vastdb(cfg: VastConfig, limit: int | None = None) -> list[dict]:
    """All vss-collection rows except vector columns (pattern from vastdb-read/SKILL.md and query.py)."""
    import vastdb  # FRIDAY-VERIFY: `pip install vastdb pyarrow` on the VM / in the pod

    endpoint = cfg.vdb_endpoint if "://" in cfg.vdb_endpoint else f"http://{cfg.vdb_endpoint}"
    session = vastdb.connect(endpoint=endpoint, access=cfg.access_key, secret=cfg.secret_key, ssl_verify=False)
    rows: list[dict] = []
    with session.transaction() as tx:
        table = tx.bucket(cfg.bucket).schema(cfg.schema).table(cfg.collection)
        columns = table.columns() if callable(table.columns) else table.columns
        names = [c.name for c in columns if c.name not in VECTOR_COLUMNS]
        # vastdb 2.1 select() takes an ibis predicate and limit_rows (not shown in the skill docs).
        # FRIDAY-VERIFY: the server accepts the camera_id pushdown; on error we filter client-side.
        kwargs = {"limit_rows": limit} if limit else {}
        if cfg.camera_ids and "camera_id" in names:
            from ibis import _

            kwargs["predicate"] = _.camera_id.isin(cfg.camera_ids)
        try:
            reader = table.select(columns=names, **kwargs)
        except Exception as e:  # noqa: BLE001 - any pushdown failure: read everything instead
            log.warning("VastDB predicate pushdown failed (%s); filtering client-side", e)
            reader = table.select(columns=names, **({"limit_rows": limit} if limit else {}))
        for batch in reader:
            rows.extend(batch.to_pylist())
            if limit and len(rows) >= limit:
                break
    return rows[:limit] if limit else rows


def load_rows_file(path: str) -> list[dict]:
    """Rows snapshot written by `scripts.vast_probe --snapshot` (.json or .json.gz), for a pod that can't
    reach VastDB directly."""
    import gzip

    raw = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    with raw as f:
        return json.load(f)


def _num(row: dict, keys: Iterable[str]) -> float | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


# FRIDAY-VERIFY: timing column names are not documented; these are guesses plus segment_number * length.
START_KEYS = ("start_sec", "start_time", "segment_start_sec", "segment_start", "start_offset")
END_KEYS = ("end_sec", "end_time", "segment_end_sec", "segment_end", "end_offset")


def row_to_segment(row: dict, parser: DetectionParser = parse_detections) -> Segment | None:
    source = row.get("source")
    if not source:
        return None
    start = _num(row, START_KEYS)
    if start is None:
        n = _num(row, ("segment_number",))
        start = (n - 1) * SEGMENT_SECONDS_GUESS if n else 0.0
    end = _num(row, END_KEYS) or start + SEGMENT_SECONDS_GUESS
    tracks, keyframe_only = parser(row.get("perception_json"))
    return Segment(
        segment_id=segment_key(source), video_id=str(row.get("original_video") or source),
        camera_id=str(row.get("camera_id") or row.get("location") or "unknown"),
        start=start, end=max(end, start + 0.5), caption=row.get("reasoning_content"),
        tracks=tracks, keyframe_only=keyframe_only,
    )


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
class VastSource:
    def __init__(
        self,
        config: VastConfig | None = None,
        http: httpx.Client | None = None,
        rows_loader: Callable[[VastConfig], list[dict]] | None = None,
        parser: DetectionParser = parse_detections,
    ):
        self.cfg = config or VastConfig()
        if not self.cfg.ingress_url:
            raise RuntimeError("SOURCE=vast needs INGRESS_URL (or VSS_URL) plus USERNAME/PASSWORD")
        self.http = http or httpx.Client(timeout=60, verify=False)  # FRIDAY-VERIFY: internal TLS certs
        rows_file = os.getenv("VAST_ROWS_FILE")
        self.rows_loader = rows_loader or ((lambda cfg: load_rows_file(rows_file)) if rows_file else load_rows_vastdb)
        self.parser = parser
        self.index_source = "vast"
        self.label_aliases: dict[str, list[str]] = {}  # FRIDAY-VERIFY: fill from dashboard objects[]
        self.caption_labels: set[str] = self.cfg.caption_labels
        self._token: str | None = None
        self._lock = threading.Lock()
        self.sources: dict[str, str] = {}  # segment_id -> S3 source URI
        self.segments: dict[str, Segment] = {}
        self._loaded = False

    # ---------- auth + HTTP ----------
    def login(self) -> str:
        r = self.http.post(f"{self.cfg.ingress_url}{API}/auth/login",
                           json={"username": self.cfg.username, "password": self.cfg.password})
        r.raise_for_status()
        self._token = r.json()["access_token"]
        return self._token

    def token(self) -> str:
        with self._lock:
            return self._token or self.login()

    def request(self, method: str, path: str, **kw) -> httpx.Response:
        """Authenticated call to the VSS backend; re-login once on 401 (no refresh endpoint is documented)."""
        for attempt in (1, 2):
            headers = {**kw.pop("headers", {}), "Authorization": f"Bearer {self.token()}"}
            r = self.http.request(method, f"{self.cfg.ingress_url}{API}{path}", headers=headers, **kw)
            if r.status_code == 401 and attempt == 1:
                with self._lock:
                    self._token = None
                continue
            r.raise_for_status()
            return r
        raise RuntimeError("unreachable")

    # ---------- segments ----------
    def _register(self, seg: Segment, source: str) -> None:
        self.sources[seg.segment_id] = source
        self.segments[seg.segment_id] = seg

    def load(self) -> None:
        if self._loaded:
            return
        rows = self.rows_loader(self.cfg)
        for row in rows:
            if self.cfg.camera_ids and str(row.get("camera_id")) not in self.cfg.camera_ids:
                continue
            seg = row_to_segment(row, self.parser)
            if seg:
                self._register(seg, row["source"])
        self._loaded = True
        log.info("VastSource: %d segments loaded", len(self.segments))
        if self.cfg.detections == "sidecar":
            n = self.enrich_with_sidecars(list(self.segments))
            log.info("VastSource: per-frame sidecar boxes for %d of %d segments", n, len(self.segments))

    def detections_sidecar(self, segment_id: str) -> object | None:
        """Per-frame YOLO boxes from GET /videos/detections (404 = no sidecar)."""
        try:
            return self.request("GET", "/videos/detections", params={"source": self.sources[segment_id]}).json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def enrich_with_sidecars(self, segment_ids: Iterable[str], workers: int = 8) -> int:
        """Replace per-segment boxes with per-frame sidecar tracks (GET /videos/detections, 8 at a time).
        Returns how many segments got sidecar tracks."""
        from concurrent.futures import ThreadPoolExecutor

        def fetch(sid):
            try:
                return sid, self.detections_sidecar(sid)
            except httpx.HTTPError as e:
                log.warning("sidecar for %s failed: %s", sid, e)
                return sid, None

        n = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sid, raw in pool.map(fetch, list(segment_ids)):
                if raw is None:
                    continue
                tracks, keyframe_only = self.parser(raw)
                if tracks:
                    self.segments[sid] = self.segments[sid].model_copy(
                        update={"tracks": tracks, "keyframe_only": keyframe_only})
                    n += 1
        return n

    # ---------- VideoSource protocol ----------
    def list_cameras(self) -> list[dict]:
        self.load()
        cams = sorted({s.camera_id for s in self.segments.values()})
        return [{"camera_id": c, "video_id": c, "name": c, "fps": None, "duration": None, "width": None,
                 "height": None, "video_url": None, "background_url": None} for c in cams]

    def candidates(self, labels: set[str], camera_ids: list[str] | None = None,
                   segment_ids: list[str] | None = None) -> list[Segment]:
        self.load()
        segs = self.segments.values() if segment_ids is None else (
            self.segments[i] for i in segment_ids if i in self.segments)
        return [s for s in segs if (camera_ids is None or s.camera_id in camera_ids)
                and labels <= {t.label for t in s.tracks}]

    def get_segment(self, segment_id: str) -> Segment:
        self.load()
        return self.segments[segment_id]

    def text_search(self, query: str, k: int = 200) -> list[tuple[str, float]]:
        """POST /api/v1/search (hybrid caption + visual). llm_top_n=0 skips the LLM synthesis."""
        body = {"query": query, "top_k": min(k, 100), "llm_top_n": 0, "min_similarity": 0.1}
        if self.cfg.camera_ids:
            body["metadata_filters"] = {"camera_id": self.cfg.camera_ids[0]}  # FRIDAY-VERIFY: single value only?
        data = self.request("POST", "/search", json=body).json()
        out = []
        for hit in data.get("results", []):
            source = hit.get("source")
            if not source:
                continue
            sid = segment_key(source)
            self.sources.setdefault(sid, source)
            out.append((sid, float(hit.get("similarity_score") or 0.0)))
        return out

    def caption_tagged(self, label: str, k: int = 100) -> set[str]:
        """Segments whose caption search hits mention a label (fallback when YOLO has no such class,
        e.g. forklift). FRIDAY-VERIFY: check that reasoning_content really names forklifts."""
        out = set()
        for sid, _ in self.text_search(label, k=k):
            seg = self.segments.get(sid)
            if seg is None or not seg.caption or label.lower() in seg.caption.lower():
                out.add(sid)
        return out

    def clip_bytes(self, segment_id: str) -> bytes:
        return self.request("GET", "/videos/stream", params={"source": self.sources[segment_id],
                                                             "token": self.token()}).content

    def ask(self, segment_id: str, question: str) -> str:
        """Cosmos3-Reason on exactly this segment; falls back to Gemini (same clip bytes) when Cosmos is
        not configured or fails and GEMINI_API_KEY is set."""
        clip = self.clip_bytes(segment_id)
        try:
            if not self.cfg.cosmos_url:
                raise RuntimeError("COSMOS3_REASON_URL is not set")
            return self._ask_cosmos(clip, question)
        except Exception as e:
            if not os.getenv("GEMINI_API_KEY"):
                raise
            log.warning("Cosmos3-Reason unavailable (%s); asking Gemini instead", e)
            from app.vision.gemini import ask_video_bytes

            text, model = ask_video_bytes(clip, question)
            try:
                data = json.loads(text[text.find("{"):text.rfind("}") + 1])
                return json.dumps({**data, "_model": model})
            except ValueError:
                return text

    def _ask_cosmos(self, clip: bytes, question: str) -> str:
        """The clip goes inline as a base64 data URI (the production-style payload in gpu/model-smoke-test)."""
        video = base64.b64encode(clip).decode()
        payload = {
            "model": self.cfg.cosmos_model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video}"}},
            ]}],
            "max_tokens": 300,
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self.cfg.gpu_token}"} if self.cfg.gpu_token else {}
        r = self.http.post(f"{self.cfg.cosmos_url}/v1/chat/completions", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        if not content:  # documented failure mode: overloaded / rejected prompt
            raise RuntimeError("Cosmos3-Reason returned empty content")
        try:  # tag the model like LocalSource does, when the answer is JSON
            data = json.loads(content[content.find("{"):content.rfind("}") + 1])
            return json.dumps({**data, "_model": self.cfg.cosmos_model})
        except ValueError:
            return content

    def clip_url(self, segment_id: str) -> str:
        return f"api/vast/clip/{segment_id}"  # proxied by app.main so the JWT never reaches the browser

    def frame_url(self, segment_id: str, t: float = 0.5) -> str:
        return ""  # no keyframe images in VAST; the UI falls back to the video's first frame

    def stream_clip(self, segment_id: str, range_header: str | None = None) -> httpx.Response:
        """Open a (Range-capable) stream of one segment for the clip proxy."""
        headers = {"Range": range_header} if range_header else {}
        req = self.http.build_request(
            "GET", f"{self.cfg.ingress_url}{API}/videos/stream",
            params={"source": self.sources[segment_id], "token": self.token()}, headers=headers)
        return self.http.send(req, stream=True)
