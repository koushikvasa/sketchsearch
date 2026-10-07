"""VastSource against mocked VSS / Cosmos responses shaped like the vast-builders-challenge skill docs."""

import base64
import json

import httpx
import pytest

from app.matcher import Matcher
from app.models import Box, Sketch, SketchObject
from app.sources.vast import VastConfig, VastSource, parse_detections, row_to_segment, segment_key

BASE = "https://vss.example"
COSMOS = "http://gpu.example:8001"
SEG_A = "s3://team-x-vss-chunks-segments/team-x/20260101_120000_wh_chunk_0000_seg_003.mp4"
SEG_B = "s3://team-x-vss-chunks-segments/team-x/20260101_120000_wh_chunk_0000_seg_004.mp4"

# perception_json shaped like the YOLO /v1/infer output the docs hint at (frames with per-frame boxes).
PERCEPTION_A = {
    "width": 1920, "height": 1080, "object_classes": ["person", "forklift"],
    "frames": [
        {"timestamp": 0.0, "detections": [
            {"label": "person", "bbox": [192, 540, 288, 864], "confidence": 0.9},
            {"label": "forklift", "bbox": [860, 100, 960, 220], "confidence": 0.8}]},
        {"timestamp": 1.0, "detections": [
            {"label": "person", "bbox": [384, 540, 480, 864], "confidence": 0.9},
            {"label": "forklift", "bbox": [860, 100, 960, 220], "confidence": 0.8}]},
        {"timestamp": 2.0, "detections": [
            {"label": "person", "bbox": [576, 540, 672, 864], "confidence": 0.9},
            {"label": "forklift", "bbox": [862, 100, 962, 220], "confidence": 0.8}]},
    ],
}
ROWS = [
    {"source": SEG_A, "original_video": "s3://team-x-vss-chunks/team-x/wh.mp4", "camera_id": "sdg_warehouse_cam-2",
     "segment_number": 4, "reasoning_content": "A worker walks toward a forklift.",
     "perception_json": json.dumps(PERCEPTION_A)},
    {"source": SEG_B, "original_video": "s3://team-x-vss-chunks/team-x/wh.mp4", "camera_id": "sdg_warehouse_cam-2",
     "segment_number": 5, "reasoning_content": "Empty aisle.",
     "perception_json": json.dumps({"object_classes": [], "object_counts": {}})},
    {"source": "s3://x/other.mp4", "camera_id": "i24_cam-1", "perception_json": None},
]


class FakeBackend:
    """Minimal VSS backend + Cosmos3-Reason endpoint."""

    def __init__(self):
        self.logins = 0
        self.expire_next = False
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path == "/api/v1/auth/login":
            body = json.loads(request.content)
            assert body == {"username": "team-x", "password": "pw"}
            self.logins += 1
            return httpx.Response(200, json={"access_token": f"tok{self.logins}", "token_type": "bearer",
                                             "username": "team-x"})
        if path.startswith("/api/v1/"):
            auth = request.headers.get("authorization") or f"Bearer {request.url.params.get('token')}"
            if self.expire_next:
                self.expire_next = False
                return httpx.Response(401, json={"detail": "expired"})
            assert auth.startswith("Bearer tok")
        if path == "/api/v1/search":
            body = json.loads(request.content)
            assert body["llm_top_n"] == 0
            return httpx.Response(200, json={
                "results": [{"source": SEG_A, "similarity_score": 0.82, "reasoning_content": "forklift"},
                            {"source": SEG_B, "similarity_score": 0.41}],
                "chunk_results": [], "sql_query": "SELECT ..."})
        if path == "/api/v1/videos/stream":
            assert request.url.params["source"] == SEG_A
            return httpx.Response(200, content=b"\x00\x00\x00\x18ftypmp42fake")
        if path == "/api/v1/videos/detections":
            if request.url.params["source"] == SEG_B:
                return httpx.Response(404, json={"detail": "no sidecar"})
            return httpx.Response(200, json=PERCEPTION_A)
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            content = body["messages"][0]["content"]
            assert body["model"] == "nvidia/cosmos3-reason"
            assert content[0]["type"] == "text" and content[1]["type"] == "video_url"
            assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
            assert base64.b64decode(content[1]["video_url"]["url"].split(",", 1)[1]).endswith(b"fake")
            return httpx.Response(200, json={"choices": [{"message": {
                "content": '{"verdict": "YES", "reason": "The worker approaches the forklift."}'}}]})
        return httpx.Response(404)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def vast(backend):
    cfg = VastConfig(ingress_url=BASE, username="team-x", password="pw", cosmos_url=COSMOS,
                     camera_ids=["sdg_warehouse_cam-2"], gpu_token="")
    return VastSource(cfg, http=httpx.Client(transport=httpx.MockTransport(backend)), rows_loader=lambda c: ROWS)


def test_env_names_follow_config_example(monkeypatch):
    for k, v in {"INGRESS_URL": "", "VSS_URL": "https://pod.example/", "VSS_USERNAME": "team-7",
                 "VSS_PASSWORD": "pw", "ACCESS_KEY": "a", "VAST_ACCESS_KEY": "va", "S3_ENDPOINT": "s3.example",
                 "VASTDB_BUCKET": "team-7-vss-db", "COSMOS3_REASON_URL": "http://gpu:8001/",
                 "VAST_CAMERA_IDS": "sdg_warehouse_cam-2,x"}.items():
        monkeypatch.setenv(k, v)
    cfg = VastConfig()
    assert cfg.ingress_url == "https://pod.example"  # the deployed pod only gets VSS_*
    assert (cfg.username, cfg.access_key, cfg.vdb_endpoint) == ("team-7", "va", "s3.example")
    assert (cfg.schema, cfg.collection) == ("vss-schema", "vss-collection")
    assert cfg.cosmos_url == "http://gpu:8001" and cfg.camera_ids == ["sdg_warehouse_cam-2", "x"]


def test_login_once_and_relogin_on_401(vast, backend):
    vast.text_search("forklift")
    vast.text_search("forklift")
    assert backend.logins == 1
    backend.expire_next = True
    vast.text_search("forklift")
    assert backend.logins == 2


def test_rows_become_segments_with_tracks(vast):
    segs = {s.segment_id: s for s in vast.candidates(set())}
    assert set(segs) == {segment_key(SEG_A), segment_key(SEG_B)}  # other camera filtered out
    a = segs[segment_key(SEG_A)]
    assert (a.start, a.end) == (15.0, 20.0)  # segment_number 4 * 5 s guess
    labels = sorted(t.label for t in a.tracks)
    assert labels == ["forklift", "person"] and not a.keyframe_only
    person = next(t for t in a.tracks if t.label == "person")
    assert [round(p.box.x, 3) for p in person.points] == [0.1, 0.2, 0.3]  # pixels -> normalized, linked across frames by nearest centre
    assert person.points[0].box.w == pytest.approx(0.05) and person.points[0].box.h == pytest.approx(0.3)
    assert segs[segment_key(SEG_B)].tracks == [] and segs[segment_key(SEG_B)].keyframe_only
    assert vast.candidates({"forklift"}) == [a]


def test_matcher_runs_on_vast_segments(vast):
    sketch = Sketch(objects=[
        SketchObject(id="w", label="person", start_box=Box(x=0.1, y=0.5, w=0.05, h=0.3),
                     end_box=Box(x=0.25, y=0.5, w=0.05, h=0.3)),
        SketchObject(id="f", label="forklift", start_box=Box(x=0.45, y=0.09, w=0.05, h=0.11)),
    ])
    results = Matcher(vast.candidates(set())).search(sketch)
    assert results[0]["segment_id"] == segment_key(SEG_A) and results[0]["score"] > 0.8


def test_text_search_maps_sources_to_ids(vast):
    hits = vast.text_search("forklift near a worker", k=500)
    assert hits == [(segment_key(SEG_A), 0.82), (segment_key(SEG_B), 0.41)]
    assert vast.sources[segment_key(SEG_A)] == SEG_A


def test_search_request_body(vast, backend):
    vast.text_search("forklift", k=500)
    body = json.loads(next(c for c in backend.calls if c.url.path == "/api/v1/search").content)
    assert body["top_k"] == 100 and body["metadata_filters"] == {"camera_id": "sdg_warehouse_cam-2"}


def test_caption_tagging_fallback(vast):
    vast.load()
    assert vast.caption_tagged("forklift") == {segment_key(SEG_A)}  # SEG_B's caption has no forklift


def test_ask_sends_the_clip_to_cosmos(vast, backend):
    vast.load()
    out = json.loads(vast.ask(segment_key(SEG_A), "Does the worker approach the forklift?"))
    assert out == {"verdict": "YES", "reason": "The worker approaches the forklift.", "_model": "nvidia/cosmos3-reason"}
    cosmos = next(c for c in backend.calls if c.url.path == "/v1/chat/completions")
    assert "authorization" not in cosmos.headers  # no GPU_BEARER_TOKEN -> no auth header


def test_ask_sends_gpu_token_when_set(backend):
    cfg = VastConfig(ingress_url=BASE, username="team-x", password="pw", cosmos_url=COSMOS, gpu_token="g123",
                     camera_ids=[])
    src = VastSource(cfg, http=httpx.Client(transport=httpx.MockTransport(backend)), rows_loader=lambda c: ROWS)
    src.load()
    src.ask(segment_key(SEG_A), "q")
    cosmos = next(c for c in backend.calls if c.url.path == "/v1/chat/completions")
    assert cosmos.headers["authorization"] == "Bearer g123"


def test_sidecar_enrichment_and_404(vast):
    vast.load()
    assert vast.enrich_with_sidecars([segment_key(SEG_A), segment_key(SEG_B)]) == 1


def test_clip_url_is_proxied(vast):
    assert vast.clip_url("v_abc") == "api/vast/clip/v_abc" and vast.frame_url("v_abc") == ""


@pytest.mark.parametrize("raw, n_tracks, keyframe_only", [
    ([{"class": "person", "xywh": [0.1, 0.1, 0.1, 0.2]}], 1, True),  # flat list, normalized xywh
    ({"detections": [{"name": "Person", "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 90}}]}, 1, True),
    ({"frames": [{"frame_index": 0, "detections": [{"label": "person", "bbox": [0.1, 0.1, 0.2, 0.3], "track_id": 7}]},
                 {"frame_index": 9, "detections": [{"label": "person", "bbox": [0.5, 0.1, 0.6, 0.3], "track_id": 7}]}]},
     1, False),  # detector track ids win over IoU linking (boxes don't overlap)
    ({"object_classes": ["person"], "object_counts": {"person": 2}}, 0, True),  # counts only
    ("not json", 0, True),
    (None, 0, True),
])
def test_parse_detections_shapes(raw, n_tracks, keyframe_only):
    tracks, kf = parse_detections(raw)
    assert (len(tracks), kf) == (n_tracks, keyframe_only)


def test_row_without_source_is_skipped():
    assert row_to_segment({"camera_id": "x"}) is None


def test_ask_falls_back_to_gemini_without_cosmos(backend, monkeypatch):
    import app.vision.gemini as gemini

    cfg = VastConfig(ingress_url=BASE, username="team-x", password="pw", cosmos_url="", camera_ids=[])
    src = VastSource(cfg, http=httpx.Client(transport=httpx.MockTransport(backend)), rows_loader=lambda c: ROWS)
    src.load()
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    seen = {}

    def fake_gemini(data, question):
        seen["bytes"] = data
        return '{"verdict": "NO", "reason": "Nobody moves."}', "gemini-fake"

    monkeypatch.setattr(gemini, "ask_video_bytes", fake_gemini)
    out = json.loads(src.ask(segment_key(SEG_A), "q"))
    assert out == {"verdict": "NO", "reason": "Nobody moves.", "_model": "gemini-fake"}
    assert seen["bytes"].endswith(b"fake")


def test_caption_labels_fallback_in_search(vast, monkeypatch):
    """No forklift class in the detector: forklift sketches search caption-tagged segments instead."""
    import app.search as search

    vast.load()
    vast.caption_labels = {"forklift"}
    # Strip forklift tracks to mimic a COCO-only detector.
    for sid, seg in list(vast.segments.items()):
        vast.segments[sid] = seg.model_copy(update={"tracks": [t for t in seg.tracks if t.label != "forklift"]})
    monkeypatch.setattr(search, "get_source", lambda: vast)
    monkeypatch.setattr(search, "get_matcher", lambda: Matcher(list(vast.segments.values())))
    sketch = Sketch(objects=[
        SketchObject(id="w", label="person", start_box=Box(x=0.1, y=0.5, w=0.05, h=0.3)),
        SketchObject(id="f", label="forklift", start_box=Box(x=0.45, y=0.09, w=0.05, h=0.11)),
    ])
    res = search.run_search(sketch, top_n=5)
    assert [r["segment_id"] for r in res["results"]] == [segment_key(SEG_A)]
    assert set(res["results"][0]["assignment"]) == {"w"}  # forklift came from the caption, not a box


def test_sidecar_mode_and_rows_file(backend, tmp_path, monkeypatch):
    import gzip

    snap = tmp_path / "rows.json.gz"
    with gzip.open(snap, "wt", encoding="utf-8") as f:
        json.dump([{**ROWS[0], "perception_json": None}, ROWS[1]], f)
    monkeypatch.setenv("VAST_ROWS_FILE", str(snap))
    cfg = VastConfig(ingress_url=BASE, username="team-x", password="pw", camera_ids=[], detections="sidecar")
    src = VastSource(cfg, http=httpx.Client(transport=httpx.MockTransport(backend)))
    a = src.get_segment(segment_key(SEG_A))  # rows came from the file, tracks from the sidecar
    assert sorted(t.label for t in a.tracks) == ["forklift", "person"] and not a.keyframe_only
    assert src.get_segment(segment_key(SEG_B)).tracks == []  # sidecar 404
