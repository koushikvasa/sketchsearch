"""LocalSource: segments/tracks from data/index.json (GT) or data/index_yolo.json (YOLO), built by build_index.py."""

import json

import numpy as np

from app.config import CLIP_EMBEDDINGS, CLIP_IDS, INDEX_FILES, INDEX_SOURCE
from app.models import Segment


class LocalSource:
    def __init__(self, index_source: str = INDEX_SOURCE):
        if index_source not in INDEX_FILES:
            raise ValueError(f"INDEX_SOURCE must be one of {sorted(INDEX_FILES)}, got {index_source!r}")
        path = INDEX_FILES[index_source]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run: uv run python -m scripts.build_index")
        data = json.loads(path.read_text())
        self.index_source = index_source
        self.meta: dict = data["meta"]
        self.cameras: list[dict] = data["cameras"]
        self.segments: dict[str, Segment] = {
            s["segment_id"]: Segment.model_validate(s) for s in data["segments"]
        }
        self._clip: tuple[np.ndarray, list[str]] | None = None

    def list_cameras(self) -> list[dict]:
        return self.cameras

    def candidates(
        self,
        labels: set[str],
        camera_ids: list[str] | None = None,
        segment_ids: list[str] | None = None,
    ) -> list[Segment]:
        if segment_ids is None:
            segs = self.segments.values()
        else:
            segs = (self.segments[i] for i in segment_ids if i in self.segments)
        return [
            s
            for s in segs
            if (camera_ids is None or s.camera_id in camera_ids) and labels <= {t.label for t in s.tracks}
        ]

    def get_segment(self, segment_id: str) -> Segment:
        return self.segments[segment_id]

    def text_search(self, query: str, k: int = 200) -> list[tuple[str, float]]:
        from app.clip_embed import embed_text

        if self._clip is None:
            ids = json.loads(CLIP_IDS.read_text())["ids"]
            self._clip = (np.load(CLIP_EMBEDDINGS), ids)
        emb, ids = self._clip
        scores = emb @ embed_text(query)
        ranked = [(ids[i], float(scores[i])) for i in np.argsort(-scores) if ids[i] in self.segments]
        return ranked[:k]

    def ask(self, segment_id: str, question: str) -> str:
        raise NotImplementedError("Phase 4: Gemini verification")

    def clip_url(self, segment_id: str) -> str:
        return f"clips/{segment_id}.mp4"

    def frame_url(self, segment_id: str, t: float = 0.5) -> str:
        return f"frames/{segment_id}.jpg"  # only the mid-frame keyframe is extracted
