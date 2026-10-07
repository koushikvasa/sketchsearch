"""Core data model. All coordinates are normalized 0-1, boxes are top-left x/y plus width/height."""

from pydantic import BaseModel


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float


class TrackPoint(BaseModel):
    t: float  # seconds from segment start
    box: Box


class Track(BaseModel):
    label: str
    track_id: str
    points: list[TrackPoint]


class Segment(BaseModel):
    segment_id: str
    video_id: str
    camera_id: str
    start: float
    end: float
    caption: str | None = None
    tracks: list[Track]
    keyframe_only: bool = False


class SketchObject(BaseModel):
    id: str
    label: str
    start_box: Box
    end_box: Box | None = None
    path: list[tuple[float, float]] | None = None
    absent: bool = False  # "nobody here" box: label must NOT appear in this region


class Sketch(BaseModel):
    objects: list[SketchObject]
    text: str | None = None
    camera_ids: list[str] | None = None
