"""Plain spatial words for sketches and tracks (left/right, up/down the aisle, toward/away, near/far).

Shared by the verification questions (what the user drew) and the result explanations (what matched),
so both describe a scene the same way.
"""

import math

from app.models import Box, SketchObject

STILL = 0.03  # displacement below this is "standing still"
TOWARD = 0.03  # distance change that counts as toward / away
NOUNS = {"person": "person", "forklift": "forklift", "robot": "small mobile robot", "transporter": "low transport cart"}


def center(b: Box) -> tuple[float, float]:
    return b.x + b.w / 2, b.y + b.h / 2


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def horizontal(x: float) -> str:
    return "left" if x < 0.4 else "right" if x > 0.6 else "centre"


def region(c) -> str:
    """'top-left', 'centre', 'bottom-right', ... of the frame."""
    h = horizontal(c[0])
    v = "top" if c[1] < 0.33 else "bottom" if c[1] > 0.66 else "middle"
    if v == "middle":
        return "centre" if h == "centre" else f"middle-{h}"
    return f"{v}-{h}" if h != "centre" else f"{v}-centre"


def aisle_position(c) -> str:
    if c[1] < 0.25:
        return "at the far end"
    if c[1] > 0.65:
        return "near the camera"
    return {"left": "on the left", "right": "on the right", "centre": "mid-aisle"}[horizontal(c[0])]


def direction(d, aisle: bool = False, height: float | None = None) -> str | None:
    """'right', 'left and up the frame', ... or None when not moving.

    With an object height, "moving" is relative to its size (perspective: a far person walking covers
    far fewer pixels than a near one)."""
    dx, dy = d
    still = STILL if height is None else max(0.01, 0.25 * height)
    if math.hypot(dx, dy) < still:
        return None
    parts = []
    if abs(dx) >= still * 0.66:
        parts.append("right" if dx > 0 else "left")
    if abs(dy) >= still * 0.66:
        if aisle:
            parts.append("up the aisle" if dy < 0 else "down the aisle")
        else:
            parts.append("up the frame (away from the camera)" if dy < 0 else "down the frame (toward the camera)")
    return " and ".join(parts)


def names(objects: list[SketchObject]) -> dict[str, str]:
    """id -> 'the forklift' / 'person A' (letters only when a label repeats)."""
    counts: dict[str, int] = {}
    for o in objects:
        counts[o.label] = counts.get(o.label, 0) + 1
    seen: dict[str, int] = {}
    out = {}
    for o in objects:
        noun = NOUNS.get(o.label, o.label)
        if counts[o.label] == 1:
            out[o.id] = f"the {noun}"
        else:
            out[o.id] = f"{noun} {chr(ord('A') + seen.get(o.label, 0))}"
            seen[o.label] = seen.get(o.label, 0) + 1
    return out


def relative_motion(a0, a1, b0, b1) -> str | None:
    """'toward' / 'away from' when the distance between two objects changes, else None."""
    change = dist(a1, b1) - dist(a0, b0)
    if change < -TOWARD:
        return "toward"
    if change > TOWARD:
        return "away from"
    return None


def zone_anchor(zone: Box, objects: list[tuple[str, tuple[float, float]]], pad: float = 0.05) -> str | None:
    """Id of the object whose centre lies in (or right next to) an absent zone."""
    for oid, c in objects:
        if zone.x - pad <= c[0] <= zone.x + zone.w + pad and zone.y - pad <= c[1] <= zone.y + zone.h + pad:
            return oid
    return None
