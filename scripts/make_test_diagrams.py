"""Draw three hand-drawn-style incident diagrams for testing 📄 Diagram -> Sketch.

  uv run python -m scripts.make_test_diagrams      -> data/test_diagrams/*.png
"""

import math
import random

from PIL import Image, ImageDraw, ImageFont

from app.config import DATA_DIR

OUT = DATA_DIR / "test_diagrams"
W, H = 1280, 720
INK = (25, 25, 35)
RED = (200, 30, 30)
rng = random.Random(7)


def font(size: int):
    for name in ("segoepr.ttf", "comic.ttf", "arial.ttf"):  # Segoe Print / Comic look hand-written
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wobbly_line(d: ImageDraw.ImageDraw, p0, p1, width=6, color=INK, steps=12):
    pts = []
    for i in range(steps + 1):
        f = i / steps
        jitter = 0 if i in (0, steps) else rng.uniform(-3, 3)
        pts.append((p0[0] + f * (p1[0] - p0[0]) + jitter, p0[1] + f * (p1[1] - p0[1]) + jitter))
    d.line(pts, fill=color, width=width, joint="curve")


def box(d, x0, y0, x1, y1, **kw):
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        wobbly_line(d, a, b, **kw)


def stick_figure(d, cx, top, height):
    head = height * 0.16
    d.ellipse([cx - head, top, cx + head, top + 2 * head], outline=INK, width=6)
    neck, hip = top + 2 * head, top + height * 0.62
    wobbly_line(d, (cx, neck), (cx, hip))
    wobbly_line(d, (cx - height * 0.22, neck + height * 0.12), (cx + height * 0.22, neck + height * 0.12))
    wobbly_line(d, (cx, hip), (cx - height * 0.16, top + height))
    wobbly_line(d, (cx, hip), (cx + height * 0.16, top + height))


def arrow(d, pts, color=INK):
    for a, b in zip(pts, pts[1:]):
        wobbly_line(d, a, b, color=color)
    (ax, ay), (bx, by) = pts[-2], pts[-1]
    ang = math.atan2(by - ay, bx - ax)
    for side in (-1, 1):
        a2 = ang + math.pi + side * 0.45
        wobbly_line(d, (bx, by), (bx + 34 * math.cos(a2), by + 34 * math.sin(a2)), color=color, steps=4)


def label(d, xy, text, size=40, color=INK):
    d.text(xy, text, fill=color, font=font(size))


def canvas(title: str):
    img = Image.new("RGB", (W, H), (250, 248, 240))
    d = ImageDraw.Draw(img)
    label(d, (30, 18), title, size=30, color=(110, 110, 120))
    return img, d


def approach():
    img, d = canvas("Incident 1 - worker walks toward the forklift")
    box(d, 840, 250, 1010, 420)
    label(d, (850, 425), "FORKLIFT")
    stick_figure(d, 260, 280, 200)
    label(d, (205, 490), "worker")
    arrow(d, [(320, 380), (560, 370), (790, 340)])
    return img


def nobody():
    img, d = canvas("Incident 2 - forklift working alone, no spotter")
    box(d, 560, 120, 720, 270)
    label(d, (565, 275), "forklift")
    d.ellipse([400, 70, 880, 420], outline=RED, width=5)
    wobbly_line(d, (420, 380), (500, 300), color=RED)
    wobbly_line(d, (420, 300), (500, 380), color=RED)
    label(d, (420, 430), "X = nobody here", color=RED)
    stick_figure(d, 1100, 450, 190)
    label(d, (1040, 650), "person")
    return img


def group():
    img, d = canvas("Incident 3 - group of three, one walks away")
    for cx in (870, 960, 1050):
        stick_figure(d, cx, 230, 170)
    label(d, (880, 420), "3 people")
    stick_figure(d, 300, 430, 200)
    label(d, (250, 640), "person")
    arrow(d, [(300, 420), (310, 300), (330, 170)])
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("1_worker_to_forklift", approach), ("2_forklift_nobody", nobody), ("3_group_and_walker", group)):
        path = OUT / f"{name}.png"
        make().save(path)
        print(path.relative_to(DATA_DIR.parent))


if __name__ == "__main__":
    main()
