"""Geometric sketch matcher (plan section 7).

Every segment is scanned with a sliding 1.5 s window. Per-window track features ("rows") are computed
once in Matcher.__init__. A search then
  1. scores every sketch object against every row in one vectorized pass (position, size, motion, End layout),
  2. assigns objects to tracks per window with the Hungarian algorithm (the only per-window Python step),
  3. adds pairwise terms (relations, converging/diverging) and absent-box checks vectorized over windows,
  4. keeps each segment's best window.
"""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.models import Segment, Sketch, SketchObject

PLAN_WEIGHTS = {"relations": 0.35, "position": 0.25, "motion": 0.25, "size": 0.10, "keyframe": 0.05}
# Tuned once on scripts/eval_matcher.py (seed 42, validated on seed 7): with hand-drawn noise, the
# End layout (keyframe) and size carry the motion signal more reliably than path shape/direction.
DEFAULT_WEIGHTS = {"relations": 0.15, "position": 0.35, "motion": 0.05, "size": 0.20, "keyframe": 0.25}
COMPONENTS = list(PLAN_WEIGHTS)
# A sketch that clearly draws movement gets motion back at this weight (others renormalized).
CLEAR_MOTION = 0.10
CLEAR_MOTION_WEIGHT = 0.25
WINDOW = 1.5  # seconds a sketch (Start -> End) represents
WINDOW_STEP = 0.5
PATH_POINTS = 16
POS_SIGMA = 0.15
SHAPE_SIGMA = 0.10  # mean distance between start-aligned paths
STILL = 0.02  # displacement below this is "not moving"
REL_MARGIN = 0.03  # left/right, above/below only judged when the sketch separates objects by more
REL_SCALE = 0.05  # track separation (same sign) that counts as full agreement
NEAR_FAR_SCALE = 0.3
CONVERGE_MARGIN = 0.05  # sketch pair distance change that counts as converging/diverging
CONVERGE_BONUS = 0.15
ABSENT_IOU = 0.1
ABSENT_PENALTY = 0.2
_INVALID = 1e3
_EPS = 1e-6


def resample_path(points) -> np.ndarray:
    """Resample a polyline to PATH_POINTS points evenly spaced by arc length."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if len(pts) == 1 or cum[-1] < _EPS:
        return np.repeat(pts[:1], PATH_POINTS, axis=0)
    s = np.linspace(0.0, cum[-1], PATH_POINTS)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)


def _center(box) -> np.ndarray:
    return np.array([box.x + box.w / 2, box.y + box.h / 2])


def object_displacement(o: SketchObject) -> float:
    """How far the sketch says an object moves: path start->end, or start box -> end box."""
    d = 0.0
    if o.path and len(o.path) >= 2:
        d = float(np.hypot(o.path[-1][0] - o.path[0][0], o.path[-1][1] - o.path[0][1]))
    if o.end_box is not None:
        d = max(d, float(np.linalg.norm(_center(o.end_box) - _center(o.start_box))))
    return d


def has_clear_motion(sketch: Sketch) -> bool:
    return any(object_displacement(o) >= CLEAR_MOTION for o in sketch.objects if not o.absent)


def effective_weights(sketch: Sketch, weights: dict[str, float] | None = None, adaptive: bool | None = None) -> dict:
    """Weights a search uses. Adaptive (the default when no weights are given): sketches with clear
    motion get motion = CLEAR_MOTION_WEIGHT and the other weights scaled to keep the same total."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    if adaptive is None:
        adaptive = weights is None
    if adaptive and has_clear_motion(sketch) and w["motion"] < CLEAR_MOTION_WEIGHT:
        total = sum(w.values())
        rest = total - w["motion"]
        scale = (total - CLEAR_MOTION_WEIGHT) / rest if rest > 0 else 0.0
        w = {k: (CLEAR_MOTION_WEIGHT if k == "motion" else v * scale) for k, v in w.items()}
    return w


def _xyxy(box) -> np.ndarray:
    return np.array([box.x, box.y, box.x + box.w, box.y + box.h])


class _Query:
    """Vectorized features of the sketch's present (non-absent) objects."""

    def __init__(self, objects: list[SketchObject]):
        k = len(objects)
        self.objects = objects
        self.start = np.array([_center(o.start_box) for o in objects]).reshape(k, 2)
        self.area = np.array([o.start_box.w * o.start_box.h for o in objects])
        self.has_end = np.array([o.end_box is not None for o in objects], dtype=bool)
        self.end = np.array([_center(o.end_box or o.start_box) for o in objects]).reshape(k, 2)
        self.has_motion = np.zeros(k, dtype=bool)
        self.path = np.zeros((k, PATH_POINTS, 2))
        self.disp = np.zeros((k, 2))
        for i, o in enumerate(objects):
            if o.path and len(o.path) >= 2:
                pts = o.path
            elif o.end_box is not None:
                pts = [self.start[i], self.end[i]]
            else:
                continue
            p = resample_path(pts)
            self.has_motion[i] = True
            self.path[i] = p - p[0]
            self.disp[i] = p[-1] - p[0]
        # Where the sketch says each object ends up (for converging/diverging checks).
        self.final = np.where(self.has_end[:, None], self.end, self.start + self.disp)
        self.pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]


class Matcher:
    def __init__(self, segments: list[Segment], aliases: dict[str, list[str]] | None = None):
        self.segments = segments
        self.aliases = aliases or {}
        self.label_ids = {lbl: i for i, lbl in enumerate(sorted({t.label for s in segments for t in s.tracks}))}
        self.seg_counts = [Counter(t.label for t in s.tracks) for s in segments]
        self.seg_cameras = np.array([s.camera_id for s in segments])
        self.seg_ids = np.array([s.segment_id for s in segments])

        cols: dict[str, list] = {k: [] for k in
                                 ("track_id", "label", "assignable", "moving", "start", "end", "area", "path", "disp")}
        boxes, ptr = [], [0]
        win: dict[str, list] = {"seg": [], "t0": [], "t1": [], "r0": [], "r1": []}
        for si, seg in enumerate(segments):
            dur = seg.end - seg.start
            tracks = [(tr, np.array([p.t for p in tr.points]),
                       np.array([[p.box.x, p.box.y, p.box.w, p.box.h] for p in tr.points])) for tr in seg.tracks]
            if seg.keyframe_only or dur <= WINDOW + _EPS:
                starts = [0.0]
            else:
                starts = np.arange(0.0, dur - WINDOW + _EPS, WINDOW_STEP)
            for t0 in starts:
                t1 = dur if seg.keyframe_only else min(t0 + WINDOW, dur)
                r0 = len(cols["track_id"])
                for tr, t, b in tracks:
                    m = (t >= t0 - _EPS) & (t <= t1 + _EPS)
                    n = int(m.sum())
                    if not n:
                        continue
                    bw = b[m]
                    c = bw[:, :2] + bw[:, 2:] / 2
                    moving = n >= 2 and not seg.keyframe_only
                    path = resample_path(c)
                    cols["track_id"].append(tr.track_id)
                    cols["label"].append(self.label_ids[tr.label])
                    cols["assignable"].append(moving or seg.keyframe_only)
                    cols["moving"].append(moving)
                    cols["start"].append(c[0])
                    cols["end"].append(c[-1])
                    cols["area"].append(bw[0, 2] * bw[0, 3])
                    cols["path"].append(path - path[0])
                    cols["disp"].append(c[-1] - c[0])
                    boxes.append(np.column_stack([bw[:, 0], bw[:, 1], bw[:, 0] + bw[:, 2], bw[:, 1] + bw[:, 3]]))
                    ptr.append(ptr[-1] + n)
                for key, v in (("seg", si), ("t0", float(t0)), ("t1", float(t1)), ("r0", r0),
                               ("r1", len(cols["track_id"]))):
                    win[key].append(v)

        n = len(cols["track_id"])
        self.track_id = cols["track_id"]
        self.label = np.array(cols["label"], dtype=int)
        self.assignable = np.array(cols["assignable"], dtype=bool)
        self.moving = np.array(cols["moving"], dtype=bool)
        self.start = np.array(cols["start"]).reshape(n, 2)
        self.end = np.array(cols["end"]).reshape(n, 2)
        self.area = np.array(cols["area"])
        self.path = np.array(cols["path"]).reshape(n, PATH_POINTS, 2)
        self.disp = np.array(cols["disp"]).reshape(n, 2)
        self.boxes = np.concatenate(boxes) if boxes else np.zeros((0, 4))
        self.box_ptr = np.array(ptr)
        self.win_seg = np.array(win["seg"], dtype=int)
        self.win_t0 = np.array(win["t0"])
        self.win_t1 = np.array(win["t1"])
        self.win_r0 = np.array(win["r0"], dtype=int)
        self.win_r1 = np.array(win["r1"], dtype=int)

    @property
    def n_windows(self) -> int:
        return len(self.win_seg)

    # ---------- helpers ----------

    def resolve(self, label: str) -> list[int]:
        """Index label ids a sketch label may match (LABEL_ALIASES; identity by default)."""
        return [self.label_ids[l] for l in self.aliases.get(label, [label]) if l in self.label_ids]

    def _window_sum(self, per_row: np.ndarray) -> np.ndarray:
        """Sum a per-row quantity (..., N) over each window's rows -> (..., W)."""
        cs = np.concatenate([np.zeros(per_row.shape[:-1] + (1,)), np.cumsum(per_row, axis=-1)], axis=-1)
        return cs[..., self.win_r1] - cs[..., self.win_r0]

    def _allowed(self, present, segment_ids, camera_ids) -> np.ndarray:
        allowed = np.ones(len(self.segments), dtype=bool)
        if segment_ids is not None:
            allowed &= np.isin(self.seg_ids, list(segment_ids))
        if camera_ids:
            allowed &= np.isin(self.seg_cameras, list(camera_ids))
        names = {v: k for k, v in self.label_ids.items()}
        for label, k in Counter(o.label for o in present).items():  # count-aware label filter
            labels = [names[i] for i in self.resolve(label)]
            have = np.array([sum(c[l] for l in labels) for c in self.seg_counts])
            allowed &= have >= k
        return allowed

    def _unary(self, q: _Query, w: dict):
        """Per (object, row) component matrices (k, N); NaN where a component doesn't apply."""
        d2 = ((self.start[None] - q.start[:, None]) ** 2).sum(-1)
        pos = np.exp(-d2 / (2 * POS_SIGMA**2))
        a, b = self.area[None], q.area[:, None]
        size = np.sqrt(np.minimum(a, b) / np.maximum(np.maximum(a, b), _EPS))

        key = np.full_like(pos, np.nan)
        key_ok = q.has_end[:, None] & self.moving[None]
        if key_ok.any():
            e2 = ((self.end[None] - q.end[:, None]) ** 2).sum(-1)
            key[key_ok] = np.exp(-e2 / (2 * POS_SIGMA**2))[key_ok]

        motion = np.full_like(pos, np.nan)
        mot_ok = q.has_motion[:, None] & self.moving[None]
        if mot_ok.any():
            tn = np.linalg.norm(self.disp, axis=1)[None]
            sn = np.linalg.norm(q.disp, axis=1)[:, None]
            cos = (q.disp @ self.disp.T) / np.maximum(tn * sn, _EPS)
            t_still, s_still = tn < STILL, sn < STILL
            direction = np.where(t_still & s_still, 1.0, np.where(t_still | s_still, 0.25, (cos + 1) / 2))
            dist = np.linalg.norm(self.path[None] - q.path[:, None], axis=-1).mean(-1)
            shape = np.exp(-(dist**2) / (2 * SHAPE_SIGMA**2))
            motion[mot_ok] = (0.5 * direction + 0.5 * shape)[mot_ok]

        num = w["position"] * pos + w["size"] * size
        den = np.full_like(pos, w["position"] + w["size"])
        for comp, name in ((motion, "motion"), (key, "keyframe")):
            has = ~np.isnan(comp)
            num = num + np.where(has, w[name] * np.nan_to_num(comp), 0.0)
            den = den + np.where(has, w[name], 0.0)
        unary = np.where(den > 0, num / np.maximum(den, _EPS), 1.0)

        valid = np.zeros(pos.shape, dtype=bool)
        for i, o in enumerate(q.objects):
            valid[i] = self.assignable & np.isin(self.label, self.resolve(o.label))
        cost = np.where(valid, 1.0 - unary, _INVALID)
        return cost, valid, {"position": pos, "size": size, "motion": motion, "keyframe": key}

    def _assign(self, cost: np.ndarray, valid: np.ndarray, widx: np.ndarray, k: int):
        """Hungarian assignment per window -> (kept window indices, rows (W', k))."""
        rows = np.zeros((len(widx), k), dtype=int)
        ok = np.ones(len(widx), dtype=bool)
        if k == 0:
            return widx, rows
        for n, wi in enumerate(widx):
            r0, r1 = self.win_r0[wi], self.win_r1[wi]
            sub = cost[:, r0:r1]
            if k == 1:
                rows[n, 0] = r0 + int(np.argmin(sub[0]))
                continue
            _, cols = linear_sum_assignment(sub)  # row indices come back as 0..k-1 (k <= window size)
            if sub[np.arange(k), cols].max() >= _INVALID:
                ok[n] = False
            rows[n] = r0 + cols
        return widx[ok], rows[ok]

    @staticmethod
    def _nanmean(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        has = ~np.isnan(x)
        cnt = has.sum(1)
        return np.where(cnt > 0, np.nan_to_num(x).sum(1) / np.maximum(cnt, 1), 0.0), cnt > 0

    def _relations(self, q: _Query, rows: np.ndarray) -> np.ndarray:
        """Pairwise left/right, above/below, near/far agreement, averaged over pairs -> (W',)."""
        total = np.zeros(len(rows))
        for i, j in q.pairs:
            ds = q.start[j] - q.start[i]
            dt = self.start[rows[:, j]] - self.start[rows[:, i]]
            terms = [np.maximum(0.0, 1.0 - np.abs(np.hypot(*ds) - np.hypot(dt[:, 0], dt[:, 1])) / NEAR_FAR_SCALE)]
            for axis in (0, 1):
                if abs(ds[axis]) > REL_MARGIN:
                    scale = min(abs(ds[axis]), REL_SCALE)
                    terms.append(np.clip(0.5 + 0.5 * np.sign(ds[axis]) * dt[:, axis] / scale, 0.0, 1.0))
            total += sum(terms) / len(terms)
        return total / len(q.pairs)

    def _converging(self, q: _Query, rows: np.ndarray) -> np.ndarray:
        """Mean vote (+1 agree / -1 disagree / 0 still) on pairs drawn converging or diverging -> (W',)."""
        votes, counts = np.zeros(len(rows)), np.zeros(len(rows))
        for i, j in q.pairs:
            if not (q.has_motion[i] and q.has_motion[j]):
                continue
            d_s = np.hypot(*(q.final[j] - q.final[i])) - np.hypot(*(q.start[j] - q.start[i]))
            if abs(d_s) <= CONVERGE_MARGIN:
                continue
            ri, rj = rows[:, i], rows[:, j]
            d_e, d_0 = self.end[rj] - self.end[ri], self.start[rj] - self.start[ri]
            d_t = np.hypot(d_e[:, 0], d_e[:, 1]) - np.hypot(d_0[:, 0], d_0[:, 1])
            judged = self.moving[ri] & self.moving[rj]
            vote = np.where(np.abs(d_t) <= STILL, 0.0, np.where(np.sign(d_t) == np.sign(d_s), 1.0, -1.0))
            votes += np.where(judged, vote, 0.0)
            counts += judged
        return np.where(counts > 0, votes / np.maximum(counts, 1), 0.0)

    def _absent_violations(self, absent: list[SketchObject]) -> np.ndarray:
        """(N,) bool: row is a track of an absent label that touches its absent box during the window."""
        viol = np.zeros(len(self.label), dtype=bool)
        if not len(self.boxes):
            return viol
        B = self.boxes
        cx, cy = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2
        area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        for o in absent:
            bx = _xyxy(o.start_box)
            inside = (cx >= bx[0]) & (cx <= bx[2]) & (cy >= bx[1]) & (cy <= bx[3])
            iw = np.clip(np.minimum(B[:, 2], bx[2]) - np.maximum(B[:, 0], bx[0]), 0, None)
            ih = np.clip(np.minimum(B[:, 3], bx[3]) - np.maximum(B[:, 1], bx[1]), 0, None)
            inter = iw * ih
            iou = inter / np.maximum(area_b + (bx[2] - bx[0]) * (bx[3] - bx[1]) - inter, _EPS)
            hit_pt = inside | (iou > ABSENT_IOU)
            hit_row = np.logical_or.reduceat(hit_pt, self.box_ptr[:-1])
            viol |= hit_row & np.isin(self.label, self.resolve(o.label))
        return viol

    # ---------- search ----------

    def search(
        self,
        sketch: Sketch,
        weights: dict[str, float] | None = None,
        segment_ids: list[str] | None = None,
        camera_ids: list[str] | None = None,
        top_n: int | None = None,
        adaptive: bool | None = None,
    ) -> list[dict]:
        w = effective_weights(sketch, weights, adaptive)
        present = [o for o in sketch.objects if not o.absent]
        absent = [o for o in sketch.objects if o.absent]
        k = len(present)
        allowed = self._allowed(present, segment_ids, camera_ids if camera_ids is not None else sketch.camera_ids)
        cand = allowed[self.win_seg] & ((self.win_r1 - self.win_r0) >= k)
        if not cand.any():
            return []
        q = _Query(present)
        cost, valid, comps = self._unary(q, w)
        if k:
            cand &= (self._window_sum(valid.astype(float)) >= 1).all(0)  # every object has a candidate track
        widx, rows = self._assign(cost, valid, np.flatnonzero(cand), k)
        if not len(widx):
            return []

        nw = len(widx)
        vals: dict[str, np.ndarray] = {}
        avail: dict[str, np.ndarray] = {}
        if k:
            obj = np.arange(k)[None]
            vals["position"] = comps["position"][obj, rows].mean(1)
            vals["size"] = comps["size"][obj, rows].mean(1)
            avail["position"] = avail["size"] = np.ones(nw, dtype=bool)
            m, avail["motion"] = self._nanmean(comps["motion"][obj, rows])
            if k >= 2:
                m = m + CONVERGE_BONUS * self._converging(q, rows)
            vals["motion"] = np.clip(m, 0.0, 1.0)
            vals["keyframe"], avail["keyframe"] = self._nanmean(comps["keyframe"][obj, rows])
            if k >= 2:
                vals["relations"] = self._relations(q, rows)
                avail["relations"] = np.ones(nw, dtype=bool)
        num, den = np.zeros(nw), np.zeros(nw)
        for name in vals:
            num += np.where(avail[name], w[name] * vals[name], 0.0)
            den += np.where(avail[name], w[name], 0.0)
        score = np.where(den > 0, num / np.maximum(den, _EPS), 1.0)

        absence_ok = None
        if absent:
            viol = self._absent_violations(absent)
            in_window = self._window_sum(viol.astype(float))[widx]
            assigned = viol[rows].sum(1) if k else 0
            absence_ok = (in_window - assigned) < 0.5
            score = np.where(absence_ok, score, score * ABSENT_PENALTY)

        # Best window per segment, then rank segments.
        seg = self.win_seg[widx]
        order = np.lexsort((-score, seg))
        _, first = np.unique(seg[order], return_index=True)
        best = order[first]
        best = best[np.lexsort((self.seg_ids[seg[best]], -score[best]))]
        if top_n:
            best = best[:top_n]

        results = []
        for b in best:
            wi = widx[b]
            results.append({
                "segment_id": self.segments[seg[b]].segment_id,
                "score": round(float(score[b]), 4),
                "components": {n: round(float(vals[n][b]), 4) for n in COMPONENTS if n in vals and avail[n][b]},
                "assignment": {present[i].id: self.track_id[rows[b, i]] for i in range(k)},
                "window": [float(self.win_t0[wi]), float(self.win_t1[wi])],
                "absence_ok": None if absence_ok is None else bool(absence_ok[b]),
            })
        return results
