"""Simple bird-in-flight tracker.

Detects moving birds against the sky with background subtraction, then links
detections across frames into tracks with a greedy nearest-neighbour matcher.

Usage:
    python tracker.py data/flying_birds.webm --show
    python tracker.py data/geese.webm -o out/geese_tracked.mp4 --csv out/geese.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------- detection


class DiffDetector:
    """Three-frame differencing with camera-motion compensation.

    Each of the two previous frames is warped into the current frame's
    coordinates (affine fit over sparse optical flow on background texture),
    so a handheld or panning camera no longer lights up the whole mask.
    Taking the AND of the two differences removes the "ghost" blob a bird
    leaves at its previous position.
    """

    def __init__(self, min_area=12, max_area=20000, threshold=18):
        self.min_area = min_area
        self.max_area = max_area
        self.threshold = threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.prev = []  # up to two previous greyscale frames

    @staticmethod
    def _warp_to(src_gray, dst_gray):
        """Warp src_gray into dst_gray's coordinate frame."""
        pts0 = cv2.goodFeaturesToTrack(src_gray, maxCorners=400, qualityLevel=0.01,
                                       minDistance=12, blockSize=7)
        if pts0 is None or len(pts0) < 12:
            return src_gray
        pts1, status, _ = cv2.calcOpticalFlowPyrLK(src_gray, dst_gray, pts0, None,
                                                   winSize=(21, 21), maxLevel=3)
        if pts1 is None:
            return src_gray
        good0 = pts0[status.ravel() == 1]
        good1 = pts1[status.ravel() == 1]
        if len(good0) < 12:
            return src_gray
        m, _ = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC,
                                           ransacReprojThreshold=3)
        if m is None:
            return src_gray
        h, w = dst_gray.shape[:2]
        return cv2.warpAffine(src_gray, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def detect(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if len(self.prev) < 2:
            self.prev.append(gray)
            return [], np.zeros_like(gray)

        masks = []
        for p in self.prev:
            d = cv2.absdiff(self._warp_to(p, gray), gray)
            _, m = cv2.threshold(d, self.threshold, 255, cv2.THRESH_BINARY)
            masks.append(m)
        mask = cv2.bitwise_and(masks[0], masks[1])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        self.prev = [self.prev[-1], gray]
        return _boxes_from_mask(mask, self.min_area, self.max_area), mask


def _boxes_from_mask(mask, min_area, max_area):
    # Camera shake/exposure changes should not create hundreds of candidates.
    if cv2.countNonZero(mask) > mask.size * 0.15:
        return []
    # Join nearby fragments only for rejection, keeping original small-bird
    # boxes separate. A large foreground object often has holes in its mask.
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    regions, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    allowed = mask.copy()
    area_limit = min(max_area, mask.size * 0.025)
    for region in regions:
        if cv2.contourArea(region) > area_limit:
            cv2.drawContours(allowed, [region], -1, 0, cv2.FILLED)
    contours, _ = cv2.findContours(allowed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        if aspect > 6.0 or aspect < 1 / 6.0:  # drop long thin streaks
            continue
        boxes.append((x, y, w, h))
    return boxes


class BirdDetector:
    """Finds small moving blobs (birds) in each frame. Best on a fixed camera."""

    def __init__(self, min_area=12, max_area=20000, history=200, var_threshold=24,
                 warmup_frames=20):
        self.min_area = min_area
        self.max_area = max_area
        self.warmup_frames = warmup_frames
        self.frames = 0
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame):
        """Return (boxes, mask); boxes are (x, y, w, h) in frame pixels."""
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = self.bg.apply(blur)
        self.frames += 1
        if self.frames <= self.warmup_frames:
            return [], np.zeros_like(mask)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        return _boxes_from_mask(mask, self.min_area, self.max_area), mask


# ----------------------------------------------------------------- tracking


class Track:
    _next_id = 1

    def __init__(self, box, frame_idx):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.centroid = _centre(box)
        self.velocity = (0.0, 0.0)
        self.trail = deque([self.centroid], maxlen=64)
        self.hits = 1
        self.confirmed = False
        self.missing = 0
        self.first_frame = frame_idx
        self.last_frame = frame_idx

    @property
    def predicted(self):
        cx, cy = self.centroid
        vx, vy = self.velocity
        return cx + vx, cy + vy

    def update(self, box, frame_idx):
        new_c = _centre(box)
        old_c = self.centroid
        # smooth the velocity so a noisy frame doesn't throw the prediction off
        vx = 0.6 * (new_c[0] - old_c[0]) + 0.4 * self.velocity[0]
        vy = 0.6 * (new_c[1] - old_c[1]) + 0.4 * self.velocity[1]
        self.velocity = (vx, vy)
        self.box = box
        self.centroid = new_c
        self.trail.append(new_c)
        self.hits += 1
        self.missing = 0
        self.last_frame = frame_idx

    def coast(self):
        """No detection this frame: carry the track forward on its velocity."""
        cx, cy = self.predicted
        self.centroid = (cx, cy)
        x, y, w, h = self.box
        self.box = (int(cx - w / 2), int(cy - h / 2), w, h)
        self.trail.append(self.centroid)
        self.missing += 1

    @property
    def speed(self):
        vx, vy = self.velocity
        return math.hypot(vx, vy)


def safe_fps(value, default=25.0):
    """Cameras often report 0 or -1 fps; VideoWriter silently fails on those."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not (1.0 < value < 240.0):  # also catches nan
        return default
    return value


def _centre(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


class BirdTracker:
    """Greedy nearest-neighbour association on predicted positions."""

    def __init__(self, max_distance=80, max_missing=12, min_hits=5):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self.total_confirmed = 0

    def update(self, boxes, frame_idx):
        unmatched = set(range(len(boxes)))
        pairs = []
        for ti, track in enumerate(self.tracks):
            px, py = track.predicted
            for bi, box in enumerate(boxes):
                old_area = track.box[2] * track.box[3]
                new_area = box[2] * box[3]
                if min(old_area, new_area) / max(old_area, new_area, 1) < 0.25:
                    continue
                bx, by = _centre(box)
                d = math.hypot(bx - px, by - py)
                if d <= self.max_distance:
                    pairs.append((d, ti, bi))
        pairs.sort()

        used_tracks, used_boxes = set(), set()
        for d, ti, bi in pairs:
            if ti in used_tracks or bi in used_boxes:
                continue
            self.tracks[ti].update(boxes[bi], frame_idx)
            used_tracks.add(ti)
            used_boxes.add(bi)
            unmatched.discard(bi)

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.coast()

        for bi in sorted(unmatched):
            self.tracks.append(Track(boxes[bi], frame_idx))

        # Tentative tracks must be seen on consecutive frames. Do not let
        # unrelated flickers accumulate enough hits to become a bird.
        self.tracks = [t for t in self.tracks
                       if t.missing <= (self.max_missing if t.confirmed else 0)]
        for track in self.tracks:
            if not track.confirmed and track.hits >= self.min_hits:
                track.confirmed = True
                self.total_confirmed += 1
        return self.confirmed

    @property
    def confirmed(self):
        return [t for t in self.tracks if t.confirmed and t.missing == 0]


# ------------------------------------------------------------------ drawing

PALETTE = [
    (66, 135, 245), (46, 204, 113), (241, 196, 15), (231, 76, 60),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (52, 152, 219),
]


def draw(frame, tracks, frame_idx, total_seen, hud=True):
    for t in tracks:
        colour = PALETTE[t.id % len(PALETTE)]
        x, y, w, h = t.box
        pad = 6
        cv2.rectangle(frame, (x - pad, y - pad), (x + w + pad, y + h + pad), colour, 2)
        cv2.putText(frame, f"#{t.id}  {t.speed:.0f}px/f", (x - pad, y - pad - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        pts = np.array(t.trail, dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, colour, 2, cv2.LINE_AA)

    if not hud:  # the GUI shows these numbers in its own status bar
        return frame

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.putText(frame, f"frame {frame_idx}   birds in view: {len(tracks)}   tracks seen: {total_seen}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------- main


def run(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    scale = args.width / float(src_w) if args.width and src_w > args.width else 1.0
    out_w, out_h = int(src_w * scale), int(src_h * scale)
    print(f"input : {args.video}  {src_w}x{src_h} @ {fps:.0f}fps")
    print(f"working at {out_w}x{out_h}")

    if args.detector == "diff":
        detector = DiffDetector(min_area=args.min_area, max_area=args.max_area,
                                threshold=args.diff_threshold)
    else:
        detector = BirdDetector(min_area=args.min_area, max_area=args.max_area)
    print(f"detector: {args.detector}")
    tracker = BirdTracker(max_distance=args.max_distance)

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (out_w, out_h))
        if not writer.isOpened():
            raise SystemExit(f"Could not open {args.output} for writing")

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        boxes, mask = detector.detect(frame)
        live = tracker.update(boxes, frame_idx)

        for t in live:
            x, y, w, h = t.box
            rows.append([frame_idx, round(frame_idx / fps, 3), t.id,
                         x, y, w, h, round(t.centroid[0], 1), round(t.centroid[1], 1),
                         round(t.speed, 2)])

        vis = draw(frame.copy(), live, frame_idx, tracker.total_confirmed)
        if writer is not None:
            writer.write(vis)
        if args.show:
            cv2.imshow("bird tracker", vis)
            if args.show_mask:
                cv2.imshow("motion mask", mask)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF in (27, ord("q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_s", "track_id", "x", "y", "w", "h",
                        "cx", "cy", "speed_px_per_frame"])
            w.writerows(rows)

    ids = {r[2] for r in rows}
    print(f"frames processed : {frame_idx}")
    print(f"tracks confirmed : {len(ids)}")
    print(f"detections logged: {len(rows)}")
    if args.output:
        print(f"video written    : {args.output}")
    if args.csv:
        print(f"csv written      : {args.csv}")


def main():
    p = argparse.ArgumentParser(description="Track birds in flight in a video.")
    p.add_argument("video", help="path to a video file (or a camera index like 0)")
    p.add_argument("-o", "--output", help="write an annotated .mp4 here")
    p.add_argument("--csv", help="write per-frame track positions here")
    p.add_argument("--show", action="store_true", help="show a live window")
    p.add_argument("--show-mask", action="store_true", help="also show the motion mask")
    p.add_argument("--width", type=int, default=1280, help="downscale to this width (0 = native)")
    p.add_argument("--detector", choices=("mog2", "diff"), default="mog2",
                   help="mog2 = background model (fixed camera); "
                        "diff = motion-compensated frame differencing (handheld/panning)")
    p.add_argument("--diff-threshold", type=int, default=18,
                   help="brightness change counted as motion, diff detector only")
    p.add_argument("--min-area", type=int, default=12, help="smallest blob to keep, px^2")
    p.add_argument("--max-area", type=int, default=20000, help="largest blob to keep, px^2")
    p.add_argument("--max-distance", type=float, default=80,
                   help="max px a bird may move between frames and stay the same track")
    args = p.parse_args()
    if args.video.isdigit():
        args.video = int(args.video)
    run(args)


if __name__ == "__main__":
    main()
