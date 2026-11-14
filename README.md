# Bird tracker

A small OpenCV tool that finds birds in flight in a video, follows each one
across frames, draws its box and flight trail, and writes the tracks to CSV.

No training data and no model weights: birds are found as small moving blobs
against the sky, which is what works best for distant birds that are only a few
pixels across.

- [`gui.py`](gui.py) — a small window: pick a video or the laptop camera, watch it track.
- [`tracker.py`](tracker.py) — the detection and tracking code, also a CLI.
- [`get_videos.py`](get_videos.py) — fetches two freely licensed sample clips.

## How it works

1. **Detect** moving pixels, in one of two ways:
   - `mog2` (default) — an adaptive background model (`createBackgroundSubtractorMOG2`).
     Best when the camera is on a tripod.
   - `diff` — three-frame differencing with camera-motion compensation. Each of the
     two previous frames is warped into the current frame with an affine fit over
     sparse optical flow, so a panning or handheld camera no longer lights up the
     whole mask. The AND of the two differences removes the ghost blob a bird would
     otherwise leave at its old position.
2. **Filter** blobs by area and aspect ratio. Ignore frames with more than 15%
   foreground motion and reject nearby fragments belonging to regions larger
   than 2.5% of the image (or `max_area`, whichever is smaller). This reduces
   camera-shake and large-object false detections, but can also hide dense flocks
   or close birds. The background model warms up for 20 frames before tracking.
3. **Track**: a greedy nearest-neighbour matcher links each detection to the
   *predicted* position of an existing track (last position + smoothed velocity).
   Unmatched tracks coast along their velocity for a few frames, which carries a
   bird through a brief occlusion; unmatched detections start a new track. A track
   is only drawn after five consecutive detections. Tentative tracks are dropped
   immediately when missed; confirmed tracks can survive brief occlusions.
   Matches reject extreme changes in blob size. The total counts only confirmed
   tracks, not every tentative motion candidate.

## Setup

```bash
pip install -r requirements.txt
```

```bash
python get_videos.py
```

## The interface

On this computer, double-click **BirdTracker.exe** in the project folder.
It opens the GUI without a console window using the installed Python 3.14.
Keep the executable beside `gui.py` and `tracker.py`; it is a local launcher,
not a standalone distribution for other computers.

```bash
python gui.py
```

A single window: **Open…** picks a video, **Camera** switches to the laptop
camera and starts straight away, the dropdown picks the detector, the slider
sets the minimum blob size, **save mp4** writes the annotated result to
`out/`, and **Play / Pause / Stop** control the run. The status line under the
frame shows the frame number, how many birds are in view, how many tracks have
been seen in total, and the processing rate.

Tracking runs on a worker thread, so the window stays responsive and always
shows the newest finished frame — on a slow clip it drops frames rather than
lagging behind. You can also start it on something directly:

```bash
python gui.py data/gulls.webm
```

```bash
python gui.py 0
```

### Live camera

**Camera** (or `python gui.py 0`) tracks the laptop camera in real time; on
Windows it opens through DirectShow, which starts much faster than the default
backend. Leave the detector on `mog2` — the camera is sitting still, so the
background model is the right choice, and it needs a second or two of video to
settle before it stops flagging things. Point it out of a window at the sky and
it will track what flies past; indoors it will happily track your hand, since
what it really detects is motion, not birds.

Large nearby motion is suppressed using overall foreground coverage and grouped
fragment size. This reduces the many boxes a person near the lens can otherwise
produce. Small isolated moving objects may still be tracked: these filters do
not classify birds. Use **diff** for a moving camera.

Another camera (an external USB one) is `python gui.py 1`, and so on.

## Run the demos from the command line

Fixed camera, a flock of ~100 birds — the background model handles this well:

```bash
python tracker.py data/flying_birds.webm --show
```

Panning camera, a single gull soaring over the dunes — needs motion compensation:

```bash
python tracker.py data/gulls.webm --detector diff --min-area 25 --show
```

Write an annotated video and a CSV of positions instead of watching it live:

```bash
python tracker.py data/gulls.webm --detector diff --min-area 25 -o out/gulls_tracked.mp4 --csv out/gulls.csv
```

Press `q` or `Esc` to quit the live window. `--show-mask` opens the motion mask
next to it, which is the quickest way to see why a clip is over- or
under-detecting.

## Options

| flag | default | what it does |
|---|---|---|
| `-o, --output` | – | write an annotated `.mp4` |
| `--csv` | – | write per-frame track positions |
| `--show` / `--show-mask` | off | live window / motion mask window |
| `--detector` | `mog2` | `mog2` for a fixed camera, `diff` for a moving one |
| `--diff-threshold` | `18` | brightness change counted as motion (`diff` only) |
| `--width` | `1280` | downscale to this width before processing (0 = native) |
| `--min-area` | `12` | smallest blob kept, px² — raise it if noise is tracked |
| `--max-area` | `20000` | largest blob kept, px² |
| `--max-distance` | `80` | how far a bird may move per frame and stay the same track |

A camera works here too: `python tracker.py 0 --show`.

## Tuning

- **Noise tracked as birds** → raise `--min-area`, or raise `--diff-threshold`.
- **Birds missed** → lower `--min-area`; check the mask with `--show-mask`.
- **One bird keeps getting a new ID** → raise `--max-distance` (fast birds, low fps).
- **Two birds crossing swap IDs** — expected from a nearest-neighbour matcher; a
  Kalman filter or an appearance descriptor would be the next step.

## CSV output

One row per confirmed track per frame:

```
frame, time_s, track_id, x, y, w, h, cx, cy, speed_px_per_frame
```

Enough to plot flight paths, count birds crossing a line, or measure speeds once
you know the scale.

## Sample videos

Both clips come from Wikimedia Commons and are **CC BY-SA 4.0**:

- [Flying Birds 1 2022-12-05](https://commons.wikimedia.org/wiki/File:Flying_Birds_1_2022-12-05.webm) by FASTILY
- [Möwenflug (Dünenkette)](https://commons.wikimedia.org/wiki/File:M%C3%B6wenflug_(D%C3%BCnenkette).webm) by Rhetos

## Limits

This tracks *motion*, not birds — anything small that moves against a plain
background is a candidate. On busy ground, in a dense murmuration, or with heavy
camera shake it will over-detect. For species-level work the next step is a
detector (YOLO fine-tuned on birds) in place of `BirdDetector`; the tracking and
drawing code takes boxes from anywhere.
