"""Minimal desktop interface for the bird tracker.

    python gui.py                      # then Open... a video, or hit Camera
    python gui.py data/gulls.webm      # start with a clip loaded
    python gui.py 0                    # start on the laptop camera

Everything heavy runs on a worker thread; the window only draws frames.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from tracker import BirdDetector, BirdTracker, DiffDetector, Track, draw, safe_fps

BG = "#151719"
FG = "#e8e8e8"
MUTED = "#8a9096"
ACCENT = "#4a90d9"
VIEW_W = 880
APP_DIR = os.path.dirname(os.path.abspath(
    sys.executable if getattr(sys, "frozen", False) else __file__))


class Worker(threading.Thread):
    """Reads the video, tracks birds, hands annotated frames to the UI."""

    def __init__(self, path, detector_name, min_area, save_to, out_queue,
                 live=False):
        super().__init__(daemon=True)
        self.path = path
        self.live = live  # camera: no frame count, no end of file
        self.detector_name = detector_name
        self.min_area = min_area
        self.save_to = save_to
        self.q = out_queue
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self.error = None

    def run(self):
        if self.live:
            # CAP_DSHOW opens far quicker than MSMF on Windows laptops
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.path, backend)
        else:
            cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            what = "camera" if self.live else os.path.basename(str(self.path))
            self.error = f"Could not open {what}"
            self.q.put(("done", None))
            return

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or VIEW_W
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
        total = 0 if self.live else max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        scale = min(1.0, VIEW_W / float(src_w))
        w, h = int(src_w * scale), int(src_h * scale)

        if self.detector_name == "diff":
            detector = DiffDetector(min_area=self.min_area)
        else:
            detector = BirdDetector(min_area=self.min_area)
        Track._next_id = 1
        tracker = BirdTracker()

        writer = None
        if self.save_to:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_to)), exist_ok=True)
            writer = cv2.VideoWriter(self.save_to, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            if not writer.isOpened():
                writer = None
                self.error = f"Could not write {os.path.basename(self.save_to)}"

        idx = 0
        misses = 0
        t0 = time.time()
        while not self.stopped.is_set():
            while self.paused.is_set() and not self.stopped.is_set():
                time.sleep(0.05)
            ok, frame = cap.read()
            if not ok:
                if self.live and misses < 30:  # camera can drop a frame
                    misses += 1
                    time.sleep(0.03)
                    continue
                break
            misses = 0
            idx += 1
            if scale != 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            boxes, _ = detector.detect(frame)
            live = tracker.update(boxes, idx)
            vis = draw(frame, live, idx, tracker.total_confirmed, hud=False)
            if writer is not None:
                writer.write(vis)
            stats = {
                "frame": idx, "total": total, "birds": len(live),
                "tracks": tracker.total_confirmed,
                "fps": idx / max(1e-6, time.time() - t0),
            }
            try:
                self.q.put(("frame", (vis, stats)), timeout=1.0)
            except queue.Full:
                pass

        cap.release()
        if writer is not None:
            writer.release()
        self.q.put(("done", None))


class App(tk.Tk):
    def __init__(self, initial=None):
        super().__init__()
        self.title("Bird tracker")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.live = isinstance(initial, int)
        self.camera_index = initial if self.live else 0
        self.path = initial
        self.worker = None
        self.q = queue.Queue(maxsize=2)
        self.photo = None

        self._build()
        self._tick()

    # ---------------------------------------------------------------- layout
    def _build(self):
        pad = {"padx": 6, "pady": 8}
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=14, pady=(12, 4))

        self._button(bar, "Open…", self.open_file).pack(side="left", **pad)
        self._button(bar, "Camera", self.use_camera).pack(side="left", pady=8)

        self.detector = tk.StringVar(value="mog2")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("T.TMenubutton", background="#23262a", foreground=FG,
                        borderwidth=0, focuscolor=BG)
        menu = ttk.OptionMenu(bar, self.detector, "mog2", "mog2", "diff",
                              style="T.TMenubutton")
        menu.pack(side="left", **pad)
        tk.Label(bar, text="detector", bg=BG, fg=MUTED).pack(side="left")

        self.min_area = tk.IntVar(value=12)
        tk.Label(bar, text="min blob", bg=BG, fg=MUTED).pack(side="left", padx=(18, 4))
        tk.Scale(bar, from_=4, to=120, orient="horizontal", variable=self.min_area,
                 length=120, bg=BG, fg=MUTED, troughcolor="#23262a", highlightthickness=0,
                 bd=0, sliderrelief="flat", activebackground=ACCENT,
                 showvalue=True, font=("TkDefaultFont", 8)).pack(side="left")

        self.save = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="save mp4", variable=self.save, bg=BG, fg=MUTED,
                       selectcolor="#23262a", activebackground=BG, activeforeground=FG,
                       highlightthickness=0, bd=0).pack(side="left", padx=6)

        self.play_btn = self._button(bar, "Play", self.toggle)
        self.play_btn.pack(side="right", **pad)
        self._button(bar, "Stop", self.stop).pack(side="right", **pad)

        self.view = tk.Label(self, bg="#0d0f10", width=VIEW_W, height=int(VIEW_W * 9 / 16),
                             text="Open a video to start", fg=MUTED)
        self.view.pack(padx=14, pady=4)

        self.progress = tk.Canvas(self, height=3, bg="#23262a", highlightthickness=0,
                                  width=VIEW_W)
        self.progress.pack(padx=14, fill="x")
        self.bar_fill = self.progress.create_rectangle(0, 0, 0, 3, fill=ACCENT, width=0)

        self.status = tk.Label(self, text="no video loaded", bg=BG, fg=MUTED,
                               anchor="w", font=("TkDefaultFont", 9))
        self.status.pack(fill="x", padx=16, pady=(6, 12))

        if self.live:
            self.title(f"Bird tracker — camera {self.camera_index}")
            self.status.config(text="camera — press Play")
        elif self.path is not None:
            self._set_path(self.path)

    def _button(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, bg="#23262a", fg=FG,
                         activebackground=ACCENT, activeforeground="#ffffff",
                         relief="flat", bd=0, padx=14, pady=5,
                         highlightthickness=0, cursor="hand2")

    # --------------------------------------------------------------- actions
    def open_file(self):
        p = filedialog.askopenfilename(
            title="Pick a video",
            initialdir=os.path.join(APP_DIR, "data"),
            filetypes=[("Video", "*.mp4 *.webm *.ogv *.avi *.mov *.mkv"), ("All", "*.*")])
        if p:
            self.stop()
            self._set_path(p)

    def use_camera(self):
        """Track live from the laptop camera (index 0)."""
        self.stop()
        self.path = self.camera_index
        self.live = True
        self.title(f"Bird tracker — camera {self.camera_index}")
        self.status.config(text="camera — press Play")
        self.start()

    def _set_path(self, p):
        self.path = p
        self.live = False
        self.title(f"Bird tracker — {os.path.basename(p)}")
        self.status.config(text=f"{os.path.basename(p)} — press Play")

    def toggle(self):
        if self.worker is None or not self.worker.is_alive():
            self.start()
        elif self.worker.paused.is_set():
            self.worker.paused.clear()
            self.play_btn.config(text="Pause")
        else:
            self.worker.paused.set()
            self.play_btn.config(text="Play")

    def start(self):
        if self.path is None:
            self.open_file()
            if self.path is None:
                return
        save_to = None
        if self.save.get():
            stem = ("camera" if self.live
                    else os.path.splitext(os.path.basename(self.path))[0])
            save_to = os.path.join(APP_DIR, "out", f"{stem}_tracked.mp4")
        self.q = queue.Queue(maxsize=2)
        self.worker = Worker(self.path, self.detector.get(), self.min_area.get(),
                             save_to, self.q, live=self.live)
        self.worker.start()
        self.play_btn.config(text="Pause")
        self.status.config(text="starting…")

    def stop(self):
        if self.worker is not None and self.worker.is_alive():
            self.worker.stopped.set()
            self.worker.paused.clear()
        self.play_btn.config(text="Play")

    # ------------------------------------------------------------- ui update
    def _tick(self):
        item = None
        while True:  # drain, keep only the newest frame
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
        if item is not None:
            kind, payload = item
            if kind == "frame":
                frame, s = payload
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.view.config(image=self.photo, text="", width=self.photo.width(),
                                 height=self.photo.height())
                self.status.config(
                    text=f"frame {s['frame']}" + (f"/{s['total']}" if s['total'] > 0 else "")
                         + f"   ·   birds in view {s['birds']}"
                         + f"   ·   tracks seen {s['tracks']}"
                         + f"   ·   {s['fps']:.0f} fps")
                if s["total"] > 0:
                    self.progress.coords(self.bar_fill, 0, 0,
                                         VIEW_W * s["frame"] / s["total"], 3)
            else:
                self.play_btn.config(text="Play")
                err = self.worker.error if self.worker else None
                self.status.config(text=err or "finished")
                self.progress.coords(self.bar_fill, 0, 0, 0, 3)
        self.after(10, self._tick)


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else None
    if source is not None and source.isdigit():  # camera index
        source = int(source)
    App(source).mainloop()
