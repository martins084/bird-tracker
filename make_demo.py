"""Render an honest, slowed replay of recorded detector/tracker output."""
import json
import os
import subprocess

import cv2
import numpy as np

OUT = 'out/bird_tracking_start.mp4'
FPS = 25
rows = json.load(open('out/demo_analysis.json'))
cap = cv2.VideoCapture('data/gulls.webm')
cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
frames = []
for _ in range(75):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('Source clip ended early')
    frames.append(cv2.resize(frame, (880, 495)))
cap.release()
writer = cv2.VideoWriter('out/demo_intermediate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1280, 720))
assert writer.isOpened()
WHITE = (240, 242, 245)
MUTED = (170, 180, 190)
GREEN = (110, 230, 80)
YELLOW = (70, 210, 255)


def text(img, line, xy, size=.65, color=WHITE):
    cv2.putText(img, line, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def base(title, subtitle):
    img = np.full((720, 1280, 3), (25, 22, 19), np.uint8)
    text(img, 'BIRD TRACKER  /  STARTUP DEMO', (30, 35), .5, MUTED)
    text(img, title, (30, 78), 1)
    text(img, subtitle, (30, 110), .6, MUTED)
    text(img, 'Footage: Rhetos / Wikimedia Commons / CC BY-SA 4.0', (30, 696), .5, MUTED)
    return img


def emit(img, count):
    for _ in range(count):
        writer.write(img)


for i, frame in enumerate(frames):
    img = base('One gull, clear view', 'Original footage at normal speed. Next: see tracking begin, slowed down.')
    img[140:635, 30:910] = frame
    text(img, 'SOURCE PREVIEW', (940, 180), .6)
    text(img, '1x speed', (940, 220), .7, GREEN)
    text(img, 'No tracking overlay', (940, 270), .55)
    text(img, 'Source: 10-13 seconds', (940, 310), .5, MUTED)
    emit(img, 1)

for i in range(24):
    r = rows[i][0]
    visible = r['tracks']
    bird = [t for t in visible if t['box'][1] < 350]
    focus = max(bird, key=lambda t: t['hits']) if bird else None
    live = any(t['hits'] >= 5 for t in visible)
    title = 'Tracking confirmed' if live else 'Waiting for five consecutive detections'
    if i < 2:
        title = 'First, collect two reference frames'
    img = base(title, 'Actual diff detector output. Yellow = tentative; green = confirmed. Replay is slowed.')
    view = frames[i].copy()
    for t in visible:
        x, y, w, h = t['box']
        color = GREEN if t['hits'] >= 5 else YELLOW
        cv2.rectangle(view, (x-4, y-4), (x+w+4, y+h+4), color, 2)
    img[140:635, 30:910] = view
    text(img, 'BIRD DETAIL / 3x', (940, 165), .55)
    cx, cy = (620, 185) if focus is None else (focus['box'][0]+focus['box'][2]//2, focus['box'][1]+focus['box'][3]//2)
    cx, cy = min(830, max(50, cx)), min(460, max(35, cy))
    img[190:400, 940:1240] = cv2.resize(view[cy-35:cy+35, cx-50:cx+50], (300, 210))
    hits = min(focus['hits'], 5) if focus else 0
    text(img, 'Confirmation: %d / 5' % hits, (940, 445), .6, GREEN if live else YELLOW)
    for n in range(5):
        cv2.rectangle(img, (940+n*59, 470), (990+n*59, 486), GREEN if live else (YELLOW if n < hits else (65, 65, 65)), -1)
    text(img, 'Frame %02d  |  source %.2fs' % (i+1, 10+i/25), (940, 525), .48, MUTED)
    text(img, 'Tracking starts at frame 13.', (30, 663), .65, GREEN)
    text(img, 'Extra fragments and missed frames are shown as detected.', (520, 663), .5, MUTED)
    if i in [8, 11, 12, 20]:
        cv2.imwrite('out/demo_check_%02d.jpg' % i, img)
    emit(img, 60 if i == 12 else (12 if i < 13 else 5))

img = base('Try it in BirdTracker.exe', 'The demo labels and zoom are explanatory overlays, not controls in the app.')
for y, line in [(210, '1. Double-click BirdTracker.exe.'), (275, '2. Open data/gulls.webm.'), (340, '3. Choose diff and set min blob to 25.'), (405, '4. Press Play. The source title card lasts about 8 seconds.'), (470, '5. A track appears after five consecutive detections.')]:
    text(img, line, (45, y), .78)
text(img, 'Motion tracking can still split one bird into several boxes or briefly lose it.', (45, 570), .65, YELLOW)
text(img, 'This video shows real output; it does not claim perfect bird recognition.', (45, 610), .65, MUTED)
emit(img, 150)
img = base('Source and credits', 'Adaptation: excerpt, resized video, tracking annotations, slowed replay; original audio omitted.')
for y, line in [(220, 'Original: Mowenflug (Dunenkette), by Rhetos'), (280, 'Wikimedia Commons | 2024'), (350, 'Original and this video adaptation: CC BY-SA 4.0'), (410, 'https://creativecommons.org/licenses/by-sa/4.0/'), (480, 'Source: commons.wikimedia.org'), (520, 'File: Mowenflug (Dunenkette).webm (German umlauts in title)')]:
    text(img, line, (45, y), .7)
emit(img, 100)
writer.release()
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', 'out/demo_intermediate.mp4', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], check=True)
check = cv2.VideoCapture(OUT)
count = 0
while check.read()[0]:
    count += 1
check.release()
print(OUT, 'validated frames:', count, 'duration:', count/FPS, 'bytes:', os.path.getsize(OUT))
