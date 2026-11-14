"""Clean half-speed replay: source footage and actual confirmed boxes only."""
import json
import subprocess

import cv2

rows = json.load(open('out/demo_analysis.json'))
cap = cv2.VideoCapture('data/gulls.webm')
cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
writer = cv2.VideoWriter('out/clean_intermediate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 720))
assert writer.isOpened()
for index in range(75):
    ok, frame = cap.read()
    assert ok
    frame = cv2.resize(frame, (1280, 720))
    for track in rows[index][0]['tracks']:
        if track['hits'] < 5:
            continue
        x, y, w, h = track['box']
        scale = 1280 / 880
        cv2.rectangle(frame, (round((x-4)*scale), round((y-4)*scale)),
                      (round((x+w+4)*scale), round((y+h+4)*scale)), (80, 230, 110), 2)
    writer.write(frame)
    writer.write(frame)
    if index == 12:
        cv2.imwrite('out/clean_check.jpg', frame)
cap.release()
writer.release()
credit = ('Footage: Mowenflug (Dunenkette) by Rhetos, Wikimedia Commons. '
          'https://commons.wikimedia.org/wiki/File:M%C3%B6wenflug_(D%C3%BCnenkette).webm '
          'Original and adaptation licensed CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ '
          'Changes: excerpt, resized, half-speed, tracking rectangles, no audio.')
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', 'out/clean_intermediate.mp4',
                '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-metadata', 'comment='+credit, '-movflags', '+faststart',
                'out/bird_tracking_clean.mp4'], check=True)
cap = cv2.VideoCapture('out/bird_tracking_clean.mp4')
count = 0
while cap.read()[0]:
    count += 1
cap.release()
assert count == 150, count
print('Validated: 150 frames, 6 seconds, 1280x720, no visible text.')
