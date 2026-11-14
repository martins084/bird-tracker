"""Download the two freely licensed sample clips into data/.

Both are from Wikimedia Commons, CC BY-SA 4.0:
  flying_birds.webm - "Flying Birds 1 2022-12-05" by FASTILY (fixed camera, flock)
  gulls.webm        - "Mowenflug (Dunenkette)" by Rhetos (panning camera, one gull)
"""

import os
import urllib.request

VIDEOS = {
    "flying_birds.webm":
        "https://upload.wikimedia.org/wikipedia/commons/f/f9/Flying_Birds_1_2022-12-05.webm",
    "gulls.webm":
        "https://upload.wikimedia.org/wikipedia/commons/c/cd/"
        "M%C3%B6wenflug_%28D%C3%BCnenkette%29.webm",
}

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(here, exist_ok=True)

for name, url in VIDEOS.items():
    dest = os.path.join(here, name)
    if os.path.exists(dest):
        print(f"{name}: already here")
        continue
    print(f"{name}: downloading ...")
    req = urllib.request.Request(url, headers={"User-Agent": "bird-tracker-demo/1.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    print(f"{name}: {os.path.getsize(dest) / 1e6:.1f} MB")
