#!/usr/bin/env python3
"""Fetch a Sentinel-2 RGB overlay for the Ocean Beach Celeris domain (satellite backdrop).
Saved as overlay.jpg matching the config corner coordinates.

Usage:  python make_overlay.py [output_dir]
Default output_dir: ./data/celeris_oceanbeach

KNOWN ISSUE: the BBOX below does NOT currently match the BBOX in build_oceanbeach_celeris.py
([-122.524, 37.734, -122.508, 37.754]). Make them equal before relying on the overlay for
spatial alignment with the simulation grid. Left unchanged here rather than silently edited.

Sentinel-2 L2A is fetched from the public AWS Earth Search STAC (no account needed).
"""
import sys
from pathlib import Path
import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import cv2

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data" / "celeris_oceanbeach"
OUT.mkdir(parents=True, exist_ok=True)
BBOX = [-122.530, 37.730, -122.508, 37.758]     # must match build_oceanbeach_celeris.py

resp = requests.post("https://earth-search.aws.element84.com/v1/search", timeout=90, json={
    "collections": ["sentinel-2-l2a"], "bbox": BBOX,
    "query": {"eo:cloud_cover": {"lt": 15}},
    "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}], "limit": 8,
})
feats = resp.json()["features"]
print("scenes:", [(f['id'][:24], round(f['properties']['eo:cloud_cover'], 1)) for f in feats[:5]])
assets = feats[0]["assets"]

def band(key):
    with rasterio.open(assets[key]["href"]) as s:
        wb = transform_bounds("EPSG:4326", s.crs, *BBOX)
        win = from_bounds(*wb, s.transform)
        return s.read(1, window=win).astype(np.float32)

r, g, b = band("red"), band("green"), band("blue")
rgb = np.dstack([r, g, b])
rgb = np.clip((rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) - np.percentile(rgb, 2) + 1e-6), 0, 1) ** 0.8
bgr = (rgb[..., ::-1] * 255).astype(np.uint8)
cv2.imwrite(str(OUT / "overlay.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"wrote {OUT / 'overlay.jpg'}  {bgr.shape[1]}x{bgr.shape[0]}  (scene {feats[0]['id']})")
