"""Precompute the two non-RGB input channels for the 5-channel encoder.

WHY
---
The advisor's suggestion: instead of handing DINOv2 a 3-channel RGB frame, widen
its patch embedding to 5 channels and supply physically meaningful extras
alongside the colour. Both extras are things the RGB channels do not express
directly:

  ch0  background-subtracted deformation.  Each session has a non-contact
       reference frame (the gel at rest). Subtracting it removes the gel's fixed
       appearance -- its wear, its tint, its lighting falloff -- and leaves only
       what this press did. As a standalone view this was worth +1.21 points.

  ch1  marker-size map.  The gel carries ~100 printed markers on a 58 px
       lattice. A marker's APPARENT SIZE grows as it moves toward the camera, so
       size change is a depth cue, and it is the one geometric quantity measured
       in this project that is not redundant with contact area (r = 0.39, versus
       r = 0.97 for lateral marker motion). Rendered as a dense map: 128 means
       unchanged, brighter means the marker grew (pressed closer), darker means
       it shrank.

Computed once and cached, because tracking ~100 markers per frame with
matchTemplate is far too slow to run inside a dataloader.

Output: one memmap of shape (N, 480, 640, 2) uint8 plus rows.csv for alignment,
which is ~3.8 GB -- cheaper than 6,115 separate PNGs to open.

Usage:
    python make_extra_channels.py --jobs 2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import CLASS_NAMES, select_rows
from marker_field import find_markers, track
from nested_cv import EXCLUDE

BG_REF_DIR = Path("C:/datasets/touch_and_go/bg_refs")
BG_GAIN = 2.0           # matches extract_features.py's background-subtracted view
H, W = 480, 640
DISC = 13               # radius of the blob painted per marker, ~ a quarter pitch


def _disc(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


DISC_MASK = _disc(DISC)


def load_gray(p):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return im if im.shape == (H, W) else cv2.resize(im, (W, H))


def marker_map(ref_gray, img_gray, pts):
    """Dense map of per-marker size ratio, painted at each marker's new position."""
    out = np.full((H, W), 128, dtype=np.uint8)
    if len(pts) == 0:
        return out
    u, sz = track(ref_gray, img_gray, pts)
    for (yf, xf), (dy, dx), s in zip(pts, u, sz):
        if not np.isfinite(s) or not np.isfinite(dy):
            continue
        y, x = int(round(yf + dy)), int(round(xf + dx))
        y0, y1 = max(0, y - DISC), min(H, y + DISC + 1)
        x0, x1 = max(0, x - DISC), min(W, x + DISC + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        m = DISC_MASK[DISC - (y - y0):DISC + (y1 - y), DISC - (x - x0):DISC + (x1 - x)]
        v = int(np.clip(128.0 + (s - 1.0) * 255.0, 0, 255))
        patch = out[y0:y1, x0:x1]
        patch[m] = np.maximum(patch[m], v) if v >= 128 else np.minimum(patch[m], v)
    return out


def one(task):
    """(index, frame path, video_id) -> (index, HxWx2 uint8) ; cached ref per call."""
    i, fpath, vid = task
    img = load_gray(fpath)
    ref = load_gray(BG_REF_DIR / f"{vid}.png")
    if img is None:
        return i, None
    two = np.zeros((H, W, 2), dtype=np.uint8)
    if ref is not None:
        d = np.abs(img.astype(np.float32) - ref.astype(np.float32)) * BG_GAIN
        two[:, :, 0] = np.clip(d, 0, 255).astype(np.uint8)
        two[:, :, 1] = marker_map(ref, img, find_markers(ref))
    else:
        two[:, :, 1] = 128
    return i, two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--like", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--out", type=Path, default=Path("../cache_extra"))
    ap.add_argument("--jobs", type=int, default=2,
                    help="Kept low by default: a fine-tune may be using the CPU "
                         "to decode frames for the GPU at the same time")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    sel = select_rows(df, "run")
    rows = pd.read_csv(args.like / "rows.csv")
    sel = sel[sel.run_id.isin(set(rows.run_id))]
    sel = sel[[Path(p).exists() for p in sel.gelsight_frame_path]]
    sel = sel[[CLASS_NAMES[int(c)] not in EXCLUDE for c in sel.label]]
    n = len(sel)
    print(f"{n:,} presses -> {args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    sel[["run_id", "label", "video_id", "frame_idx"]].to_csv(
        args.out / "rows.csv", index=False)
    mm = np.lib.format.open_memmap(args.out / "extra.npy", mode="w+",
                                   dtype=np.uint8, shape=(n, H, W, 2))

    tasks = [(k, p, v) for k, (p, v) in
             enumerate(zip(sel.gelsight_frame_path, sel.video_id))]
    done = 0
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for i, two in pool.imap_unordered(one, tasks, chunksize=16):
                if two is not None:
                    mm[i] = two
                done += 1
                if done % 250 == 0:
                    print(f"  {done:,}/{n:,}", flush=True)
    else:
        for t in tasks:
            i, two = one(t)
            if two is not None:
                mm[i] = two
            done += 1
            if done % 250 == 0:
                print(f"  {done:,}/{n:,}", flush=True)
    mm.flush()
    print(f"wrote {args.out/'extra.npy'}  "
          f"{mm.nbytes/1e9:.2f} GB  shape {mm.shape}")


if __name__ == "__main__":
    main()
