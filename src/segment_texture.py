"""Step 1 of the morphology track: turn a gel imprint into separated contact regions.

PIPELINE
--------
    frame - session reference        isolate deformation (markers/lighting cancel)
    |residual| > threshold           binary mask: contact vs no contact
    opening(disk 2)                  drop speckle noise
    closing(disk 2)                  fill pinholes inside a contact patch
    distance transform               distance from each contact pixel to background
    watershed on -distance           SPLIT touching patches that thresholding merged

Watershed is the load-bearing step. Plain thresholding merges neighbouring grains
into one blob wherever they touch, which destroys exactly the size distribution we
are trying to measure -- the earlier blob_stats.py run had that problem and
reported near-identical distributions for all eighteen materials.

PER-REGION MEASURES
    area     pixels in the region
    volume   sum of the residual over the region -- how deep as well as how wide.
             A proxy for indentation depth, since the residual grows with gel
             deformation. NOT calibrated depth: GelSight RGB encodes surface SLOPE
             under coloured lights, so this is monotone-ish in depth rather than
             metric. Named "volume" for the intuition, with that caveat attached.
    centroid used for nearest-neighbour spacing (how close the dots are)

Usage:
    python segment_texture.py --per-class 120 --out ../results/segments.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import disk, binary_opening, binary_closing
from skimage.segmentation import watershed

from common import CLASS_NAMES
from nested_cv import EXCLUDE

BG_REF_DIR = Path("C:/datasets/touch_and_go/bg_refs")
THRESH = 25.0        # fixed cut on the residual; see blob_stats.py docstring
MIN_AREA = 6         # px


def segment(img, ref, thresh=THRESH):
    """Return (labels, residual). Labels are watershed-separated contact regions."""
    d = np.abs(img - ref)
    mask = d > thresh
    mask = binary_opening(mask, disk(2))
    mask = binary_closing(mask, disk(2))
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), d

    dist = ndimage.distance_transform_edt(mask)
    # Seeds = local maxima of the distance transform, one per grain. A 7 px
    # footprint suppresses duplicate seeds inside a single grain.
    peaks = (dist == ndimage.maximum_filter(dist, size=7)) & (dist > 1.5)
    markers, _ = ndimage.label(peaks)
    labels = watershed(-dist, markers, mask=mask)
    return labels.astype(np.int32), d


def region_measures(labels, resid):
    """area, volume and centroid for every region."""
    n = int(labels.max())
    if n == 0:
        return np.array([]), np.array([]), np.zeros((0, 2))
    idx = np.arange(1, n + 1)
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.float64)
    vol = np.asarray(ndimage.sum(resid, labels, idx), dtype=np.float64)
    cen = np.array(ndimage.center_of_mass(resid, labels, idx))
    keep = area >= MIN_AREA
    return area[keep], vol[keep], cen[keep]


def nn_spacing(cen):
    """Median nearest-neighbour distance between region centroids."""
    if len(cen) < 2:
        return np.nan
    d, _ = cKDTree(cen).query(cen, k=2)
    return float(np.median(d[:, 1]))


def downsample(v, H):
    a, b = v.shape
    if a % H == 0 and b % H == 0:
        return v.reshape(H, a // H, H, b // H).mean(axis=(1, 3))
    return np.asarray(Image.fromarray(v).resize((H, H), Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=Path("../features_dinop9/rows.csv"))
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("../results/segments.npz"))
    ap.add_argument("--heat-size", type=int, default=64)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows)
    rows = rows[[CLASS_NAMES[int(c)] not in EXCLUDE for c in rows.label]]
    parts = [d.sample(min(len(d), args.per_class), random_state=42)
             for _, d in rows.groupby("label")]
    keep = pd.concat(parts).reset_index(drop=True)
    classes = sorted(keep.label.unique().tolist())
    print("%d presses, %d classes" % (len(keep), len(classes)))

    H = args.heat_size
    heat = {c: np.zeros((H, H)) for c in classes}
    heat_n = {c: 0 for c in classes}

    keys = ("label", "n", "spacing", "cover", "area_med", "area_p90",
            "vol_med", "vol_sum", "vol_p90", "area_iqr", "vol_iqr")
    P = {k: [] for k in keys}
    RL, RA, RV = [], [], []

    refs = {}
    for i, (_, r) in enumerate(keep.iterrows()):
        sess = Path(r.gelsight_frame_path).parent.parent.name
        if sess not in refs:
            f = BG_REF_DIR / ("%s.png" % sess)
            refs[sess] = (np.asarray(Image.open(f).convert("L"), dtype=np.float32)
                          if f.exists() else None)
        ref = refs[sess]
        if ref is None:
            continue
        img = np.asarray(Image.open(r.gelsight_frame_path).convert("L"),
                         dtype=np.float32)
        if img.shape != ref.shape:
            continue

        labels, resid = segment(img, ref)
        area, vol, cen = region_measures(labels, resid)
        if len(area) == 0:
            continue

        c = int(r.label)
        heat[c] += downsample(np.where(labels > 0, resid, 0.0).astype(np.float32), H)
        heat_n[c] += 1

        P["label"].append(c)
        P["n"].append(len(area))
        P["spacing"].append(nn_spacing(cen))
        P["cover"].append(float(area.sum()) / img.size)
        P["area_med"].append(float(np.median(area)))
        P["area_p90"].append(float(np.percentile(area, 90)))
        P["area_iqr"].append(float(np.percentile(area, 75) - np.percentile(area, 25)))
        P["vol_med"].append(float(np.median(vol)))
        P["vol_sum"].append(float(vol.sum()))
        P["vol_p90"].append(float(np.percentile(vol, 90)))
        P["vol_iqr"].append(float(np.percentile(vol, 75) - np.percentile(vol, 25)))
        RL.append(np.full(len(area), c))
        RA.append(area)
        RV.append(vol)
        if i % 250 == 0:
            print("\r  %d/%d" % (i, len(keep)), end="", flush=True)
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {("press_" + k): np.array(v) for k, v in P.items()}
    payload["region_label"] = np.concatenate(RL)
    payload["region_area"] = np.concatenate(RA)
    payload["region_volume"] = np.concatenate(RV)
    payload["heat"] = np.stack([heat[c] / max(heat_n[c], 1) for c in classes])
    payload["heat_classes"] = np.array(classes)
    np.savez(args.out, **payload)
    print("%d presses, %d regions -> %s"
          % (len(P["label"]), len(payload["region_area"]), args.out))

    lab = np.array(P["label"])
    print("\n%-20s%9s%10s%10s%9s"
          % ("class", "regions", "med area", "med vol", "spacing"))
    for c in classes:
        m = lab == c
        print("%-20s%9.0f%10.1f%10.0f%9.1f"
              % (CLASS_NAMES[c],
                 np.mean(np.array(P["n"])[m]),
                 np.median(np.array(P["area_med"])[m]),
                 np.median(np.array(P["vol_med"])[m]),
                 np.nanmedian(np.array(P["spacing"])[m])))


if __name__ == "__main__":
    main()
