"""Contact-blob size statistics per material: how many bumps, and how big.

The idea this tests: a gel imprint is a field of contact patches, and materials
should differ in the SIZE DISTRIBUTION of those patches. Sand should give many
tiny ones; gravel a few large ones; fabric a fine regular weave. That is a
physically meaningful descriptor, unlike any of DINOv2's 1,536 unnamed numbers.

Method: subtract the session's non-contact reference (make_bg_refs.py) so only
deformation remains, threshold at a robust multiple of the frame's own noise
level, label connected components, and record each blob's area.

The threshold is per-frame and robust (median + k*MAD) rather than global,
because sessions differ in gel wear and illumination. A global threshold would
turn session nuisance into apparent material differences -- exactly the confound
the grouped CV splits exist to catch.

Writes a tidy npz of per-press blob statistics for plotting and for use as
features.

Usage:
    python blob_stats.py --per-class 60 --out ../results/blob_stats.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from common import CLASS_NAMES
from nested_cv import EXCLUDE

BG_REF_DIR = Path("C:/datasets/touch_and_go/bg_refs")
MIN_BLOB = 4          # px; below this it is sensor noise, not a contact patch


def blob_areas(path: str, ref: np.ndarray, k: float = 3.0, absolute: float = 0.0):
    """Areas of contact blobs in one frame. Returns (areas, threshold_used).

    `absolute` thresholds the residual at a FIXED intensity, which is what you
    want. A per-frame robust threshold (median + k*MAD) self-normalises: MAD
    scales with the frame's own variability, so roughly the same fraction of
    pixels crosses it no matter what was pressed. Measured that way every class
    returned ~205 blobs and ~10% coverage -- the statistic had been normalised
    away. Background subtraction has already removed the session's baseline, so a
    fixed cut on the residual is both meaningful and comparable across sessions.
    """
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if ref is None or img.shape != ref.shape:
        return np.array([]), 0.0
    d = np.abs(img - ref)
    if absolute > 0:
        thr = absolute
    else:
        med = np.median(d)
        mad = np.median(np.abs(d - med)) * 1.4826      # -> sd-equivalent
        thr = med + k * max(mad, 1e-3)
    mask = d > thr
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.array([]), thr
    areas = np.bincount(lab.ravel())[1:]
    return areas[areas >= MIN_BLOB], thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=Path("../features_dinop9/rows.csv"))
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--abs-threshold", type=float, default=25.0,
                    help="fixed residual cut; 0 falls back to the per-frame MAD "
                         "rule, which self-normalises and is not recommended")
    ap.add_argument("--out", type=Path, default=Path("../results/blob_stats.npz"))
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    rows = pd.read_csv(args.rows)
    rows = rows[[CLASS_NAMES[int(c)] not in EXCLUDE for c in rows.label]]
    parts = [d.sample(min(len(d), args.per_class), random_state=42)
             for _, d in rows.groupby("label")]
    keep = pd.concat(parts).reset_index(drop=True)
    print(f"{len(keep):,} presses across {keep.label.nunique()} classes "
          f"(<= {args.per_class} per class)")

    refs = {}
    out_lab, out_cnt, out_area, out_med, all_areas, all_lab = [], [], [], [], [], []
    for i, (_, r) in enumerate(keep.iterrows()):
        sess = Path(r.gelsight_frame_path).parent.parent.name
        if sess not in refs:
            f = BG_REF_DIR / f"{sess}.png"
            refs[sess] = (np.asarray(Image.open(f).convert("L"), dtype=np.float32)
                          if f.exists() else None)
        areas, _ = blob_areas(r.gelsight_frame_path, refs[sess], args.k,
                              args.abs_threshold)
        if len(areas) == 0:
            continue
        out_lab.append(int(r.label))
        out_cnt.append(len(areas))
        out_area.append(float(areas.sum()))
        out_med.append(float(np.median(areas)))
        all_areas.append(areas)
        all_lab.append(np.full(len(areas), int(r.label)))
        if i % 200 == 0:
            print(f"\r  {i:,}/{len(keep):,}", end="", flush=True)
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             label=np.array(out_lab), n_blobs=np.array(out_cnt),
             total_area=np.array(out_area), median_area=np.array(out_med),
             blob_area=np.concatenate(all_areas),
             blob_label=np.concatenate(all_lab))
    print(f"{len(out_lab):,} presses, {sum(map(len, all_areas)):,} blobs -> {args.out}")

    print(f"\n{'class':<22}{'presses':>8}{'blobs/press':>13}{'median area':>13}"
          f"{'coverage %':>12}")
    lab = np.array(out_lab)
    for c in sorted(set(out_lab)):
        m = lab == c
        print(f"{CLASS_NAMES[c]:<22}{m.sum():>8}"
              f"{np.mean(np.array(out_cnt)[m]):>13.0f}"
              f"{np.median(np.array(out_med)[m]):>13.1f}"
              f"{100*np.mean(np.array(out_area)[m])/(480*640):>12.1f}")


if __name__ == "__main__":
    main()
