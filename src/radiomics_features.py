"""Hand-crafted texture descriptors -- the "no image" model.

Everything else in this project hands a neural network an image. This computes a
short list of named, physically interpretable numbers instead, in the radiomics
tradition (first-order statistics, GLCM, run-length, shape). The point is not to
beat DINOv2 -- it will not -- but to establish how much of the signal is reachable
without a learned encoder, and whether these numbers add anything when bolted
onto one.

FEATURE GROUPS
  first-order   mean/sd/skew/kurtosis/percentiles/histogram entropy of the residual
  GLCM          contrast, dissimilarity, homogeneity, ASM, energy, correlation and
                entropy, averaged over 4 angles at 3 distances (13 px covers the
                marker pitch, which matters -- see the caveat below)
  run-length    short/long-run emphasis and run-percentage on a quantised image
  spectral      radially-averaged power in 6 frequency bands (roughness by scale)
  morphology    region count, coverage, area/volume quantiles, centroid spacing
                (from segment_texture.py -- "closeness of dots to each other")

CAVEAT THAT GOVERNS THE INTERPRETATION
The GelSight gel carries ~99 PRINTED MARKER DOTS on a 58 px lattice. They move
when the gel deforms, so subtracting a static reference leaves a ring at every
marker rather than removing them. Any descriptor computed on the residual is
therefore dominated by marker displacement, not by the material's own texture.
That is a real tactile signal -- marker fields are how GelSight measures shear and
normal force -- but it must not be reported as a texture measurement. Features are
also computed with the markers masked out (--mask-markers) so the two can be
separated.

Usage:
    python radiomics_features.py --out ../features_radiomics
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops

from common import CLASS_NAMES
from segment_texture import BG_REF_DIR, nn_spacing, region_measures, segment

LEVELS = 32
DISTANCES = (1, 5, 13)          # 13 px straddles the 58 px marker pitch
ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def quantise(x, levels=LEVELS):
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    q = np.clip((x - lo) / (hi - lo), 0, 1) * (levels - 1)
    return q.astype(np.uint8)


def first_order(d):
    p = np.percentile(d, [10, 25, 50, 75, 90, 99])
    h, _ = np.histogram(d, bins=64, density=True)
    h = h[h > 0]
    ent = float(-(h * np.log2(h)).sum())
    m, s = float(d.mean()), float(d.std())
    z = (d - m) / max(s, 1e-6)
    return [m, s, float((z ** 3).mean()), float((z ** 4).mean()), ent, *p]


def glcm_feats(q):
    g = graycomatrix(q, distances=list(DISTANCES), angles=list(ANGLES),
                     levels=LEVELS, symmetric=True, normed=True)
    out = []
    for prop in ("contrast", "dissimilarity", "homogeneity", "ASM", "energy",
                 "correlation"):
        out.extend(graycoprops(g, prop).mean(axis=1).tolist())
    # entropy is not in graycoprops; compute it per (distance, angle) and average
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -(g * np.log2(np.where(g > 0, g, 1))).sum(axis=(0, 1))
    out.extend(e.mean(axis=1).tolist())
    return out


def run_length(q, levels=LEVELS):
    """Short/long-run emphasis and run percentage, horizontal runs only."""
    diff = np.diff(q, axis=1) != 0
    starts = np.concatenate([np.ones((q.shape[0], 1), bool), diff], axis=1)
    idx = np.flatnonzero(starts.ravel())
    lens = np.diff(np.append(idx, q.size)).astype(np.float64)
    lens = lens[lens > 0]
    if len(lens) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    sre = float((1.0 / lens ** 2).mean())
    lre = float((lens ** 2).mean())
    rp = float(len(lens) / q.size)
    return [sre, lre, rp, float(lens.mean())]


def radial_spectrum(d, bands=6):
    f = np.abs(np.fft.fftshift(np.fft.fft2(d - d.mean()))) ** 2
    h, w = f.shape
    y, x = np.ogrid[:h, :w]
    r = np.hypot(y - h / 2, x - w / 2)
    rmax = min(h, w) / 2
    edges = np.linspace(0, rmax, bands + 1)
    tot = f.sum() + 1e-9
    return [float(f[(r >= a) & (r < b)].sum() / tot)
            for a, b in zip(edges[:-1], edges[1:])]


def marker_mask(ref, pct=12):
    """Pixels that are dark in the UNDEFORMED gel = the printed markers."""
    m = ref < np.percentile(ref, pct)
    return ndimage.binary_dilation(m, np.ones((5, 5)))


NAMES = (
    ["fo_mean", "fo_sd", "fo_skew", "fo_kurt", "fo_entropy",
     "fo_p10", "fo_p25", "fo_p50", "fo_p75", "fo_p90", "fo_p99"]
    + [f"glcm_{p}_d{d}" for p in ("contrast", "dissim", "homog", "asm",
                                  "energy", "corr", "entropy") for d in DISTANCES]
    + ["rl_short", "rl_long", "rl_pct", "rl_mean"]
    + [f"spec_band{i}" for i in range(6)]
    + ["seg_n", "seg_cover", "seg_area_med", "seg_area_p90", "seg_area_iqr",
       "seg_vol_med", "seg_vol_p90", "seg_spacing"]
)


def describe(img, ref, mask_markers=False):
    d = np.abs(img - ref)
    if mask_markers:
        mm = marker_mask(ref)
        d = d.copy()
        d[mm] = np.median(d[~mm]) if (~mm).any() else 0.0
    q = quantise(d)
    labels, resid = segment(img, ref)
    area, vol, cen = region_measures(labels, resid)
    if len(area) == 0:
        seg = [0, 0, 0, 0, 0, 0, 0, np.nan]
    else:
        seg = [len(area), area.sum() / d.size, np.median(area),
               np.percentile(area, 90),
               np.percentile(area, 75) - np.percentile(area, 25),
               np.median(vol), np.percentile(vol, 90), nn_spacing(cen)]
    return first_order(d) + glcm_feats(q) + run_length(q) + \
        radial_spectrum(d) + [float(v) for v in seg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=Path("../features_dinop9/rows.csv"))
    ap.add_argument("--out", type=Path, default=Path("../features_radiomics"))
    ap.add_argument("--mask-markers", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows)
    if args.limit:
        rows = rows.iloc[:args.limit]
    print(f"{len(rows):,} presses, {len(NAMES)} features, "
          f"mask_markers={args.mask_markers}")

    refs, X = {}, np.full((len(rows), len(NAMES)), np.nan, dtype=np.float32)
    for i, (_, r) in enumerate(rows.iterrows()):
        sess = Path(r.gelsight_frame_path).parent.parent.name
        if sess not in refs:
            f = BG_REF_DIR / f"{sess}.png"
            refs[sess] = (np.asarray(Image.open(f).convert("L"), dtype=np.float32)
                          if f.exists() else None)
        ref = refs[sess]
        if ref is None:
            continue
        img = np.asarray(Image.open(r.gelsight_frame_path).convert("L"),
                         dtype=np.float32)
        if img.shape != ref.shape:
            continue
        try:
            X[i] = describe(img, ref, args.mask_markers)
        except Exception:
            pass
        if i % 100 == 0:
            print(f"\r  {i:,}/{len(rows):,}", end="", flush=True)
    print()

    # impute the few failures with the column median so downstream code is simple
    med = np.nanmedian(X, axis=0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(med, np.where(bad)[1])

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "touch.npy", X)
    for f in ("labels.npy", "groups.npy", "rows.csv"):
        src = args.rows.parent / f
        if src.exists():
            import shutil
            shutil.copy2(src, args.out / f)
    (args.out / "feature_names.txt").write_text("\n".join(NAMES))
    print(f"{X.shape} -> {args.out}   finite: {np.isfinite(X).all()}")


if __name__ == "__main__":
    main()
