"""Treat the GelSight as a force transducer: track its printed markers.

THE IDEA
--------
Every analysis in this project so far has treated a gel frame as a picture and
asked an image encoder what it depicts. That works for materials whose surface
has texture and fails for the smooth/hard family -- Metal, Plastic, Glass, Tile,
Paper, Rubber -- which have no texture to photograph. Those six are 22% of the
dataset and sit at 0.13-0.40 F1.

What separates them is MECHANICS: stiffness, compliance, friction. A GelSight
measures those with the ~100 black dots printed on its gel. Press it and they
move apart and swell; drag it and they shear.

TWO CHANNELS
------------
lateral   where each marker moved in the image plane -> shear and spreading.
depth     how much each marker GREW. Markers nearer the camera project larger,
          so apparent size is a per-marker height readout -- roughly a hundred
          depth samples per frame, which is a sparse topographic map of the
          indentation. This is the channel that should separate hard from soft,
          because it measures normal indentation rather than shear.

Size must be measured against a LOCAL background, never a global threshold.
Contact darkens the whole frame, so any fixed or percentile cut makes every dark
blob measure bigger under load -- an artifact that mimics the depth signal
exactly. Here each marker is thresholded midway between the darkest pixel in its
own neighbourhood and that neighbourhood's background, so uniform darkening
cancels and only genuine growth survives.

WHAT AN EARLIER VERSION GOT WRONG
A first pass used lateral displacement alone, a +/-13 px search window and a
fixed detection percentile. It failed: adding it to the encoder features cost
2.3 points of accuracy and 2.7 points on the smooth family it was meant to
rescue. Two defects contributed and are fixed here -- 18% of presses failed
tracking and were silently filled with column medians, and the search window was
narrower than the lattice allows, so heavy contacts could not be matched. A
failure mask is now returned instead of imputing, and the window is widened to
the half-pitch limit.

Usage:
    python marker_field.py --limit 40
    python marker_field.py --out ../features_marker
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from common import select_pool_frames

BG_REF_DIR = Path("C:/datasets/touch_and_go/bg_refs")
PATCH = 11          # half-size of the template cut around a marker's rest position
SEARCH = 20         # half-size of the search window; lattice pitch is ~58 px, so
                    # going beyond ~29 would risk matching the neighbouring marker
HALO = 16           # half-size of the window used for the local-background size read
MIN_MARKERS = 25


def find_markers(ref: np.ndarray) -> np.ndarray:
    """Rest positions of the printed markers.

    The detection percentile adapts: worn or dimly lit gels yield fewer dark
    pixels, and a fixed cut was leaving some sessions below the usable minimum.
    """
    for pct in (12.0, 16.0, 20.0, 25.0):
        m = ref < np.percentile(ref, pct)
        m = ndimage.binary_opening(m, np.ones((3, 3)))
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        cen = np.array(ndimage.center_of_mass(m, lab, np.arange(1, n + 1)))
        keep = (area >= 6) & (area <= 600)
        cen = cen[keep]
        if len(cen) >= MIN_MARKERS:
            return cen
    return np.zeros((0, 2))


def local_size(img: np.ndarray, y: int, x: int) -> float:
    """Marker area at (y, x), thresholded against its own neighbourhood.

    Immune to overall darkening: the cut sits halfway between the darkest pixel
    in the halo and the halo's own bright background, so scaling the whole frame
    leaves the measured area unchanged.
    """
    h, w = img.shape
    y0, y1 = max(0, y - HALO), min(h, y + HALO + 1)
    x0, x1 = max(0, x - HALO), min(w, x + HALO + 1)
    win = img[y0:y1, x0:x1]
    if win.size < 50:
        return np.nan
    bg = np.percentile(win, 75)
    lo = win.min()
    if bg - lo < 6:                       # no marker-like contrast here
        return 0.0
    return float((win < (lo + bg) / 2.0).sum())


def track(ref, img, pts):
    """Lateral displacement and size ratio for every marker. NaN where unreadable."""
    h, w = ref.shape
    u = np.full((len(pts), 2), np.nan, dtype=np.float32)
    sz = np.full(len(pts), np.nan, dtype=np.float32)
    for i, (yf, xf) in enumerate(pts):
        y, x = int(round(yf)), int(round(xf))
        if not (PATCH + SEARCH < y < h - PATCH - SEARCH and
                PATCH + SEARCH < x < w - PATCH - SEARCH):
            continue
        tpl = ref[y - PATCH:y + PATCH + 1, x - PATCH:x + PATCH + 1]
        win = img[y - PATCH - SEARCH:y + PATCH + SEARCH + 1,
                  x - PATCH - SEARCH:x + PATCH + SEARCH + 1]
        if tpl.std() < 1e-3:
            continue
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx < 0.15:
            continue
        dy, dx = loc[1] - SEARCH, loc[0] - SEARCH
        u[i] = (dy, dx)
        s0 = local_size(ref, y, x)
        s1 = local_size(img, y + dy, x + dx)       # measured where it moved TO
        if s0 and np.isfinite(s0) and np.isfinite(s1) and s0 > 2:
            sz[i] = s1 / s0
    return u, sz


def field_stats(pts, u, sz, resid):
    ok = np.isfinite(u).all(axis=1)
    if ok.sum() < MIN_MARKERS:
        return None
    p, d = pts[ok], u[ok]
    mag = np.hypot(d[:, 0], d[:, 1])

    c = p.mean(axis=0)
    rad = p - c
    rhat = rad / np.clip(np.linalg.norm(rad, axis=1, keepdims=True), 1e-6, None)
    radial = (d * rhat).sum(axis=1)
    tang = np.abs(d[:, 0] * rhat[:, 1] - d[:, 1] * rhat[:, 0])

    s = sz[ok]
    s = s[np.isfinite(s)]
    if len(s) < 10:
        s = np.array([1.0])
    area = float((resid > 25).mean())

    return dict(
        mag_mean=float(mag.mean()), mag_p90=float(np.percentile(mag, 90)),
        mag_std=float(mag.std()),
        radial_mean=float(radial.mean()), tang_mean=float(tang.mean()),
        shear_ratio=float(tang.mean() / max(abs(radial.mean()), 1e-3)),
        # --- depth channel ---
        size_mean=float(s.mean()), size_p90=float(np.percentile(s, 90)),
        size_std=float(s.std()),
        size_frac_grown=float((s > 1.05).mean()),
        # how concentrated the growth is: a stiff surface indents a small patch
        # deeply, a compliant one spreads the load
        size_concentration=float(np.percentile(s, 90) / max(s.mean(), 1e-3)),
        area=area, energy=float(resid.mean()),
        area_per_disp=float(area / max(mag.mean(), 1e-3)),
        size_per_disp=float((s.mean() - 1.0) / max(mag.mean(), 1e-3)),
    )


FRAME_KEYS = ["mag_mean", "mag_p90", "mag_std", "radial_mean", "tang_mean",
              "shear_ratio", "size_mean", "size_p90", "size_std",
              "size_frac_grown", "size_concentration", "area", "energy",
              "area_per_disp", "size_per_disp"]


def press_features(seq):
    S = np.array([[f[k] for k in FRAME_KEYS] for f in seq], dtype=np.float64)
    t = np.linspace(0, 1, len(S))
    out, names = [], []
    for j, k in enumerate(FRAME_KEYS):
        v = S[:, j]
        slope = np.polyfit(t, v, 1)[0] if len(v) > 2 else 0.0
        out += [v.mean(), v.max() - v.min(), slope]
        names += [f"{k}_mean", f"{k}_range", f"{k}_slope"]

    a = S[:, FRAME_KEYS.index("area")]
    m = S[:, FRAME_KEYS.index("mag_mean")]
    g = S[:, FRAME_KEYS.index("size_mean")]
    for num, den, nm in ((a, m, "compliance"), (g, m, "depth_per_shear"),
                         (g, a, "depth_per_area")):
        if den.std() > 1e-6:
            k_fit = float(np.polyfit(den, num, 1)[0])
            res = float(np.abs(num - np.polyval(np.polyfit(den, num, 1), den)).mean())
        else:
            k_fit, res = 0.0, 0.0
        out += [k_fit, res]
        names += [f"{nm}_slope", f"{nm}_hyst"]
    return np.array(out, dtype=np.float32), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--like", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--out", type=Path, default=Path("../features_marker"))
    ap.add_argument("--frames-per-run", type=int, default=9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0,
                    help="index of the first press to process")
    ap.add_argument("--count", type=int, default=0,
                    help="how many presses to process; 0 = all remaining. Chunking "
                         "exists because a full pass exceeds the harness background "
                         "time limit; chunks are merged by merge_marker.py")
    ap.add_argument("--part", type=Path, default=None,
                    help="write a chunk here as an .npz instead of a feature dir")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    sel = select_pool_frames(df, args.frames_per_run).sort_values(["run_id", "frame_idx"])
    rows = pd.read_csv(args.like / "rows.csv")
    want = list(rows.run_id)
    if args.limit:
        want = want[:args.limit]
    all_want = want
    if args.count:
        want = want[args.start:args.start + args.count]
    elif args.start:
        want = want[args.start:]
    sel = sel[sel.run_id.isin(set(want))]
    print(f"{len(want):,} presses, {len(sel):,} frames, search +/-{SEARCH} px")

    refs, markers, feats, names = {}, {}, {}, None
    done = 0
    for rid, g in sel.groupby("run_id", sort=False):
        g = g.sort_values("frame_idx")
        sess = Path(g.iloc[0].gelsight_frame_path).parent.parent.name
        if sess not in refs:
            f = BG_REF_DIR / f"{sess}.png"
            if f.exists():
                r = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
                refs[sess], markers[sess] = r, find_markers(r)
            else:
                refs[sess], markers[sess] = None, None
        ref, pts = refs[sess], markers[sess]
        if ref is None or pts is None or len(pts) < MIN_MARKERS:
            continue
        seq = []
        for p in g.gelsight_frame_path:
            img = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            if img.shape != ref.shape:
                break
            u, sz = track(ref, img, pts)
            st = field_stats(pts, u, sz, np.abs(img - ref))
            if st is None:
                break
            seq.append(st)
        if len(seq) < 3:
            continue
        v, names = press_features(seq)
        feats[rid] = v
        done += 1
        if done % 100 == 0:
            print(f"\r  {done:,}", end="", flush=True)
    print()

    if args.part:
        np.savez(args.part, run_id=np.array(list(feats.keys()), dtype=object),
                 feat=np.array(list(feats.values()), dtype=np.float32),
                 names=np.array(names, dtype=object))
        print(f"chunk {args.start}..{args.start+len(want)}: "
              f"{len(feats):,}/{len(want):,} tracked -> {args.part}")
        return

    X = np.zeros((len(want), len(names)), dtype=np.float32)
    ok = np.zeros(len(want), dtype=bool)
    for i, rid in enumerate(want):
        if rid in feats:
            X[i], ok[i] = feats[rid], True
    # rows that failed keep the column mean of the SUCCESSFUL rows; the mask is
    # saved so downstream code can exclude them rather than trust a filled value
    if ok.any():
        X[~ok] = np.nan_to_num(X[ok].mean(axis=0))

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "touch.npy", X)
    np.save(args.out / "tracked.npy", ok)
    for f in ("labels.npy", "groups.npy", "rows.csv"):
        src = args.like / f
        if src.exists():
            import shutil
            shutil.copy2(src, args.out / f)
    (args.out / "feature_names.txt").write_text("\n".join(names))
    nm = np.mean([len(v) for v in markers.values() if v is not None])
    print(f"{X.shape} -> {args.out}")
    print(f"tracked {ok.sum():,}/{len(want):,} ({100*ok.mean():.1f}%)   "
          f"markers/session {nm:.0f}")


if __name__ == "__main__":
    main()
