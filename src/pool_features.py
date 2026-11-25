"""Re-pool cached per-frame embeddings into a new feature directory.

extract_features.py used to mean-pool the k frames of a press inside the
extraction loop and discard the individual frames, so every new pooling idea
cost another full 55k-frame pass (~55 min on this machine). With --save-frames
it now caches the unpooled embeddings, and this script turns them into feature
directories in seconds.

WHY TEMPORAL STATISTICS
-----------------------
Mean-pooling answers "what did this press look like on average". It destroys, by
construction, how the gel deformed DURING the press -- and that motion is not
noise, it is compliance and stiction: how far the surface gives under load and
whether it grips. Those are material properties. A press into foam and a press
into stone can average out similarly while their deformation trajectories differ
completely.

The std over a run's frames is a cheap proxy for that trajectory. It costs one
extra grouped pass over data already on disk.

Caveat worth stating: runs have variable length (median 27 frames, min 11) and
select_pool_frames takes k evenly spaced frames, so the sampling INTERVAL differs
between a short press and a long one. The std is therefore not measured over a
fixed time window, and part of its variance across presses reflects run length
rather than material. Grouped CV keeps this honest -- run length is not
class-constant -- but it is a real confound worth naming rather than hiding.

Usage:
    python pool_features.py --frames ../features_dinop9_patch \
        --recipe meanstd --out ../features_dinop9_patch_ms
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


def grouped_moments(emb: np.ndarray, run_ids: np.ndarray):
    """Per-run mean, std and max in one pass. Returns (mean, std, max, order).

    Uses grouped sums rather than a boolean mask per run: masking would be
    O(n_runs * n_frames), which at 6k runs x 55k frames is 340M comparisons.
    std comes from E[x^2] - E[x]^2, clipped at 0 because that identity can go
    slightly negative in float arithmetic when a run's frames are near-identical.
    """
    order, inverse = np.unique(run_ids, return_inverse=True)
    n, d = len(order), emb.shape[1]
    counts = np.bincount(inverse, minlength=n)[:, None].astype(np.float64)

    sums = np.zeros((n, d)); np.add.at(sums, inverse, emb.astype(np.float64))
    sqs = np.zeros((n, d)); np.add.at(sqs, inverse, emb.astype(np.float64) ** 2)
    mean = sums / counts
    std = np.sqrt(np.clip(sqs / counts - mean ** 2, 0.0, None))

    mx = np.full((n, d), -np.inf)
    np.maximum.at(mx, inverse, emb.astype(np.float64))
    return (mean.astype(np.float32), std.astype(np.float32),
            mx.astype(np.float32), order)


RECIPES = {
    "mean":       lambda m, s, x: m,
    "meanstd":    lambda m, s, x: np.hstack([m, s]),
    "meanmax":    lambda m, s, x: np.hstack([m, x]),
    "meanstdmax": lambda m, s, x: np.hstack([m, s, x]),
    "std":        lambda m, s, x: s,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True,
                    help="feature dir written with --save-frames")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recipe", choices=list(RECIPES), default="meanstd")
    ap.add_argument("--modality", default="touch", choices=["touch", "vision"])
    args = ap.parse_args()

    emb = np.load(args.frames / f"{args.modality}_frames.npy")
    run_ids = np.load(args.frames / "frame_run_ids.npy", allow_pickle=True)
    if len(emb) != len(run_ids):
        raise SystemExit(f"{len(emb)} frames but {len(run_ids)} run ids")

    mean, std, mx, order = grouped_moments(emb, run_ids)
    pooled = RECIPES[args.recipe](mean, std, mx)

    # The row order must match the source dir's, or labels silently misalign.
    ref_order = np.load(args.frames / "labels.npy")
    if len(order) != len(ref_order):
        raise SystemExit(f"{len(order)} runs pooled but {len(ref_order)} labels")

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"{args.modality}.npy", pooled)
    for f in ("labels.npy", "groups.npy", "rows.csv"):
        shutil.copy2(args.frames / f, args.out / f)

    frames_per_run = len(emb) / len(order)
    print(f"{len(emb):,} frames -> {len(order):,} presses "
          f"({frames_per_run:.1f} per press)")
    print(f"recipe={args.recipe}  {emb.shape[1]}-d -> {pooled.shape[1]}-d")
    print(f"finite: {np.isfinite(pooled).all()}  -> {args.out}")


if __name__ == "__main__":
    main()
