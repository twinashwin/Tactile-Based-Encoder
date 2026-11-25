"""Measure the real per-channel mean/std of Touch and Go gel frames.

finetune_dino.py normalises with ImageNet statistics, which describe photographs
of the world. A GelSight frame is a sheet of elastomer lit by three fixed
coloured LEDs: its colour distribution is narrow, strongly biased toward
whichever LEDs dominate, and nothing like ImageNet's. Normalising with the wrong
statistics puts every input in the wrong part of the encoder's expected range
before a single weight is trained.

Samples presses rather than reading all 6,115, since the mean converges quickly.

Usage:
    python frame_stats.py --n 800
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import CLASS_NAMES, select_rows
from nested_cv import EXCLUDE, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--like", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    sel = select_rows(df, "run")
    rows = pd.read_csv(args.like / "rows.csv")
    sel = sel[sel.run_id.isin(set(rows.run_id))]
    sel = sel[[Path(p).exists() for p in sel.gelsight_frame_path]]
    sel = sel[[CLASS_NAMES[int(c)] not in EXCLUDE for c in sel.label]]

    rng = np.random.default_rng(SEED)
    paths = sel.gelsight_frame_path.to_numpy()
    pick = rng.choice(len(paths), size=min(args.n, len(paths)), replace=False)
    print(f"sampling {len(pick):,} of {len(paths):,} presses")

    # accumulate sums, not a list of arrays -- 800 full frames is 2.9 GB
    s = np.zeros(3, dtype=np.float64)
    ss = np.zeros(3, dtype=np.float64)
    npx = 0
    for k, i in enumerate(pick):
        with Image.open(paths[i]) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        a = a.reshape(-1, 3)
        s += a.sum(0); ss += (a ** 2).sum(0); npx += len(a)
        if (k + 1) % 200 == 0:
            print(f"  {k+1:,}", flush=True)

    mean = s / npx
    std = np.sqrt(np.maximum(ss / npx - mean ** 2, 1e-12))
    print("\nGelSight (measured)")
    print(f"  mean [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print("ImageNet (what we have been using)")
    print("  mean [0.4850, 0.4560, 0.4060]")
    print("  std  [0.2290, 0.2240, 0.2250]")
    print("\nGELSIGHT_STATS = ([%.4f, %.4f, %.4f], [%.4f, %.4f, %.4f])"
          % (*mean, *std))


if __name__ == "__main__":
    main()
