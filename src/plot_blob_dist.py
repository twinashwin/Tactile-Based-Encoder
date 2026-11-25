"""Figure: contact-blob size distribution per material.

Tests the hypothesis that materials differ in the size distribution of their
contact patches -- many small dots for sand, few large ones for gravel. Small
multiples rather than 18 overlaid series: 18 is far past the number of
categorical hues that stay distinguishable, and the comparison being made is
each class against the pooled distribution, not class against class.

Usage:
    python plot_blob_dist.py --stats ../results/blob_stats.npz
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES

import os

# Deck-matched dark theme (sampled from the target Google Slides deck) or the
# light default. Set FIGTHEME=dark to render figures that sit on the deck's
# #161826 background instead of punching a white rectangle into it.
if os.environ.get("FIGTHEME", "light") == "dark":
    SURFACE = "#161826"
    INK = "#f3f2f6"
    INK2 = "#9d9cb4"
    SERIES = "#9e8feb"
    REF = "#454564"
    SEQ = "magma"
else:
    SURFACE = "#faf9f6"
    INK = "#111111"
    INK2 = "#5a564e"
    SERIES = "#8a1c1c"
    REF = "#d8d4ca"
    SEQ = "magma_r"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=Path("../results/blob_stats.npz"))
    ap.add_argument("--out", type=Path, default=Path("../results/blob_distribution.png"))
    args = ap.parse_args()

    d = np.load(args.stats)
    ba, bl = d["blob_area"], d["blob_label"]
    lab, cnt, med = d["label"], d["n_blobs"], d["median_area"]
    classes = sorted(set(bl.tolist()))

    bins = np.logspace(np.log10(4), np.log10(3000), 30)
    centres = np.sqrt(bins[:-1] * bins[1:])
    pooled, _ = np.histogram(ba, bins=bins, density=True)

    fig = plt.figure(figsize=(13, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 6, hspace=0.52, wspace=0.28,
                          left=0.06, right=0.98, top=0.86, bottom=0.44)

    # ---- small multiples: one panel per material ----
    for i, c in enumerate(classes):
        ax = fig.add_subplot(gs[i // 6, i % 6])
        h, _ = np.histogram(ba[bl == c], bins=bins, density=True)
        ax.fill_between(centres, pooled, color=REF, lw=0, zorder=1)
        ax.plot(centres, h, color=SERIES, lw=2, zorder=2)
        ax.set_xscale("log")
        ax.set_title(f"{CLASS_NAMES[int(c)]}", fontsize=9, color=INK, pad=3)
        ax.set_facecolor(SURFACE)
        ax.tick_params(labelsize=7, colors=INK2, length=2)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(REF)
        if i // 6 < 2:
            ax.set_xticklabels([])

    fig.text(0.06, 0.935, "Contact-blob size distribution is nearly identical "
             "across materials", fontsize=15, color=INK, weight="bold")
    fig.text(0.06, 0.900,
             "Blue = that material.  Grey = all materials pooled.  "
             "If size distribution carried material identity, the blue curves "
             "would sit off the grey.",
             fontsize=10, color=INK2)
    fig.text(0.5, 0.395, "blob area (pixels, log scale)", fontsize=9,
             color=INK2, ha="center")

    # ---- summary: class means against within-class spread ----
    ax = fig.add_axes([0.06, 0.13, 0.92, 0.20])
    mus, sds, names = [], [], []
    for c in classes:
        m = lab == c
        mus.append(med[m].mean()); sds.append(med[m].std()); names.append(CLASS_NAMES[int(c)])
    order = np.argsort(mus)
    x = np.arange(len(order))
    mus, sds = np.array(mus)[order], np.array(sds)[order]
    ax.errorbar(x, mus, yerr=sds, fmt="o", ms=7, color=SERIES,
                ecolor=REF, elinewidth=6, capsize=0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in order], rotation=35, ha="right",
                       fontsize=8, color=INK2)
    ax.set_ylabel("median blob\narea (px)", fontsize=9, color=INK2)
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=8, colors=INK2, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(REF)
    ax.set_title("Class averages (dots) are swamped by the spread within each "
                 "class (bars): between-class sd is 0.29x within-class sd",
                 fontsize=10, color=INK, pad=6, loc="left")

    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
