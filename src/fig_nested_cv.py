"""Figure: the nested, session-grouped cross-validation protocol.

A schematic, not a chart: it encodes set membership, not magnitude, so the only
quantitative thing it carries is the relative width of each partition.

Two panels. The top shows the outer loop, five folds over 124 recording
sessions, with a different fifth held out for test each time. The bottom expands
one outer fold's training portion to show the inner four-way split, of which
only the first pairing is used. That last detail is the one readers get wrong:
nested cross-validation usually implies both loops are iterated, which would be
twenty encoders, whereas this protocol trains five.

Colours are categorical slots 1, 2, 3 in fixed order, validated for CVD
separation against the chart surface. Every block carries a direct label, which
is what licenses the aqua's sub-3:1 contrast against the surface.

Usage:
    python fig_nested_cv.py --out ../results/fig_nested_cv.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
TRAIN = "#2a78d6"      # categorical slot 1
VAL = "#eb6834"        # slot 2
TEST = "#1baf7a"       # slot 3
UNUSED = "#d6d4cf"

N_SESS = 124
N_OUTER = 5
N_INNER = 4
GAP = 0.012            # surface gap between adjacent fills


def block(ax, x, y, w, h, color, label=None, text_color="#ffffff", fontsize=8):
    """One rounded partition block with an optional direct label."""
    ax.add_patch(FancyBboxPatch(
        (x + GAP / 2, y), max(w - GAP, 0.001), h,
        boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=0, facecolor=color, mutation_aspect=0.35))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="medium")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("../results/fig_nested_cv.png"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 4.4), dpi=args.dpi,
        gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.30})
    fig.patch.set_facecolor(SURFACE)

    # ---------------- outer loop ----------------
    ax1.set_facecolor(SURFACE)
    row_h, pitch = 0.1, 0.155
    seg = 1.0 / N_OUTER

    for i in range(N_OUTER):
        y = 1.0 - (i + 1) * pitch
        for j in range(N_OUTER):
            x = j * seg
            if j == i:
                block(ax1, x, y, seg, row_h, TEST,
                      "test" if i == 0 else None)
            else:
                block(ax1, x, y, seg, row_h, TRAIN,
                      "train" if (i == 0 and j == 2) else None)
        ax1.text(-0.018, y + row_h / 2, f"fold {i+1}", ha="right", va="center",
                 fontsize=8.5, color=INK_2)

    ax1.text(0, 1.0 - 0.3 * pitch + 0.012,
             "124 recording sessions, grouped: every press from one session stays on one side",
             ha="left", va="bottom", fontsize=8.5, color=INK_2)
    ax1.text(0, 1.0 + 0.085, "Outer loop: 5 folds, 80 / 20",
             ha="left", va="bottom", fontsize=10.5, color=INK, fontweight="semibold")
    ax1.annotate("", xy=(1.004, 1.0 - N_OUTER * pitch + 0.004),
                 xytext=(1.004, 1.0 - pitch + row_h - 0.004),
                 arrowprops=dict(arrowstyle="-", color="#b8b6b1", lw=0.9))
    ax1.text(1.016, 1.0 - (N_OUTER * pitch + row_h) / 2 - 0.02,
             "5 encoders\ntrained,\none per fold", ha="left", va="center",
             fontsize=8, color=INK_2, linespacing=1.4)

    ax1.set_xlim(-0.1, 1.2); ax1.set_ylim(1.0 - N_OUTER * pitch - 0.005, 1.135)
    ax1.axis("off")

    # ---------------- inner split ----------------
    ax2.set_facecolor(SURFACE)
    iseg = 1.0 / N_INNER
    irow_h, ipitch = 0.115, 0.17

    for i in range(N_INNER):
        y = 1.0 - (i + 1) * ipitch
        used = i == 0
        for j in range(N_INNER):
            x = j * iseg
            if not used:
                block(ax2, x, y, iseg, irow_h, UNUSED)
            elif j == 0:
                block(ax2, x, y, iseg, irow_h, VAL, "validation")
            else:
                block(ax2, x, y, iseg, irow_h, TRAIN,
                      "training" if j == 2 else None)
        tag = "used" if used else "discarded"
        ax2.text(-0.018, y + irow_h / 2, f"pairing {i+1}", ha="right",
                 va="center", fontsize=8.5,
                 color=INK if used else "#9a9892")
        ax2.text(1.016, y + irow_h / 2, tag, ha="left", va="center",
                 fontsize=8, color=INK_2 if used else "#9a9892",
                 fontweight="semibold" if used else "normal")

    ax2.text(0, 1.0 + 0.1,
             "Inner split of one outer fold's training data: 4 pairings, first one kept",
             ha="left", va="bottom", fontsize=10.5, color=INK,
             fontweight="semibold")
    ax2.text(0, 1.0 + 0.012,
             "the validation set chooses the training epoch; the test fold is scored once, at the end",
             ha="left", va="bottom", fontsize=8.5, color=INK_2)

    ax2.text(0, 1.0 - N_INNER * ipitch - 0.055,
             "Outer fold 1:  3,672 training presses   ·   1,208 validation   ·   1,235 test",
             ha="left", va="center", fontsize=9, color=INK)

    ax2.set_xlim(-0.1, 1.2); ax2.set_ylim(1.0 - N_INNER * ipitch - 0.09, 1.175)
    ax2.axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
