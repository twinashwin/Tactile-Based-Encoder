"""Result figures for the deck: the accuracy ladder and the head comparison.

Numbers are hard-coded from the confirmation runs rather than recomputed, because
they come from several separate nested-CV runs (touch_heads_confirm_*.json,
touch_heads_cls_*.json) and the point of this file is presentation, not analysis.
Each value is annotated with its source so it can be traced back.

Usage:
    FIGTHEME=dark python plot_results.py --outdir ../results/slides
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if os.environ.get("FIGTHEME", "light") == "dark":
    SURFACE, INK, INK2 = "#161826", "#f3f2f6", "#9d9cb4"
    SERIES, REF, GOOD = "#9e8feb", "#454564", "#5fd3a6"
else:
    SURFACE, INK, INK2 = "#faf9f6", "#111111", "#5a564e"
    SERIES, REF, GOOD = "#8a1c1c", "#d8d4ca", "#1f6b4a"

# (label, accuracy, source)
LADDER = [
    ("Starting point\nnested_cv.json MLP", 0.5264, "seed 42"),
    ("+ protocol fixes\n4-fold inner loop, tuned\npreprocessing, acc+F1 objective",
     0.5999, "seeds 7/13/2024"),
    ("+ patch tokens\ninstead of CLS", 0.6033, "seeds 7/13/2024"),
    ("+ background-subtracted\nsecond view", 0.6154, "seeds 7/13/2024"),
]

# head, accuracy, macro-F1, confirmed?
HEADS = [
    ("MLP", 0.6154, 0.3917, True),
    ("SVM-linear", 0.6032, 0.3685, True),
    ("SVM-RBF", 0.6085, 0.3558, False),
    ("Logistic", 0.6003, 0.3565, False),
    ("kNN-cosine", 0.5737, 0.2925, False),
]


def style(ax, keep=("bottom", "left")):
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=9, colors=INK2, length=2)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color(REF)


def fig_ladder(out):
    fig, ax = plt.subplots(figsize=(13, 6.6), facecolor=SURFACE)
    vals = [v for _, v, _ in LADDER]
    x = np.arange(len(vals))

    base = 0.50
    for i, (lab, v, src) in enumerate(LADDER):
        prev = base if i == 0 else vals[i - 1]
        # the carried-over part, then the increment on top
        ax.bar(i, prev - base, bottom=base, width=0.62, color=REF, lw=0)
        ax.bar(i, v - prev, bottom=prev, width=0.62,
               color=SERIES if i else REF, lw=0)
        ax.text(i, v + 0.0032, f"{v:.4f}", ha="center", fontsize=13,
                color=INK, weight="bold")
        if i:
            gap = v - prev
            # a thin increment cannot hold a label; put it beside the bar instead
            if gap > 0.008:
                ax.text(i, prev + gap / 2, f"+{100*gap:.2f}", ha="center",
                        va="center", fontsize=11, color=SURFACE, weight="bold")
            else:
                ax.text(i + 0.36, prev + gap / 2, f"+{100*gap:.2f}", ha="left",
                        va="center", fontsize=10.5, color=SERIES, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in LADDER], fontsize=10, color=INK)
    ax.set_ylim(base, 0.634)
    ax.set_ylabel("touch-only accuracy", fontsize=11, color=INK2)
    style(ax)
    ax.grid(axis="y", color=REF, lw=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    fig.text(0.075, 0.945, "Where the +8.9 points came from", fontsize=18,
             color=INK, weight="bold")
    fig.text(0.075, 0.885,
             "Each bar is measured against the one before it, so the increments "
             "add up. 83% of the gain is protocol, not representation.\n"
             "Step 1 is a cross-seed comparison; steps 2 and 3 are paired on "
             "seeds 7 / 13 / 2024.",
             fontsize=10.5, color=INK2, va="top")
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.09, right=0.97)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def fig_heads(out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)
    names = [h[0] for h in HEADS]
    y = np.arange(len(names))[::-1]

    for ax, idx, title, lo in ((axes[0], 1, "accuracy", 0.55),
                               (axes[1], 2, "macro-F1", 0.25)):
        vals = [h[idx] for h in HEADS]
        cols = [SERIES if h[3] else REF for h in HEADS]
        ax.barh(y, vals, height=0.6, color=cols, lw=0)
        for yy, v in zip(y, vals):
            ax.text(v + (0.002 if idx == 1 else 0.003), yy, f"{v:.4f}",
                    va="center", fontsize=10.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names if idx == 1 else [], fontsize=11, color=INK)
        ax.set_xlim(lo, max(vals) * 1.06)
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
        style(ax, keep=("bottom",))
        ax.grid(axis="x", color=REF, lw=0.6, alpha=0.45)
        ax.set_axisbelow(True)

    fig.suptitle("The MLP wins on both metrics at once", fontsize=17, color=INK,
                 weight="bold", x=0.055, ha="left", y=0.97)
    fig.text(0.055, 0.885,
             "Solid = confirmed on three held-out seeds. Grey = seed 42 only. "
             "Tuned on accuracy and macro-F1 weighted equally.",
             fontsize=10.5, color=INK2)
    fig.subplots_adjust(top=0.74, bottom=0.09, left=0.11, right=0.97, wspace=0.08)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("../results"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_ladder(args.outdir / "res_ladder.png")
    fig_heads(args.outdir / "res_heads.png")


if __name__ == "__main__":
    main()
