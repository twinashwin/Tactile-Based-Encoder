"""Figures for the morphology track.

Produces four PNGs in results/:
  seg_pipeline.png      original -> residual -> binary mask -> watershed regions
  seg_heatmap.png       per-class aggregated contact-volume map
  seg_distributions.png per-class region area and volume distributions
  seg_iqr.png           all classes on one axis, median + IQR

Design notes: eighteen classes is far past the number of categorical hues that
stay distinguishable, so the per-class panels are small multiples against a
pooled reference rather than eighteen overlaid series, and the summary figures
use one hue with position carrying the comparison.

Usage:
    python plot_segments.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import CLASS_NAMES
from segment_texture import BG_REF_DIR, region_measures, segment

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


def style(ax, keep=("bottom", "left")):
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=8, colors=INK2, length=2)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color(REF)


# --------------------------------------------------------------------------
def fig_pipeline(out):
    """Four stages of the pipeline on two contrasting materials."""
    rows = pd.read_csv("../features_dinop9/rows.csv")
    picks = []
    for want in ("Gravel", "Synthetic Fabric", "Rock"):
        code = [k for k, v in CLASS_NAMES.items() if v == want][0]
        sub = rows[rows.label == code]
        for _, r in sub.iterrows():
            ref_f = BG_REF_DIR / (Path(r.gelsight_frame_path).parent.parent.name + ".png")
            if ref_f.exists():
                picks.append((want, r.gelsight_frame_path, ref_f))
                break

    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.3 * len(picks)),
                             facecolor=SURFACE)
    titles = ["1. raw gel image", "2. minus session reference",
              "3. binary contact mask", "4. watershed regions"]
    for r, (name, path, ref_f) in enumerate(picks):
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        ref = np.asarray(Image.open(ref_f).convert("L"), dtype=np.float32)
        labels, resid = segment(img, ref)
        mask = labels > 0
        # scramble label ids so adjacent regions get contrasting colours
        shown = np.where(mask, (labels.astype(np.int64) * 7919) % 997 + 1, 0)
        panels = [(img, "gray", None), (resid, "magma", None),
                  (mask, "gray", None), (shown, "nipy_spectral", None)]
        for c, (m, cmap, _) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(m, cmap=cmap, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(REF)
            if r == 0:
                ax.set_title(titles[c], fontsize=11, color=INK, pad=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=11, color=INK)
        axes[r, 3].set_xlabel("%d regions" % labels.max(), fontsize=9, color=INK2)
    fig.suptitle("The pipeline finds the SENSOR'S MARKERS, not the material",
                 fontsize=15, color=INK, weight="bold", x=0.06, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
def fig_heatmap(d, out):
    heat, classes = d["heat"], d["heat_classes"]
    vmax = np.percentile(heat, 99.5)
    fig, axes = plt.subplots(3, 6, figsize=(13, 7.2), facecolor=SURFACE)
    for i, c in enumerate(classes):
        ax = axes[i // 6, i % 6]
        ax.imshow(heat[i], cmap=SEQ, vmin=0, vmax=vmax, interpolation="bilinear")
        ax.set_title(CLASS_NAMES[int(c)], fontsize=9.5, color=INK, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(REF)
    fig.suptitle("Aggregated contact volume by material - all eighteen maps are "
                 "the same", fontsize=15, color=INK, weight="bold",
                 x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.923,
             "Every press of a material, averaged. Brightness = contact volume.\n"
             "The maps encode where the SENSOR contacts and how its marker "
             "lattice deforms - averaging cancels material texture, which is "
             "never in the same place twice.",
             fontsize=10, color=INK2)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("mean contact volume", fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.03, right=0.9,
                        hspace=0.25, wspace=0.08)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
def fig_distributions(d, out):
    ra, rv, rl = d["region_area"], d["region_volume"], d["region_label"]
    classes = sorted(set(rl.tolist()))
    fig, axes = plt.subplots(3, 6, figsize=(13, 7.4), facecolor=SURFACE)
    bins = np.logspace(np.log10(6), np.log10(4000), 34)
    centres = np.sqrt(bins[:-1] * bins[1:])
    pooled, _ = np.histogram(ra, bins=bins, density=True)
    for i, c in enumerate(classes):
        ax = axes[i // 6, i % 6]
        h, _ = np.histogram(ra[rl == c], bins=bins, density=True)
        ax.fill_between(centres, pooled, color=REF, lw=0)
        ax.plot(centres, h, color=SERIES, lw=2)
        ax.set_xscale("log")
        ax.set_title(CLASS_NAMES[int(c)], fontsize=9.5, color=INK, pad=3)
        ax.set_yticks([])
        style(ax, keep=("bottom",))
        if i // 6 < 2:
            ax.set_xticklabels([])
    fig.suptitle("Region-area distribution per material, after watershed "
                 "separation", fontsize=15, color=INK, weight="bold",
                 x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.923,
             "Purple = that material. Grey = all materials pooled.\n"
             "The regions "
             "counted are displaced MARKER DOTS, not grains - Gravel and Rock "
             "separate because they load the gel differently, not by grain size.",
             fontsize=10, color=INK2)
    fig.text(0.5, 0.035, "region area (pixels, log scale)", fontsize=9.5,
             color=INK2, ha="center")
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.04, right=0.98,
                        hspace=0.42, wspace=0.15)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
def fig_iqr(d, out):
    """All classes on one axis: median + IQR for area, volume and spacing."""
    lab = d["press_label"]
    classes = sorted(set(lab.tolist()))
    panels = [("press_area_med", "median region area (px)"),
              ("press_vol_med", "median region volume"),
              ("press_spacing", "nearest-neighbour spacing (px)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4), facecolor=SURFACE)
    order = None
    for k, (key, title) in enumerate(panels):
        ax = axes[k]
        v = np.nan_to_num(d[key], nan=np.nanmedian(d[key]))
        stats = [(np.median(v[lab == c]),
                  np.percentile(v[lab == c], 25),
                  np.percentile(v[lab == c], 75), c) for c in classes]
        if order is None:
            order = [s[3] for s in sorted(stats)]
        stats = {s[3]: s for s in stats}
        y = np.arange(len(order))
        med = [stats[c][0] for c in order]
        lo = [stats[c][1] for c in order]
        hi = [stats[c][2] for c in order]
        ax.hlines(y, lo, hi, color=REF, lw=7, zorder=1)
        ax.plot(med, y, "o", ms=8, color=SERIES, zorder=3, lw=0)
        ax.set_yticks(y)
        ax.set_yticklabels([CLASS_NAMES[int(c)] for c in order] if k == 0 else [],
                           fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
        ax.invert_yaxis()
        style(ax, keep=("bottom",))
        ax.grid(axis="x", color=REF, lw=0.6, alpha=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Median and interquartile range per material - the boxes "
                 "overlap almost completely", fontsize=15, color=INK,
                 weight="bold", x=0.035, ha="left", y=0.975)
    fig.text(0.035, 0.923,
             "Dot = class median, bar = interquartile range across presses, "
             "measured on displaced marker regions.\n"
             "Gravel and Rock separate at "
             "the extremes; the fourteen classes between them do not. "
             "Classifying on these: 0.154 vs a 0.070 baseline.",
             fontsize=10, color=INK2)
    fig.subplots_adjust(top=0.85, bottom=0.06, left=0.12, right=0.98, wspace=0.12)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
def fig_markers(out):
    """The discovery: the segmented dots are the gel's printed markers."""
    from scipy import ndimage as ndi
    rows = pd.read_csv("../features_dinop9/rows.csv")
    # bg_refs covers more sessions than rows.csv; take the first that overlaps
    for f in sorted(BG_REF_DIR.glob("*.png")):
        sub = rows[rows.gelsight_frame_path.str.contains(f.stem, regex=False)]
        if len(sub):
            break
    ref = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
    img = np.asarray(Image.open(sub.iloc[0].gelsight_frame_path).convert("L"),
                     dtype=np.float32)

    m = ref < np.percentile(ref, 12)
    m = ndi.binary_opening(m, np.ones((3, 3)))
    lab, n = ndi.label(m)
    cen = np.array(ndi.center_of_mass(m, lab, np.arange(1, n + 1)))

    F = np.abs(np.fft.fftshift(np.fft.fft2(ref - ref.mean())))
    cy, cx = np.array(F.shape) // 2
    F[cy - 3:cy + 4, cx - 3:cx + 4] = 0

    fig, ax = plt.subplots(1, 4, figsize=(15, 5.0), facecolor=SURFACE)
    ax[0].imshow(ref, cmap="gray")
    ax[0].set_title("Undeformed gel (no contact)", fontsize=11, color=INK)
    ax[1].imshow(ref, cmap="gray")
    ax[1].plot(cen[:, 1], cen[:, 0], "o", mfc="none", mec="#eb6834", mew=1.6,
               ms=11, lw=0)
    ax[1].set_title("%d printed markers detected" % n, fontsize=11, color=INK)
    ax[2].imshow(np.abs(img - ref), cmap="magma")
    ax[2].set_title("A press, minus the reference", fontsize=11, color=INK)
    k = 90
    sl = np.log1p(F[cy - k:cy + k, cx - k:cx + k])
    ax[3].imshow(sl, cmap="magma", vmin=np.percentile(sl, 55),
                 vmax=np.percentile(sl, 99.8))
    ax[3].set_title("Fourier transform: lattice at 58 px", fontsize=11, color=INK)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(REF)
    fig.suptitle("The gel carries a printed marker lattice, and subtraction "
                 "cannot remove it", fontsize=15, color=INK, weight="bold",
                 x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.855,
             "Markers MOVE when the gel deforms, so subtracting a static "
             "reference leaves a ring at every marker (panel 3).\n"
             "Those rings - "
             "not the material - are what the segmentation was counting.",
             fontsize=10, color=INK2)
    fig.subplots_adjust(top=0.70, bottom=0.03, left=0.02, right=0.98, wspace=0.06)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=Path("../results/segments.npz"))
    ap.add_argument("--outdir", type=Path, default=Path("../results"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.stats)
    fig_markers(args.outdir / "seg_markers.png")
    fig_pipeline(args.outdir / "seg_pipeline.png")
    fig_heatmap(d, args.outdir / "seg_heatmap.png")
    fig_distributions(d, args.outdir / "seg_distributions.png")
    fig_iqr(d, args.outdir / "seg_iqr.png")


if __name__ == "__main__":
    main()
