"""Visualise the train / val / test split.

Three panels, because the split has three things worth seeing and they need
different forms:

  A  presses by split      - pie. Part-to-whole at a glance, 3 segments. Val and
                             test are within 0.4 points of each other, which a pie
                             reads poorly, so both carry direct labels; that also
                             discharges the light-mode contrast relief rule.
  B  sessions by split     - stacked bar. The SESSION is the unit actually being
                             split; presses only follow. Stacked bar because
                             part-to-whole reads precisely here.
  C  per class             - stacked bars, one per material. The real question a
                             split visualisation should answer: did stratification
                             survive the grouping constraint, or did some class end
                             up with nothing to test on?

Series colours are categorical slots 1-3 of the validated reference palette,
re-validated against THIS project's surfaces (#F3F2F6 / #131219):
all checks pass in both modes, all-pairs.

Usage:
    python plot_split.py --out ../results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPLITS = ["train", "val", "test"]

THEME = {
    "light": dict(surface="#F3F2F6", ink="#17161E", muted="#676375",
                  grid="#DFDDE7", series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark":  dict(surface="#131219", ink="#ECEBF1", muted="#9C98AB",
                  grid="#2E2C39", series=["#3987e5", "#d95926", "#199e70"]),
}
GAP = 2.0          # surface-coloured gap between stacked segments, per mark specs


def style(fig, axes, t):
    fig.patch.set_facecolor(t["surface"])
    for ax in axes:
        ax.set_facecolor(t["surface"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=t["muted"], length=0, labelsize=8)


def panel_pie(ax, data, t):
    vals = [data["presses"][s] for s in SPLITS]
    total = sum(vals)
    wedges, _ = ax.pie(
        vals, radius=0.86, colors=t["series"], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor=t["surface"], linewidth=GAP))
    # Direct labels: val and test differ by 0.4 points, unreadable from angle alone.
    for w, s, v in zip(wedges, SPLITS, vals):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        r = 1.34
        ax.text(r * np.cos(ang), r * np.sin(ang),
                f"{s}\n{v:,}  ({v / total:.1%})",
                ha="center", va="center", fontsize=8, color=t["ink"], linespacing=1.45)
    ax.text(0, 0, f"{total:,}\npresses", ha="center", va="center",
            fontsize=11, color=t["ink"], linespacing=1.3, weight="bold")
    ax.set_title("A · presses by split", fontsize=9.5, color=t["ink"],
                 loc="left", pad=14)


def panel_sessions(ax, data, t):
    vals = [data["sessions"][s] for s in SPLITS]
    total = sum(vals)
    left = 0.0
    for v, s, c in zip(vals, SPLITS, t["series"]):
        ax.barh(0, v, left=left, height=0.42, color=c,
                edgecolor=t["surface"], linewidth=GAP)
        ax.text(left + v / 2, 0, f"{v}", ha="center", va="center",
                fontsize=9, color="white", weight="bold")
        ax.text(left + v / 2, -0.36, f"{s}\n{v / total:.1%}", ha="center",
                va="top", fontsize=8, color=t["muted"], linespacing=1.4)
        left += v
    ax.set_xlim(0, total)
    ax.set_ylim(-0.95, 0.45)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"B · sessions by split — the unit actually split ({total} total)",
                 fontsize=9.5, color=t["ink"], loc="left", pad=14)


def panel_classes(ax, data, t):
    names = data["names"]
    mat = np.array([data["per_class"][s] for s in SPLITS], dtype=float)  # 3 x C
    totals = mat.sum(0)
    order = np.argsort(-totals)                    # most-supported class first
    y = np.arange(len(order))

    left = np.zeros(len(order))
    for row, s, c in zip(mat[:, order], SPLITS, t["series"]):
        ax.barh(y, row, left=left, height=0.68, color=c, label=s,
                edgecolor=t["surface"], linewidth=GAP)
        left += row

    ax.set_yticks(y, [names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("presses", fontsize=8.5, color=t["muted"])
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for i, idx in enumerate(order):               # total at the bar end
        ax.text(totals[idx] + max(totals) * 0.012, i, f"{int(totals[idx]):,}",
                va="center", fontsize=7.5, color=t["muted"])
    ax.set_xlim(0, max(totals) * 1.09)
    ax.set_title("C · per class — every class keeps presses in all three splits",
                 fontsize=9.5, color=t["ink"], loc="left", pad=10)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5, ncol=3)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])


def build(data, mode, out_path):
    t = THEME[mode]
    fig = plt.figure(figsize=(11, 8.8), dpi=130)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.55], width_ratios=[1, 1.35],
                          hspace=0.30, wspace=0.18,
                          left=0.13, right=0.97, top=0.845, bottom=0.07)
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_ses = fig.add_subplot(gs[0, 1])
    ax_cls = fig.add_subplot(gs[1, :])
    style(fig, [ax_pie, ax_ses, ax_cls], t)

    panel_pie(ax_pie, data, t)
    panel_sessions(ax_ses, data, t)
    panel_classes(ax_cls, data, t)

    fig.suptitle("Train / validation / test split — outer fold 1 of 5, grouped by session",
                 fontsize=12.5, color=t["ink"], x=0.085, ha="left", y=0.965)
    fig.text(0.085, 0.928,
             "Whole sessions are assigned to one split, so no press from a recording "
             "appears on both sides. Every press is tested exactly once across the 5 folds.",
             fontsize=8.5, color=t["muted"], ha="left")

    fig.savefig(out_path, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("../results"))
    args = ap.parse_args()
    data = json.loads((args.out / "split_composition.json").read_text())
    for mode in ("light", "dark"):
        build(data, mode, args.out / f"split_{mode}.png")


if __name__ == "__main__":
    main()
