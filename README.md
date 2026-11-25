# A GelSight-specific tactile encoder for material classification

Code for adapting a DINOv2 ViT-S/14 encoder to GelSight imprints and measuring
what that adaptation is worth, on the Touch and Go dataset (Yang et al., 2022).

The question the repository answers: published tactile-only results on this
dataset freeze their encoder and train only a linear layer on material labels.
GelSight frames look nothing like the natural images such encoders were
pretrained on, since the colour of a pixel encodes the local slope of the gel
rather than the colour of the object. Unfreezing the encoder and fine-tuning it
on gel imprints is worth **+9.35 ± 2.03 points** at frame level under a
session-disjoint split.

## Results

All figures are the mean over five outer folds of a session-grouped nested
cross-validation, with the standard deviation across folds, unless the fold
count says otherwise. 6,142 presses, 19 material classes, 124 recording
sessions.

### Tactile only

| Readout | Encoder | Session split | Random split |
|---|---|---|---|
| Frame-level linear probe | frozen | 57.28 ± 4.85% | 70.13 ± 0.48% |
| Frame-level linear probe | adapted | **66.63 ± 2.99%** | 75.46% (1 fold) |
| Press-level MLP | frozen | 62.22 ± 0.22% (3 seeds) | — |
| Press-level MLP | adapted | **69.42 ± 3.68%** | 81.85 ± 0.90% |

Adapting the encoder is worth +9.35 ± 2.03 at frame level and +7.20 at press
level, positive on every fold in both cases.

### Visual and visuo-tactile

One decision per press, five folds.

| Configuration | Session split | Macro-F1 | Random split | Macro-F1 |
|---|---|---|---|---|
| Visual only | 85.04 ± 1.82% | 0.723 | 93.08 ± 0.90% | 0.884 |
| Tactile only | 70.15 ± 2.75% | 0.493 | 81.78 ± 1.00% | 0.679 |
| Concatenation | **88.25 ± 1.83%** | 0.749 | 94.16 ± 0.41% | 0.877 |
| Late fusion (w = 0.4–0.5) | 85.14 ± 2.12% | 0.690 | 93.62 ± 0.98% | 0.888 |
| Oracle ceiling | 91.42 ± 2.02% | n/a | 96.17 ± 0.73% | n/a |

Concatenation beats vision alone by +3.21 ± 1.11 on the session split and
+1.07 ± 0.67 on the random split, positive on all five folds in both. Touch
contributes three times as much once the split stops letting vision recognise
individual presses.

## The two splits, and why both are reported

**Session-grouped** is the honest protocol. Whole recording sessions are
assigned to one side, so no session contributes to both training and test. Every
claim above rests on this split.

**Press-level random** keeps each press intact but scatters presses across
sessions. It exists only to quantify leakage, and it reproduces the measured
leakage structure of the original Touch and Go split, where Gungor et al. (2025)
found the nearest training frame to a test frame sat 85 frames away, roughly
three presses, in the same video.

A third, looser option is deliberately *not* used: splitting the nine frames of
a press independently. That inflates frame-level accuracy by about 11.6 points
over the press-level random split and is heavier than anything in the published
protocols.

Per-fold accuracy carries a session-clustered standard error of roughly ±3 to
±4 points, because the effective sample size per fold is about 12 recording
sessions rather than 1,200 presses. Unpaired margins smaller than that are not
resolved by this cross-validation. The paired within-fold comparisons, such as
frozen against adapted and vision against concatenation, are unaffected, since
session-sampling noise cancels.

## Pipeline

```
build_index.py          index the dataset, drop "Inconclusive" labels
make_bg_refs.py         per-video background reference frames
finetune_dino.py        adapt the encoder, one per outer fold
extract_features.py     per-press or per-frame descriptors from an encoder
score_ft_features.py    press-level MLP, tuned per fold
frame_probe_grouped.py  frame-level linear probe, encoder k scores fold k
fuse_ft.py              vision, touch, concatenation, late fusion, oracle
fig_nested_cv.py        the cross-validation schematic
```

`nested_cv.py` holds the protocol constants (`OUTER_FOLDS`, `SEED`,
`EXCLUDE`) that every script imports, and `touch_heads.py` holds the shared
feature preprocessing. The `*.sh` drivers chain these per fold and are
resumable, skipping any feature set already on disk.

Reproducing the headline numbers, in order:

```bash
cd src
bash grouped_all_folds.sh        # press-level, session split, 5 folds
bash extract_frames_grouped.sh   # per-frame features + frame-level probe
bash run_fuse_v2.sh              # fusion, both splits, 5 folds
```

## Setup

The dataset is not included. Download Touch and Go from
https://touch-and-go.github.io and set the paths at the top of `src/common.py`.
Several scripts also carry a hardcoded interpreter path; they expect
`C:/dev/tactile-gpu/.venv/Scripts/python.exe` and will need editing.

```
torch==2.4.1
torch-directml==0.2.5.dev240914
numpy==2.5.3
scikit-learn==1.9.1
pandas==3.0.6
pillow==12.3.0
matplotlib
```

Inference ran on an AMD Radeon RX 6750 XT through `torch-directml`, not CUDA, at
about 55 frames per second for feature extraction and roughly 7 images per
second for fine-tuning. On NVIDIA hardware, drop `torch-directml` and pass
`--device cuda` instead of `--device directml`.

One practical note from building this: the scikit-learn probes use all available
cores through BLAS, and running one alongside a DirectML extraction starves the
loader badly enough to cut throughput from 55 frames per second to 6. Cap
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` if the two have to overlap.

## What is not here

Fine-tuned checkpoints (5.4 GB) and cached feature arrays (several GB) are
excluded. Every script that consumes them will regenerate them. The
`results/*.json` files are included and contain every number reported above.

## References

Gungor, C., Eppinger, D., & Kovashka, A. (2025). *Towards generalization of
tactile image generation: Reference-free evaluation in a leakage-free setting*
(arXiv:2503.06860).

Oquab, M., et al. (2023). *DINOv2: Learning robust visual features without
supervision* (arXiv:2304.07193).

Yang, F., Ma, C., Zhang, J., Zhu, J., Yuan, W., & Owens, A. (2022). Touch and
Go: Learning from human-collected vision and touch. *Advances in Neural
Information Processing Systems 35, Datasets and Benchmarks Track*.
