#!/usr/bin/env bash
# Frame-level probe on the PRESS-LEVEL RANDOM split, 5 folds.
# Frozen first: its features are encoder-independent and already cached, so all
# five folds are valid with no extraction. Adapted can only do fold 0, because
# features_ftf_* came from ft_ckpt_leaky10/fold0_best.pt alone; folds 1-4 would
# need their own per-frame extraction.
set -u
PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
cd "$(dirname "$0")" || exit 1
echo "=== frozen, random split, 5 folds :: $(date '+%H:%M:%S') ==="
"$PY" -u frame_probe_grouped.py --tag frozen --random \
      --template ../features_dinop9_patch ../features_patch_bg \
      > ../results/frame_probe_random_frozen.log 2>&1
echo "  exit $? :: $(date '+%H:%M:%S')"
tail -10 ../results/frame_probe_random_frozen.log
echo "=== adapted, random split, fold 0 only :: $(date '+%H:%M:%S') ==="
"$PY" -u frame_probe_grouped.py --tag adapted --random --folds 0 \
      --template ../features_ftf_patch ../features_ftf_bg \
      > ../results/frame_probe_random_adapted.log 2>&1
echo "  exit $? :: $(date '+%H:%M:%S')"
tail -8 ../results/frame_probe_random_adapted.log
echo "=== done :: $(date '+%H:%M:%S') ==="
