#!/usr/bin/env bash
# Frozen frame-level probe, all five session-grouped folds.
# Waits for the fusion chain to finish first so the two sklearn jobs do not
# contend for cores while the DirectML extraction needs a loader thread.
# Frozen features are encoder-independent, so one fixed pair of dirs is valid
# for every fold; only the fold boundary changes.
set -u
PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
cd "$(dirname "$0")" || exit 1
until [ -f ../results/fuse_grouped_f4.json ]; do sleep 30; done
echo "=== fusion done, frozen frame probe :: $(date '+%H:%M:%S') ==="
"$PY" -u frame_probe_grouped.py --tag frozen \
      --template ../features_dinop9_patch ../features_patch_bg \
      > ../results/frame_probe_grouped_frozen.log 2>&1
echo "  exit $? :: $(date '+%H:%M:%S')"
tail -12 ../results/frame_probe_grouped_frozen.log
