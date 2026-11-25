#!/usr/bin/env bash
# Frozen frame-level probe, run STRICTLY AFTER the adapted chain finishes.
#
# The first attempt ran it concurrently with extraction and both suffered: the
# sklearn probe's BLAS took all 16 cores and starved the DirectML data loader,
# dropping extraction from 64 frames/s to 6.1. Measured, not guessed: fold 0's
# raw view did 55,287 frames in 14m22s alone, while the background-subtracted
# view managed 9,184 frames in 25 min against the probe. The Oct 8 run shows
# background subtraction itself costs nothing, so contention was the whole gap.
#
# Waits on the driver's own done line rather than on the adapted json, which is
# rewritten after every fold and so appears long before all five are in.
set -u
PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
cd "$(dirname "$0")" || exit 1
until grep -q "^=== done ::" ../results/extract_frames_grouped.log 2>/dev/null; do
  sleep 60
done
echo "=== adapted chain finished, starting frozen probe :: $(date '+%H:%M:%S') ==="
"$PY" -u frame_probe_grouped.py --tag frozen \
      --template ../features_dinop9_patch ../features_patch_bg \
      > ../results/frame_probe_grouped_frozen.log 2>&1
echo "  exit $? :: $(date '+%H:%M:%S')"
tail -14 ../results/frame_probe_grouped_frozen.log
