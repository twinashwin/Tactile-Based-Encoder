#!/usr/bin/env bash
# Per-FRAME features from the fine-tuned encoder, for a frame-level linear probe.
#
# WHY: the published comparison table (STT supplementary, Table 6) reports
# frame-level LINEAR PROBE accuracy -- a single linear layer on frozen features,
# one decision per frame. Our 81.85% uses a tuned MLP over nine pooled frames.
# Both differences favour us, so the leaderboard comparison measures the readout
# as much as the encoder. These features let us report one row measured the same
# way on both sides.
set -u
PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
CK="../results/ft_ckpt_leaky10/fold0_best.pt"
cd "$(dirname "$0")" || exit 1
for view in patch bg; do
  OUT="../features_ftf_${view}"
  [ -f "$OUT/touch_frames.npy" ] && { echo "$view: cached"; continue; }
  EXTRA=""; [ "$view" = "bg" ] && EXTRA="--bg-subtract"
  echo "=== per-frame extraction, $view :: $(date '+%H:%M:%S') ==="
  "$PY" -u extract_features.py --backbone dinov2_vits14 --frames-per-run 9 \
        --pool-tokens patchmeanstd --embed touch --require both $EXTRA \
        --save-frames --init "$CK" --device directml --out "$OUT" \
        > "../results/extract_ftf_${view}.log" 2>&1
  echo "  exit $? :: $(date '+%H:%M:%S')"
done
echo "=== done :: $(date '+%H:%M:%S') ==="
