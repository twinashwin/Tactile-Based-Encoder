#!/usr/bin/env bash
# Fusion on the SESSION-GROUPED split, all five folds.
# Fold k is scored with encoder k (features_ftg{k}), so no session the encoder
# trained on appears in its own test split. Fills the two gaps the results
# section flags: a visual-only number on the session split, and a fusion
# number comparable to the published visuo-tactile figures.
set -u
PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
cd "$(dirname "$0")" || exit 1
for k in 0 1 2 3 4; do
  OUT="../results/fuse_grouped_f${k}.json"
  [ -f "$OUT" ] && { echo "fold $k: cached"; continue; }
  echo "=== fusion grouped fold $k :: $(date '+%H:%M:%S') ==="
  "$PY" -u fuse_ft.py --grouped --fold "$k" \
        --vision ../features_dinop9 \
        --touch "../features_ftg${k}_patch" "../features_ftg${k}_bg" \
        --out "$OUT" > "../results/fuse_grouped_f${k}.log" 2>&1
  echo "  exit $? :: $(date '+%H:%M:%S')"
  tail -9 "../results/fuse_grouped_f${k}.log"
done
echo "=== done :: $(date '+%H:%M:%S') ==="
