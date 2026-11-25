#!/usr/bin/env bash
# Re-run fusion with the corrected late-fusion weight selection, both protocols.
# vision / touch / concat / oracle are unchanged by the fix and should reproduce
# the earlier values exactly, which doubles as a consistency check. Only the
# late_fusion row can move.
# Threads are capped so this does not starve the DirectML extraction's loader,
# the mistake that cut extraction from 64 frames/s to 6.1 earlier today.
set -u
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
cd "$(dirname "$0")" || exit 1
for k in 0 1 2 3 4; do
  echo "=== grouped fold $k :: $(date '+%H:%M:%S') ==="
  "$PY" -u fuse_ft.py --grouped --fold "$k" --vision ../features_dinop9 \
        --touch "../features_ftg${k}_patch" "../features_ftg${k}_bg" \
        --out "../results/fuse_grouped_v2_f${k}.json" \
        > "../results/fuse_grouped_v2_f${k}.log" 2>&1
  grep -E "weight chosen|^concat|^vision|^touch|^late_fusion|^oracle" \
       "../results/fuse_grouped_v2_f${k}.log"
done
for k in 0 1 2 3 4; do
  SUF=""; [ "$k" -gt 0 ] && SUF="$k"
  echo "=== random fold $k :: $(date '+%H:%M:%S') ==="
  "$PY" -u fuse_ft.py --fold "$k" --vision ../features_dinop9 \
        --touch "../features_ftl${SUF}_patch" "../features_ftl${SUF}_bg" \
        --out "../results/fuse_random_v2_f${k}.json" \
        > "../results/fuse_random_v2_f${k}.log" 2>&1
  grep -E "weight chosen|^concat|^vision|^touch|^late_fusion|^oracle" \
       "../results/fuse_random_v2_f${k}.log"
done
echo "=== done :: $(date '+%H:%M:%S') ==="
