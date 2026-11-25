#!/usr/bin/env bash
# Per-FRAME features from the SESSION-GROUPED fine-tune, one encoder per fold.
#
# extract_frames_ft.sh built features_ftf_* from ft_ckpt_leaky10/fold0_best.pt,
# the encoder fine-tuned on the RANDOM split. Probing those features on a
# video-disjoint split is not leak-free: that encoder trained on presses from
# all 124 sessions, the held-out ones included. This rebuilds the same features
# from ft_ckpt_clean, where encoder k never saw fold k's sessions, so
# frame_probe_grouped.py can score fold k with encoder k.
#
# Resumable: skips any fold/view already on disk. Roughly 20 min per view at the
# measured ~52 img/s inference rate, so about 3.5 hours for all ten.
set -u
PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
cd "$(dirname "$0")" || exit 1

for k in 0 1 2 3 4; do
  CK="../results/ft_ckpt_clean/fold${k}_best.pt"
  if [ ! -f "$CK" ]; then echo "fold $k: no checkpoint, skipping"; continue; fi
  for view in patch bg; do
    OUT="../features_ftfg${k}_${view}"
    if [ -f "$OUT/touch_frames.npy" ]; then
      echo "fold $k $view: cached"; continue
    fi
    EXTRA=""
    [ "$view" = "bg" ] && EXTRA="--bg-subtract"
    echo "=== per-frame grouped fold $k, $view :: $(date '+%H:%M:%S') ==="
    "$PY" -u extract_features.py --backbone dinov2_vits14 --frames-per-run 9 \
          --pool-tokens patchmeanstd --embed touch --require both $EXTRA \
          --save-frames --init "$CK" --device directml --out "$OUT" \
          > "../results/extract_ftfg${k}_${view}.log" 2>&1
    echo "  exit $? :: $(date '+%H:%M:%S')"
    tail -2 "../results/extract_ftfg${k}_${view}.log"
  done
done

echo "=== scoring adapted, all folds :: $(date '+%H:%M:%S') ==="
"$PY" -u frame_probe_grouped.py --tag adapted \
      --template '../features_ftfg{k}_patch' '../features_ftfg{k}_bg' \
      > ../results/frame_probe_grouped_adapted.log 2>&1
echo "  exit $? :: $(date '+%H:%M:%S')"
tail -12 ../results/frame_probe_grouped_adapted.log
echo "=== done :: $(date '+%H:%M:%S') ==="
