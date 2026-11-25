#!/usr/bin/env bash
# Combine the two leaky pipelines: fine-tuned encoder + 9-frame pooling +
# raw/background-subtracted views + tuned MLP head.
#
#   frozen encoder,    9 frames, 2 views, tuned MLP  -> 0.7658
#   fine-tuned encoder, 1 frame, 1 view,  linear     -> 0.7138
#
# Neither has the other's advantages. This runs the frozen pipeline's machinery
# on the fine-tuned encoder's weights.
set -u

PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
CK="../results/ft_ckpt_leaky10/fold0_best.pt"
cd "$(dirname "$0")" || exit 1

# wait for the raw-view extraction already in flight (started on CPU before
# --device existed; ~60 min for 55k frames at ~15 img/s)
echo "=== waiting for features_ftl_patch :: $(date '+%H:%M:%S') ==="
for i in $(seq 1 180); do
  [ -f ../features_ftl_patch/touch.npy ] && break
  sleep 30
done
if [ ! -f ../features_ftl_patch/touch.npy ]; then
  echo "raw view never appeared; aborting"; exit 1
fi
echo "raw view ready :: $(date '+%H:%M:%S')"

# second view: background-subtracted, now on the GPU (~52 img/s vs ~15 on CPU)
if [ ! -f ../features_ftl_bg/touch.npy ]; then
  echo "=== bg-subtracted extraction (GPU) :: $(date '+%H:%M:%S') ==="
  "$PY" -u extract_features.py --backbone dinov2_vits14 --frames-per-run 9 \
        --pool-tokens patchmeanstd --embed touch --require both \
        --bg-subtract --init "$CK" --device directml \
        --out ../features_ftl_bg > ../results/extract_ftl_bg.log 2>&1
  echo "--- bg extraction exit $? :: $(date '+%H:%M:%S')"
  tail -3 ../results/extract_ftl_bg.log
fi

echo "=== scoring: fine-tuned encoder, 9 frames, both views, tuned MLP ==="
"$PY" -u score_ft_features.py \
      --features ../features_ftl_patch ../features_ftl_bg \
      --fold 0 --out ../results/ft_features_leaky.json \
      > ../results/score_ftl.log 2>&1
echo "--- scoring exit $? :: $(date '+%H:%M:%S')"
tail -12 ../results/score_ftl.log

# raw view alone, as the ablation that separates encoder gain from second view
echo "=== scoring: raw view only ==="
"$PY" -u score_ft_features.py --features ../features_ftl_patch \
      --fold 0 --out ../results/ft_features_leaky_rawonly.json \
      > ../results/score_ftl_raw.log 2>&1
tail -6 ../results/score_ftl_raw.log
echo "=== done :: $(date '+%H:%M:%S') ==="
