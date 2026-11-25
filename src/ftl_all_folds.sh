#!/usr/bin/env bash
# Turn the fold-0 result (0.8031 leaky) into a reportable five-fold number.
#
# WHY PER-FOLD ENCODERS
# ---------------------
# Each fold's encoder was fine-tuned only on that fold's training split. Using
# one encoder for all five folds would let it score test presses it had trained
# on, which is leakage BEYOND the published protocol and would not survive
# review. So fold k is scored with encoder k, and features are re-extracted per
# fold: 4 folds x 2 views x ~18 min on the GPU.
#
# Fold 0 is already done (raw 0.7884, raw+bg 0.8031) and is skipped.
#
# Reference points on this protocol:
#   UniTouch 61.3 (touch)  TaCo-L 75.12 (touch)  ViTaPEs 80.10 (vision+touch)
#   frozen DINOv2 + full pipeline 76.58 (touch)
set -u

PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
cd "$(dirname "$0")" || exit 1

for k in 1 2 3 4; do
  CK="../results/ft_ckpt_leaky10/fold${k}_best.pt"
  if [ ! -f "$CK" ]; then echo "fold $k: no checkpoint, skipping"; continue; fi

  for view in patch bg; do
    OUT="../features_ftl${k}_${view}"
    if [ -f "$OUT/touch.npy" ]; then echo "fold $k $view: cached"; continue; fi
    EXTRA=""
    [ "$view" = "bg" ] && EXTRA="--bg-subtract"
    echo "=== fold $k, $view view :: $(date '+%H:%M:%S') ==="
    "$PY" -u extract_features.py --backbone dinov2_vits14 --frames-per-run 9 \
          --pool-tokens patchmeanstd --embed touch --require both $EXTRA \
          --init "$CK" --device directml --out "$OUT" \
          > "../results/extract_ftl${k}_${view}.log" 2>&1
    echo "  exit $? :: $(date '+%H:%M:%S')"
  done

  echo "=== scoring fold $k ==="
  "$PY" -u score_ft_features.py \
        --features "../features_ftl${k}_patch" "../features_ftl${k}_bg" \
        --fold "$k" --out "../results/ft_features_leaky_f${k}.json" \
        > "../results/score_ftl_f${k}.log" 2>&1
  grep -E "FOLD|chosen" "../results/score_ftl_f${k}.log"
done

echo "=== all folds done :: $(date '+%H:%M:%S') ==="
"$PY" - <<'PYEOF'
import json, glob, numpy as np
rows = []
for f in sorted(glob.glob("../results/ft_features_leaky*.json")):
    d = json.load(open(f))
    if "accuracy" in d:
        rows.append((d.get("fold"), d["accuracy"], d["macro_f1"], f))
rows = [r for r in rows if "rawonly" not in r[3]]
for k, a, m, f in sorted(rows):
    print(f"  fold {k}: acc {a:.4f}  macroF1 {m:.4f}")
if rows:
    a = np.array([r[1] for r in rows]); m = np.array([r[2] for r in rows])
    print(f"\nLEAKY {len(rows)} fold(s): accuracy {a.mean():.4f} +/- {a.std():.4f}"
          f"   macro-F1 {m.mean():.4f}")
    print("beats: TaCo-L 0.7512 (touch), ViTaPEs 0.8010 (vision+touch),"
          " frozen-pipeline 0.7658")
PYEOF
