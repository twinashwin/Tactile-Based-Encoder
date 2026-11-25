#!/usr/bin/env bash
# The combination that gave 0.8144 leaky, measured on the SESSION-GROUPED
# protocol -- the number that survives a reviewer who knows about TaG-NoLeak.
#
# Grouped best so far is 0.6484 (1-frame fine-tune + 9-frame test-time
# averaging). This instead feeds the fine-tuned encoder's 9-frame pooled
# raw+bg features into a tuned MLP, which on the leaky protocol was worth
# +7.46 over the 1-frame linear-head version.
#
# Encoders come from ft_ckpt_clean -- the GROUPED 5-fold fine-tune. Fold k is
# scored with encoder k, so no session the encoder trained on appears in its
# own test split. Resumable: skips any feature set already on disk.
set -u
PY="C:/dev/tactile-gpu/.venv/Scripts/python.exe"
cd "$(dirname "$0")" || exit 1

for k in 0 1 2 3 4; do
  CK="../results/ft_ckpt_clean/fold${k}_best.pt"
  if [ ! -f "$CK" ]; then echo "fold $k: no checkpoint, skipping"; continue; fi
  for view in patch bg; do
    OUT="../features_ftg${k}_${view}"
    if [ -f "$OUT/touch.npy" ]; then echo "fold $k $view: cached"; continue; fi
    EXTRA=""
    [ "$view" = "bg" ] && EXTRA="--bg-subtract"
    echo "=== grouped fold $k, $view :: $(date '+%H:%M:%S') ==="
    "$PY" -u extract_features.py --backbone dinov2_vits14 --frames-per-run 9 \
          --pool-tokens patchmeanstd --embed touch --require both $EXTRA \
          --init "$CK" --device directml --out "$OUT" \
          > "../results/extract_ftg${k}_${view}.log" 2>&1
    echo "  exit $? :: $(date '+%H:%M:%S')"
  done
  echo "=== scoring grouped fold $k ==="
  "$PY" -u score_ft_features.py --grouped \
        --features "../features_ftg${k}_patch" "../features_ftg${k}_bg" \
        --fold "$k" --out "../results/ft_features_grouped_f${k}.json" \
        > "../results/score_ftg_f${k}.log" 2>&1
  grep -E "FOLD|chosen" "../results/score_ftg_f${k}.log"
done

echo "=== grouped summary :: $(date '+%H:%M:%S') ==="
"$PY" - <<'PYEOF'
import json, glob, numpy as np
rows = []
for f in sorted(glob.glob("../results/ft_features_grouped_f*.json")):
    d = json.load(open(f))
    rows.append((d["fold"], d["accuracy"], d["macro_f1"]))
for k, a, m in sorted(rows):
    print(f"  fold {k}: acc {a:.4f}  macroF1 {m:.4f}")
if rows:
    a = np.array([r[1] for r in rows]); m = np.array([r[2] for r in rows])
    print(f"\nGROUPED {len(rows)} fold(s): accuracy {a.mean():.4f} +/- {a.std():.4f}"
          f"   macro-F1 {m.mean():.4f}")
    print("previous grouped best: 0.6484 (1-frame fine-tune + 9-frame TTA)")
    print("frozen grouped baseline: 0.6222 +/- 0.0022")
PYEOF
