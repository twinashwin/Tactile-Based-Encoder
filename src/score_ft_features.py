"""Score 9-frame pooled features from a FINE-TUNED encoder with a tuned head.

WHY
---
On the leaky protocol the two pipelines each win on different things and had
never been combined:

    frozen DINOv2  + 9 frames + raw&bg views + tuned MLP   0.7658
    fine-tuned DINOv2 + 1 frame + raw view + linear head   0.7138

The frozen pipeline's pooling, second view and tuned head are worth more than
fine-tuning's adaptation -- but there is no reason to choose. This scores the
fine-tuned encoder's features through the frozen pipeline's machinery.

PROTOCOL HONESTY
----------------
The encoder was fine-tuned on leaky fold 0's training split, so it has seen
presses that land in folds 1-4's test sets. Scoring all five folds would add
leakage beyond the published protocol, so this reports FOLD 0 ONLY. That is
defensible here because the measured leaky fold std is 0.0081 (against 0.0296
grouped) -- one leaky fold is within a fifth of a point of the five-fold mean.

Usage:
    python score_ft_features.py --features ../features_ftl_patch ../features_ftl_bg
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

from nested_cv import OUTER_FOLDS, SEED
from touch_heads import preprocess

warnings.filterwarnings("ignore", category=ConvergenceWarning)

GRID = [{"pca": None, "alpha": 1e-4}, {"pca": None, "alpha": 1e-3},
        {"pca": 192, "alpha": 1e-4}, {"pca": 256, "alpha": 1e-4}]


def load(dirs):
    Xs, blocks = [], []
    y = g = None
    for d in dirs:
        t = np.load(Path(d) / "touch.npy")
        Xs.append(t)
        blocks.append(t.shape[1])
        yy = np.load(Path(d) / "labels.npy")
        gg = np.load(Path(d) / "groups.npy", allow_pickle=True)
        if y is None:
            y, g = yy, gg
        else:
            assert np.array_equal(y, yy), f"{d} label order differs"
    return np.concatenate(Xs, axis=1), y, g, blocks


def fit_score(Xtr, ytr, Xte, yte, cfg, blocks):
    A, B = preprocess(Xtr, [Xte], pca=cfg["pca"], whiten=False,
                      blocks=blocks, seed=SEED)
    clf = MLPClassifier(hidden_layer_sizes=(512,), alpha=cfg["alpha"],
                        max_iter=400, random_state=SEED, early_stopping=False)
    p = clf.fit(A, ytr).predict(B)
    return accuracy_score(yte, p), f1_score(yte, p, average="macro",
                                            zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--grouped", action="store_true",
                    help="Session-grouped split instead of the leaky one")
    ap.add_argument("--out", type=Path, default=Path("../results/ft_features.json"))
    args = ap.parse_args()

    X, y_raw, groups, blocks = load(args.features)
    # touch_heads.py drops any class present in fewer than two recording
    # sessions, which takes the set from 6,143/19 to 6,115/18. Replicate it or
    # this number is not comparable to the 0.7658 it is meant to beat.
    keep = np.array([len(np.unique(groups[y_raw == c])) >= 2 for c in y_raw])
    dropped = sorted(set(y_raw[~keep].tolist()))
    if dropped:
        print(f"dropped {len(y_raw)-keep.sum()} presses in {len(dropped)} "
              f"single-session class(es) to match touch_heads.py")
    X, y_raw, groups = X[keep], y_raw[keep], groups[keep]
    cls = sorted(set(y_raw.tolist()))
    m = {c: i for i, c in enumerate(cls)}
    y = np.array([m[c] for c in y_raw])
    split_groups = groups if args.grouped else np.arange(len(y))
    print(f"{X.shape[0]:,} presses | {X.shape[1]} dims (blocks {blocks}) | "
          f"{len(cls)} classes | {'GROUPED' if args.grouped else 'LEAKY'} split")

    cv = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    tr, te = list(cv.split(np.zeros(len(y)), y, split_groups))[args.fold]
    print(f"fold {args.fold}: train {len(tr):,} test {len(te):,}")

    # pick the config on inner folds of the training split only
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
    isplits = list(inner.split(np.zeros(len(tr)), y[tr], split_groups[tr]))
    best, best_acc = None, -1.0
    for cfg in GRID:
        accs = []
        for i_fit, i_val in isplits[:2]:          # two inner folds is enough here
            a, _ = fit_score(X[tr[i_fit]], y[tr[i_fit]],
                             X[tr[i_val]], y[tr[i_val]], cfg, blocks)
            accs.append(a)
        mu = float(np.mean(accs))
        print(f"  inner  pca={str(cfg['pca']):<5} alpha={cfg['alpha']:<7} "
              f"acc {mu:.4f}", flush=True)
        if mu > best_acc:
            best, best_acc = cfg, mu
    print(f"chosen: {best}  (inner acc {best_acc:.4f})")

    acc, f1 = fit_score(X[tr], y[tr], X[te], y[te], best, blocks)
    print(f"\nFOLD {args.fold} TEST  accuracy {acc:.4f}  macro-F1 {f1:.4f}")
    print("references on this protocol:")
    print("  frozen DINOv2 + tuned MLP      0.7658")
    print("  fine-tuned, 1 frame, linear    0.7138")
    print("  TaCo-L (published, touch)      0.7512")
    print("  ViTaPEs (published, vis+touch) 0.8010")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"features": [str(d) for d in args.features], "fold": args.fold,
         "protocol": "grouped" if args.grouped else "leaky",
         "config": best, "accuracy": acc, "macro_f1": f1}, indent=1))


if __name__ == "__main__":
    main()
