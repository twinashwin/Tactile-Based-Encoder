"""Re-run the six experiments and dump everything the report needs as JSON.

Kept separate from run_experiments.py so the experiment script stays focused on
producing results and reporting concerns don't leak into it. Imports the split,
models and seed from run_experiments, so the numbers match it exactly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from common import CLASS_NAMES
from run_experiments import MODELS, SEED, build_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("features"))
    ap.add_argument("--out", type=Path, default=Path("results/report_data.json"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    args = ap.parse_args()

    np.random.seed(SEED)

    y = np.load(args.features / "labels.npy")
    groups = np.load(args.features / "groups.npy", allow_pickle=True)
    vision = np.load(args.features / "vision.npy")
    touch = np.load(args.features / "touch.npy")

    dropped = []
    keep = np.ones(len(y), bool)
    for c in np.unique(y):
        if len(np.unique(groups[y == c])) < 2:
            keep &= y != c
            dropped.append(CLASS_NAMES[int(c)])
    vision, touch, y, groups = vision[keep], touch[keep], y[keep], groups[keep]

    kept = sorted(np.unique(y).tolist())
    names = [CLASS_NAMES[c] for c in kept]
    y = np.array([{c: i for i, c in enumerate(kept)}[c] for c in y])

    tr, va, test = build_split(y, groups)

    report = {
        "n_examples": int(len(y)),
        "n_classes": len(kept),
        "n_sessions": int(len(np.unique(groups))),
        "dropped_classes": dropped,
        "class_names": names,
        "test_support": np.bincount(y[test], minlength=len(kept)).tolist(),
        "splits": {
            "train": {"n": int(len(tr)), "sessions": int(len(np.unique(groups[tr])))},
            "val": {"n": int(len(va)), "sessions": int(len(np.unique(groups[va])))},
            "test": {"n": int(len(test)), "sessions": int(len(np.unique(groups[test])))},
        },
        "runs": [],
    }

    feature_sets = {
        "Vision only": vision,
        "Touch only": touch,
        "Vision + Touch": np.concatenate([vision, touch], axis=1),
    }

    for fname, X in feature_sets.items():
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva, Xte = (scaler.transform(X[i]).astype(np.float32)
                         for i in (tr, va, test))
        for mname, fit in MODELS.items():
            pred = fit(Xtr, y[tr], Xva, y[va], args).predict(Xte)
            acc = accuracy_score(y[test], pred)
            f1 = f1_score(y[test], pred, average="macro", zero_division=0)
            report["runs"].append({
                "feature_set": fname,
                "model": mname,
                "dim": int(X.shape[1]),
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1), 4),
                "per_class_f1": [round(float(v), 3) for v in
                                 f1_score(y[test], pred, average=None, zero_division=0)],
                "confusion": confusion_matrix(y[test], pred,
                                              labels=range(len(kept))).tolist(),
            })
            print(f"{fname:<15} {mname:<9} acc {acc:.4f}  macro-F1 {f1:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
