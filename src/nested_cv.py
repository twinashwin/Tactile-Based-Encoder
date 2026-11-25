"""Nested cross-validation over the full experiment matrix.

PROTOCOL (following the Touch and Go literature)
------------------------------------------------
Split       Grouped by recording video/session -- what Seeing Through Touch
            (arXiv 2604.11579) does, "ensuring no video overlap between the train
            and test sets". Random frame-level splits leak the same press across
            the boundary and inflate accuracy.

Nesting     OUTER 5-fold StratifiedGroupKFold = an 80/20 train/test split, repeated
            so every press is tested exactly once.
            INNER 4-fold StratifiedGroupKFold *inside each outer-train fold*, used
            only to choose hyperparameters. Every candidate is scored on all four
            inner folds and averaged.
            The chosen setting is then refit on the FULL outer-train fold and
            evaluated once on the outer test fold, which nothing in selection ever
            touched. That separation is the entire point of nesting: tuning and
            evaluation never share data.

Classes     18 classes: the 20 material labels minus Natural Fabric (1 press in 1
            session, unsplittable group-wise) and minus Others, which Seeing
            Through Touch also excludes as a catch-all with no coherent appearance.

Normalise   StandardScaler fit on the training fold only, then L2 row
            normalisation -- the convention in linear-probe evaluation.

MODELS
------
  Logistic    multinomial logistic regression (L-BFGS), tuned over C
  MLP         256->128 ReLU (torch), tuned over weight decay
  Gated       learned per-dimension gate mixing vision and touch   [fusion only]
  CrossAttn   2-token multi-head cross-modal attention             [fusion only]

Torch heads early-stop on the inner validation fold; the epoch that won is carried
to the final refit as a fixed budget, so the refit uses all outer-train data
without ever validating on data it trained on.

Usage:
    python nested_cv.py --features ../features_dinop9 --out ../results
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from common import CLASS_NAMES
from fusion import (CrossAttnFusion, GatedFusion, MlpHead, predict,
                    train_torch_head)

SEED = 42
OUTER_FOLDS = 5            # 5 folds == 80/20 train/test per fold
INNER_FOLDS = 4
EXCLUDE = {"Natural Fabric", "Others"}

GRIDS = {
    "Logistic":  [{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)],
    "MLP":       [{"weight_decay": w} for w in (1e-4, 1e-2)],
    "Gated":     [{"weight_decay": w} for w in (1e-4, 1e-2)],
    "CrossAttn": [{"weight_decay": w} for w in (1e-4, 1e-2)],
}
TORCH_KINDS = {"MLP": MlpHead, "Gated": GatedFusion,
               "CrossAttn": CrossAttnFusion}


def normalise(train_X, other_Xs, l2=True):
    """Standardise on TRAIN statistics only, then L2-normalise rows."""
    scaler = StandardScaler().fit(train_X)
    out = [scaler.transform(X).astype(np.float32) for X in [train_X, *other_Xs]]
    if l2:
        out = [X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
               for X in out]
    return out


def train_one(kind, params, Xtr, ytr, d_v, n_classes, Xva=None, yva=None,
              epochs=None):
    """Train a single model. Returns (model, val_f1, best_epoch).

    Xva=None means a fixed-budget refit with no validation data involved.
    """
    if kind == "Logistic":
        m = LogisticRegression(max_iter=2000, C=params["C"],
                               class_weight="balanced").fit(Xtr, ytr)
    else:
        cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)
        # MlpHead consumes the whole vector; the fusion heads split it in half,
        # so they take the PER-MODALITY width instead.
        d_head = Xtr.shape[1] if kind == "MLP" else d_v
        model = TORCH_KINDS[kind](d_head, n_classes)
        return train_torch_head(model, Xtr, ytr, Xva, yva,
                                epochs=epochs or 200,
                                weight_decay=params["weight_decay"],
                                class_weight=cw, seed=SEED)

    if Xva is None:
        return m, float("nan"), None
    f1 = f1_score(yva, m.predict(Xva), average="macro", zero_division=0)
    return m, f1, None


def infer(model, X):
    return predict(model, X) if hasattr(model, "forward") else model.predict(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--out", type=Path, default=Path("../results"))
    ap.add_argument("--no-l2", action="store_true")
    args = ap.parse_args()
    l2 = not args.no_l2
    np.random.seed(SEED)

    y_raw = np.load(args.features / "labels.npy")
    groups = np.load(args.features / "groups.npy", allow_pickle=True)
    V = np.load(args.features / "vision.npy")
    T = np.load(args.features / "touch.npy")

    keep = np.array([CLASS_NAMES[int(c)] not in EXCLUDE for c in y_raw])
    for c in np.unique(y_raw):
        if len(np.unique(groups[y_raw == c])) < 2:
            keep &= y_raw != c
    V, T, y_raw, groups = V[keep], T[keep], y_raw[keep], groups[keep]

    kept = sorted(np.unique(y_raw).tolist())
    names = [CLASS_NAMES[c] for c in kept]
    y = np.array([{c: i for i, c in enumerate(kept)}[c] for c in y_raw])
    d_v, n_classes = V.shape[1], len(names)
    print(f"{len(y):,} presses | {n_classes} classes | "
          f"{len(np.unique(groups)):,} session groups | "
          f"excluded: {', '.join(sorted(EXCLUDE))}")
    print(f"outer {OUTER_FOLDS}-fold (80/20) x inner {INNER_FOLDS}-fold | "
          f"normalise: StandardScaler(train){' + L2' if l2 else ''}\n")

    EXPERIMENTS = [
        ("Vision only",    "Logistic",  V),
        ("Vision only",    "MLP",       V),
        ("Touch only",     "Logistic",  T),
        ("Touch only",     "MLP",       T),
        ("Vision + Touch", "Logistic",  np.hstack([V, T])),
        ("Vision + Touch", "MLP",       np.hstack([V, T])),
        ("Vision + Touch", "Gated",     np.hstack([V, T])),
        ("Vision + Touch", "CrossAttn", np.hstack([V, T])),
    ]

    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                 random_state=SEED)
    outer_splits = list(outer.split(np.zeros(len(y)), y, groups))

    results = []
    for fs, kind, X in EXPERIMENTS:
        fold_acc, fold_f1, chosen = [], [], []
        pooled = np.full(len(y), -1)

        for tr_idx, te_idx in outer_splits:
            assert not (set(groups[tr_idx]) & set(groups[te_idx]))

            # ---------- INNER: score every candidate on all inner folds ----------
            inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                         random_state=SEED)
            inner_splits = list(inner.split(np.zeros(len(tr_idx)), y[tr_idx],
                                            groups[tr_idx]))
            scores = {i: [] for i in range(len(GRIDS[kind]))}
            epochs_seen = {i: [] for i in range(len(GRIDS[kind]))}

            for ia, ib in inner_splits:
                a, b = tr_idx[ia], tr_idx[ib]
                Xa, Xb = normalise(X[a], [X[b]], l2)
                for i, params in enumerate(GRIDS[kind]):
                    _, f1, ep = train_one(kind, params, Xa, y[a], d_v, n_classes,
                                          Xva=Xb, yva=y[b])
                    scores[i].append(f1)
                    if ep:
                        epochs_seen[i].append(ep)

            bi = max(scores, key=lambda i: float(np.mean(scores[i])))
            params = GRIDS[kind][bi]
            budget = (int(np.median(epochs_seen[bi])) if epochs_seen[bi] else None)

            # ---------- OUTER: refit on ALL outer-train, score once on test ------
            Xtr, Xte = normalise(X[tr_idx], [X[te_idx]], l2)
            model, _, _ = train_one(kind, params, Xtr, y[tr_idx], d_v, n_classes,
                                    epochs=budget)
            pred = infer(model, Xte)
            pooled[te_idx] = pred
            fold_acc.append(accuracy_score(y[te_idx], pred))
            fold_f1.append(f1_score(y[te_idx], pred, average="macro",
                                    zero_division=0))
            chosen.append({**params, **({"epochs": budget} if budget else {})})

        cov = pooled >= 0
        acc = accuracy_score(y[cov], pooled[cov])
        f1 = f1_score(y[cov], pooled[cov], average="macro", zero_division=0)
        results.append({
            "feature_set": fs, "model": kind, "dim": int(X.shape[1]),
            "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
            "accuracy_std": round(float(np.std(fold_acc)), 4),
            "macro_f1_std": round(float(np.std(fold_f1)), 4),
            "fold_accuracy": [round(float(v), 4) for v in fold_acc],
            "fold_macro_f1": [round(float(v), 4) for v in fold_f1],
            "per_class_f1": [round(float(v), 3) for v in
                             f1_score(y[cov], pooled[cov], average=None,
                                      zero_division=0)],
            "confusion": confusion_matrix(y[cov], pooled[cov],
                                          labels=range(n_classes)).tolist(),
            "hyperparams": chosen,
        })
        print(f"{fs:<15} {kind:<10} acc {acc:.4f} +/-{np.std(fold_acc):.3f}   "
              f"macro-F1 {f1:.4f} +/-{np.std(fold_f1):.3f}   {chosen[0]}",
              flush=True)

    meta = {
        "protocol": f"nested CV: outer {OUTER_FOLDS}-fold (80/20) grouped by "
                    f"session, inner {INNER_FOLDS}-fold for hyperparameters",
        "n_examples": int(len(y)), "n_classes": n_classes,
        "n_groups": int(len(np.unique(groups))),
        "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "normalisation": "StandardScaler(train-only)" + (" + L2" if l2 else ""),
        "class_weight": "balanced",
        "excluded_classes": sorted(EXCLUDE),
        "class_names": names,
        "support": np.bincount(y, minlength=n_classes).tolist(),
        "runs": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "nested_cv.json").write_text(json.dumps(meta))
    print(f"\nwrote {args.out / 'nested_cv.json'}")


if __name__ == "__main__":
    main()
