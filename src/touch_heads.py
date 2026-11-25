"""Which classifier head, and which preprocessing, is best for touch-only features?

The MLP was chosen by convention, not by evidence. On ~4,900 training presses of
384-d frozen features, kernel methods and ensembles are plausible competitors --
and the tactile literature actually uses them: TaCo-L reports 75.12% on Touch and
Go with an SVM over frozen features, and the ResNet-ViT-SVM line of work classifies
frozen deep features with an SVM rather than a learned head.

PREPROCESSING IS A HYPERPARAMETER
---------------------------------
PCA dimensionality, whitening and class_weight are selected on the inner folds
alongside C and gamma, never by looking at outer-fold scores. That matters here:
exploratory runs showed PCA-whitening moving touch accuracy by ~+0.02, which is
exactly the size of effect you can manufacture by peeking. Putting the choice in
the inner loop makes the outer number honest by construction.

The grid is staged rather than crossed -- preprocessing first at a default model
setting, then model hyperparameters at the chosen preprocessing. A full cross of
7 x 2 x 25 would be 350 fits per inner fold for the SVM alone. Staging is
coordinate descent on the inner objective, which is still selection-on-train.

SELECTION METRIC
----------------
--select-on decides BOTH what the inner folds optimise and which head is declared
the winner, so the thing being tuned is always the thing being reported. The
previous version selected on macro-F1 and was read for accuracy.

Usage:
    python touch_heads.py --features ../features_dinop9 --out ../results
    python touch_heads.py --features ../features_dinop9 ../features_dinobp9 \
        --select-on accuracy --seed 7
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from common import CLASS_NAMES
from fusion import MlpHead, predict, train_torch_head
from nested_cv import EXCLUDE, INNER_FOLDS, OUTER_FOLDS, SEED

# ---------------------------------------------------------------------------
# Grids. STAGE A tunes preprocessing at a fixed default model setting; STAGE B
# tunes the model at the preprocessing STAGE A chose.
# ---------------------------------------------------------------------------

# pca=None means "no projection"; whiten is meaningless then, so it is not crossed.
PREP = ([{"pca": None, "whiten": False}] +
        [{"pca": p, "whiten": w} for p in (64, 128, 192) for w in (True, False)])

# Heads that cannot express a class prior are not crossed with class_weight.
NO_CLASS_WEIGHT = {"kNN-cosine", "HistGB"}

STAGE_A_DEFAULT = {
    "Logistic":     {"C": 1.0},
    "SVM-RBF":      {"C": 10.0, "gamma": "scale"},
    "SVM-linear":   {"C": 1.0},
    "kNN-cosine":   {"n_neighbors": 15},
    "RandomForest": {"n_estimators": 500, "min_samples_leaf": 1},
    "HistGB":       {"max_iter": 200, "learning_rate": 0.1},
    "MLP":          {"weight_decay": 1e-4},
}

# gamma: "scale" or a float m, meaning m / n_features (resolved at fit time, since
# n_features depends on the PCA dimensionality STAGE A picked).
STAGE_B = {
    "Logistic":     [{"C": c} for c in (0.03, 0.3, 1.0, 3.0, 10.0)],
    "SVM-RBF":      [{"C": c, "gamma": g} for c in (1.0, 3.0, 10.0, 30.0, 100.0)
                     for g in ("scale", 0.3, 1.0, 3.0, 10.0)],
    "SVM-linear":   [{"C": c} for c in (0.1, 1.0, 10.0)],
    "kNN-cosine":   [{"n_neighbors": k} for k in (5, 15, 30)],
    "RandomForest": [{"n_estimators": 500, "min_samples_leaf": m} for m in (1, 3)],
    "HistGB":       [{"max_iter": 200, "learning_rate": lr} for lr in (0.05, 0.1)],
    "MLP":          [{"weight_decay": w} for w in (1e-4, 1e-2)],
}

HEADS = list(STAGE_B)
def _macro_f1(a, b):
    return f1_score(a, b, average="macro", zero_division=0)


def _acc_f1(a, b):
    """Accuracy and macro-F1 weighted equally.

    These pull in opposite directions on this dataset: three classes hold 55% of
    the presses, so accuracy is mostly "did you get Synthetic Fabric, Wood and
    Concrete right" while macro-F1 is mostly "did you get Sand (12 presses) and
    Gravel (35) right". Tuning for accuracy alone drove macro-F1 from 0.365 to
    0.33; this metric refuses that trade.

    A plain arithmetic mean, deliberately. The two are on different scales here
    (~0.61 vs ~0.34), so an equal weighting on the RAW values means one point of
    macro-F1 is worth exactly one point of accuracy -- and since macro-F1 is the
    lower and more improvable of the two, selection leans toward the rare
    classes. That is the intended behaviour, not a bug, but it is the reason this
    is not the same as "maximise accuracy subject to macro-F1 not dropping".
    """
    return 0.5 * (accuracy_score(a, b) + _macro_f1(a, b))


METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "macro_f1": _macro_f1,
    "acc_f1": _acc_f1,
}


# ---------------------------------------------------------------------------
# Preprocessing. Fit on the training fold only, applied to everything else.
# ---------------------------------------------------------------------------

def preprocess(Xtr, others, pca=None, whiten=False, blocks=None, seed=SEED):
    """StandardScaler -> (equalise blocks) -> optional PCA -> L2 rows.

    `blocks` is a list of widths when several feature directories were stacked.
    After standardisation every dimension has unit variance, so a 1024-d block
    would carry 1024 units of variance against a 384-d block's 384 and dominate
    the kernel. Dividing each block by sqrt(width) equalises their contribution.
    """
    sc = StandardScaler().fit(Xtr)
    out = [sc.transform(X).astype(np.float32) for X in [Xtr, *others]]

    if blocks and len(blocks) > 1:
        scales = np.concatenate([np.full(w, 1.0 / np.sqrt(w)) for w in blocks])
        out = [X * scales.astype(np.float32) for X in out]

    if pca:
        n = min(pca, out[0].shape[0], out[0].shape[1])
        p = PCA(n_components=n, whiten=whiten, random_state=seed).fit(out[0])
        out = [p.transform(X).astype(np.float32) for X in out]

    return [X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
            for X in out]


def build(kind, params, class_weight, n_features, seed=SEED):
    p = dict(params)
    cw = {"class_weight": class_weight} if kind not in NO_CLASS_WEIGHT else {}
    if kind == "Logistic":
        return LogisticRegression(max_iter=2000, **cw, **p)
    if kind in ("SVM-RBF", "SVM-linear"):
        if kind == "SVM-RBF" and p.get("gamma", "scale") != "scale":
            p["gamma"] = float(p["gamma"]) / n_features
        kernel = "rbf" if kind == "SVM-RBF" else "linear"
        return SVC(kernel=kernel, **cw, **p)
    if kind == "kNN-cosine":
        return KNeighborsClassifier(metric="cosine", **p)
    if kind == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **cw, **p)
    if kind == "HistGB":
        return HistGradientBoostingClassifier(random_state=seed, **p)
    raise ValueError(kind)


def fit_predict(kind, params, class_weight, Xtr, ytr, Xte, n_classes,
                Xva=None, yva=None, epochs=None, seed=SEED):
    """Returns (predictions on Xte, best_epoch or None)."""
    if kind == "MLP":
        cw = (compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)
              if class_weight == "balanced" else np.ones(n_classes))
        m = MlpHead(Xtr.shape[1], n_classes)
        m, _, ep = train_torch_head(m, Xtr, ytr, Xva, yva, epochs=epochs or 200,
                                    weight_decay=params["weight_decay"],
                                    class_weight=cw, seed=seed)
        return predict(m, Xte), ep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = build(kind, params, class_weight, Xtr.shape[1], seed).fit(Xtr, ytr)
    return m.predict(Xte), None


# ---------------------------------------------------------------------------
# Nested selection. Every candidate is scored on ALL inner folds and averaged,
# matching nested_cv.py. The previous version used next(inner.split(...)) -- a
# single 3:1 split -- which made selection noisy enough to pick the wrong C and
# cost roughly 0.03 accuracy on the SVM.
# ---------------------------------------------------------------------------

def score_candidate(kind, prep, params, class_weight, X, y, tr, inner_splits,
                    n_classes, metric, blocks, seed):
    """Mean inner-fold score for one candidate. Also returns a median epoch budget."""
    if kind == "MLP":      # one thread per worker; joblib supplies the parallelism
        import torch
        torch.set_num_threads(1)
    scores, epochs = [], []
    for ia, ib in inner_splits:
        a, b = tr[ia], tr[ib]
        Xa, Xb = preprocess(X[a], [X[b]], blocks=blocks, seed=seed, **prep)
        pred, ep = fit_predict(kind, params, class_weight, Xa, y[a], Xb,
                               n_classes, Xva=Xb, yva=y[b], seed=seed)
        scores.append(METRICS[metric](y[b], pred))
        if ep:
            epochs.append(ep)
    return float(np.mean(scores)), (int(np.median(epochs)) if epochs else None)


def evaluate(X, y, groups, n_classes, label, metric, blocks, seed, n_jobs,
             heads=None):
    outer = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(np.zeros(len(y)), y, groups))
    rows = []

    for kind in (heads or HEADS):
        # Each MLP worker is pinned to one torch thread (see score_candidate), so
        # the pool must stay small enough that the processes do not thrash.
        jobs = min(n_jobs, 6) if kind == "MLP" else n_jobs
        cws = [None] if kind in NO_CLASS_WEIGHT else [None, "balanced"]
        pooled = np.full(len(y), -1)
        fold_scores, chosen = [], []

        for tr, te in outer_splits:
            assert not (set(groups[tr]) & set(groups[te]))
            inner = StratifiedGroupKFold(INNER_FOLDS, shuffle=True,
                                         random_state=seed)
            inner_splits = list(inner.split(np.zeros(len(tr)), y[tr], groups[tr]))
            run = lambda prep, params, cw: delayed(score_candidate)(
                kind, prep, params, cw, X, y, tr, inner_splits, n_classes,
                metric, blocks, seed)

            # ---- STAGE A: preprocessing + class prior, at a default model ----
            grid_a = [(p, c) for p in PREP for c in cws]
            res_a = Parallel(n_jobs=jobs)(
                run(p, STAGE_A_DEFAULT[kind], c) for p, c in grid_a)
            prep, cw = grid_a[int(np.argmax([s for s, _ in res_a]))]

            # ---- STAGE B: model hyperparameters, at the chosen preprocessing --
            grid_b = STAGE_B[kind]
            res_b = Parallel(n_jobs=jobs)(run(prep, p, cw) for p in grid_b)
            bi = int(np.argmax([s for s, _ in res_b]))
            params, budget = grid_b[bi], res_b[bi][1]

            # ---- OUTER: refit on all outer-train, score once on the test fold -
            Xtr, Xte = preprocess(X[tr], [X[te]], blocks=blocks, seed=seed, **prep)
            pred, _ = fit_predict(kind, params, cw, Xtr, y[tr], Xte, n_classes,
                                  epochs=budget, seed=seed)
            pooled[te] = pred
            fold_scores.append(METRICS[metric](y[te], pred))
            chosen.append({**prep, **params, "class_weight": cw,
                           **({"epochs": budget} if budget else {})})

        row = {"features": label, "head": kind,
               "selected_on": metric,
               **{m: round(float(f(y, pooled)), 4) for m, f in METRICS.items()},
               "fold_std": round(float(np.std(fold_scores)), 4),
               "hyperparams": chosen}
        rows.append(row)
        print(f"  {kind:<14}{row['accuracy']:>10.4f}{row['balanced_accuracy']:>10.4f}"
              f"{row['macro_f1']:>10.4f}   +/-{row['fold_std']:.3f}   "
              f"pca={chosen[0]['pca']} cw={chosen[0]['class_weight']}", flush=True)
    return rows


def load_touch(dirs):
    """Stack touch features from one or more feature dirs, checking alignment."""
    labels = [np.load(d / "labels.npy") for d in dirs]
    for lab in labels[1:]:
        if not np.array_equal(lab, labels[0]):
            raise SystemExit("feature dirs are not row-aligned -- labels differ")
    Ts = [np.load(d / "touch.npy") for d in dirs]
    return (np.hstack(Ts) if len(Ts) > 1 else Ts[0],
            labels[0],
            np.load(dirs[0] / "groups.npy", allow_pickle=True),
            [T.shape[1] for T in Ts])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, nargs="+",
                    default=[Path("../features_dinop9")],
                    help="one or more feature dirs; touch.npy is stacked")
    ap.add_argument("--out", type=Path, default=Path("../results"))
    ap.add_argument("--select-on", choices=list(METRICS), default="accuracy",
                    help="inner-fold objective AND the reported headline")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ungrouped", action="store_true",
                    help="Reproduce the leaky split used by the published "
                         "baselines: random stratified folds, sessions NOT held "
                         "out. For comparability only.")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--heads", nargs="+", default=None, choices=HEADS,
                    help="restrict the sweep to these heads")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args()
    np.random.seed(args.seed)

    T, y_raw, groups, blocks = load_touch(args.features)
    if args.ungrouped:
        # Give every press its own group, which collapses StratifiedGroupKFold to
        # an ordinary stratified random split -- i.e. the LEAKY protocol the
        # published Touch and Go numbers were measured on, where presses from the
        # same session (often the same object, seconds apart) land on both sides
        # of the split. Measured here purely so our number can be compared
        # like-for-like with prior work; it is NOT the headline protocol.
        groups = np.arange(len(y_raw))
        print("*** UNGROUPED (LEAKY) SPLIT -- sessions span train and test ***")
    keep = np.array([CLASS_NAMES[int(c)] not in EXCLUDE for c in y_raw])
    for c in np.unique(y_raw):
        if len(np.unique(groups[y_raw == c])) < 2:
            keep &= y_raw != c
    T, y_raw, groups = T[keep], y_raw[keep], groups[keep]
    kept = sorted(np.unique(y_raw).tolist())
    y = np.array([{c: i for i, c in enumerate(kept)}[c] for c in y_raw])
    n_classes = len(kept)

    label = "+".join(d.name.replace("features_", "") for d in args.features)
    print(f"touch-only, {len(y):,} presses, {n_classes} classes, "
          f"{T.shape[1]}-d frozen features {blocks}")
    print(f"seed {args.seed} | selecting on {args.select_on} | n_jobs {args.n_jobs}")
    print(f"  {'head':<14}{'accuracy':>10}{'bal-acc':>10}{'macro-F1':>10}")
    rows = evaluate(T, y, groups, n_classes, label, args.select_on, blocks,
                    args.seed, args.n_jobs, args.heads)

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"touch_heads{('_' + args.tag) if args.tag else ''}.json"
    (args.out / name).write_text(json.dumps(rows, indent=1))
    best = max(rows, key=lambda r: r[args.select_on])
    print(f"\nbest head: {best['head']}  acc {best['accuracy']:.4f}  "
          f"bal-acc {best['balanced_accuracy']:.4f}  "
          f"macro-F1 {best['macro_f1']:.4f}")
    print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
