"""Steps 3-5 - group-aware split, then 2 models x 3 feature sets.

Classifiers are pure scikit-learn. (PyTorch is still required by
extract_features.py for the frozen backbone - scikit-learn has no pretrained CNN -
but nothing in this file imports it.)

Split: StratifiedGroupKFold on the grouping key. Group-aware so no group's presses
straddle the split (the main way accuracy gets inflated on this dataset), and
stratified so rare classes appear on both sides where the group structure allows.

  --group-by session  one group per recording (default)
  --group-by date     one group per calendar day. Stricter: sessions recorded on
                      the same day are usually the same place, so this partly
                      controls for one physical surface appearing on both sides.

  --folds 1  one fixed split - fast, but a point estimate with no error bar
  --folds 5  every example tested exactly once across folds; reports mean +/- std
             over folds AND pooled per-class metrics over all predictions

Models:
  Logistic - LogisticRegression, L-BFGS, multinomial, run to convergence.
  MLP      - MLPClassifier, 256 -> 128, ReLU, Adam. Stepped one epoch at a time via
             warm_start and early-stopped on the group-aware validation set,
             because sklearn's own early_stopping splits at random and ignores
             groups. sklearn has no dropout, so it is regularised by L2 (alpha).

  --tune          grid-search regularisation on VALIDATION macro-F1
  --class-weight  'balanced' reweights classes inversely to frequency, targeting
                  macro-F1 directly under this dataset's 1338:1 imbalance

Usage:
    python run_experiments.py --tune --class-weight balanced --folds 5
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from common import CLASS_NAMES

SEED = 42
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
ALPHA_GRID = [1e-4, 1e-2, 1e-1, 1.0]


def _fit_mlp_early_stopping(clf, Xtr, ytr, Xva, yva, args):
    """One epoch per .fit() call via warm_start; keep the best-validation weights."""
    best_f1, best, since = -1.0, None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(args.epochs):
            clf.fit(Xtr, ytr)
            f1 = f1_score(yva, clf.predict(Xva), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best = ([w.copy() for w in clf.coefs_],
                        [b.copy() for b in clf.intercepts_])
                since = 0
            else:
                since += 1
                if since >= args.patience:
                    break
    if best is not None:
        clf.coefs_, clf.intercepts_ = best
    return clf, best_f1


def fit_logistic(Xtr, ytr, Xva, yva, args):
    grid = C_GRID if args.tune else [args.C]
    best = None
    for C in grid:
        m = LogisticRegression(max_iter=2000, C=C,
                               class_weight=args.class_weight).fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro", zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, m, C)
    return best[1], {"C": best[2]}


def fit_mlp(Xtr, ytr, Xva, yva, args):
    """MLPClassifier has no class_weight, so balancing is applied by resampling.

    Sample weights cannot be passed to MLPClassifier either, so 'balanced' is
    approximated by duplicating rare-class rows in proportion to their weight.
    """
    Xfit, yfit = Xtr, ytr
    if args.class_weight == "balanced":
        w = compute_sample_weight("balanced", ytr)
        reps = np.maximum(1, np.round(w / w.min())).astype(int)
        idx = np.repeat(np.arange(len(ytr)), reps)
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)
        Xfit, yfit = Xtr[idx], ytr[idx]

    grid = ALPHA_GRID if args.tune else [args.alpha]
    best = None
    for alpha in grid:
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
            learning_rate_init=args.lr, batch_size=args.batch_size,
            alpha=alpha, max_iter=1, warm_start=True, random_state=SEED,
        )
        m, f1 = _fit_mlp_early_stopping(clf, Xfit, yfit, Xva, yva, args)
        if best is None or f1 > best[0]:
            best = (f1, m, alpha)
    return best[1], {"alpha": best[2]}


MODELS = {"Logistic": fit_logistic, "MLP": fit_mlp}


def make_groups(sessions: np.ndarray, how: str) -> np.ndarray:
    """session id '20220601_182052' -> itself, or its '20220601' date prefix."""
    if how == "session":
        return sessions
    return np.array([str(s).split("_")[0] for s in sessions])


def iter_splits(y, groups, n_folds: int):
    """Yield (train, val, test) index arrays, group-disjoint in all three."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (trainval, test) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        if i >= n_folds:
            return
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        a, b = next(inner.split(np.zeros(len(trainval)), y[trainval],
                                groups[trainval]))
        yield trainval[a], trainval[b], test


def build_split(y, groups):
    """First fold only - kept so single-split callers keep working."""
    return next(iter_splits(y, groups, 1))


def plot_confusion(y_true, y_pred, names, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    with np.errstate(invalid="ignore"):
        norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.55),) * 2)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_and_filter(args):
    y = np.load(args.features / "labels.npy")
    sessions = np.load(args.features / "groups.npy", allow_pickle=True)
    groups = make_groups(sessions, args.group_by)

    vision = (np.load(args.features / "vision.npy")
              if (args.features / "vision.npy").exists() else None)
    touch = (np.load(args.features / "touch.npy")
             if (args.features / "touch.npy").exists() else None)
    if vision is None and touch is None:
        raise SystemExit(f"No vision.npy or touch.npy in {args.features}")

    keep = np.ones(len(y), bool)
    for c in np.unique(y):
        n_groups = len(np.unique(groups[y == c]))
        n_examples = int((y == c).sum())
        why = None
        if n_groups < args.min_groups:
            why = f"spans {n_groups} group(s)"
        elif n_examples < args.min_support:
            why = f"{n_examples} examples < min-support {args.min_support}"
        if why:
            keep &= y != c
            print(f"dropped {CLASS_NAMES[int(c)]}: {why}")

    vision = None if vision is None else vision[keep]
    touch = None if touch is None else touch[keep]
    y, groups = y[keep], groups[keep]

    kept = sorted(np.unique(y).tolist())
    names = [CLASS_NAMES[c] for c in kept]
    y = np.array([{c: i for i, c in enumerate(kept)}[c] for c in y])
    return vision, touch, y, groups, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("../features"))
    ap.add_argument("--out", type=Path, default=Path("../results"))
    ap.add_argument("--folds", type=int, default=1,
                    help="1 = single fixed split; 5 = full CV with error bars")
    ap.add_argument("--group-by", choices=["session", "date"], default="session")
    ap.add_argument("--tune", action="store_true",
                    help="grid-search regularisation on validation macro-F1")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--min-groups", type=int, default=2)
    ap.add_argument("--min-support", type=int, default=0,
                    help="drop classes with fewer than this many examples; 50 "
                         "gives the 'measurable classes only' variant")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()
    args.class_weight = None if args.class_weight == "none" else args.class_weight

    np.random.seed(SEED)
    vision, touch, y, groups, names = load_and_filter(args)
    print(f"{len(y):,} examples | {len(names)} classes | "
          f"{len(np.unique(groups)):,} {args.group_by} groups")

    feature_sets = {}
    if vision is not None:
        feature_sets["Vision only"] = vision
    if touch is not None:
        feature_sets["Touch only"] = touch
    if vision is not None and touch is not None:
        feature_sets["Vision + Touch"] = np.concatenate([vision, touch], axis=1)

    splits = list(iter_splits(y, groups, args.folds))
    for i, (tr, va, te) in enumerate(splits):
        leaks = (set(groups[tr]) | set(groups[va])) & set(groups[te])
        assert not leaks, f"fold {i}: group leak into test: {leaks}"
    print(f"{len(splits)} fold(s), group-disjoint | fold 0: "
          f"train {len(splits[0][0]):,} val {len(splits[0][1]):,} "
          f"test {len(splits[0][2]):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for fname, X in feature_sets.items():
        for mname, fit in MODELS.items():
            fold_acc, fold_f1, chosen = [], [], []
            pooled_pred = np.full(len(y), -1)

            for tr, va, te in splits:
                scaler = StandardScaler().fit(X[tr])
                Xtr, Xva, Xte = (scaler.transform(X[i]).astype(np.float32)
                                 for i in (tr, va, te))
                model, params = fit(Xtr, y[tr], Xva, y[va], args)
                pred = model.predict(Xte)
                pooled_pred[te] = pred
                fold_acc.append(accuracy_score(y[te], pred))
                fold_f1.append(f1_score(y[te], pred, average="macro",
                                        zero_division=0))
                chosen.append(params)

            covered = pooled_pred >= 0
            pooled_acc = accuracy_score(y[covered], pooled_pred[covered])
            pooled_f1 = f1_score(y[covered], pooled_pred[covered],
                                 average="macro", zero_division=0)

            slug = (f"{fname}_{mname}{args.tag}".lower()
                    .replace(" + ", "_").replace(" ", "_"))
            plot_confusion(y[covered], pooled_pred[covered], names,
                           f"{fname} / {mname}  "
                           f"(acc {pooled_acc:.3f}, macro-F1 {pooled_f1:.3f})",
                           args.out / f"confusion_{slug}.png")

            results.append({
                "feature_set": fname, "model": mname, "dim": int(X.shape[1]),
                "folds": len(splits),
                "accuracy": round(float(pooled_acc), 4),
                "macro_f1": round(float(pooled_f1), 4),
                "accuracy_mean": round(float(np.mean(fold_acc)), 4),
                "accuracy_std": round(float(np.std(fold_acc)), 4),
                "macro_f1_mean": round(float(np.mean(fold_f1)), 4),
                "macro_f1_std": round(float(np.std(fold_f1)), 4),
                "per_class_f1": [round(float(v), 3) for v in
                                 f1_score(y[covered], pooled_pred[covered],
                                          average=None, zero_division=0)],
                "confusion": confusion_matrix(y[covered], pooled_pred[covered],
                                              labels=range(len(names))).tolist(),
                "hyperparams": chosen,
            })
            spread = f"  +/- {np.std(fold_f1):.3f}" if len(splits) > 1 else ""
            print(f"{fname:<15} {mname:<9} acc {pooled_acc:.4f}  "
                  f"macro-F1 {pooled_f1:.4f}{spread}   {chosen[0]}")

    support = (np.bincount(y, minlength=len(names)) if args.folds >= 5
               else np.bincount(y[splits[0][2]], minlength=len(names)))
    meta = {
        "n_examples": int(len(y)), "n_classes": len(names),
        "n_groups": int(len(np.unique(groups))), "group_by": args.group_by,
        "folds": len(splits), "tuned": bool(args.tune),
        "class_weight": args.class_weight or "none",
        "min_support": args.min_support,
        "class_names": names,
        "test_support": support.tolist(),
        "runs": results,
    }
    (args.out / f"report_data{args.tag}.json").write_text(json.dumps(meta))
    (args.out / f"results{args.tag}.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'Feature set':<15} | {'Logistic acc / F1':>19} | {'MLP acc / F1':>19}")
    print("-" * 60)
    for fname in feature_sets:
        row = {r["model"]: r for r in results if r["feature_set"] == fname}
        cells = " | ".join(
            f"{row[m]['accuracy']:.3f} / {row[m]['macro_f1']:.3f}".rjust(19)
            for m in ("Logistic", "MLP"))
        print(f"{fname:<15} | {cells}")
    print(f"\nwrote {args.out}/report_data{args.tag}.json and "
          f"{len(results)} confusion matrices")


if __name__ == "__main__":
    main()
