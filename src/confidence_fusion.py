"""Confidence-gated late fusion: turn vision down when vision is unsure.

The proposal: score vision's confidence per example and, when it is low, fall back
on the independent tactile prediction. This is decision-level fusion conditioned on
uncertainty, as opposed to the feature-level gating already tested.

It rests on a premise that must be checked first, not assumed:
    does vision's confidence actually predict vision's errors?
If confidence carries no signal about correctness, no routing rule built on it can
work, however sophisticated. That is measured here as an AUROC.

Three rules are compared, all tuned on inner folds only:
  hard    use touch when vision confidence < threshold
  soft    p = a*p_vision + (1-a)*p_touch, with a rising with vision confidence
  fixed   p = (1-w)*p_vision + w*p_touch, w constant  (the baseline already run)

and against an ORACLE router that always picks the modality that happens to be
correct - an upper bound no confidence rule can pass.

Usage:
    python confidence_fusion.py --features ../features_dinop9 --out ../results
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from common import CLASS_NAMES
from nested_cv import EXCLUDE, INNER_FOLDS, OUTER_FOLDS, SEED


def prep(Xtr, others):
    sc = StandardScaler().fit(Xtr)
    out = [sc.transform(X).astype(np.float32) for X in [Xtr, *others]]
    return [X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
            for X in out]


def probs(Xtr, ytr, Xte, n_classes):
    m = LogisticRegression(max_iter=2000, C=1.0,
                           class_weight="balanced").fit(Xtr, ytr)
    p = np.zeros((len(Xte), n_classes))
    p[:, m.classes_] = m.predict_proba(Xte)
    return p


def blend_hard(pv, pt, conf, thr):
    """Below threshold, hand the decision to touch entirely."""
    out = pv.copy()
    out[conf < thr] = pt[conf < thr]
    return out


def blend_soft(pv, pt, conf, k):
    """Weight rises with vision confidence; k sets how sharply."""
    a = (conf ** k)[:, None]
    return a * pv + (1 - a) * pt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--touch-features", type=Path, default=None,
                    help="Take touch.npy from here instead, so the upgraded touch "
                         "encoder can be fused against the same vision features.")
    ap.add_argument("--out", type=Path, default=Path("../results"))
    args = ap.parse_args()
    np.random.seed(SEED)

    y_raw = np.load(args.features / "labels.npy")
    groups = np.load(args.features / "groups.npy", allow_pickle=True)
    V = np.load(args.features / "vision.npy")
    if args.touch_features:
        import pandas as _pd
        a = _pd.read_csv(args.features / "rows.csv").run_id.tolist()
        b = _pd.read_csv(args.touch_features / "rows.csv").run_id.tolist()
        assert a == b, "row order differs; cannot fuse these two feature sets"
        T = np.load(args.touch_features / "touch.npy")
        print(f"touch from {args.touch_features.name}  dim {T.shape[1]}")
    else:
        T = np.load(args.features / "touch.npy")

    keep = np.array([CLASS_NAMES[int(c)] not in EXCLUDE for c in y_raw])
    for c in np.unique(y_raw):
        if len(np.unique(groups[y_raw == c])) < 2:
            keep &= y_raw != c
    V, T, y_raw, groups = V[keep], T[keep], y_raw[keep], groups[keep]
    kept = sorted(np.unique(y_raw).tolist())
    y = np.array([{c: i for i, c in enumerate(kept)}[c] for c in y_raw])
    K = len(kept)

    THR = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    KS = [0.5, 1.0, 2.0, 4.0, 8.0]
    WS = [0.1, 0.2, 0.3, 0.4, 0.5]

    preds = {k: np.full(len(y), -1) for k in
             ("vision", "touch", "concat", "hard", "soft", "fixed", "oracle")}
    confs = np.zeros(len(y))
    picks = []

    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                 random_state=SEED)
    for tr, te in outer.split(np.zeros(len(y)), y, groups):
        # --- inner fold: choose threshold / sharpness / weight ---
        inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                     random_state=SEED)
        ia, ib = next(inner.split(np.zeros(len(tr)), y[tr], groups[tr]))
        a, b = tr[ia], tr[ib]
        Va, Vb = prep(V[a], [V[b]])
        Ta, Tb = prep(T[a], [T[b]])
        pv_i, pt_i = probs(Va, y[a], Vb, K), probs(Ta, y[a], Tb, K)
        c_i = pv_i.max(1)
        f1 = lambda p: f1_score(y[b], p.argmax(1), average="macro", zero_division=0)
        best_thr = max(THR, key=lambda t: f1(blend_hard(pv_i, pt_i, c_i, t)))
        best_k = max(KS, key=lambda k: f1(blend_soft(pv_i, pt_i, c_i, k)))
        best_w = max(WS, key=lambda w: f1((1 - w) * pv_i + w * pt_i))
        picks.append({"thr": best_thr, "k": best_k, "w": best_w})

        # --- outer fold: refit on all of train, apply chosen rule once ---
        Vtr, Vte = prep(V[tr], [V[te]])
        Ttr, Tte = prep(T[tr], [T[te]])
        pv, pt = probs(Vtr, y[tr], Vte, K), probs(Ttr, y[tr], Tte, K)
        cf = np.hstack([V, T])
        Ctr, Cte = prep(cf[tr], [cf[te]])
        pc = probs(Ctr, y[tr], Cte, K)
        conf = pv.max(1)
        confs[te] = conf

        preds["vision"][te] = pv.argmax(1)
        preds["touch"][te] = pt.argmax(1)
        preds["concat"][te] = pc.argmax(1)
        preds["hard"][te] = blend_hard(pv, pt, conf, best_thr).argmax(1)
        preds["soft"][te] = blend_soft(pv, pt, conf, best_k).argmax(1)
        preds["fixed"][te] = ((1 - best_w) * pv + best_w * pt).argmax(1)
        vok = pv.argmax(1) == y[te]
        preds["oracle"][te] = np.where(vok, pv.argmax(1), pt.argmax(1))

    # ---- does confidence predict vision errors at all? ----
    v_wrong = (preds["vision"] != y).astype(int)
    auroc = roc_auc_score(v_wrong, -confs)
    print("PREMISE CHECK — is vision confidence informative about vision errors?")
    print(f"  AUROC(low confidence -> vision wrong) = {auroc:.3f}"
          f"   (0.5 = useless, 1.0 = perfect)")
    print(f"  mean confidence when vision RIGHT = {confs[v_wrong == 0].mean():.3f}")
    print(f"  mean confidence when vision WRONG = {confs[v_wrong == 1].mean():.3f}")

    print("\nFUSION RULES (nested, tuned on inner folds only)")
    print(f"  {'rule':<34}{'accuracy':>10}{'macro-F1':>10}")
    rows = {}
    for k in ("vision", "touch", "concat", "fixed", "hard", "soft", "oracle"):
        acc = accuracy_score(y, preds[k])
        f1 = f1_score(y, preds[k], average="macro", zero_division=0)
        rows[k] = {"accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4)}
        label = {"vision": "vision only", "touch": "touch only",
                 "concat": "concatenation (feature-level)",
                 "fixed": "late fusion, fixed weight",
                 "hard": "confidence routing (hard threshold)",
                 "soft": "confidence weighting (soft)",
                 "oracle": "ORACLE router (upper bound)"}[k]
        print(f"  {label:<34}{acc:>10.4f}{f1:>10.4f}")

    out = {"auroc_confidence_vs_error": round(float(auroc), 4),
           "mean_conf_right": round(float(confs[v_wrong == 0].mean()), 4),
           "mean_conf_wrong": round(float(confs[v_wrong == 1].mean()), 4),
           "results": rows, "chosen_per_fold": picks}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "confidence_fusion.json").write_text(json.dumps(out))
    print(f"\nwrote {args.out / 'confidence_fusion.json'}")


if __name__ == "__main__":
    main()
