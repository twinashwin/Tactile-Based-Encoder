"""Fuse vision with the FINE-TUNED touch features, on the leaky protocol.

Touch-only reached 0.8031 leaky (fine-tuned encoder, 9 frames, raw+bg views,
tuned MLP), which already edges ViTaPEs' 0.8010 visuo-tactile figure. The obvious
next question is what the same system does WITH vision, and the answer was never
measured: every fusion number in this project used the old weak touch features
(0.4664-0.4940 accuracy) under the grouped protocol.

Fusion gain here is known to scale with touch quality -- it went +0.87 -> +1.31
when touch improved from 0.4664 to 0.4940 -- so with touch an order of magnitude
better the complementarity question is genuinely open.

Rules scored, matching confidence_fusion.py so the numbers are comparable:
  vision only / touch only          the two single modalities
  concat                            feature-level, block-equalised
  late fusion (fixed weight)        the rule that won before (0.8633 grouped)
  oracle                            upper bound: right if EITHER is right

CAVEAT: the vision tower is FROZEN DINOv2. Only touch was fine-tuned, so this
understates what a symmetric system would do.

Usage:
    python fuse_ft.py --vision ../features_dinop9 \
        --touch ../features_ftl_patch ../features_ftl_bg
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

from nested_cv import OUTER_FOLDS, SEED
from touch_heads import preprocess

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--touch", type=Path, nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/fuse_ft.json"))
    args = ap.parse_args()

    vr = pd.read_csv(args.vision / "rows.csv").run_id.tolist()
    V = np.load(args.vision / "vision.npy")
    Ts, blocks = [], []
    for d in args.touch:
        assert pd.read_csv(d / "rows.csv").run_id.tolist() == vr, \
            f"{d} row order differs from the vision set"
        t = np.load(d / "touch.npy")
        Ts.append(t); blocks.append(t.shape[1])
    T = np.concatenate(Ts, axis=1)
    y_raw = np.load(args.vision / "labels.npy")
    groups = np.load(args.vision / "groups.npy", allow_pickle=True)

    keep = np.array([len(np.unique(groups[y_raw == c])) >= 2 for c in y_raw])
    V, T, y_raw, groups = V[keep], T[keep], y_raw[keep], groups[keep]
    cls = sorted(set(y_raw.tolist())); m = {c: i for i, c in enumerate(cls)}
    y = np.array([m[c] for c in y_raw])
    sg = groups if args.grouped else np.arange(len(y))
    print(f"{len(y):,} presses | vision {V.shape[1]}d | touch {T.shape[1]}d "
          f"{blocks} | {len(cls)} classes | "
          f"{'GROUPED' if args.grouped else 'LEAKY'}")

    cv = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    tr, te = list(cv.split(np.zeros(len(y)), y, sg))[args.fold]
    print(f"fold {args.fold}: train {len(tr):,} test {len(te):,}\n")

    def probs(X, bl):
        A, B = preprocess(X[tr], [X[te]], pca=None, whiten=False,
                          blocks=bl, seed=SEED)
        clf = MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, max_iter=400,
                            random_state=SEED)
        clf.fit(A, y[tr])
        return clf.predict_proba(B)

    Pv = probs(V, [V.shape[1]])
    Pt = probs(T, blocks)
    Pc = probs(np.concatenate([V, T], axis=1), [V.shape[1], *blocks])

    yt = y[te]
    rows = {}

    def rec(tag, P):
        p = P.argmax(1)
        rows[tag] = {"accuracy": float(accuracy_score(yt, p)),
                     "macro_f1": float(f1_score(yt, p, average="macro",
                                                zero_division=0))}

    rec("vision", Pv); rec("touch", Pt); rec("concat", Pc)

    # late fusion: weight chosen on a HELD-OUT slice of train.
    #
    # The previous version fit both towers on all of `tr` and then swept the
    # weight against `np.array_split(tr, 4)[0]`, a slice of `tr` itself. Both
    # towers memorise their training data, so every weight from 0.0 to 1.0 tied
    # at exactly 1.0000 accuracy on that slice. The loop's strict `>` kept the
    # first entry, so w=0.0 was never selected on evidence, it won a degenerate
    # tie, and late fusion silently collapsed to vision alone on every fold of
    # both protocols. Fitting on train-minus-slice instead gives a real curve
    # that peaks near w=0.4.
    #
    # The slice is also grouped now. array_split cuts by array position, which
    # can put presses from one session on both sides of the boundary.
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
    i_fit, i_sel = next(inner.split(np.zeros(len(tr)), y[tr], sg[tr]))
    fit_idx, sel_idx = tr[i_fit], tr[i_sel]
    Av, Bv = preprocess(V[fit_idx], [V[sel_idx]], pca=None, whiten=False,
                        blocks=[V.shape[1]], seed=SEED)
    At, Bt = preprocess(T[fit_idx], [T[sel_idx]], pca=None, whiten=False,
                        blocks=blocks, seed=SEED)
    cv_ = MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, max_iter=400,
                        random_state=SEED).fit(Av, y[fit_idx])
    ct_ = MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-4, max_iter=400,
                        random_state=SEED).fit(At, y[fit_idx])
    pv_, pt_ = cv_.predict_proba(Bv), ct_.predict_proba(Bt)
    best_w, best_a = 0.0, -1.0
    sweep = []
    for w in np.arange(0.0, 1.01, 0.1):
        a = accuracy_score(y[sel_idx], (pv_ * (1 - w) + pt_ * w).argmax(1))
        sweep.append((float(w), float(a)))
        if a > best_a:
            best_w, best_a = float(w), a
    print(f"late-fusion weight sweep on held-out train slice "
          f"({len(sel_idx):,} presses):")
    print("  " + "  ".join(f"{w:.1f}:{a:.4f}" for w, a in sweep))
    print(f"late-fusion touch weight chosen: {best_w:.1f} (acc {best_a:.4f})")
    rec("late_fusion", Pv * (1 - best_w) + Pt * best_w)

    orac = (Pv.argmax(1) == yt) | (Pt.argmax(1) == yt)
    rows["oracle"] = {"accuracy": float(orac.mean()), "macro_f1": None}

    print(f"{'rule':<14}{'accuracy':>10}{'macro-F1':>10}")
    for k in ("vision", "touch", "concat", "late_fusion", "oracle"):
        f = rows[k]["macro_f1"]
        print(f"{k:<14}{rows[k]['accuracy']:>10.4f}"
              f"{(f'{f:.4f}' if f is not None else '-'):>10}")
    print(f"\nfusion gain over vision alone: "
          f"{(rows['late_fusion']['accuracy']-rows['vision']['accuracy'])*100:+.2f} pts")
    print("ViTaPEs (published, vision+touch, leaky): 0.8010")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"fold": args.fold, "protocol": "grouped" if args.grouped else "leaky",
         "touch_weight": best_w, "weight_sweep": sweep, "results": rows},
        indent=1))


if __name__ == "__main__":
    main()
