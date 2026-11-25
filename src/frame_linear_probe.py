"""Frame-level linear probe, matching the protocol of the published comparison.

WHY THIS EXISTS
---------------
Our headline tactile-only number (81.85%) uses a tuned MLP over nine pooled
frames per press. The figures it gets compared against -- VT CMC 54.7 through
STT 67.77 -- come from Table 6 of the STT supplementary, which is captioned
"Material Classification Linear Probing Accuracy" and evaluates one decision per
FRAME on the original Touch-and-Go split. Two differences, a stronger readout and
nine-frame pooling, both favour us. Quoting 81.85 against 67.77 therefore
measures the readout as much as the encoder.

This script removes both advantages: a single linear layer (no hidden units, no
hyperparameter search beyond regularisation strength chosen on training data),
one prediction per frame, no pooling. It is the row of the comparison that is
measured the same way on both sides.

It reports two splits, labelled, because the field does not agree on one:
  random     -- frames split at random, the character of the official split,
                which Gungor et al. (arXiv:2503.06860) showed leaks: test frames
                sit as few as 85 frames from training frames
  video      -- whole videos assigned to one side, the construction both that
                paper and STT's own Section 7 argue for

Frozen and fine-tuned encoders are both scored, so the encoder effect is
isolated at the same readout.

Usage:
    python frame_linear_probe.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from nested_cv import SEED

# (label, [feature dirs]) -- each dir must carry touch_frames.npy + frame_run_ids.npy
ENCODERS = [
    ("frozen DINOv2", ["../features_dinop9_patch", "../features_patch_bg"]),
    ("fine-tuned DINOv2", ["../features_ftf_patch", "../features_ftf_bg"]),
]


def load_frames(dirs):
    """Per-frame features, with labels and video ids expanded from rows.csv."""
    X, blocks = [], []
    run_ids = None
    for d in dirs:
        d = Path(d)
        f = np.load(d / "touch_frames.npy", mmap_mode="r")
        r = np.load(d / "frame_run_ids.npy", allow_pickle=True)
        if run_ids is None:
            run_ids = r
        else:
            assert np.array_equal(run_ids, r), f"{d} frame order differs"
        X.append(np.asarray(f, dtype=np.float32))
        blocks.append(X[-1].shape[1])
    rows = pd.read_csv(Path(dirs[0]) / "rows.csv")
    lab = dict(zip(rows.run_id, rows.label))
    vid = dict(zip(rows.run_id, rows.video_id))
    y_raw = np.array([lab[r] for r in run_ids])
    groups = np.array([vid[r] for r in run_ids])
    return np.concatenate(X, axis=1), y_raw, groups, blocks


def probe(Xtr, ytr, Xte, yte, blocks, C):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    if len(blocks) > 1:
        s = np.concatenate([np.full(w, 1.0 / np.sqrt(w)) for w in blocks])
        A, B = A * s, B * s
    A /= np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-8, None)
    B /= np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-8, None)
    clf = LogisticRegression(max_iter=3000, C=C, n_jobs=-1)
    p = clf.fit(A, ytr).predict(B)
    return (accuracy_score(yte, p),
            f1_score(yte, p, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="Substring of an encoder label; runs just that one. The "
                         "frozen result is already cached, so re-running it costs "
                         "an hour for no new information.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--eval-folds", type=int, default=1,
                    help="Folds actually scored. One is enough on the random "
                         "split, where measured fold std is under 0.01.")
    ap.add_argument("--out", type=Path,
                    default=Path("../results/frame_linear_probe.json"))
    args = ap.parse_args()

    out = []
    for name, dirs in ENCODERS:
        if args.only and args.only.lower() not in name.lower():
            continue
        if not all((Path(d) / "touch_frames.npy").exists() for d in dirs):
            print(f"{name}: per-frame features missing, skipping")
            continue
        X, y_raw, groups, blocks = load_frames(dirs)
        # match touch_heads.py: drop classes present in fewer than two sessions
        keep = np.array([len(np.unique(groups[y_raw == c])) >= 2 for c in y_raw])
        X, y_raw, groups = X[keep], y_raw[keep], groups[keep]
        cls = sorted(set(y_raw.tolist()))
        m = {c: i for i, c in enumerate(cls)}
        y = np.array([m[c] for c in y_raw])
        print(f"\n=== {name} ===")
        print(f"{len(y):,} frames | {X.shape[1]} dims {blocks} | "
              f"{len(cls)} classes | {len(set(groups))} videos")

        for split, sg in (("random (official-split character)",
                           np.arange(len(y))),
                          ("video-disjoint (leak-free)", groups)):
            cv = StratifiedGroupKFold(args.folds, shuffle=True,
                                      random_state=SEED)
            accs, f1s = [], []
            for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, sg)):
                if fold >= args.eval_folds:
                    break
                # regularisation picked on a held-out slice of TRAIN only
                sub = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
                i_fit, i_val = next(sub.split(np.zeros(len(tr)), y[tr], sg[tr]))
                best_C, best = 1.0, -1.0
                for C in (0.1, 1.0, 10.0):
                    a, _ = probe(X[tr[i_fit]], y[tr[i_fit]],
                                 X[tr[i_val]], y[tr[i_val]], blocks, C)
                    if a > best:
                        best_C, best = C, a
                a, f = probe(X[tr], y[tr], X[te], y[te], blocks, best_C)
                print(f"  {split:<36} fold {fold}  C={best_C:<5} "
                      f"acc {a:.4f}  macroF1 {f:.4f}", flush=True)
                accs.append(a); f1s.append(f)
            out.append({"encoder": name, "split": split,
                        "accuracy": float(np.mean(accs)),
                        "accuracy_std": float(np.std(accs)),
                        "macro_f1": float(np.mean(f1s)),
                        "n_folds": len(accs), "n_frames": int(len(y))})

    print("\n" + "=" * 78)
    print(f"{'encoder':<20}{'split':<38}{'accuracy':>10}{'macro-F1':>10}")
    for r in out:
        print(f"{r['encoder']:<20}{r['split']:<38}"
              f"{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}")
    print("\npublished, frame-level linear probe, original split "
          "(STT suppl. Table 6):")
    print("  VT CMC 0.547 | MViTac 0.576 | UniTouch 0.613 | ViT-Lens-2 0.630")
    print("  TLV-Link 0.672 | OmniBind 0.6745 | STT 0.6777")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": out}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
