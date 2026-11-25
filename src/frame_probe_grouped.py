"""Frame-level linear probe on the SESSION-GROUPED split, scored honestly.

WHY THIS REPLACES frame_linear_probe.py
---------------------------------------
That script reported 0.7127 for the fine-tuned encoder on its "video-disjoint
(leak-free)" split. The split was leak-free with respect to the LINEAR PROBE's
training frames, but the features it probed came from
extract_frames_ft.sh, which loads

    ../results/ft_ckpt_leaky10/fold0_best.pt

the encoder fine-tuned on the RANDOM (press-level) split. That encoder trained on
presses drawn from every one of the 124 sessions, including the sessions the
probe then held out as test. The encoder had already seen the test sessions, so
0.7127 is contaminated and the +9.78 gain over frozen is inflated. The frozen row
(0.6149) is unaffected, since frozen DINOv2 never saw this dataset.

WHAT THIS DOES INSTEAD
----------------------
Fold k is probed with encoder k from ft_ckpt_clean, the session-grouped 5-fold
fine-tune, which is the pairing grouped_all_folds.sh already uses for the
press-level result. No session an encoder trained on appears in its own test
split.

The fold boundary is derived at PRESS level and then mapped onto frames by
run_id. Deriving it from the frames directly would be wrong: StratifiedGroupKFold
balances classes over the samples it is given, so a frame-level call can return a
different session partition than the press-level call that defined which sessions
encoder k trained on, which would quietly reintroduce the leak.

Frozen features are encoder-independent, so --template may point at one fixed
pair of directories and all five folds are still valid.

Usage:
    python frame_probe_grouped.py --tag frozen \
        --template ../features_dinop9_patch ../features_patch_bg
    python frame_probe_grouped.py --tag adapted \
        --template '../features_ftfg{k}_patch' '../features_ftfg{k}_bg'
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from nested_cv import OUTER_FOLDS, SEED


def press_table(d: Path):
    """Press-level run_id / label / video_id, in the order the row file lists."""
    rows = pd.read_csv(d / "rows.csv")
    return (rows.run_id.to_numpy(),
            rows.label.to_numpy(),
            rows.video_id.to_numpy())


def press_split(d: Path, fold: int, random_split: bool = False):
    """Test/train run_ids for one outer fold, decided at press level.

    Mirrors score_ft_features.py exactly: same class filter, same
    StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED), grouped
    by video_id for the session split and by press for the random one. Fold k
    here is therefore the fold k that encoder k was trained against.

    The random split keeps each press intact and scatters presses across the
    boundary, which is the press-level protocol the rest of the project reports.
    It is deliberately not the frame-level random split the first version of this
    probe used, where the nine frames of one press were dealt out independently.
    That is a heavier leak than anything in the published protocols and is not
    the comparison we want to put next to the session numbers.
    """
    run_id, label, video = press_table(d)
    keep = np.array([len(np.unique(video[label == c])) >= 2 for c in label])
    run_id, label, video = run_id[keep], label[keep], video[keep]
    cls = sorted(set(label.tolist()))
    y = np.array([{c: i for i, c in enumerate(cls)}[c] for c in label])

    sg = np.arange(len(y)) if random_split else video
    cv = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    tr, te = list(cv.split(np.zeros(len(y)), y, sg))[fold]
    return set(run_id[tr].tolist()), set(run_id[te].tolist()), cls, len(y)


def load_frames(dirs):
    """Per-frame features concatenated across views, plus each frame's run_id."""
    X, blocks, run_ids = [], [], None
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
    return np.concatenate(X, axis=1), run_ids, blocks


def probe(Xtr, ytr, Xte, yte, blocks, C):
    """Single linear layer, block-equalised, L2-normalised rows.

    Identical preprocessing to frame_linear_probe.py so the corrected numbers
    stay comparable to the ones already recorded. Scaling uses training-fold
    statistics only.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.clip(sd, 1e-8, None)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
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
    ap.add_argument("--tag", required=True, help="Label for the output record")
    ap.add_argument("--template", nargs="+", required=True,
                    help="Feature dirs; '{k}' is replaced by the fold index")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--random", action="store_true",
                    help="Press-level random split instead of session-grouped")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    kind = "random" if args.random else "grouped"
    out = args.out or Path(f"../results/frame_probe_{kind}_{args.tag}.json")

    recs = []
    for k in args.folds:
        dirs = [Path(t.format(k=k)) for t in args.template]
        if not all((d / "touch_frames.npy").exists() for d in dirs):
            print(f"fold {k}: per-frame features missing ({dirs[0]}), skipping",
                  flush=True)
            continue

        tr_ids, te_ids, cls, n_press = press_split(dirs[0], k, args.random)
        X, run_ids, blocks = load_frames(dirs)
        lab = dict(zip(*press_table(dirs[0])[:2]))
        cmap = {c: i for i, c in enumerate(cls)}

        # a frame belongs to whichever side its press was assigned
        side = np.array([2 if r in te_ids else (1 if r in tr_ids else 0)
                         for r in run_ids])
        y = np.array([cmap.get(lab[r], -1) for r in run_ids])
        tr = np.where((side == 1) & (y >= 0))[0]
        te = np.where((side == 2) & (y >= 0))[0]
        assert not (set(run_ids[tr].tolist()) & set(run_ids[te].tolist())), \
            "a press landed on both sides"

        print(f"\n=== {args.tag} fold {k} ===", flush=True)
        print(f"  {n_press:,} presses kept, {len(cls)} classes | "
              f"{X.shape[1]} dims {blocks}", flush=True)
        print(f"  train {len(tr):,} frames | test {len(te):,} frames", flush=True)

        # regularisation chosen on an inner split of TRAIN only, grouped by press
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
        i_fit, i_val = next(inner.split(np.zeros(len(tr)), y[tr],
                                        run_ids[tr]))
        best_C, best = 1.0, -1.0
        for C in (0.1, 1.0, 10.0):
            a, _ = probe(X[tr[i_fit]], y[tr[i_fit]],
                         X[tr[i_val]], y[tr[i_val]], blocks, C)
            print(f"    inner C={C:<5} acc {a:.4f}", flush=True)
            if a > best:
                best_C, best = C, a

        a, f = probe(X[tr], y[tr], X[te], y[te], blocks, best_C)
        print(f"  FOLD {k}  C={best_C}  acc {a:.4f}  macroF1 {f:.4f}", flush=True)
        recs.append({"fold": k, "C": best_C, "accuracy": float(a),
                     "macro_f1": float(f), "n_train": int(len(tr)),
                     "n_test": int(len(te))})
        del X

        # written after every fold so a power cut keeps what finished
        acc = np.array([r["accuracy"] for r in recs])
        f1s = np.array([r["macro_f1"] for r in recs])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"tag": args.tag, "template": args.template,
             "split": f"press-level {kind}, encoder k scores fold k",
             "folds": recs,
             "accuracy_mean": float(acc.mean()),
             "accuracy_std": float(acc.std()),
             "macro_f1_mean": float(f1s.mean()),
             "macro_f1_std": float(f1s.std()),
             "n_folds": len(recs)}, indent=1))

    if not recs:
        print("nothing scored")
        return
    acc = np.array([r["accuracy"] for r in recs])
    f1s = np.array([r["macro_f1"] for r in recs])
    print("\n" + "=" * 62)
    for r in recs:
        print(f"  fold {r['fold']}: acc {r['accuracy']:.4f}  "
              f"macroF1 {r['macro_f1']:.4f}")
    print(f"\n{args.tag}: accuracy {acc.mean()*100:.2f} +/- {acc.std()*100:.2f}"
          f"   macro-F1 {f1s.mean():.4f} +/- {f1s.std():.4f}"
          f"   ({len(recs)} folds)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
