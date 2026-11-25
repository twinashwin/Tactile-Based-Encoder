"""Is touch wrong in the same places vision is wrong?

Fusion can only help where one modality is right and the other is not. This
cross-tabulates per-example predictions from vision-only and touch-only under the
identical nested-CV split, and reports:

  - the 2x2 agreement table
  - P(touch right | vision wrong), which is fusion's entire opportunity
  - an ORACLE bound: accuracy if a perfect referee always picked the correct
    modality. No real fusion mechanism can beat it, so it bounds what fusion
    could ever be worth on these features.
  - the correlation between the two error patterns. Independent errors are what
    makes fusion pay; correlated errors mean both sensors fail on the same hard
    instances, for a shared underlying reason.

Usage:
    python complementarity.py --features ../features_dinop9 --out ../results
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from common import CLASS_NAMES
from nested_cv import EXCLUDE, OUTER_FOLDS, SEED, normalise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--touch-features", type=Path, default=None,
                    help="Take touch.npy from here instead, so the upgraded touch "
                         "encoder can be fused against the same vision features. "
                         "Row order is asserted identical.")
    ap.add_argument("--out", type=Path, default=Path("../results"))
    args = ap.parse_args()

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
    names = [CLASS_NAMES[c] for c in kept]
    y = np.array([{c: i for i, c in enumerate(kept)}[c] for c in y_raw])

    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                 random_state=SEED)
    pred = {k: np.full(len(y), -1) for k in ("vision", "touch", "fused")}
    for tr, te in outer.split(np.zeros(len(y)), y, groups):
        for k, X in (("vision", V), ("touch", T),
                     ("fused", np.hstack([V, T]))):
            Xtr, Xte = normalise(X[tr], [X[te]], True)
            m = LogisticRegression(max_iter=2000, C=1.0,
                                   class_weight="balanced").fit(Xtr, y[tr])
            pred[k][te] = m.predict(Xte)

    cv, ct, cf = (pred[k] == y for k in ("vision", "touch", "fused"))
    n = len(y)
    cells = {
        "both_right":   int((cv & ct).sum()),
        "vision_only":  int((cv & ~ct).sum()),
        "touch_only":   int((~cv & ct).sum()),
        "both_wrong":   int((~cv & ~ct).sum()),
    }
    p_v_err, p_t_err = float((~cv).mean()), float((~ct).mean())
    oracle = float((cv | ct).mean())

    rescue = []
    for i, nm in enumerate(names):
        mask = y == i
        verr = int((~cv & mask).sum())
        resc = int((~cv & ct & mask).sum())
        rescue.append({"name": nm, "n": int(mask.sum()), "vision_errors": verr,
                       "rescued": resc,
                       "rate": round(resc / verr, 3) if verr else None})

    out = {
        "n": n,
        "cells": cells,
        "vision_accuracy": round(float(cv.mean()), 4),
        "touch_accuracy": round(float(ct.mean()), 4),
        "fused_accuracy": round(float(cf.mean()), 4),
        "p_touch_right_given_vision_wrong":
            round(cells["touch_only"] / max(1, int((~cv).sum())), 3),
        "p_touch_right_given_vision_right":
            round(cells["both_right"] / max(1, int(cv.sum())), 3),
        "oracle_accuracy": round(oracle, 4),
        "headroom": round(oracle - float(cv.mean()), 4),
        "captured": round(float(cf.mean()) - float(cv.mean()), 4),
        "captured_fraction":
            round((float(cf.mean()) - float(cv.mean())) /
                  max(1e-9, oracle - float(cv.mean())), 3),
        "phi": round(float(np.corrcoef((~cv).astype(int),
                                       (~ct).astype(int))[0, 1]), 3),
        "p_both_wrong_observed": round(cells["both_wrong"] / n, 4),
        "p_both_wrong_independent": round(p_v_err * p_t_err, 4),
        "rescue_by_class": rescue,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "complementarity.json").write_text(json.dumps(out))

    print(f"{'':<20}{'touch RIGHT':>13}{'touch WRONG':>13}")
    print(f"{'vision RIGHT':<20}{cells['both_right']:>13,}{cells['vision_only']:>13,}")
    print(f"{'vision WRONG':<20}{cells['touch_only']:>13,}{cells['both_wrong']:>13,}")
    print(f"\noracle {oracle:.4f} | vision {cv.mean():.4f} | fused {cf.mean():.4f}")
    print(f"captured {out['captured']:+.4f} of {out['headroom']:+.4f} headroom "
          f"({out['captured_fraction']:.0%})")
    print(f"phi(errors) = {out['phi']:+.3f}")
    print(f"\nwrote {args.out / 'complementarity.json'}")


if __name__ == "__main__":
    main()
