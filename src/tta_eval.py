"""Re-score already-trained fine-tune checkpoints with test-time frame averaging.

WHY THIS IS CHEAP
-----------------
The fine-tune trains on ONE frame per press because nine frames would be nine
times the training cost. But nothing forces inference to use one frame. Each press
is a contiguous run of frames of the same material, so averaging the model's
predictions across nine of them costs only inference -- and inference on DirectML
is ~52 img/s against ~7 img/s for training. Five folds is about twenty minutes.

Crucially this needs NO retraining: it loads the fold checkpoints the clean 5-fold
run already wrote and only changes how they are scored. If it helps, it is the
best accuracy-per-GPU-hour change available; if it does not, it cost twenty
minutes.

Averaging LOGITS rather than votes, and reporting both, because a hard vote throws
away the model's confidence and nine frames is few enough that ties matter.

Usage:
    python tta_eval.py --ckpt-dir ../results/ft_ckpt_clean --frames 9
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from common import CLASS_NAMES, select_pool_frames, select_rows
from finetune_dino import (EVAL_TF, Net, build_transforms, expand_patch_embed,
                           pick_device)
from nested_cv import EXCLUDE, OUTER_FOLDS, SEED


class Flat(Dataset):
    """Flat list of frames; the caller regroups them by press afterwards."""

    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--like", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("../results/ft_ckpt_clean"))
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--norm", default="imagenet")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("../results/tta.json"))
    args = ap.parse_args()

    dev, kind = pick_device(args.device)
    _, eval_tf = build_transforms(args.size, args.norm, "none")
    print(f"device: {kind} | {args.frames} frames per press at test time")

    df = pd.read_csv(args.index)
    rows = pd.read_csv(args.like / "rows.csv")

    # the 1-frame selection defines the folds, exactly as in training, so the
    # split is identical and the only thing that changes is test-time pooling
    one = select_rows(df, "run")
    one = one[one.run_id.isin(set(rows.run_id))]
    one = one[[Path(p).exists() for p in one.gelsight_frame_path]]
    one = one[[CLASS_NAMES[int(c)] not in EXCLUDE for c in one.label]]
    groups = one.video_id.to_numpy()
    kept = sorted(one.label.unique().tolist())
    m = {c: i for i, c in enumerate(kept)}
    y = np.array([m[c] for c in one.label])
    n_classes = len(kept)
    run_order = one.run_id.tolist()

    # multi-frame selection, grouped by run so each press keeps its own frames
    pool = select_pool_frames(df, args.frames)
    pool = pool[pool.run_id.isin(set(run_order))]
    pool = pool[[Path(p).exists() for p in pool.gelsight_frame_path]]
    by_run = pool.groupby("run_id").gelsight_frame_path.apply(list).to_dict()
    avail = np.array([len(by_run.get(r, [])) for r in run_order])
    print(f"{len(run_order):,} presses | frames per press: "
          f"min {avail.min()} max {avail.max()} mean {avail.mean():.1f}")

    cv = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    out = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        if fold >= args.folds:
            break
        bp = args.ckpt_dir / f"fold{fold}_best.pt"
        if not bp.exists():
            print(f"fold {fold}: no checkpoint, skipping")
            continue
        backbone = torch.hub.load("facebookresearch/dinov2", args.backbone,
                                  verbose=False)
        expand_patch_embed(backbone, 3)
        model = Net(backbone, n_classes)
        try:
            model.load_state_dict(torch.load(bp, map_location="cpu"))
        except Exception as e:
            print(f"fold {fold}: checkpoint unreadable ({type(e).__name__}); skipping")
            continue
        model = model.to(dev).eval()

        # flatten every test press's frames into one pass, remembering owners
        paths, owner = [], []
        for j, i in enumerate(te):
            fs = by_run.get(run_order[i], [])
            if not fs:
                fs = [one.gelsight_frame_path.iloc[i]]
            paths.extend(fs)
            owner.extend([j] * len(fs))
        owner = np.array(owner)

        logits = []
        dl = DataLoader(Flat(paths, eval_tf), batch_size=args.batch,
                        shuffle=False, num_workers=0)
        with torch.no_grad():
            for bi, xb in enumerate(dl):
                logits.append(model(xb.to(dev)).detach().cpu().numpy())
                if bi % 60 == 0:
                    print(f"  fold {fold}: {bi*args.batch:,}/{len(paths):,}",
                          flush=True)
        L = np.concatenate(logits)
        yt = y[te]

        n = len(te)
        mean_logit = np.zeros((n, n_classes), dtype=np.float64)
        mean_prob = np.zeros((n, n_classes), dtype=np.float64)
        votes = np.zeros((n, n_classes), dtype=np.float64)
        P = torch.softmax(torch.from_numpy(L), 1).numpy()
        for j in range(n):
            s = owner == j
            mean_logit[j] = L[s].mean(0)
            mean_prob[j] = P[s].mean(0)
            for c in L[s].argmax(1):
                votes[j, c] += 1
        # single-frame reference: the middle frame of each press, which is what
        # the training run was scored on
        mid = np.zeros((n, n_classes), dtype=np.float64)
        for j in range(n):
            idx = np.flatnonzero(owner == j)
            mid[j] = L[idx[len(idx) // 2]]

        row = {"fold": fold}
        for tag, M in (("1frame", mid), ("mean_logit", mean_logit),
                       ("mean_prob", mean_prob), ("vote", votes)):
            p = M.argmax(1)
            row[tag] = {"accuracy": float(accuracy_score(yt, p)),
                        "macro_f1": float(f1_score(yt, p, average="macro",
                                                   zero_division=0))}
        print(f"  fold {fold}  1frame {row['1frame']['accuracy']:.4f}  "
              f"mean_logit {row['mean_logit']['accuracy']:.4f}  "
              f"mean_prob {row['mean_prob']['accuracy']:.4f}  "
              f"vote {row['vote']['accuracy']:.4f}", flush=True)
        out.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"frames": args.frames, "folds": out},
                                       indent=1))

    if out:
        print()
        for tag in ("1frame", "mean_logit", "mean_prob", "vote"):
            a = [r[tag]["accuracy"] for r in out]
            f = [r[tag]["macro_f1"] for r in out]
            print(f"{tag:<12} accuracy {np.mean(a):.4f} +/- {np.std(a):.4f}   "
                  f"macro-F1 {np.mean(f):.4f}")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
