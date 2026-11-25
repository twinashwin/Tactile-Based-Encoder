"""Multiple-instance learning over DINOv2 patch tokens.

THE IDEA
--------
Material identity is a texture property, and texture is statistically
stationary: if a press is Gravel then essentially every 14x14 patch of that
imprint is Gravel. The supervised pipeline throws that structure away -- it
attaches one label to one pooled 768-d vector per press, so 3,672 training
presses give 3,672 training examples against a 22M-parameter encoder, and we
measure a 25 point train/validation gap.

Treating each patch as its own labelled instance turns the same data into
~940,000 training examples (3,672 x 256), and pooling the per-patch predictions
back up to the press is a permutation-invariant aggregation -- the correct
inductive bias for a stationary texture, where global self-attention over the
whole imprint is modelling long-range structure that does not determine the
answer.

WHAT THIS DOES AND DOES NOT TEST
--------------------------------
It runs on FROZEN DINOv2 features, so it is comparable to the frozen baseline
(0.6222 +/- 0.0022, 3 seeds), not to the fine-tuned 0.6409. It isolates one
thing: whether patch-level supervision plus pooling beats press-level
supervision on the same representation. The `--mean-pool` control trains the
identical head on mean-pooled patches, which separates "more training signal"
from "better aggregation" -- without it a gain could be either.

Usage:
    python mil_patches.py --extract          # ~2 min on GPU, writes 1.2 GB
    python mil_patches.py                    # trains + scores, CPU
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from common import CLASS_NAMES, select_rows
from finetune_dino import build_transforms, pick_device
from nested_cv import EXCLUDE, OUTER_FOLDS, SEED

NPATCH = 256          # (224 / 14) ** 2
DIM = 384


def selection(index, like):
    df = pd.read_csv(index)
    sel = select_rows(df, "run")
    rows = pd.read_csv(like / "rows.csv")
    sel = sel[sel.run_id.isin(set(rows.run_id))]
    sel = sel[[Path(p).exists() for p in sel.gelsight_frame_path]]
    sel = sel[[CLASS_NAMES[int(c)] not in EXCLUDE for c in sel.label]]
    return sel


class Frames(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def extract(args, sel):
    dev, kind = pick_device(args.device)
    _, eval_tf = build_transforms(224, "imagenet", "none")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           verbose=False).eval().to(dev)
    for p in model.parameters():
        p.requires_grad = False
    paths = sel.gelsight_frame_path.tolist()
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(args.cache, mode="w+", dtype=np.float16,
                                   shape=(len(paths), NPATCH, DIM))
    dl = DataLoader(Frames(paths, eval_tf), batch_size=args.batch,
                    shuffle=False, num_workers=0)
    print(f"device: {kind} | extracting {len(paths):,} x {NPATCH} patch tokens")
    k = 0
    with torch.no_grad():
        for bi, xb in enumerate(dl):
            t = model.forward_features(xb.to(dev))["x_norm_patchtokens"]
            a = t.detach().cpu().numpy().astype(np.float16)
            mm[k:k + len(a)] = a
            k += len(a)
            if bi % 40 == 0:
                print(f"  {k:,}/{len(paths):,}", flush=True)
    mm.flush()
    print(f"wrote {args.cache}  {mm.nbytes/1e9:.2f} GB")


class Head(nn.Module):
    def __init__(self, n_classes, d=DIM, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden),
                                 nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--like", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--cache", type=Path, default=Path("../cache_patches/tok.npy"))
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--patches-per-press", type=int, default=64,
                    help="Training subsample. All 256 are used at test time.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-batch", type=int, default=1024)
    ap.add_argument("--mean-pool", action="store_true",
                    help="Control: same head on mean-pooled patches, i.e. "
                         "press-level supervision on the same features")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("../results/mil.json"))
    args = ap.parse_args()

    sel = selection(args.index, args.like)
    if args.extract:
        extract(args, sel)
        return

    groups = sel.video_id.to_numpy()
    kept = sorted(sel.label.unique().tolist())
    m = {c: i for i, c in enumerate(kept)}
    y = np.array([m[c] for c in sel.label])
    n_classes = len(kept)
    T = np.load(args.cache, mmap_mode="r")
    assert T.shape[0] == len(y), f"cache has {T.shape[0]} rows, selection {len(y)}"
    print(f"{len(y):,} presses | tokens {T.shape} | {n_classes} classes")

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    cv = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    out = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        if fold >= args.folds:
            break
        # build the training matrix: either k patches per press (MIL) or the
        # press mean (control)
        if args.mean_pool:
            Xtr = np.asarray(T[tr], dtype=np.float32).mean(1)
            ytr = y[tr]
        else:
            pick = rng.integers(0, NPATCH, size=(len(tr), args.patches_per_press))
            Xtr = np.stack([np.asarray(T[i], dtype=np.float32)[p]
                            for i, p in zip(tr, pick)]).reshape(-1, DIM)
            ytr = np.repeat(y[tr], args.patches_per_press)
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        print(f"fold {fold}: train rows {len(Xtr):,} "
              f"({'mean-pool' if args.mean_pool else 'MIL patches'})")

        head = Head(n_classes)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        Xt = torch.from_numpy(Xtr)
        yt = torch.from_numpy(ytr.astype(np.int64))
        lossf = nn.CrossEntropyLoss()
        n = len(Xt)
        for ep in range(args.epochs):
            head.train()
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, args.train_batch):
                idx = perm[s:s + args.train_batch]
                opt.zero_grad()
                l = lossf(head(Xt[idx]), yt[idx])
                l.backward(); opt.step()
                tot += float(l.detach()) * len(idx)
            print(f"  ep {ep+1}/{args.epochs} loss {tot/n:.4f}", flush=True)

        # score: average the per-patch probabilities over all 256 patches
        head.eval()
        preds_mil, preds_mean = [], []
        with torch.no_grad():
            for i in te:
                P = (np.asarray(T[i], dtype=np.float32) - mu) / sd
                lg = head(torch.from_numpy(P))
                preds_mil.append(int(lg.softmax(1).mean(0).argmax()))
                pm = (np.asarray(T[i], dtype=np.float32).mean(0, keepdims=True)
                      - mu) / sd
                preds_mean.append(int(head(torch.from_numpy(pm)).argmax()))
        row = {"fold": fold}
        for tag, p in (("pooled_prob", preds_mil), ("mean_token", preds_mean)):
            row[tag] = {
                "accuracy": float(accuracy_score(y[te], p)),
                "macro_f1": float(f1_score(y[te], p, average="macro",
                                           zero_division=0))}
        print(f"  fold {fold}  pooled {row['pooled_prob']['accuracy']:.4f}  "
              f"mean-token {row['mean_token']['accuracy']:.4f}", flush=True)
        out.append(row)

    print()
    for tag in ("pooled_prob", "mean_token"):
        a = [r[tag]["accuracy"] for r in out]
        f = [r[tag]["macro_f1"] for r in out]
        print(f"{tag:<12} accuracy {np.mean(a):.4f} +/- {np.std(a):.4f}   "
              f"macro-F1 {np.mean(f):.4f}")
    print("frozen baseline 0.6222 +/- 0.0022 | fine-tuned 0.6409 +/- 0.0296")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"mode": "mean_pool" if args.mean_pool else "mil",
         "patches_per_press": args.patches_per_press, "folds": out}, indent=1))


if __name__ == "__main__":
    main()
