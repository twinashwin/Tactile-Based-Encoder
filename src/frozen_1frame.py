"""Frozen-probe baseline on ONE midpoint frame per press -- the control for fine-tuning.

Fine-tuning can only afford one frame per press (nine frames x five folds is 43
hours on DirectML). Comparing a one-frame fine-tune against the nine-frame frozen
headline of 0.6154 would therefore confound two changes at once: unfreezing the
encoder, and throwing away eight ninths of the input.

This script supplies the missing control -- the SAME encoder, FROZEN, on the SAME
single midpoint frame, scored on the SAME grouped fold. The difference between
this and the fine-tuned run isolates exactly one thing: whether training the
encoder's weights helps.

Runs on GPU where available; the forward pass is ~52 img/s on DirectML, so the
whole thing takes a couple of minutes.

Usage:
    python frozen_1frame.py --folds 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import CLASS_NAMES, select_rows
from nested_cv import EXCLUDE, OUTER_FOLDS, SEED

TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def pick_device(name):
    if name == "cpu":
        return torch.device("cpu"), "cpu"
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if name in ("auto", "directml"):
        try:
            import torch_directml
            return torch_directml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


class Frames(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return TF(im.convert("RGB"))


def normalise(train_X, others):
    sc = StandardScaler().fit(train_X)
    out = [sc.transform(X).astype(np.float32) for X in [train_X, *others]]
    return [X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None) for X in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--like", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("../results/frozen_1frame.json"))
    args = ap.parse_args()

    dev, kind = pick_device(args.device)
    print(f"device: {kind}")

    df = pd.read_csv(args.index)
    sel = select_rows(df, "run")
    rows = pd.read_csv(args.like / "rows.csv")
    sel = sel[sel.run_id.isin(set(rows.run_id))]
    sel = sel[[Path(p).exists() for p in sel.gelsight_frame_path]]
    sel = sel[[CLASS_NAMES[int(c)] not in EXCLUDE for c in sel.label]]
    groups = sel.video_id.to_numpy()
    kept = sorted(sel.label.unique().tolist())
    m = {c: i for i, c in enumerate(kept)}
    y = np.array([m[c] for c in sel.label])
    paths = sel.gelsight_frame_path.tolist()
    print(f"{len(paths):,} presses | {len(kept)} classes | {len(set(groups)):,} sessions")

    model = torch.hub.load("facebookresearch/dinov2", args.backbone, verbose=False).eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(dev)

    feats = []
    loader = DataLoader(Frames(paths), batch_size=args.batch, shuffle=False, num_workers=0)
    with torch.no_grad():
        for bi, xb in enumerate(loader):
            f = model.forward_features(xb.to(dev))
            cls, patch = f["x_norm_clstoken"], f["x_norm_patchtokens"]
            # same representation the fine-tuned head sees, so the only
            # difference between the two runs is whether weights are trained
            v = torch.cat([cls, patch.mean(1), patch.std(1)], dim=1)
            feats.append(v.detach().cpu().numpy())
            if bi % 40 == 0:
                print(f"  {bi*args.batch:,}/{len(paths):,}", flush=True)
    X = np.concatenate(feats).astype(np.float32)
    print(f"features {X.shape}")

    cv = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    out = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        if fold >= args.folds:
            break
        Xtr, Xte = normalise(X[tr], [X[te]])
        for tag, clf in (("Logistic", LogisticRegression(max_iter=2000, C=1.0)),
                         ("SVM-RBF", SVC(kernel="rbf", C=10.0, gamma="scale"))):
            p = clf.fit(Xtr, y[tr]).predict(Xte)
            acc = accuracy_score(y[te], p)
            f1 = f1_score(y[te], p, average="macro", zero_division=0)
            print(f"  fold {fold}  {tag:<9} acc {acc:.4f}  macroF1 {f1:.4f}  "
                  f"acc_f1 {0.5*(acc+f1):.4f}")
            out.append({"fold": fold, "head": tag, "accuracy": acc,
                        "macro_f1": f1, "acc_f1": 0.5 * (acc + f1)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"note": "FROZEN encoder, 1 midpoint frame per press - the control for "
                 "finetune_dino.py, which uses the same frames and folds",
         "device": kind, "runs": out}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
