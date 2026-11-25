"""Fine-tune DINOv2 on gel imprints instead of probing it frozen.

WHY THIS EXISTS
---------------
Every result in this project so far comes from a FROZEN encoder: DINOv2's weights
have never seen a gel image. Thirteen ideas were tested on top of those frozen
features and the pattern was consistent -- anything applied after the encoder
returned roughly zero, three independent hand-crafted descriptor sets all
plateaued near 0.33, and the representation tweaks that did work were worth about
a point each.

Meanwhile the published numbers being chased are fine-tuned: TaCo-L 75.1%,
ViTaPEs 80.1%. UniTouch's 61.3% is a probe, but over an encoder that was
pretrained on tactile data. Nobody reaches those numbers with a frozen
photograph encoder, and that is the one lever this project has never pulled.

HARDWARE
--------
Runs on DirectML (AMD) or CUDA, falling back to CPU. On an RX 6750 XT via
DirectML a training step costs ~1.46 s for a batch of 8, i.e. 5.5 img/s --
about 3 hours for 6,143 presses x 10 epochs. Note that DirectML's backward
kernels are far slower relative to its forward pass (51.9 img/s inference), so
this is roughly 4x slower than the inference speedup would suggest.

PROTOCOL
--------
Session-grouped splits, exactly as everywhere else in this project: no recording
session ever appears in both train and test. Default is ONE held-out fold, not
the full 5-fold nest, because five folds would be ~12 hours; the single fold is
for deciding whether unfreezing helps at all, and is NOT comparable to the
nested-CV headline. Use --folds 5 for the real number once it looks worth it.

ONE FRAME PER PRESS. The frozen pipeline pools nine frames, which is worth real
accuracy, so fine-tuning starts at a handicap and has to beat that before any
gain is genuine. Clearing ~0.65 on single frames would be strong evidence the
nine-frame version is worth a GPU with proper CUDA.

Usage:
    python finetune_dino.py --epochs 10 --batch 8
    python finetune_dino.py --folds 5 --epochs 12        # the real run
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import CLASS_NAMES, select_rows
from nested_cv import EXCLUDE, OUTER_FOLDS, SEED

IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# measured over Touch and Go gel frames by src/frame_stats.py -- the gel is lit by
# three fixed coloured LEDs, so its colour distribution is nothing like ImageNet's
GELSIGHT_STATS = ([0.5438, 0.5149, 0.5478], [0.1448, 0.1172, 0.1349])
# std is ~60% of ImageNet's: the gel is a low-contrast surface, so ImageNet
# normalisation squashes every frame below the range the encoder expects


class PhotometricHFlip:
    """Mirror the image AND remap colour channels to keep the lighting physical.

    A GelSight encodes surface slope as the ratio of three coloured LEDs placed at
    fixed angles. Mirroring the pixels mirrors the geometry but leaves the implied
    light directions untouched, which produces a frame no real sensor could
    output: the standard RandomHorizontalFlip is a physically invalid augmentation
    here. Reflecting the LED triangle maps two of its vertices onto each other, so
    the geometric flip has to be paired with swapping the two corresponding
    channels. Which pair depends on the sensor's actual LED layout, so `swap` is a
    parameter to be determined empirically rather than assumed.
    """

    def __init__(self, swap=(0, 2), p=0.5):
        self.swap, self.p = swap, p

    def __call__(self, t):                      # t is CHW, post-ToTensor
        if torch.rand(1).item() >= self.p:
            return t
        t = torch.flip(t, dims=[2])
        i, j = self.swap
        t = t.clone()
        t[[i, j]] = t[[j, i]]
        return t


def build_transforms(size: int, norm: str, flip: str):
    """DINOv2 was trained at 224 on photographs resized from a short side of 256.

    Gel frames are 640x480, so the default pipeline downsamples ~2.9x and then
    crops the edges away -- discarding exactly the high-frequency detail that
    distinguishes one texture from another. Keeping the 8:7 resize-to-crop ratio
    lets --size scale the whole pipeline; patch count grows as (size/14)^2, so 448
    costs 4x the compute of 224.
    """
    mean, std = GELSIGHT_STATS if norm == "gelsight" else IMAGENET_STATS
    nm = transforms.Normalize(mean, std)
    resize = int(round(size * 8 / 7))

    geo = [transforms.Resize(resize),
           transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1))]
    # a gel imprint has no canonical up, and a vertical flip keeps each LED on its
    # own side of the mirror axis far more often than a horizontal one does
    post = [transforms.RandomVerticalFlip()]
    if flip == "plain":
        post.insert(0, transforms.RandomHorizontalFlip())
        train = transforms.Compose(geo + post + [transforms.ToTensor(), nm])
    elif flip == "photometric":
        train = transforms.Compose(
            geo + post + [transforms.ToTensor(), PhotometricHFlip(), nm])
    elif flip == "none":
        train = transforms.Compose(geo + [transforms.ToTensor(), nm])
    else:
        raise ValueError(flip)

    ev = transforms.Compose([transforms.Resize(resize),
                             transforms.CenterCrop(size),
                             transforms.ToTensor(), nm])
    return train, ev


TRAIN_TF, EVAL_TF = build_transforms(224, "imagenet", "plain")


# ch0 = background-subtracted deformation, ch1 = marker-size map (128 = unchanged).
# The exact values matter little: the two new patch-embedding channels are
# zero-initialised, so the model starts numerically identical to pretrained and
# has to learn its way into using them.
EXTRA_STATS = ([0.1151, 0.5197], [0.1869, 0.1164])   # measured over cache_extra


def build_transforms5(size: int, norm: str, flip: str):
    """Same geometry as build_transforms, but applied to a 5-channel tensor.

    torchvision geometric transforms accept CHW tensors, so RGB and the two extra
    channels are cropped and flipped together. They must stay registered to each
    other or the marker positions stop matching the pixels they describe.
    """
    mean, std = GELSIGHT_STATS if norm == "gelsight" else IMAGENET_STATS
    nm = transforms.Normalize(list(mean) + EXTRA_STATS[0],
                              list(std) + EXTRA_STATS[1])
    resize = int(round(size * 8 / 7))
    geo = [transforms.Resize(resize, antialias=True),
           transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1),
                                        antialias=True)]
    post = [transforms.RandomVerticalFlip()]
    if flip == "plain":
        post.insert(0, transforms.RandomHorizontalFlip())
    elif flip == "photometric":
        post.append(PhotometricHFlip())
    train = transforms.Compose(geo + post + [nm])
    ev = transforms.Compose([transforms.Resize(resize, antialias=True),
                             transforms.CenterCrop(size), nm])
    return train, ev


class Presses5(Dataset):
    """RGB stacked with the two precomputed physical channels."""

    def __init__(self, paths, labels, idx, extra, train: bool):
        self.paths, self.labels, self.idx = paths, labels, idx
        self.extra = extra
        self.tf = TRAIN_TF5 if train else EVAL_TF5

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        ex = np.asarray(self.extra[self.idx[i]])
        if ex.shape[:2] != rgb.shape[:2]:
            import cv2
            ex = cv2.resize(ex, (rgb.shape[1], rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        a = np.concatenate([rgb, ex], axis=2)            # H W 5
        t = torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0)
        return self.tf(t), int(self.labels[i])


TRAIN_TF5, EVAL_TF5 = build_transforms5(224, "imagenet", "plain")


def expand_patch_embed(backbone, in_ch: int):
    """Widen DINOv2 first layer from 3 input channels to in_ch.

    The patch embedding is literally Conv2d(3, 384, kernel=14, stride=14), so the
    advisor suggestion of changing the input from 3 to 5 is exactly right. The
    pretrained kernels are copied into the first three channels and the new ones
    start at zero, so step 0 reproduces the pretrained model and nothing is
    destroyed; the extra channels can only earn their way in by reducing loss.
    """
    pe = backbone.patch_embed.proj
    if pe.in_channels == in_ch:
        return
    new = nn.Conv2d(in_ch, pe.out_channels, kernel_size=pe.kernel_size,
                    stride=pe.stride, padding=pe.padding,
                    bias=pe.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :pe.in_channels].copy_(pe.weight)
        if pe.bias is not None:
            new.bias.copy_(pe.bias)
    backbone.patch_embed.proj = new
    backbone.patch_embed.in_chans = in_ch


def pick_device(name: str):
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


class Presses(Dataset):
    def __init__(self, paths, labels, train: bool):
        self.paths, self.labels, self.tf = paths, labels, TRAIN_TF if train else EVAL_TF

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), int(self.labels[i])


class Net(nn.Module):
    """DINOv2 backbone with a linear head over CLS + mean/std of the patch tokens.

    The frozen pipeline found patch statistics beat the CLS summary on gel
    imprints, because CLS is trained to be invariant to exactly the local texture
    that identifies a material. Keeping all three gives the head the same
    information and lets fine-tuning decide how to weight it.
    """

    def __init__(self, backbone, n_classes: int, d: int = 384):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(d * 3, n_classes)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        cls, patch = f["x_norm_clstoken"], f["x_norm_patchtokens"]
        return self.head(torch.cat([cls, patch.mean(1), patch.std(1)], dim=1))


def evaluate(model, loader, dev, n_classes):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(dev))
            P.append(logits.detach().cpu().argmax(1).numpy())
            Y.append(yb.numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro", zero_division=0)
    return acc, f1, 0.5 * (acc + f1), p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--like", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--freeze-blocks", type=int, default=0,
                    help="freeze the first N transformer blocks; with only ~4,900 "
                         "training presses this is a cheap guard against "
                         "overfitting and speeds each step up")
    ap.add_argument("--folds", type=int, default=1,
                    help="how many of the 5 grouped folds to actually run")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "directml", "cpu"])
    ap.add_argument("--out", type=Path, default=Path("../results/finetune.json"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("../results/ft_ckpt"),
                    help="model/optimizer state is saved here after every epoch AND "
                         "every --save-steps batches, so a run killed by a wall-clock "
                         "limit resumes exactly where it stopped")
    ap.add_argument("--save-steps", type=int, default=120)
    ap.add_argument("--class-weight", default="balanced",
                    choices=["balanced", "sqrt", "none"],
                    help="How hard to upweight rare classes in the loss. "
                         "'balanced' maximises macro-F1 but spends accuracy on "
                         "the 55%% of presses that sit in the top three classes; "
                         "'none' is pure accuracy; 'sqrt' is the midpoint")
    ap.add_argument("--size", type=int, default=224,
                    help="Input resolution. 448 keeps ~4x the pixels of the "
                         "default and costs ~4x the GPU time per epoch")
    ap.add_argument("--norm", default="imagenet", choices=["imagenet", "gelsight"])
    ap.add_argument("--channels", type=int, default=3, choices=[3, 5],
                    help="5 widens the patch embedding and adds the "
                         "background-subtracted deformation and marker-size "
                         "maps from make_extra_channels.py")
    ap.add_argument("--extra", type=Path, default=Path("../cache_extra"))
    ap.add_argument("--init", type=Path, default=None,
                    help="Checkpoint from pretrain_mae.py whose 'encoder' weights "
                         "replace the ImageNet-pretrained backbone. Loaded BEFORE "
                         "the patch embedding is widened, since the MAE trains on "
                         "3 channels")
    ap.add_argument("--flip", default="plain",
                    choices=["plain", "photometric", "none"],
                    help="plain = the usual RandomHorizontalFlip, which is "
                         "physically invalid on a gel frame; photometric pairs the "
                         "mirror with an LED channel swap; none drops it")
    ap.add_argument("--ungrouped", action="store_true",
                    help="Reproduce the LEAKY protocol the published Touch and Go "
                         "numbers use: random stratified folds, sessions NOT held "
                         "out, so near-duplicate presses from the same session sit "
                         "on both sides. Measured for like-for-like comparison with "
                         "prior work; never the headline. The frozen pipeline gains "
                         "+14.11 points this way.")
    ap.add_argument("--inner-val", type=int, default=4,
                    help="Carve a session-grouped validation split out of the "
                         "TRAINING data and pick the best epoch on that, so the "
                         "outer test fold is never used for selection. 0 restores "
                         "the old leaky behaviour of selecting on test.")
    ap.add_argument("--select-on", default="accuracy",
                    choices=["accuracy", "macro_f1", "acc_f1"],
                    help="Which validation score picks the best epoch. This must "
                         "match the headline metric or we tune for one thing and "
                         "report another")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="checkpoint and exit cleanly after this long; 0 = run to "
                         "completion. One epoch is ~15 min here, longer than the "
                         "harness background limit, so mid-epoch checkpointing is "
                         "what makes an unattended multi-hour run possible at all")
    args = ap.parse_args()

    dev, kind = pick_device(args.device)
    global TRAIN_TF, EVAL_TF, TRAIN_TF5, EVAL_TF5
    TRAIN_TF, EVAL_TF = build_transforms(args.size, args.norm, args.flip)
    TRAIN_TF5, EVAL_TF5 = build_transforms5(args.size, args.norm, args.flip)
    print(f"device: {kind} | size {args.size} | norm {args.norm} | "
          f"flip {args.flip} | channels {args.channels}")
    torch.manual_seed(SEED)

    df = pd.read_csv(args.index)
    sel = select_rows(df, "run")                       # midpoint = peak contact
    rows = pd.read_csv(args.like / "rows.csv")
    sel = sel[sel.run_id.isin(set(rows.run_id))]
    sel = sel[[Path(p).exists() for p in sel.gelsight_frame_path]]
    keep = np.array([CLASS_NAMES[int(c)] not in EXCLUDE for c in sel.label])
    sel = sel[keep]
    groups = sel.video_id.to_numpy()
    kept = sorted(sel.label.unique().tolist())
    m = {c: i for i, c in enumerate(kept)}
    y = np.array([m[c] for c in sel.label])
    paths = sel.gelsight_frame_path.tolist()
    names = [CLASS_NAMES[c] for c in kept]
    n_classes = len(kept)
    print(f"{len(paths):,} presses | {n_classes} classes | "
          f"{len(set(groups)):,} sessions | 1 frame per press")

    EXTRA = None
    if args.channels == 5:
        er = pd.read_csv(args.extra / "rows.csv")
        assert er.run_id.tolist() == list(sel.run_id), (
            "cache_extra/rows.csv does not match this selection; "
            "rerun make_extra_channels.py")
        EXTRA = np.load(args.extra / "extra.npy", mmap_mode="r")
        print(f"extra channels {EXTRA.shape} from {args.extra}")

    split_groups = groups
    if args.ungrouped:
        # every press is its own group -> StratifiedGroupKFold degenerates to a
        # stratified random split. Applied to BOTH the outer folds and the inner
        # selection split, so the whole protocol is leaky, matching the baselines.
        split_groups = np.arange(len(y))
        print("*** UNGROUPED (LEAKY) SPLIT -- sessions span train and test ***")

    cv = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    results = []
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, split_groups)):
        if fold >= args.folds:
            break
        assert not (set(split_groups[tr]) & set(split_groups[te]))
        if args.inner_val:
            # select the epoch on held-out TRAINING sessions, never on test
            inner = StratifiedGroupKFold(args.inner_val, shuffle=True, random_state=SEED)
            i_fit, i_sel = next(inner.split(np.zeros(len(tr)), y[tr],
                                             split_groups[tr]))
            fit, sel_idx = tr[i_fit], tr[i_sel]
            assert not (set(split_groups[fit]) & set(split_groups[sel_idx]))
            assert not (set(split_groups[fit]) & set(split_groups[te]))
        else:
            fit, sel_idx = tr, te
        def make(ix, train):
            if args.channels == 5:
                return Presses5([paths[i] for i in ix], y[ix], ix, EXTRA, train)
            return Presses([paths[i] for i in ix], y[ix], train)
        tl = DataLoader(make(fit, True), batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=True)
        sl = DataLoader(make(sel_idx, False), batch_size=args.batch,
                        shuffle=False, num_workers=0)
        vl = DataLoader(make(te, False), batch_size=args.batch,
                        shuffle=False, num_workers=0)
        print(f"  fold {fold}: fit {len(fit):,} | select {len(sel_idx):,} | "
              f"test {len(te):,}"
              + ("" if args.inner_val else "   *** SELECTING ON TEST (leaky) ***"))

        backbone = torch.hub.load("facebookresearch/dinov2", args.backbone, verbose=False)
        if args.init:
            st = torch.load(args.init, map_location="cpu", weights_only=False)
            sd = st["encoder"] if "encoder" in st else st
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            assert not unexpected, f"unexpected keys in --init: {unexpected[:5]}"
            print(f"  loaded MAE encoder from {args.init.name} "
                  f"(step {st.get('step','?')}, {len(missing)} keys left at init)")
        expand_patch_embed(backbone, args.channels)
        if args.freeze_blocks:
            for blk in backbone.blocks[:args.freeze_blocks]:
                for p in blk.parameters():
                    p.requires_grad = False
        model = Net(backbone, n_classes).to(dev)

        if args.class_weight == "none":
            lossf = nn.CrossEntropyLoss()
        else:
            cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y[fit])
            if args.class_weight == "sqrt":
                # halves the exponent on the 111:1 imbalance, so Sand (n=12) is
                # worth ~10x Synthetic Fabric instead of ~111x
                cw = np.sqrt(cw)
            lossf = nn.CrossEntropyLoss(
                weight=torch.tensor(cw, dtype=torch.float32).to(dev))
        opt = torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad],
             "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ], weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        cpath = args.ckpt_dir / f"fold{fold}.pt"
        bpath = args.ckpt_dir / f"fold{fold}_best.pt"
        best, start_ep, skip_steps = None, 0, 0
        if cpath.exists():
            try:
                st = torch.load(cpath, map_location="cpu", weights_only=False)
                model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
                sched.load_state_dict(st["sched"])
                best, start_ep, skip_steps = st["best"], st["epoch"], st["step"]
                print(f"  fold {fold}: resuming at epoch {start_ep+1}, "
                      f"step {skip_steps}")
            except Exception as e:
                print(f"  fold {fold}: checkpoint unreadable ({type(e).__name__}: "
                      f"{e}); discarding it and training this fold from scratch",
                      flush=True)
                cpath.unlink(missing_ok=True)
                bpath.unlink(missing_ok=True)
                best, start_ep, skip_steps = None, 0, 0

        def atomic_save(obj, path):
            tmp = path.with_name(path.name + ".tmp")
            torch.save(obj, tmp)
            os.replace(tmp, path)          # atomic on NTFS within a volume

        def save(ep, step):
            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "best": best,
                         "epoch": ep, "step": step}, cpath)

        t_start = time.time()
        out_of_time = False
        for ep in range(start_ep, args.epochs):
            model.train()
            # nb counts only the batches THIS invocation ran; dividing by
            # len(tl) would under-report the loss after a resume
            t0, seen, tot, nb = time.time(), 0, 0.0, 0
            for bi, (xb, yb) in enumerate(tl):
                if bi < skip_steps:
                    continue
                opt.zero_grad()
                loss = lossf(model(xb.to(dev)), yb.to(dev))
                loss.backward()
                opt.step()
                tot += float(loss.detach().cpu()); seen += len(xb); nb += 1
                if (bi + 1) % args.save_steps == 0:
                    save(ep, bi + 1)
                if args.max_minutes and (time.time() - t_start) / 60 >= args.max_minutes:
                    save(ep, bi + 1)
                    print(f"  fold {fold} ep {ep+1}: time limit, checkpointed at "
                          f"step {bi+1}/{len(tl)}", flush=True)
                    out_of_time = True
                    break
            if out_of_time:
                break
            skip_steps = 0
            sched.step()
            acc, f1, combo, _, _ = evaluate(model, sl, dev, n_classes)
            rate = seen / max(1e-9, time.time() - t0)
            print(f"  fold {fold} ep {ep+1}/{args.epochs}  loss {tot/max(nb,1):.3f} "
                  f"(over {nb} batches)  val acc {acc:.4f}  macroF1 {f1:.4f}  "
                  f"{rate:.1f} img/s", flush=True)
            scores = {"accuracy": acc, "macro_f1": f1, "acc_f1": combo}
            # best stores val_-prefixed keys to keep selection scores visibly
            # distinct from the test scores added after the fold finishes
            vkey = f"val_{args.select_on}"
            if best is None or scores[args.select_on] > best[vkey]:
                best = {"epoch": ep + 1, **{f"val_{k}": v for k, v in scores.items()}}
                atomic_save(model.state_dict(), bpath)
            save(ep + 1, 0)
        if out_of_time:
            print("stopped on the time limit; rerun the same command to continue")
            return
        if bpath.exists():
            try:
                model.load_state_dict(torch.load(bpath, map_location="cpu"))
                model = model.to(dev)
            except Exception as e:
                print(f"  fold {fold}: best-weights file unreadable "
                      f"({type(e).__name__}); scoring the last epoch instead",
                      flush=True)
        t_acc, t_f1, t_combo, _, _ = evaluate(model, vl, dev, n_classes)
        print(f"  fold {fold} selected epoch {best['epoch']} "
              f"(val acc {best['val_accuracy']:.4f})  ->  "
              f"TEST acc {t_acc:.4f}  macroF1 {t_f1:.4f}", flush=True)
        results.append({"fold": fold, **best, "accuracy": t_acc,
                        "macro_f1": t_f1, "acc_f1": t_combo})

        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "device": kind, "backbone": args.backbone, "epochs": args.epochs,
            "frames_per_press": 1, "protocol": f"{args.folds} of {OUTER_FOLDS} "
            f"session-grouped folds (NOT the nested-CV headline)",
            "class_weight": args.class_weight, "select_on": args.select_on,
            "size": args.size, "norm": args.norm, "flip": args.flip,
            "channels": args.channels,
            "init": str(args.init) if args.init else None,
            "inner_val_folds": args.inner_val,
            "ungrouped_leaky": bool(args.ungrouped),
            "class_names": names, "folds": results}, indent=1))

    if results:
        a = np.mean([r["accuracy"] for r in results])
        f = np.mean([r["macro_f1"] for r in results])
        print(f"\nmean over {len(results)} fold(s): accuracy {a:.4f}  macro-F1 {f:.4f}")
        print("frozen 9-frame baseline for reference: 0.6154 / 0.3917 (nested, 3 seeds)")


if __name__ == "__main__":
    main()
