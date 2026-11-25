"""Masked-autoencoder pretraining of DINOv2 on unlabelled gel frames.

WHY MAE AND NOT DINOv2's OWN OBJECTIVE
--------------------------------------
The supervised fine-tune trains on 3,672 images against 21M parameters and shows
a ~25 point train/validation gap -- it is memorising. The only information-adding
resource on this machine is the 1.17M unlabelled frames already sitting in the
session videos. Using them needs a self-supervised objective, and on fixed
hardware the choice of objective is a hardware decision, not a taste one:

    DINOv2 multi-crop (2 global + 8 local)  ~10 fwd/img   infeasible (>120 h)
    2-view contrastive (SimCLR-style)        ~2 fwd+bwd   ~80 h
    MAE, 75% masking                         0.25x tokens ~28 h   <-- this

MAE is the cheapest because dropping 75% of the patches means the encoder runs on
64 tokens instead of 256. It is the only one of the three that fits.

LEAKAGE
-------
Pretraining is per-fold and sees ONLY that fold's training sessions, enforced by
unlabelled_frames.sessions_for_fold. Pretraining on all sessions and then
evaluating on held-out ones would let the encoder adapt to the exact gel wear and
lighting the test fold exists to measure. The cost of doing this honestly is that
one pretraining run buys one fold, so the result carries single-fold noise
(+/- ~3 points) -- five folds would be five pretraining runs, which does not fit
the week.

Usage:
    python pretrain_mae.py --fold 0 --steps 40000 --batch 32
    python finetune_dino.py --init ../results/mae_fold0.pt ...
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from finetune_dino import pick_device, GELSIGHT_STATS
from unlabelled_frames import UnlabelledFrames, sessions_for_fold

PATCH = 14


class Attn(nn.Module):
    """Plain multi-head attention: explicit matmul, no fused kernels.

    DirectML does not reliably provide the fused attention paths that
    nn.TransformerEncoderLayer reaches for, and this runs on DirectML.
    """

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        x = (a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp)), nn.GELU(),
                                 nn.Linear(int(d * mlp), d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


def patchify(img, p=PATCH):
    """(B,3,H,W) -> (B, N, p*p*3), raster order matching patch_embed's flatten."""
    B, C, H, W = img.shape
    h, w = H // p, W // p
    x = img.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, p * p * C)


class MAE(nn.Module):
    def __init__(self, backbone, size=224, dec_dim=192, dec_depth=4, dec_heads=3):
        super().__init__()
        self.bb = backbone
        self.size = size
        self.n = (size // PATCH) ** 2
        d = backbone.pos_embed.shape[-1]
        # DINOv2's pos_embed is stored for 518px (1369 patches + cls); ask the
        # model to interpolate it down to our resolution once, then reuse
        with torch.no_grad():
            dummy = torch.zeros(1, self.n + 1, d)
            pe = backbone.interpolate_pos_encoding(dummy, size, size)
        self.register_buffer("pe", pe)                      # 1, n+1, d
        self.to_dec = nn.Linear(d, dec_dim)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pe = nn.Parameter(torch.zeros(1, self.n, dec_dim))
        self.dec = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.head = nn.Linear(dec_dim, PATCH * PATCH * 3)
        nn.init.trunc_normal_(self.mask_tok, std=0.02)
        nn.init.trunc_normal_(self.dec_pe, std=0.02)

    def forward(self, img, mask_ratio=0.75):
        B = img.shape[0]
        keep = int(self.n * (1 - mask_ratio))

        x = self.bb.patch_embed(img) + self.pe[:, 1:]        # B, n, d
        noise = torch.rand(B, self.n, device=x.device)
        order = noise.argsort(1)                             # shuffle
        restore = order.argsort(1)
        ids = order[:, :keep]
        x = torch.gather(x, 1, ids.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        cls = self.bb.cls_token + self.pe[:, :1]
        x = torch.cat([cls.expand(B, -1, -1), x], 1)         # B, keep+1, d
        for blk in self.bb.blocks:
            x = blk(x)
        x = self.bb.norm(x)

        y = self.to_dec(x)                                   # B, keep+1, dec
        pad = self.mask_tok.expand(B, self.n - keep, -1)
        y = torch.cat([y[:, 1:], pad], 1)                    # drop cls, append masks
        y = torch.gather(y, 1, restore.unsqueeze(-1).expand(-1, -1, y.shape[-1]))
        y = y + self.dec_pe
        for blk in self.dec:
            y = blk(y)
        pred = self.head(self.dec_norm(y))                   # B, n, p*p*3

        tgt = patchify(img)
        # per-patch pixel normalisation, as in the MAE paper: without it the loss
        # is dominated by patches that merely happen to be bright
        mu = tgt.mean(-1, keepdim=True)
        sd = tgt.var(-1, keepdim=True).add(1e-6).sqrt()
        tgt = (tgt - mu) / sd

        # 1 on patches the encoder never saw, 0 on the visible ones; loss is
        # computed on the masked patches only, which is the whole point
        lost = torch.ones(B, self.n, device=x.device).scatter(1, ids, 0.0)
        loss = (((pred - tgt) ** 2).mean(-1) * lost).sum() / lost.sum().clamp(min=1)
        return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    args = ap.parse_args()

    out = args.out or Path(f"../results/mae_fold{args.fold}.pt")
    dev, kind = pick_device(args.device)
    torch.manual_seed(0)

    train_s, test_s = sessions_for_fold(args.fold)
    print(f"device: {kind} | fold {args.fold} | pretrain on {len(train_s)} "
          f"sessions, {len(test_s)} held out (never decoded)")

    mean = torch.tensor(GELSIGHT_STATS[0]).view(1, 3, 1, 1)
    std = torch.tensor(GELSIGHT_STATS[1]).view(1, 3, 1, 1)

    bb = torch.hub.load("facebookresearch/dinov2", args.backbone, verbose=False)
    model = MAE(bb, size=args.size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))

    start = 0
    if out.exists():
        st = torch.load(out, map_location="cpu", weights_only=False)
        model.load_state_dict(st["mae"]); opt.load_state_dict(st["opt"])
        start = st["step"]
        print(f"resuming at step {start:,}")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        t = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    ds = UnlabelledFrames(train_s, stride=args.stride, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=0)

    step, t0, run = start, time.time(), []
    model.train()
    while step < args.steps:
        for raw in dl:
            if step >= args.steps:
                break
            img = (raw.float().div_(255.0) - mean) / std
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = model(img.to(dev), args.mask_ratio)
            opt.zero_grad(); loss.backward(); opt.step()
            run.append(float(loss.detach().cpu())); step += 1
            if step % 100 == 0:
                r = step * args.batch / max(1e-9, time.time() - t0)
                print(f"  step {step:,}/{args.steps:,}  loss "
                      f"{np.mean(run[-100:]):.4f}  {r:.1f} img/s", flush=True)
            if step % args.save_every == 0:
                torch.save({"mae": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "encoder": model.bb.state_dict()}, out)
            if args.max_minutes and (time.time() - t0) / 60 >= args.max_minutes:
                torch.save({"mae": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "encoder": model.bb.state_dict()}, out)
                print(f"time limit at step {step:,}; rerun to continue")
                return
    torch.save({"mae": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "encoder": model.bb.state_dict()}, out)
    print(f"done {step:,} steps | final loss {np.mean(run[-200:]):.4f} | wrote {out}")


if __name__ == "__main__":
    main()
