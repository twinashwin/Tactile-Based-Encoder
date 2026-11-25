"""Encode the touch stream with UniTouch instead of DINOv2.

UniTouch (Yang et al., CVPR 2024, arXiv:2401.18084) is an ImageBind-derived model whose
touch tower was contrastively pretrained on tactile data. It reports 61.3% touch-only
material accuracy on Touch and Go, against our 52.6% with a frozen DINOv2 -- the gap
this script exists to test.

Design notes
------------
Weights are loaded HERE rather than via x2touch(pretrained=True), for three reasons:
  1. that path calls torch.load() with no map_location, so a CUDA-saved checkpoint
     dies on a CPU-only machine;
  2. it hardcodes './last_new.ckpt' relative to the working directory;
  3. it calls load_state_dict(strict=False) and discards the result, so a key-name
     mismatch leaves the model at RANDOM INIT and reports nothing. That failure would
     produce plausible-looking but meaningless embeddings, so this script inspects the
     load and refuses to continue if too few parameters matched.

ImageBind's data.py is deliberately not imported: it needs open3d, pytorchvideo and a
torchvision video API removed in 0.28. Its image transform is reproduced exactly below
-- note the CLIP normalisation statistics, which differ from the ImageNet ones used for
DINOv2. Using the wrong ones would silently degrade every embedding.

Row order is taken from an existing feature directory's rows.csv, so the resulting
touch.npy aligns element-for-element with that directory's vision.npy and labels.npy.

Usage:
    python extract_unitouch.py --like ../features_dinop9 --out ../features_unitouch
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

UNITOUCH_REPO = Path(r"C:\dev\unitouch\UniTouch")
CKPT = UNITOUCH_REPO / "last_new.ckpt"

# ImageBind's load_and_transform_vision_data, reproduced. CLIP stats, not ImageNet.
PREPROCESS = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])


class TouchFrames(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return PREPROCESS(im.convert("RGB"))


def load_unitouch(ckpt_path: Path):
    """Build x2touch and load the pretrained weights, verifying they actually landed."""
    sys.path.insert(0, str(UNITOUCH_REPO))
    from ImageBind.models.x2touch_model_part import ModalityType, x2touch

    model = x2touch(pretrained=False)          # no hardcoded path, no CUDA deserialise
    model.eval()

    print(f"loading {ckpt_path.name} ({ckpt_path.stat().st_size / 1024**3:.2f} GB)...")
    try:                                        # mmap avoids pulling 8.5 GB into RAM
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    print(f"checkpoint holds {len(state):,} tensors")

    # The repo strips 6 characters from each key. Rather than trusting that, try the
    # plausible prefixes and keep whichever actually matches the most model keys.
    target = set(model.state_dict().keys())
    best = None
    for name, fn in (("as-is", lambda k: k),
                     ("strip 6 chars", lambda k: k[6:]),
                     ("strip 'model.'", lambda k: k[len("model."):]
                      if k.startswith("model.") else k),
                     ("strip 'module.'", lambda k: k[len("module."):]
                      if k.startswith("module.") else k)):
        mapped = {fn(k): v for k, v in state.items()}
        hits = len(target & set(mapped.keys()))
        print(f"  key mapping [{name:<14}] matches {hits:,} / {len(target):,}")
        if best is None or hits > best[0]:
            best = (hits, name, mapped)

    hits, name, mapped = best
    frac = hits / max(1, len(target))
    print(f"using key mapping '{name}': {hits:,}/{len(target):,} ({frac:.1%})")

    result = model.load_state_dict(mapped, strict=False)
    print(f"  missing keys    : {len(result.missing_keys):,}")
    print(f"  unexpected keys : {len(result.unexpected_keys):,}")

    touch_missing = [k for k in result.missing_keys if "touch" in k.lower()]
    print(f"  missing keys mentioning 'touch': {len(touch_missing):,}")

    if frac < 0.5:
        raise SystemExit(
            f"ABORT: only {frac:.1%} of parameters matched the checkpoint. The model "
            "would be largely randomly initialised and any downstream number would be "
            "meaningless. Fix the key mapping before continuing.")

    del ckpt, state, mapped
    return model, ModalityType


@torch.no_grad()
def sanity_check(model, ModalityType, paths):
    """Random weights also 'work'. These checks distinguish loaded from not-loaded."""
    batch = torch.stack([PREPROCESS(Image.open(p).convert("RGB")) for p in paths[:24]])
    emb = model({ModalityType.TOUCH: batch})[ModalityType.TOUCH].numpy()

    norms = np.linalg.norm(emb, axis=1)
    e = emb / np.clip(norms[:, None], 1e-8, None)
    sim = e @ e.T
    off = sim[~np.eye(len(e), dtype=bool)]
    print(f"\nsanity: embedding dim {emb.shape[1]}, "
          f"norm mean {norms.mean():.3f} (sd {norms.std():.3f})")
    print(f"        pairwise cosine between different frames: "
          f"mean {off.mean():+.3f}, max {off.max():+.3f}")
    print(f"        per-dimension variance: {emb.var(axis=0).mean():.5f}")
    if off.mean() > 0.99:
        raise SystemExit("ABORT: all embeddings near-identical - encoder is degenerate.")
    if not np.isfinite(emb).all():
        raise SystemExit("ABORT: non-finite values in embeddings.")


@torch.no_grad()
def embed_all(model, ModalityType, paths, batch_size, workers):
    loader = DataLoader(TouchFrames(paths), batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    out, done, t0 = [], 0, time.time()
    for batch in loader:
        out.append(model({ModalityType.TOUCH: batch})[ModalityType.TOUCH].numpy())
        done += len(batch)
        rate = done / max(1e-9, time.time() - t0)
        eta = (len(paths) - done) / max(1e-9, rate) / 60
        if done % 400 < batch_size:
            print("  {:,}/{:,}  {:.1f} img/s".format(done, len(paths), rate), flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def embed_resumable(model, ModalityType, paths, batch_size, workers, buf, ndone,
                    cache, done_f, save_every=200):
    """Embed from `ndone` onward, checkpointing so a kill loses at most one block.

    A full pass is ~49 min at 2.1 img/s, which exceeds the harness background
    limit, so progress is written to disk and the next invocation resumes.
    """
    loader = DataLoader(TouchFrames(paths[ndone:]), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    i, t0 = ndone, time.time()
    since = 0
    for batch in loader:
        out = model({ModalityType.TOUCH: batch})[ModalityType.TOUCH].numpy()
        buf[i:i + len(out)] = out
        i += len(out); since += len(out)
        if since >= save_every:
            np.save(cache, buf); np.save(done_f, np.array([i])); since = 0
        rate = (i - ndone) / max(1e-9, time.time() - t0)
        if i % 400 < batch_size:
            print("  {:,}/{:,}  {:.1f} img/s".format(i, len(paths), rate), flush=True)
    print()
    np.save(cache, buf); np.save(done_f, np.array([i]))
    return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", type=Path, default=Path("../features_dinop9"),
                    help="feature dir whose rows.csv/vision.npy/labels define the rows")
    ap.add_argument("--out", type=Path, default=Path("../features_unitouch"))
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="embed only the first N rows (smoke test)")
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--frames-per-run", type=int, default=0,
                    help="0 = use --like/rows.csv unchanged (the ORIGINAL behaviour, "
                         "which takes the FIRST frame of each press - the weakest-contact "
                         "moment). 1 = the run MIDPOINT, at peak contact, which is what "
                         "common.select_rows picks and what the DINOv2 features use. "
                         "k>1 = k evenly spaced frames, mean-pooled per press.")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}\n"
                         "download last_new.ckpt from "
                         "https://huggingface.co/chfeng/Touch-LLM")

    rows = pd.read_csv(args.like / "rows.csv")
    run_ids = None
    if args.frames_per_run == 0:
        paths = rows.gelsight_frame_path.tolist()
    else:
        from common import select_pool_frames, select_rows
        df = pd.read_csv(args.index)
        sel = (select_rows(df, "run") if args.frames_per_run == 1
               else select_pool_frames(df, args.frames_per_run))
        sel = sel[sel.run_id.isin(set(rows.run_id))]
        # keep the press order of --like so labels/groups stay aligned
        order = {r: i for i, r in enumerate(rows.run_id)}
        sel = sel.assign(_o=sel.run_id.map(order)).sort_values(["_o", "frame_idx"])
        sel = sel[[Path(p).exists() for p in sel.gelsight_frame_path]]
        paths = sel.gelsight_frame_path.tolist()
        run_ids = sel.run_id.to_numpy()
        print(f"frame selection: {'midpoint' if args.frames_per_run == 1 else str(args.frames_per_run)+' pooled'}"
              f"  ->  {len(paths):,} frames over {len(set(run_ids)):,} presses")
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} gelsight frames named in rows.csv are missing, "
                         f"e.g. {missing[0]}")
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths):,} touch frames, row order taken from {args.like / 'rows.csv'}")

    args.out.mkdir(parents=True, exist_ok=True)
    cache, done_f = args.out / "_partial.npy", args.out / "_done.npy"
    if cache.exists() and done_f.exists():
        buf = np.load(cache)
        ndone = int(np.load(done_f)[0])
        print("resuming: {:,}/{:,} frames already embedded".format(ndone, len(paths)))
    else:
        buf, ndone = np.zeros((len(paths), 1024), dtype=np.float32), 0

    if ndone >= len(paths):
        touch = buf
        print("all frames already embedded")
    else:
        model, ModalityType = load_unitouch(args.ckpt)
        if ndone == 0:
            sanity_check(model, ModalityType, paths)
        print("embedding with UniTouch (batch {}) from {:,}:".format(args.batch_size, ndone))
        touch = embed_resumable(model, ModalityType, paths, args.batch_size,
                                args.workers, buf, ndone, cache, done_f)
        if int(np.load(done_f)[0]) < len(paths):
            print("stopped early; rerun the same command to resume")
            return

    args.out.mkdir(parents=True, exist_ok=True)
    if run_ids is not None and args.frames_per_run > 1:
        order_u, inv = np.unique(run_ids, return_inverse=True)
        sums = np.zeros((len(order_u), touch.shape[1]), dtype=np.float64)
        np.add.at(sums, inv, touch.astype(np.float64))
        counts = np.bincount(inv, minlength=len(order_u))[:, None]
        pos = {r: k for k, r in enumerate(order_u)}
        touch = np.stack([(sums[pos[r]] / counts[pos[r]]) for r in rows.run_id]).astype(np.float32)
        print(f"pooled to {touch.shape} over presses")
    np.save(args.out / "touch.npy", touch)
    if not args.limit:
        # Carry across the aligned vision/labels/groups so the fused experiments and
        # nested_cv.py work against this directory unchanged.
        for f in ("vision.npy", "labels.npy", "groups.npy", "rows.csv"):
            shutil.copy2(args.like / f, args.out / f)

    print(f"\ntouch {touch.shape} {touch.dtype} -> {args.out}")
    print(f"finite: {np.isfinite(touch).all()}  "
          f"norm mean {np.linalg.norm(touch, axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
