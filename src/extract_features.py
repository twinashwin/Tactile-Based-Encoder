"""Step 2 - cache frozen-backbone embeddings for both modalities.

Protocol: a pretrained vision backbone with its classifier head removed and weights
frozen, used purely as a fixed feature extractor. This is the "linear probing"
setup, so results are comparable to linear-probe numbers in the literature and NOT
to in-domain fine-tuned numbers.

Vision and touch each get their own frozen instance of the same architecture.

--backbone
    resnet18        512-d,  ImageNet supervised. Fast baseline. ~68 img/s on CPU.
    resnet50       2048-d,  ImageNet supervised. ~22 img/s.
    dinov2_vitb14   768-d,  the base DINOv2. Same family as vits14, ~4x the
                            compute, and the largest lever left in this study.
    dinov2_vits14   384-d,  self-supervised. Stronger local/texture structure,
                            which is the plausible win for GelSight imprints since
                            an ImageNet classifier is trained to discard texture
                            that a self-supervised model keeps. ~22 img/s.

--frames-per-run k
    1   the run's midpoint frame only (one embedding per press)
    k>1 embed k evenly-spaced frames per press and MEAN-POOL them into one vector.
        Still one row per press, so N and independence are unchanged - but each
        row is a lower-variance estimate. Variance falls roughly as 1/k.

--tta   also embed the horizontal flip of every frame and average. Reduces
        framing variance; largely complementary to frame pooling, which reduces
        temporal variance.

--pool-tokens (DINOv2 only)
    cls           the CLS summary vector, D-d. What this script used to emit.
    patchmeanstd  mean AND std over the per-patch tokens, 2D-d. Measured +0.032
                  accuracy over cls on a like-for-like 1-frame probe (0.4839 vs
                  0.4518, SVM-RBF, grouped 5-fold). CLS is trained to summarise
                  what an image depicts and to be invariant to texture; material
                  identity from an imprint IS the texture, which is why the
                  patch statistics carry more here.
    patchmean     mean over patch tokens only, D-d.
    cls_patchstd  CLS for the global view plus patch std for texture energy, 2D-d.

--save-frames
    also cache the UNPOOLED per-frame embeddings, so new pooling recipes (temporal
    std, max) can be tried by pool_features.py in seconds rather than by paying
    another full extraction pass.

Input geometry is deliberately NOT configurable. Resize(256)+CenterCrop(224)
discards ~42% of a 640x480 GelSight frame, which looks like a bug, so it was
tested: feeding the whole pad as Resize((252,336)) scored WORSE (0.4325 vs 0.4518
cls, 0.4754 vs 0.4839 patchmeanstd). A gel imprint is a stationary texture, so the
crop keeps a representative sample, and the wider view costs more in effective
resolution than the extra field of view returns.

Usage:
    python extract_features.py --index ../index.csv --out ../features_dino9 \
        --backbone dinov2_vits14 --frames-per-run 9
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (ResNet18_Weights, ResNet50_Weights, resnet18,
                                resnet50)

from common import select_pool_frames, select_rows

# ImageNet statistics - correct for all three backbones, DINOv2 included.
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])


BG_GAIN = float(__import__("os").environ.get("BG_GAIN", 2.0))   # residual amplification; 2.0 clips ~4% of pixels, 1.0 clips ~0.5%
BG_REF_DIR = Path("C:/datasets/touch_and_go/bg_refs")
BIN_THRESH = 25.0      # residual cut for --binarize; matches segment_texture.py


def session_references(paths: list[str], ref_dir: Path = BG_REF_DIR) -> dict:
    """Load the per-session non-contact GelSight reference built by make_bg_refs.py.

    A GelSight image is an undeformed gel plus a deformation, and only the
    deformation carries material information. Subtracting a reference removes the
    constant gel colour, marker lattice and illumination so the encoder stops
    spending capacity on them. Because the CV splits are grouped by session, that
    nuisance is exactly the signal that cannot transfer.

    This used to median over 40 LABELLED CONTACT frames, which was wrong: 55 of
    126 sessions in index.csv contain exactly one material, so for 44% of the data
    the "reference" was that class's own imprint and subtracting it deleted the
    signal -- for some classes and not others. make_bg_refs.py instead medians over
    frames carrying NO label, i.e. the sensor between presses, which is genuine
    undeformed gel. See its docstring.

    Sessions with no reference file are returned absent, and FrameDataset then
    leaves their frames unsubtracted rather than guessing.
    """
    refs = {}
    for session in {Path(p).parent.parent.name for p in paths}:
        f = ref_dir / f"{session}.png"
        if f.exists():
            with Image.open(f) as im:
                refs[session] = np.asarray(im.convert("RGB"), dtype=np.float32)
    return refs


class FrameDataset(Dataset):
    def __init__(self, paths: list[str], refs: dict | None = None,
                 binarize: bool = False):
        self.paths = paths
        self.refs = refs
        self.binarize = binarize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        path = self.paths[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.refs is not None:
                ref = self.refs.get(Path(path).parent.parent.name)
                if ref is not None:
                    arr = np.asarray(im, dtype=np.float32)
                    if arr.shape == ref.shape:
                        if self.binarize:
                            # Contact / no-contact mask, replicated to 3 channels.
                            # Everything about colour and depth is discarded; only
                            # the SHAPE of the contact regions survives.
                            d = np.abs(arr - ref).mean(axis=2)
                            m = (d > BIN_THRESH).astype(np.uint8) * 255
                            arr = np.repeat(m[:, :, None], 3, axis=2)
                        else:
                            arr = np.clip((arr - ref) * BG_GAIN + 128.0, 0.0, 255.0)
                        im = Image.fromarray(arr.astype(np.uint8))
            return PREPROCESS(im)


def build_extractor(name: str, init=None) -> nn.Module:
    """Pretrained backbone, head removed, frozen, in eval mode.

    eval() is load-bearing for the ResNets: BatchNorm in train mode normalises
    using the current batch's statistics, which would make an image's embedding
    depend on its batch-mates.
    """
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    elif name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
    elif name.startswith("dinov2_"):
        model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
    else:
        raise ValueError(f"unknown backbone {name}")

    if init is not None:
        # Load a backbone fine-tuned by finetune_dino.py. Lets the frozen
        # pipeline's 9-frame pooling, dual view and tuned head run on top of an
        # ADAPTED encoder instead of the stock one -- the two sets of gains were
        # never combined before, and each is worth more than the other alone.
        st = torch.load(init, map_location="cpu", weights_only=False)
        sd = st.get("encoder", st)
        sd = {k[len("backbone."):]: v for k, v in sd.items()
              if k.startswith("backbone.")} or sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys in --init: {unexpected[:5]}"
        print(f"  loaded fine-tuned backbone from {Path(init).name} "
              f"({len(missing)} keys left at pretrained init)")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def forward_tokens(model, batch: torch.Tensor, mode: str) -> torch.Tensor:
    """One forward pass, reduced to a vector per image according to `mode`.

    ResNets have no patch tokens, so only "cls" is meaningful for them; main()
    rejects the other modes rather than silently returning the wrong thing.
    """
    if mode == "cls":
        return model(batch)
    feats = model.forward_features(batch)
    cls, patches = feats["x_norm_clstoken"], feats["x_norm_patchtokens"]
    if mode == "patchmean":
        return patches.mean(1)
    if mode == "patchmeanstd":
        return torch.cat([patches.mean(1), patches.std(1)], dim=1)
    if mode == "cls_patchstd":
        return torch.cat([cls, patches.std(1)], dim=1)
    raise ValueError(mode)


def pick_device(name: str):
    """Inference here was CPU-only, which cost ~45 min per 55k-frame extraction
    while the GPU sat idle. DirectML does ~52 img/s against ~15 on eight CPU
    threads."""
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


@torch.no_grad()
def embed(paths: list[str], model: nn.Module, batch_size: int, workers: int,
          tta: bool, refs: dict | None = None,
          pool_tokens: str = "cls", binarize: bool = False,
          dev: torch.device | None = None) -> np.ndarray:
    loader = DataLoader(FrameDataset(paths, refs, binarize), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    chunks = []
    for i, batch in enumerate(loader):
        if dev is not None:
            batch = batch.to(dev)
        out = forward_tokens(model, batch, pool_tokens)
        if tta:
            flipped = forward_tokens(model, torch.flip(batch, dims=[3]),
                                     pool_tokens)
            out = (out + flipped) / 2
        chunks.append(out.detach().cpu().numpy())
        done = min((i + 1) * batch_size, len(paths))
        print("\r  {:,}/{:,}".format(done, len(paths)), end="", flush=True)
    print()
    return np.concatenate(chunks).astype(np.float32)


def pool_by_run(emb: np.ndarray, run_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool embeddings within each run. Returns (pooled, run_id order).

    Uses a grouped sum rather than one boolean mask per run: masking would be
    O(n_runs * n_frames), which at 6k runs x 56k frames is 340M comparisons.
    """
    order, inverse = np.unique(run_ids, return_inverse=True)
    sums = np.zeros((len(order), emb.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, emb)
    counts = np.bincount(inverse, minlength=len(order))[:, None]
    return (sums / counts).astype(np.float32), order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("../index.csv"))
    ap.add_argument("--out", type=Path, default=Path("../features"))
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50", "dinov2_vits14",
                             "dinov2_vitb14"])
    ap.add_argument("--frames-per-run", type=int, default=1)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--pool-tokens", default="cls",
                    choices=["cls", "patchmean", "patchmeanstd", "cls_patchstd"],
                    help="how to reduce a DINOv2 forward pass to one vector")
    ap.add_argument("--binarize", action="store_true",
                    help="feed DINOv2 a CONTACT MASK instead of an image: "
                         "|frame-reference| > threshold, 0/255, 3 channels. "
                         "Implies --bg-subtract. Tests how much of the signal is "
                         "carried by contact SHAPE alone, with all colour and "
                         "depth information removed.")
    ap.add_argument("--save-frames", action="store_true",
                    help="also cache UNPOOLED per-frame embeddings so new "
                         "pooling recipes cost seconds, not another full pass")
    ap.add_argument("--embed", choices=["both", "vision", "touch"], default=None,
                    help="which modalities to actually embed. Defaults to "
                         "--require. Separating them lets a touch-only "
                         "experiment keep --require both for row alignment "
                         "without paying to re-embed vision it already has.")
    ap.add_argument("--bg-subtract", action="store_true",
                    help="subtract a per-session median reference from the "
                         "GELSIGHT stream only, isolating gel deformation")
    ap.add_argument("--require", choices=["both", "vision", "touch"], default="both",
                    help="'both' keeps only presses carrying BOTH modalities, so "
                         "all three feature sets score identical examples")
    ap.add_argument("--init", default=None,
                    help="Checkpoint from finetune_dino.py whose backbone weights "
                         "replace the pretrained ones before extraction")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "directml", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    if args.pool_tokens != "cls" and not args.backbone.startswith("dinov2_"):
        raise SystemExit(f"--pool-tokens {args.pool_tokens} needs patch tokens; "
                         f"{args.backbone} has none. Use --pool-tokens cls.")
    embed_which = args.embed or args.require
    if args.binarize:
        args.bg_subtract = True      # the mask is defined against the reference

    df = pd.read_csv(args.index)
    k = args.frames_per_run
    sel = select_rows(df, "run") if k == 1 else select_pool_frames(df, k)

    has_v = sel.video_frame_path.map(lambda p: Path(p).exists())
    has_g = sel.gelsight_frame_path.map(lambda p: Path(p).exists())
    keep = {"both": has_v & has_g, "vision": has_v, "touch": has_g}[args.require]
    sel = sel[keep].reset_index(drop=True)
    if sel.empty:
        raise SystemExit(f"No frames satisfy --require {args.require}. "
                         "Run extract_frames.py first.")

    n_runs = sel.run_id.nunique()
    print(f"backbone={args.backbone} frames/run={k} tta={args.tta} "
          f"pool_tokens={args.pool_tokens} embed={embed_which}")
    print(f"{len(sel):,} frames -> {n_runs:,} presses "
          f"({len(sel) / n_runs:.1f} frames pooled per press)")

    args.out.mkdir(parents=True, exist_ok=True)
    run_ids = sel.run_id.to_numpy()
    shapes = {}

    for modality, column in (("vision", "video_frame_path"),
                             ("touch", "gelsight_frame_path")):
        if embed_which not in ("both", modality):
            continue
        paths = sel[column].tolist()
        refs = None
        if args.bg_subtract and modality == "touch":
            print("computing per-session GelSight references...")
            refs = session_references(paths)
            print(f"  {len(refs)} session references")
        print(f"{modality} embeddings:")
        model = build_extractor(args.backbone, args.init)   # per modality
        dev, kind = pick_device(args.device)
        model = model.to(dev)
        print(f"  device: {kind}")
        emb = embed(paths, model, args.batch_size, args.workers, args.tta, refs,
                    args.pool_tokens, args.binarize and modality == "touch", dev=dev)
        pooled, order = pool_by_run(emb, run_ids)
        np.save(args.out / f"{modality}.npy", pooled)
        shapes[modality] = pooled.shape
        if args.save_frames:
            np.save(args.out / f"{modality}_frames.npy", emb)
            np.save(args.out / "frame_run_ids.npy", run_ids)

    # Labels and groups follow the pooled run order, not the frame order.
    per_run = sel.drop_duplicates("run_id").set_index("run_id").loc[order]
    np.save(args.out / "labels.npy", per_run.label.to_numpy())
    np.save(args.out / "groups.npy", per_run.video_id.to_numpy())
    per_run.reset_index().to_csv(args.out / "rows.csv", index=False)

    print("\n" + " | ".join(f"{k2} {v}" for k2, v in shapes.items())
          + f" -> {args.out}")


if __name__ == "__main__":
    main()
