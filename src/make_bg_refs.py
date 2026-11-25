"""Build a true per-session GelSight background reference from UNLABELED frames.

Why this file exists
--------------------
A GelSight image is an undeformed gel plus a deformation, and only the deformation
carries material information. Subtracting a reference removes the constant gel
colour, marker lattice and illumination so the encoder stops spending capacity on
them. Because the CV splits are grouped by session, session-constant nuisance is
exactly the kind of signal that does not transfer, so removing it should help.

The reference has to be NON-CONTACT, and the previous attempt was not.
`extract_features.session_references()` medians over 40 *labelled contact* frames,
which is only a valid reference if a session contains many materials. It does not:
measured on index.csv, 55 of 126 sessions (44%) contain exactly ONE material and 61
(48%) are over 90% one material. For those sessions the "background" IS that class's
own imprint, so subtracting it deletes the signal -- for some classes and not
others, which is a class-correlated corruption that would quietly bias every
downstream number.

The fix is available for free. Only the ~9 selected frames per press were ever
extracted as JPEGs, but gelsight.mp4 is still on disk, and 60-73% of each video's
frames are not labelled at all -- the sensor travelling between presses. Those are
genuine undeformed gel. We median over those.

Frames labelled -1 (Inconclusive) are treated as CONTACT and excluded from the
reference: -1 means the annotator could not identify the material, not that nothing
was being touched.

Usage:
    python make_bg_refs.py                      # all sessions, skips existing
    python make_bg_refs.py --frames 128 --force
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_ROOT = Path(r"C:\datasets\touch_and_go\dataset")
LABELS = Path(r"C:\datasets\touch_and_go\label.txt")
OUT_DIR = Path(r"C:\datasets\touch_and_go\bg_refs")


def labeled_frames(labels_path: Path) -> dict[str, set[int]]:
    """Every frame that carries ANY label, Inconclusive included -- i.e. contact."""
    df = pd.read_csv(labels_path, header=None, names=["rel_path", "label"])
    split = df["rel_path"].str.split("/", n=1, expand=True)
    out: dict[str, set[int]] = defaultdict(set)
    for session, fname in zip(split[0], split[1]):
        out[session].add(int(fname.replace(".jpg", "")))
    return dict(out)


def chunked_median(stack: np.ndarray, strips: int = 6) -> np.ndarray:
    """Median over axis 0, computed in horizontal strips.

    np.median upcasts to float64 and sorts a copy, so a 128x480x640x3 uint8 stack
    would need ~940 MB in one go. Striping keeps the peak bounded.
    """
    h = stack.shape[1]
    edges = np.linspace(0, h, strips + 1).astype(int)
    return np.concatenate(
        [np.median(stack[:, a:b], axis=0) for a, b in zip(edges, edges[1:])],
        axis=0).astype(np.uint8)


def build_reference(video: Path, contact: set[int], n_frames: int):
    """Median of up to n_frames evenly-spaced NON-CONTACT frames. None if too few."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, 0

    free = np.array([i for i in range(total) if i not in contact])
    if len(free) < 8:                      # not enough clean gel to trust a median
        cap.release()
        return None, len(free)

    pick = set(free[np.linspace(0, len(free) - 1, min(n_frames, len(free)))
                    .astype(int)].tolist())

    # One sequential pass. Seeking per frame would re-decode from the preceding
    # keyframe each time, the same trap extract_frames.py documents.
    frames, idx, last = [], 0, max(pick)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in pick:
            frames.append(frame)
        idx += 1
    cap.release()
    if len(frames) < 8:
        return None, len(frames)
    return chunked_median(np.stack(frames)), len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--frames", type=int, default=128,
                    help="how many non-contact frames to median over")
    ap.add_argument("--force", action="store_true", help="rebuild existing refs")
    args = ap.parse_args()

    contact = labeled_frames(args.labels)
    sessions = sorted(p.name for p in args.data_root.iterdir()
                      if (p / "gelsight.mp4").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(sessions)} sessions with gelsight.mp4 -> {args.out}")

    made, skipped, failed = 0, 0, []
    for session in tqdm(sessions, desc="sessions"):
        dest = args.out / f"{session}.png"
        if dest.exists() and not args.force:
            skipped += 1
            continue
        ref, n = build_reference(args.data_root / session / "gelsight.mp4",
                                 contact.get(session, set()), args.frames)
        if ref is None:
            failed.append((session, n))
            continue
        cv2.imwrite(str(dest), ref)
        made += 1

    print(f"\nwrote {made} references, skipped {skipped} existing")
    if failed:
        print(f"{len(failed)} sessions had too few non-contact frames:")
        for s, n in failed[:10]:
            print(f"  {s}  ({n} usable)")
        print("  -> extract_features.py falls back to no subtraction for these.")


if __name__ == "__main__":
    main()
