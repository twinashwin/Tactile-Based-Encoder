"""Extract the labeled frames the experiments need from video.mp4 / gelsight.mp4.

This replaces the dataset's bundled extract_frame.py, which (a) writes every frame
of every video and (b) calls cap.set(CAP_PROP_POS_FRAMES, i) once per frame. On
H.264 each such seek re-decodes from the preceding keyframe, so per-frame seeking is
far slower than a single sequential pass. Here we decode each stream once, front to
back, and write only the frames named in the index.

Output layout matches the original so downstream code is unchanged:

    <session>/video_frame/0000000053.jpg
    <session>/gelsight_frame/0000000053.jpg

Already-extracted frames are skipped, so re-running after changing
--frames-per-run only does the new work.

Usage:
    python extract_frames.py --index ../index.csv --frames-per-run 9
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from common import select_pool_frames, select_rows


def wanted_from_labels(labels_path: Path) -> dict[str, set[int]]:
    """Every labeled frame, including Inconclusive (-1). Complete but expensive."""
    df = pd.read_csv(labels_path, header=None, names=["rel_path", "label"])
    out: dict[str, set[int]] = defaultdict(set)
    for rel in df["rel_path"]:
        session, fname = rel.split("/", 1)
        out[session].add(int(fname.replace(".jpg", "")))
    return dict(out)


def wanted_from_index(index_path: Path, unit: str, k: int) -> dict[str, set[int]]:
    """Only the frames the experiments consume.

    unit="run" with k=1 gives the run midpoint (one frame per press). k>1 gives
    k evenly-spaced frames per press, for embedding mean-pooling. Uses the same
    selectors as feature extraction, so the two cannot disagree about which
    frames exist on disk.
    """
    df = pd.read_csv(index_path)
    if unit == "frame":
        sel = df
    elif k == 1:
        sel = select_rows(df, "run")
    else:
        sel = select_pool_frames(df, k)

    out: dict[str, set[int]] = defaultdict(set)
    for session, idx in zip(sel["video_id"], sel["frame_idx"]):
        out[session].add(int(idx))
    return dict(out)


def extract_stream(video_path: Path, out_dir: Path, wanted: set[int]) -> int:
    """Sequentially decode video_path, writing frames whose index is in `wanted`."""
    if not video_path.exists():
        return 0

    remaining = {i for i in wanted if not (out_dir / f"{i:010d}.jpg").exists()}
    if not remaining:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    last = max(remaining)
    written = 0
    idx = 0
    while remaining and idx <= last:
        ok, frame = cap.read()
        if not ok:
            break  # stream ended early; caller reports the shortfall
        if idx in remaining:
            cv2.imwrite(str(out_dir / f"{idx:010d}.jpg"), frame)
            remaining.discard(idx)
            written += 1
        idx += 1
    cap.release()
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=None,
                    help="index.csv from build_index.py (recommended)")
    ap.add_argument("--unit", choices=["run", "frame"], default="run")
    ap.add_argument("--frames-per-run", type=int, default=1,
                    help="1 = midpoint only; k>1 = k evenly-spaced frames per "
                         "press for mean-pooling; 0 = every labeled frame")
    ap.add_argument("--labels", type=Path, default=None,
                    help="label.txt - extracts EVERY labeled frame incl. "
                         "Inconclusive (~45 GB). Rarely what you want.")
    ap.add_argument("--data-root", type=Path,
                    default=Path(r"C:\datasets\touch_and_go\dataset"))
    args = ap.parse_args()

    if args.index:
        wanted = wanted_from_index(args.index, args.unit, args.frames_per_run)
        source = f"{args.index} (unit={args.unit}, frames/run={args.frames_per_run})"
    elif args.labels:
        wanted = wanted_from_labels(args.labels)
        source = f"{args.labels} (all labeled frames)"
    else:
        ap.error("pass --index (recommended) or --labels")

    total = sum(len(v) for v in wanted.values())
    print(f"source: {source}")
    print(f"{len(wanted):,} sessions; {total:,} frames per stream "
          f"({total * 2:,} JPEGs at ~90 KB = ~{total * 2 * 90 / 1024 / 1024:.1f} GB)")

    complete, partial, absent, written = 0, 0, 0, 0
    for session, frames in tqdm(sorted(wanted.items()), desc="sessions"):
        sess_dir = args.data_root / session
        if not sess_dir.exists():
            absent += 1
            continue

        written += extract_stream(sess_dir / "video.mp4",
                                  sess_dir / "video_frame", frames)
        written += extract_stream(sess_dir / "gelsight.mp4",
                                  sess_dir / "gelsight_frame", frames)

        has_v = (sess_dir / "video.mp4").exists()
        has_g = (sess_dir / "gelsight.mp4").exists()
        if has_v and has_g:
            complete += 1
        elif has_v or has_g:
            partial += 1
        else:
            absent += 1

    print(f"\nwrote {written:,} new JPEGs")
    print(f"sessions with both streams : {complete:,}")
    print(f"sessions with one stream   : {partial:,}")
    print(f"sessions with no video     : {absent:,}")


if __name__ == "__main__":
    main()
