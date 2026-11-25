"""Step 1 - build the experiment index from Touch and Go's frame-level label file.

The label file is headerless CSV, one row per LABELED FRAME:

    20220601_182052/0000000053.jpg,3

Labels arrive in contiguous runs of consecutive frame indices; each run is a
single press against one surface. Frames inside a run are near-duplicates, so
this script reports counts at both granularities (frame-level and run-level)
and emits a `run_id` column, letting Step 2 choose its unit of analysis.

Label -1 is "Inconclusive" and is dropped: it is not a material class.

Usage:
    python build_index.py --labels C:/datasets/touch_and_go/label.txt
"""

import argparse
from pathlib import Path

import pandas as pd

CLASS_NAMES = {
    0: "Concrete", 1: "Plastic", 2: "Glass", 3: "Wood", 4: "Metal",
    5: "Brick", 6: "Tile", 7: "Leather", 8: "Synthetic Fabric",
    9: "Natural Fabric", 10: "Rubber", 11: "Paper", 12: "Tree",
    13: "Grass", 14: "Soil", 15: "Rock", 16: "Gravel", 17: "Sand",
    18: "Plants", 19: "Others",
}


def load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path, header=None, names=["rel_path", "label"])
    # "20220601_182052/0000000053.jpg" -> session + integer frame index
    parts = df["rel_path"].str.split("/", n=1, expand=True)
    df["video_id"] = parts[0]
    df["frame_idx"] = parts[1].str.replace(".jpg", "", regex=False).astype(int)
    return df.drop(columns=["rel_path"])


def assign_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Number contiguous (video_id, label) blocks of consecutive frames.

    A new run starts whenever the session changes, the label changes, or the
    frame index is not exactly one greater than the previous row's.
    """
    df = df.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    new_session = df["video_id"] != df["video_id"].shift()
    new_label = df["label"] != df["label"].shift()
    gap = df["frame_idx"] != df["frame_idx"].shift() + 1
    df["run_id"] = (new_session | new_label | gap).cumsum() - 1
    return df


def attach_paths(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    df["video_frame_path"] = [
        str(data_root / v / "video_frame" / f"{i:010d}.jpg")
        for v, i in zip(df["video_id"], df["frame_idx"])
    ]
    df["gelsight_frame_path"] = [
        str(data_root / v / "gelsight_frame" / f"{i:010d}.jpg")
        for v, i in zip(df["video_id"], df["frame_idx"])
    ]
    return df


def report(df: pd.DataFrame) -> None:
    print(f"\nlabeled frames (excl. -1): {len(df):,}")
    print(f"touch runs               : {df['run_id'].nunique():,}")
    print(f"sessions                 : {df['video_id'].nunique():,}")
    print(f"median frames per run    : {df.groupby('run_id').size().median():.0f}")

    print("\nclass distribution")
    print(f"{'id':>3}  {'name':<17} {'frames':>9} {'runs':>7} {'sessions':>9}")
    counts = (
        df.groupby("label")
        .agg(frames=("frame_idx", "size"),
             runs=("run_id", "nunique"),
             sessions=("video_id", "nunique"))
        .reindex(range(20), fill_value=0)
    )
    for label, row in counts.iterrows():
        print(f"{label:>3}  {CLASS_NAMES[label]:<17} {row.frames:>9,} "
              f"{row.runs:>7,} {row.sessions:>9,}")

    nz = counts[counts.runs > 0]
    if len(nz):
        print(f"\nimbalance ratio (runs, max/min): {nz.runs.max() / nz.runs.min():.1f}x")
        print("-> report macro-F1 alongside accuracy.")
    thin = counts[(counts.sessions > 0) & (counts.sessions < 2)]
    for label, row in thin.iterrows():
        print(f"WARNING: {CLASS_NAMES[label]} spans {row.sessions} session(s); "
              "a group-aware split cannot place it on both sides.")

    have_v = df["video_frame_path"].map(lambda p: Path(p).exists())
    have_g = df["gelsight_frame_path"].map(lambda p: Path(p).exists())
    print(f"\nframes present on disk: vision {have_v.sum():,} | "
          f"touch {have_g.sum():,} | both {(have_v & have_g).sum():,}")
    if not (have_v & have_g).any():
        print("WARNING: zero complete vision+touch pairs. "
              "Run extract_frames.py, and check that all dataset parts are present.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path(r"C:\datasets\touch_and_go\label.txt"))
    ap.add_argument("--data-root", type=Path,
                    default=Path(r"C:\datasets\touch_and_go\dataset"))
    ap.add_argument("--out", type=Path, default=Path("index.csv"))
    args = ap.parse_args()

    df = load_labels(args.labels)
    dropped = (df["label"] == -1).sum()
    df = df[df["label"] != -1]
    print(f"dropped {dropped:,} Inconclusive (-1) rows")

    df = assign_runs(df)
    df = attach_paths(df, args.data_root)
    report(df)

    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
