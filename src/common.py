"""Shared definitions for the Touch and Go material-classification experiments.

`select_rows` lives here because frame extraction and feature extraction must agree
exactly on which frames constitute the dataset. If they disagree, feature extraction
silently drops rows whose JPEGs were never written.
"""

import pandas as pd

CLASS_NAMES = {
    0: "Concrete", 1: "Plastic", 2: "Glass", 3: "Wood", 4: "Metal",
    5: "Brick", 6: "Tile", 7: "Leather", 8: "Synthetic Fabric",
    9: "Natural Fabric", 10: "Rubber", 11: "Paper", 12: "Tree",
    13: "Grass", 14: "Soil", 15: "Rock", 16: "Gravel", 17: "Sand",
    18: "Plants", 19: "Others",
}


def select_rows(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Reduce the index to one row per example.

    unit="frame": every labeled frame. ~27x more rows, but rows within a run are
        near-duplicate images, so this inflates N without adding information.
    unit="run": one frame per contiguous touch run, taken from the run's middle,
        which sits at peak contact rather than the noisy start/end of the press.
    """
    if unit == "frame":
        return df.reset_index(drop=True)

    # Sort within each run, then keep the row at the run's midpoint by POSITION.
    # Selecting by frame_idx value instead would misalign, since a run's frame
    # indices are session-relative rather than row-relative.
    df = df.sort_values(["run_id", "frame_idx"]).reset_index(drop=True)
    position = df.groupby("run_id").cumcount()
    run_length = df.groupby("run_id")["frame_idx"].transform("size")
    return df[position == run_length // 2].reset_index(drop=True)


def select_pool_frames(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep up to `k` evenly-spaced frames per run, for embedding mean-pooling.

    Averaging several frames' embeddings into one vector denoises the estimate of
    a press without inflating N: the result is still one example per run, so
    statistical independence is unchanged. Variance falls roughly as 1/k.

    k <= 0 keeps every labeled frame in the run.

    Spacing is done by bucketing each row's rank within its run into k bins and
    taking the first row of each bin, so short runs simply yield fewer frames
    rather than duplicates.
    """
    if k <= 0:
        return df.sort_values(["run_id", "frame_idx"]).reset_index(drop=True)

    df = df.sort_values(["run_id", "frame_idx"]).reset_index(drop=True)
    position = df.groupby("run_id").cumcount()
    run_length = df.groupby("run_id")["frame_idx"].transform("size")
    df = df.assign(_bucket=(position * k // run_length))
    return (df.drop_duplicates(subset=["run_id", "_bucket"])
              .drop(columns="_bucket")
              .reset_index(drop=True))
