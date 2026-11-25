"""Stream unlabelled gel frames straight out of the session videos.

WHY THIS EXISTS
---------------
Touch and Go holds 1,424,892 gelsight frames but only ~250k carry a material
label, and the supervised pipeline uses just 6,115 of them (one per press). The
other 1.17M frames are the only large, information-adding resource available on
this machine -- and they cost nothing, because they already sit inside the
per-session gelsight.mp4 files (9.5 MB each, ~1.3 GB for all 140 sessions).
Extracting them as JPEGs would cost tens of GB we do not have; decoding them
sequentially costs nothing and runs at hundreds of frames per second, far above
the ~7 img/s the GPU can train at. So the decoder is never the bottleneck.

LEAKAGE
-------
This is the part that matters. Pretraining on a session and then testing on it
leaks -- the encoder will have adapted to that specific gel's wear and lighting,
which is exactly the shift the test fold is supposed to measure. So the caller
MUST pass the sessions it is allowed to see, and `sessions_for_fold` derives that
list from the same StratifiedGroupKFold split the supervised runs use, with an
assertion that the held-out sessions are absent.

Usage:
    from unlabelled_frames import UnlabelledFrames, sessions_for_fold
    train_sessions, test_sessions = sessions_for_fold(0)
    ds = UnlabelledFrames(train_sessions, stride=8)
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset, get_worker_info

from common import CLASS_NAMES, select_rows
from nested_cv import EXCLUDE, OUTER_FOLDS, SEED

DATA = Path("C:/datasets/touch_and_go/dataset")


def sessions_for_fold(fold: int, index=Path("../index.csv"),
                      like=Path("../features_dinop9")):
    """(sessions safe to pretrain on, sessions held out) for one outer fold.

    Mirrors the supervised split exactly so the two stages agree on what 'unseen'
    means.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = pd.read_csv(index)
    sel = select_rows(df, "run")
    rows = pd.read_csv(like / "rows.csv")
    sel = sel[sel.run_id.isin(set(rows.run_id))]
    sel = sel[[Path(p).exists() for p in sel.gelsight_frame_path]]
    sel = sel[[CLASS_NAMES[int(c)] not in EXCLUDE for c in sel.label]]
    groups = sel.video_id.to_numpy()
    kept = sorted(sel.label.unique().tolist())
    m = {c: i for i, c in enumerate(kept)}
    y = np.array([m[c] for c in sel.label])

    cv = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        if k == fold:
            train_s = sorted(set(groups[tr]))
            test_s = sorted(set(groups[te]))
            assert not (set(train_s) & set(test_s))
            return train_s, test_s
    raise ValueError(f"fold {fold} out of range")


def count_frames(sessions):
    """Frames available per session, from the video header (no decoding)."""
    out = {}
    for s in sessions:
        v = DATA / s / "gelsight.mp4"
        if not v.exists():
            continue
        cap = cv2.VideoCapture(str(v))
        out[s] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return out


class UnlabelledFrames(IterableDataset):
    """Sequential decode, interleaved across sessions so batches stay mixed.

    Random access into an mp4 means a keyframe seek per read, which is far slower
    than a linear pass. Instead each worker walks several videos at once and
    round-robins between them, which keeps decoding sequential while still giving
    the model a batch drawn from many different gels.

    stride: keep every Nth frame. Consecutive frames are 0.037 s apart and nearly
    identical, so a stride well above 1 buys diversity for free.
    """

    def __init__(self, sessions, stride: int = 8, size: int = 224,
                 interleave: int = 8, max_frames: int | None = None):
        self.sessions = list(sessions)
        self.stride = stride
        self.size = size
        self.interleave = interleave
        self.max_frames = max_frames

    def _shard(self):
        wi = get_worker_info()
        if wi is None:
            return self.sessions
        return self.sessions[wi.id::wi.num_workers]

    def __iter__(self):
        rng = np.random.default_rng(SEED)
        mine = list(self._shard())
        rng.shuffle(mine)
        served = 0
        pos = 0
        caps = []

        def open_next():
            nonlocal pos
            while pos < len(mine):
                v = DATA / mine[pos] / "gelsight.mp4"
                pos += 1
                if v.exists():
                    c = cv2.VideoCapture(str(v))
                    if c.isOpened():
                        return c
            return None

        for _ in range(self.interleave):
            c = open_next()
            if c is None:
                break
            caps.append([c, 0])

        while caps:
            for slot in list(caps):
                cap, n = slot
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    caps.remove(slot)
                    nc = open_next()
                    if nc is not None:
                        caps.append([nc, 0])
                    continue
                slot[1] = n + 1
                if n % self.stride:
                    continue
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if g.shape[0] != self.size or g.shape[1] != self.size:
                    g = cv2.resize(g, (self.size, self.size),
                                   interpolation=cv2.INTER_AREA)
                yield np.ascontiguousarray(g.transpose(2, 0, 1))
                served += 1
                if self.max_frames and served >= self.max_frames:
                    for c, _ in caps:
                        c.release()
                    return
