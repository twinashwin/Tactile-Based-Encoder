"""Contrastive vision -> touch alignment, the UniTouch recipe on our own pairs.

UniTouch reaches 61.3% touch-only on Touch and Go by aligning tactile embeddings
to image embeddings through self-supervision, then linear-probing the frozen
tactile side. The dataset gives us exactly the pairing that needs: every press has
a vision embedding and a touch embedding of the same instant.

Method: train a projector on the FROZEN touch features so that its output matches
the paired FROZEN vision embedding, under a symmetric InfoNCE loss. Vision is a
training signal only - at test time the classifier sees projected touch and
nothing else, so this is representation learning, not leakage.

Leakage control that actually matters here: the projector is trained inside each
outer fold on TRAINING sessions only. Fitting it on all pairs first, then
cross-validating the classifier, would let test-session structure into the
representation and inflate everything downstream.

Usage:
    python align.py --features ../features_dinop9 --out ../results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from common import CLASS_NAMES
from nested_cv import EXCLUDE, INNER_FOLDS, OUTER_FOLDS, SEED

CHECKPOINTS = [25, 50, 100, 200]     # projector epochs, chosen on the inner fold
TEMP = 0.07


class TouchProjector(nn.Module):
    """Small MLP mapping frozen touch features into the vision embedding space."""

    def __init__(self, d_touch: int, d_vision: int, d_hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_touch, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_vision),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(t), dim=-1)


def info_nce(z: torch.Tensor, v: torch.Tensor, temp: float = TEMP) -> torch.Tensor:
    """Symmetric contrastive loss: press i's touch must match press i's vision."""
    logits = z @ v.T / temp
    labels = torch.arange(len(z), device=z.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.T, labels))


def train_projector(Ttr, Vtr, epochs: int, batch_size: int = 256,
                    lr: float = 1e-3, seed: int = SEED):
    """Fit the projector on training-fold pairs only. Returns the model."""
    torch.manual_seed(seed)
    d_t, d_v = Ttr.shape[1], Vtr.shape[1]
    model = TouchProjector(d_t, d_v)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    T = torch.tensor(Ttr, dtype=torch.float32)
    V = F.normalize(torch.tensor(Vtr, dtype=torch.float32), dim=-1)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(T))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 8:          # a tiny final batch makes InfoNCE unstable
                continue
            opt.zero_grad()
            info_nce(model(T[idx]), V[idx]).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def project(model, X) -> np.ndarray:
    return model(torch.tensor(X, dtype=torch.float32)).numpy()


def probe(Xtr, ytr, Xte, yte, head="logistic"):
    """Frozen-feature probe. 'logistic' is the linear-probe convention; 'svm'
    is the RBF kernel, which touch_heads.py found stronger on tactile features
    and which the tactile literature (TaCo-L, ResNet-ViT-SVM) actually uses."""
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    if head == "svm":
        m = SVC(kernel="rbf", C=10.0, gamma="scale",
                class_weight="balanced").fit(a, ytr)
    else:
        m = LogisticRegression(max_iter=2000, C=1.0,
                               class_weight="balanced").fit(a, ytr)
    pred = m.predict(b)
    return pred, f1_score(yte, pred, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("../features_dinop9"))
    ap.add_argument("--out", type=Path, default=Path("../results"))
    ap.add_argument("--head", choices=["logistic", "svm"],
                    default="logistic")
    args = ap.parse_args()
    np.random.seed(SEED)

    y_raw = np.load(args.features / "labels.npy")
    groups = np.load(args.features / "groups.npy", allow_pickle=True)
    V = np.load(args.features / "vision.npy")
    T = np.load(args.features / "touch.npy")

    keep = np.array([CLASS_NAMES[int(c)] not in EXCLUDE for c in y_raw])
    for c in np.unique(y_raw):
        if len(np.unique(groups[y_raw == c])) < 2:
            keep &= y_raw != c
    V, T, y_raw, groups = V[keep], T[keep], y_raw[keep], groups[keep]
    kept = sorted(np.unique(y_raw).tolist())
    names = [CLASS_NAMES[c] for c in kept]
    y = np.array([{c: i for i, c in enumerate(kept)}[c] for c in y_raw])
    print(f"{len(y):,} paired presses | {len(names)} classes | "
          f"touch {T.shape[1]}-d -> vision {V.shape[1]}-d\n")

    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                 random_state=SEED)
    base_pred = np.full(len(y), -1)
    proj_pred = np.full(len(y), -1)
    chosen = []

    for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        # ---- baseline: raw frozen touch features ----
        base_pred[te], _ = probe(T[tr], y[tr], T[te], y[te], args.head)

        # ---- INNER: choose projector epochs on downstream macro-F1 ----
        inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                     random_state=SEED)
        ia, ib = next(inner.split(np.zeros(len(tr)), y[tr], groups[tr]))
        a, b = tr[ia], tr[ib]
        scores = {}
        for ep in CHECKPOINTS:
            proj = train_projector(T[a], V[a], ep)
            _, f1 = probe(project(proj, T[a]), y[a], project(proj, T[b]), y[b],
                          args.head)
            scores[ep] = f1
        best_ep = max(scores, key=scores.get)
        chosen.append(best_ep)

        # ---- refit projector on the FULL outer-train, then probe ----
        proj = train_projector(T[tr], V[tr], best_ep)
        proj_pred[te], _ = probe(project(proj, T[tr]), y[tr],
                                 project(proj, T[te]), y[te], args.head)
        print(f"  fold {k}: projector epochs={best_ep} "
              f"(inner F1 {scores[best_ep]:.3f})", flush=True)

    out = {}
    for label, pred in (("touch (frozen)", base_pred),
                        ("touch (aligned to vision)", proj_pred)):
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        out[label] = {"accuracy": round(float(acc), 4),
                      "macro_f1": round(float(f1), 4),
                      "per_class_f1": [round(float(v), 3) for v in
                                       f1_score(y, pred, average=None,
                                                zero_division=0)]}
        print(f"\n{label:<28} acc {acc:.4f}   macro-F1 {f1:.4f}")

    d_acc = out["touch (aligned to vision)"]["accuracy"] - out["touch (frozen)"]["accuracy"]
    d_f1 = out["touch (aligned to vision)"]["macro_f1"] - out["touch (frozen)"]["macro_f1"]
    print(f"\nalignment delta: {d_acc:+.4f} accuracy, {d_f1:+.4f} macro-F1")
    print(f"UniTouch reference (20-class, touch-grouped split): 0.6130")

    out["projector_epochs_per_fold"] = chosen
    out["class_names"] = names
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"alignment_{args.head}.json").write_text(json.dumps(out))
    print(f"wrote {args.out}/alignment_{args.head}.json")


if __name__ == "__main__":
    main()
