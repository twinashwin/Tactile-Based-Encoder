"""Learned fusion heads for vision + touch, as alternatives to concatenation.

Concatenation cannot learn "trust touch more for this example" - it fits one fixed
weight per dimension. These two heads can:

  GatedFusion       a sigmoid gate, computed from both modalities, mixes them
                    per-dimension and per-example:  f = g*v + (1-g)*t.
                    The direct test of adaptive modality reweighting.

  CrossAttnFusion   vision and touch become two tokens; a multi-head attention
                    block lets each attend to the other before classification.
                    This is the mechanism ViTaPEs and UniTouch use, reduced to
                    the smallest form that still qualifies as cross-modal
                    attention.

Both are deliberately small. With ~4,900 training presses per outer fold, a large
fusion module would overfit long before it learned anything - the MLP already does.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class GatedFusion(nn.Module):
    """Per-dimension, per-example convex mix of the two modalities."""

    def __init__(self, d_in: int, n_classes: int, d_hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.d = d_in
        self.gate = nn.Sequential(
            nn.Linear(2 * d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_in), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_in, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, t = x[:, :self.d], x[:, self.d:]
        g = self.gate(x)
        return self.head(g * v + (1 - g) * t)

    @torch.no_grad()
    def gate_stats(self, x: torch.Tensor) -> float:
        """Mean gate value: 1.0 = pure vision, 0.0 = pure touch."""
        return float(self.gate(x).mean())


class CrossAttnFusion(nn.Module):
    """Two tokens (vision, touch) through a multi-head attention block."""

    def __init__(self, d_in: int, n_classes: int, d_model: int = 128,
                 n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.d = d_in
        self.proj_v = nn.Linear(d_in, d_model)
        self.proj_t = nn.Linear(d_in, d_model)
        # A learned per-modality embedding, so attention can tell the two tokens
        # apart - without it the block is permutation-invariant over modalities.
        self.modality = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, t = x[:, :self.d], x[:, self.d:]
        tokens = torch.stack([self.proj_v(v), self.proj_t(t)], dim=1)
        tokens = tokens + self.modality            # (B, 2, d_model)
        a, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + a)
        tokens = self.norm2(tokens + self.ff(tokens))
        return self.head(tokens.flatten(1))        # both tokens -> classifier

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """(2,2) mean attention: rows = query modality, cols = attended-to."""
        v, t = x[:, :self.d], x[:, self.d:]
        tokens = torch.stack([self.proj_v(v), self.proj_t(t)], dim=1) + self.modality
        _, w = self.attn(tokens, tokens, tokens, need_weights=True,
                         average_attn_weights=True)
        return w.mean(0).cpu().numpy()


def train_torch_head(model: nn.Module, Xtr, ytr, Xva=None, yva=None, *,
                     epochs: int = 200, lr: float = 1e-3, batch_size: int = 128,
                     patience: int = 15, weight_decay: float = 0.0,
                     class_weight=None, seed: int = 42):
    """Adam training. Returns (model, best_val_f1, best_epoch).

    With a validation set: early-stops on validation macro-F1 and restores the
    best weights, reporting the epoch that won. With Xva=None: trains for exactly
    `epochs` epochs and never looks at held-out data - used for the final refit on
    the full outer-train fold, where the epoch budget comes from the inner CV.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = None if class_weight is None else torch.tensor(class_weight,
                                                       dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = None if Xva is None else torch.tensor(Xva, dtype=torch.float32)

    if Xva is None:                       # fixed-budget refit, no validation
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(Xtr_t))
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()
                opt.step()
        return model, float("nan"), epochs

    best_f1, best_state, since, best_epoch = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(1).numpy()
        f1 = f1_score(yva, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_epoch = f1, ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, best_epoch


@torch.no_grad()
def predict(model: nn.Module, X) -> np.ndarray:
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


class MlpHead(nn.Module):
    """256 -> 128 ReLU with dropout: the same MLP as before, in torch.

    Moved off sklearn's MLPClassifier because that accepts neither class_weight
    nor sample_weight, forcing balanced training to be done by duplicating rare
    rows - which inflated the training set roughly fivefold and dominated runtime.
    A weighted cross-entropy does the same job at no cost, and shares the early
    stopping and epoch-budget logic with the fusion heads.
    """

    def __init__(self, d_in: int, n_classes: int, d_hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
