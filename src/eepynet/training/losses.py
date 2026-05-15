from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def class_weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.zeros_like(counts, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        total = counts[nonzero].sum()
        weights[nonzero] = total / (nonzero.sum() * counts[nonzero])
        weights[nonzero] /= weights[nonzero].mean()
    return torch.tensor(weights, dtype=torch.float32)


def masked_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = mask.bool() & (labels >= 0)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid], weight=class_weights)
