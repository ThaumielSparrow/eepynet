from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def class_weights_from_counts(counts: np.ndarray, power: float = 1.0) -> torch.Tensor:
    """Mean-normalized class weights ``(total / count) ** power``.

    ``power=1.0`` reproduces the inverse-frequency weighting; ``power=0.5`` (sqrt)
    is the gentler default used by the trainer to avoid over-emitting rare classes.
    """

    counts = np.asarray(counts, dtype=np.float64)
    weights = np.zeros_like(counts, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        total = counts[nonzero].sum()
        raw = (total / counts[nonzero]) ** float(power)
        weights[nonzero] = (raw / raw.mean()).astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)


def masked_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid = mask.bool() & (labels >= 0)
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits[valid],
        labels[valid],
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )


def masked_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss with optional per-class weights, applied only to valid (unmasked) epochs.

    Class weights serve as α_t in the focal formulation; the modulating factor
    ``(1 − p_t)^γ`` additionally down-weights confident predictions so the
    gradient concentrates on hard examples (e.g. N1 misclassified as W/N2).
    """
    valid = mask.bool() & (labels >= 0)
    if not valid.any():
        return logits.sum() * 0.0
    flat_logits = logits[valid]  # [N, C]
    flat_labels = labels[valid]  # [N]
    ce = F.cross_entropy(
        flat_logits,
        flat_labels,
        weight=class_weights.to(flat_logits.dtype) if class_weights is not None else None,
        reduction="none",
    )
    with torch.no_grad():
        p_t = flat_logits.softmax(dim=-1).gather(1, flat_labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t).pow(gamma)
    return (focal_weight * ce).mean()
