import numpy as np
import torch

from eepynet.models.eepynet import EepyNet
from eepynet.training.losses import class_weights_from_counts, masked_weighted_cross_entropy
from eepynet.training.metrics import compute_metrics


def test_eepynet_forward_shape_small_config():
    model = EepyNet(
        in_channels=2,
        num_classes=5,
        embedding_dim=32,
        hidden_dim=32,
        signal_base_channels=8,
        signal_branch_channels=4,
        dropout=0.0,
        group_norm_groups=4,
    )
    x = torch.randn(2, 2, 16, 3000)

    logits = model(x)

    assert logits.shape == (2, 16, 5)


def test_masked_weighted_cross_entropy_ignores_padding():
    logits = torch.randn(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 0]])
    mask = torch.tensor([[True, True, False, False]])
    weights = class_weights_from_counts(np.array([10, 2, 5, 1, 1]))

    loss = masked_weighted_cross_entropy(logits, labels, mask, weights)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_compute_metrics_returns_per_class_and_confusion_matrix():
    metrics = compute_metrics(
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 2, 4]),
    )

    assert metrics["accuracy"] == 0.8
    assert set(metrics["per_class"]) == {"W", "N1", "N2", "N3", "REM"}
    assert len(metrics["confusion_matrix"]) == 5
