"""Training-time signal augmentation for EEG/PSG chunks.

All augmentations operate on a single chunk tensor of shape [C, E, T]
(channels, epochs, time-samples).  Only applied during training — pass
``augmenter=None`` (or omit) for validation/test datasets.
"""
from __future__ import annotations

from typing import Any

import torch


class SignalAugmenter:
    """Applies three independent, configurable augmentations in sequence.

    Args:
        gaussian_noise_std: Std of additive Gaussian noise relative to the
            z-scored signal (0 = disabled). 0.05 is a good default.
        channel_dropout_prob: Probability of zeroing each channel independently.
            At least one channel is always preserved (0 = disabled).
        amplitude_scale_range: (min, max) of a per-channel uniform amplitude
            multiplier.  (1.0, 1.0) is a no-op.
    """

    def __init__(
        self,
        gaussian_noise_std: float = 0.0,
        channel_dropout_prob: float = 0.0,
        amplitude_scale_range: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.noise_std = float(gaussian_noise_std)
        self.channel_dropout_prob = float(channel_dropout_prob)
        self.amp_min, self.amp_max = float(amplitude_scale_range[0]), float(amplitude_scale_range[1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to *x* in-place-safe way (always returns a new tensor).

        Args:
            x: Float tensor of shape ``[C, E, T]``.

        Returns:
            Augmented tensor, same shape and dtype as *x*.
        """
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if self.channel_dropout_prob > 0:
            C = x.shape[0]
            keep = torch.bernoulli(torch.full((C,), 1.0 - self.channel_dropout_prob))
            if not keep.any():
                keep[torch.randint(C, (1,)).item()] = 1.0
            x = x * keep[:, None, None]

        if self.amp_min != self.amp_max:
            C = x.shape[0]
            scale = torch.empty(C).uniform_(self.amp_min, self.amp_max)
            x = x * scale[:, None, None]

        return x

    def __repr__(self) -> str:
        return (
            f"SignalAugmenter(noise_std={self.noise_std}, "
            f"channel_dropout={self.channel_dropout_prob}, "
            f"amplitude=[{self.amp_min}, {self.amp_max}])"
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SignalAugmenter | None:
        """Build from the top-level training config dict.

        Returns ``None`` (no augmentation) when the ``augmentation`` section
        is absent or all values are at their no-op defaults.
        """
        aug = config.get("augmentation") or {}
        noise = float(aug.get("gaussian_noise_std", 0.0))
        dropout = float(aug.get("channel_dropout_prob", 0.0))
        amp_range_raw = aug.get("amplitude_scale_range", [1.0, 1.0])
        amp_range = (float(amp_range_raw[0]), float(amp_range_raw[1]))

        if noise == 0.0 and dropout == 0.0 and amp_range == (1.0, 1.0):
            return None
        return cls(
            gaussian_noise_std=noise,
            channel_dropout_prob=dropout,
            amplitude_scale_range=amp_range,
        )
