from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, requested_groups: int) -> int:
    for groups in range(min(requested_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels, groups), out_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            dropout=dropout,
        )
        padding = dilation * (kernel_size - 1) // 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels, groups), channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class MultiScaleConvStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        out_channels: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvGNAct(
                    in_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    groups=groups,
                    dropout=dropout,
                )
                for kernel_size in (25, 51, 101)
            ]
        )
        self.project = ConvGNAct(
            branch_channels * len(self.branches),
            out_channels,
            kernel_size=1,
            groups=groups,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class EpochSignalEncoder(nn.Module):
    """Encode each 30-second EEG epoch into one embedding vector."""

    def __init__(
        self,
        in_channels: int = 2,
        embedding_dim: int = 192,
        base_channels: int = 64,
        branch_channels: int = 32,
        groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = MultiScaleConvStem(
            in_channels=in_channels,
            branch_channels=branch_channels,
            out_channels=base_channels,
            groups=groups,
            dropout=dropout,
        )
        self.encoder = nn.Sequential(
            ResidualBlock1D(base_channels, kernel_size=7, groups=groups, dropout=dropout),
            ConvGNAct(base_channels, base_channels * 2, kernel_size=9, stride=4, groups=groups),
            ResidualBlock1D(base_channels * 2, kernel_size=7, groups=groups, dropout=dropout),
            ConvGNAct(base_channels * 2, base_channels * 3, kernel_size=9, stride=4, groups=groups),
            ResidualBlock1D(base_channels * 3, kernel_size=5, groups=groups, dropout=dropout),
            ConvGNAct(base_channels * 3, base_channels * 4, kernel_size=9, stride=4, groups=groups),
            ResidualBlock1D(base_channels * 4, kernel_size=5, groups=groups, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,E,T], got shape {tuple(x.shape)}")
        batch, channels, epochs, samples = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * epochs, channels, samples)
        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.proj(x)
        return x.reshape(batch, epochs, -1)


class UNetDown(nn.Module):
    def __init__(self, channels: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock1D(channels, kernel_size=5, groups=groups, dropout=dropout),
            ConvGNAct(channels, channels, kernel_size=5, stride=2, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block[0](x)
        down = self.block[1](skip)
        return down, skip


class UNetUp(nn.Module):
    def __init__(self, channels: int, groups: int, dropout: float) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvGNAct(channels * 2, channels, kernel_size=3, groups=groups, dropout=dropout),
            ResidualBlock1D(channels, kernel_size=5, groups=groups, dropout=dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class EpochSequenceUNet(nn.Module):
    """1D U-Net over the epoch axis."""

    def __init__(
        self,
        embedding_dim: int = 192,
        hidden_dim: int = 192,
        groups: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_proj = ConvGNAct(
            embedding_dim,
            hidden_dim,
            kernel_size=1,
            groups=groups,
            dropout=dropout,
        )
        self.downs = nn.ModuleList(
            [UNetDown(hidden_dim, groups=groups, dropout=dropout) for _ in range(4)]
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(hidden_dim, kernel_size=5, dilation=2, groups=groups, dropout=dropout),
            ResidualBlock1D(hidden_dim, kernel_size=5, dilation=4, groups=groups, dropout=dropout),
            ResidualBlock1D(hidden_dim, kernel_size=5, dilation=8, groups=groups, dropout=dropout),
        )
        self.ups = nn.ModuleList(
            [UNetUp(hidden_dim, groups=groups, dropout=dropout) for _ in range(4)]
        )
        self.out_norm = nn.GroupNorm(_group_count(hidden_dim, groups), hidden_dim)
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,E,D], got shape {tuple(x.shape)}")
        x = x.transpose(1, 2)
        original_len = x.shape[-1]
        x = self.input_proj(x)

        skips: list[torch.Tensor] = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        if x.shape[-1] != original_len:
            x = F.interpolate(x, size=original_len, mode="linear", align_corners=False)
        x = self.out_act(self.out_norm(x))
        return x.transpose(1, 2)


class SleepStageHead(nn.Module):
    def __init__(self, hidden_dim: int = 192, num_classes: int = 5) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class EepyNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 5,
        embedding_dim: int = 192,
        hidden_dim: int = 192,
        signal_base_channels: int = 64,
        signal_branch_channels: int = 32,
        dropout: float = 0.15,
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.model_config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "signal_base_channels": signal_base_channels,
            "signal_branch_channels": signal_branch_channels,
            "dropout": dropout,
            "group_norm_groups": group_norm_groups,
        }
        self.epoch_encoder = EpochSignalEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            base_channels=signal_base_channels,
            branch_channels=signal_branch_channels,
            groups=group_norm_groups,
            dropout=dropout,
        )
        self.sequence_unet = EpochSequenceUNet(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            groups=group_norm_groups,
            dropout=dropout,
        )
        self.head = SleepStageHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.epoch_encoder(x)
        x = self.sequence_unet(x)
        return self.head(x)


def build_model(config: dict[str, Any]) -> EepyNet:
    return EepyNet(**config)


def load_model_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[EepyNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_cfg = checkpoint["model_config"]
    model = EepyNet(**model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
