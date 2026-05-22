"""Lightweight U-Net architecture for mask prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)  # type: ignore[no-any-return]


class _Down(nn.Module):
    """Downsampling: maxpool + double conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)  # type: ignore[no-any-return]


class _Up(nn.Module):
    """Upsampling: transpose conv + skip concat + double conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd input dimensions
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)  # type: ignore[no-any-return]


class UNet(nn.Module):
    """Compact 3-downsample U-Net for lithography mask prediction.

    Architecture: ``inc → down1 → down2 → down3 → up1 → up2 → up3 → outc``
    — three down-samples and three up-samples, with a 256-channel
    bottleneck. Channel widths (32 → 64 → 128 → 256) are intentionally
    half of those in ``Jiang2020_NeuralILT`` Fig. 4, and we run one
    fewer down-sample than the paper, so this is a v0.1 inference-budget
    baseline, **not** a paper-faithful re-implementation. The Identity-
    like baseline numbers reported by ``neural-ilt`` reflect that depth
    cap. See ``docs/audits/neural-ilt-architecture.md`` for the full
    audit and the path to a paper-faithful variant.

    Input:  (B, 1, H, W) design layout
    Output: (B, 1, H, W) mask logits
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.inc = _DoubleConv(in_channels, 32)
        self.down1 = _Down(32, 64)
        self.down2 = _Down(64, 128)
        self.down3 = _Down(128, 256)
        self.up1 = _Up(256, 128)
        self.up2 = _Up(128, 64)
        self.up3 = _Up(64, 32)
        self.outc = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)  # type: ignore[no-any-return]
