"""Finite-vs-periodic Fourier boundary contracts for B04."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SpectrumBoundary(str, Enum):
    PERIODIC_FIXTURE = "PERIODIC_FIXTURE"
    FINITE_ZERO_PADDED = "FINITE_ZERO_PADDED"


@dataclass(frozen=True)
class KernelSupport:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if min(self.left, self.top, self.right, self.bottom) < 0:
            raise ValueError("kernel support must be non-negative")


@dataclass(frozen=True)
class ZeroPaddingContract:
    physical_shape: tuple[int, int]
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    kernel_support: KernelSupport

    def __post_init__(self) -> None:
        if min(self.pad_left, self.pad_top, self.pad_right, self.pad_bottom) < 0:
            raise ValueError("padding must be non-negative")
        k = self.kernel_support
        if (
            self.pad_left < k.left
            or self.pad_right < k.right
            or self.pad_top < k.top
            or self.pad_bottom < k.bottom
        ):
            raise ValueError("insufficient zero padding for finite convolution")

    @property
    def padded_shape(self) -> tuple[int, int]:
        h, w = self.physical_shape
        return (
            self.pad_top + h + self.pad_bottom,
            self.pad_left + w + self.pad_right,
        )


def stable_geometric_run_sum(
    *,
    frequency_index: int,
    x0: int,
    x1: int,
    period: int,
) -> complex:
    """Production midpoint formula; proof path should use outward root disks."""
    if period <= 0 or not 0 <= x0 < x1 <= period:
        raise ValueError("invalid run/period")
    k = frequency_index % period
    length = x1 - x0
    if k == 0:
        return complex(length, 0.0)
    theta = 2.0 * math.pi * k / period
    amp = math.sin(length * theta / 2.0) / math.sin(theta / 2.0)
    phase = -theta * (x0 + (length - 1) / 2.0)
    return complex(math.cos(phase), math.sin(phase)) * amp
