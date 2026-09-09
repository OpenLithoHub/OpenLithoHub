"""Known-layout row-run interface compression for B04.

The core identity is exact for a discrete coherent kernel bank at integer
query points.  The same API can be instantiated with derivative kernel banks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class HorizontalRun:
    y: int
    x0: int
    x1: int

    def __post_init__(self) -> None:
        if self.y < 0 or self.x0 < 0 or self.x1 <= self.x0:
            raise ValueError("invalid horizontal run")

    @property
    def length(self) -> int:
        return self.x1 - self.x0


class RunSource(Protocol):
    def iter_runs(self) -> Iterable[HorizontalRun]: ...


@dataclass(frozen=True)
class RunCompressionStats:
    occupied_pixels: int
    horizontal_runs: int

    @property
    def pixel_to_run_ratio(self) -> float:
        if self.horizontal_runs == 0:
            return float("inf") if self.occupied_pixels else 1.0
        return self.occupied_pixels / self.horizontal_runs


def encode_binary_row_runs(mask: np.ndarray) -> tuple[HorizontalRun, ...]:
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    binary = np.asarray(mask) != 0
    runs = []
    h, w = binary.shape
    for y in range(h):
        x = 0
        while x < w:
            while x < w and not binary[y, x]:
                x += 1
            if x == w:
                break
            x0 = x
            while x < w and binary[y, x]:
                x += 1
            runs.append(HorizontalRun(y, x0, x))
    return tuple(runs)


def run_compression_stats(runs: Sequence[HorizontalRun]) -> RunCompressionStats:
    return RunCompressionStats(
        occupied_pixels=sum(r.length for r in runs),
        horizontal_runs=len(runs),
    )


def split_runs_outside_chebyshev(
    runs: Sequence[HorizontalRun], *, center_y: int, center_x: int, halo_px: int
) -> tuple[HorizontalRun, ...]:
    if halo_px < 0:
        raise ValueError("halo_px must be non-negative")
    out = []
    for run in runs:
        if abs(run.y - center_y) > halo_px:
            out.append(run)
            continue
        lo = center_x - halo_px
        hi = center_x + halo_px + 1
        if run.x0 < min(run.x1, lo):
            out.append(HorizontalRun(run.y, run.x0, min(run.x1, lo)))
        if max(run.x0, hi) < run.x1:
            out.append(HorizontalRun(run.y, max(run.x0, hi), run.x1))
    return tuple(out)


@dataclass
class CyclicRowPrefix:
    prefix: np.ndarray
    grid_n: int

    @classmethod
    def from_kernel_bank(cls, kernels: np.ndarray) -> CyclicRowPrefix:
        kernels = np.asarray(kernels)
        if kernels.ndim != 3 or kernels.shape[1] != kernels.shape[2]:
            raise ValueError("kernels must have shape (K,N,N)")
        k, n, _ = kernels.shape
        prefix = np.zeros((k, n, 2 * n + 1), dtype=np.complex128)
        for j in range(k):
            for yi in range(n):
                seq = np.array(
                    [kernels[j, yi, (-s) % n] for s in range(2 * n)],
                    dtype=np.complex128,
                )
                prefix[j, yi, 1:] = np.cumsum(seq)
        return cls(prefix=prefix, grid_n=n)

    def sum_runs(
        self, runs: Sequence[HorizontalRun], *, center_y: int, center_x: int
    ) -> np.ndarray:
        n = self.grid_n
        c = n // 2
        out = np.zeros(self.prefix.shape[0], dtype=np.complex128)
        for run in runs:
            yi = (c + center_y - run.y) % n
            length = run.length
            r_start = (c + center_x - run.x0) % n
            s0 = (-r_start) % n
            out += self.prefix[:, yi, s0 + length] - self.prefix[:, yi, s0]
        return out


def known_layout_far_coherent(
    prefix: CyclicRowPrefix,
    runs: Sequence[HorizontalRun],
    *,
    center_y: int,
    center_x: int,
    halo_px: int,
) -> np.ndarray:
    far = split_runs_outside_chebyshev(runs, center_y=center_y, center_x=center_x, halo_px=halo_px)
    return prefix.sum_runs(far, center_y=center_y, center_x=center_x)


def zero_padded_horizontal_variation(runs: Sequence[HorizontalRun]) -> int:
    """For disjoint binary row-runs, padded horizontal TV is exactly 2R."""
    return 2 * len(runs)
