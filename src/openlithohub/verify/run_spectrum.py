"""Direct row-run to Fourier-spectrum accumulation for B04."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .interface_runs import HorizontalRun


@dataclass(frozen=True)
class ComplexDiskGrid:
    mid: np.ndarray
    rad: np.ndarray

    def __post_init__(self) -> None:
        if self.mid.shape != self.rad.shape:
            raise ValueError("complex disk arrays must have equal shape")
        if np.any(self.rad < 0):
            raise ValueError("complex disk radii must be non-negative")


@dataclass
class RunSpectrumAccumulator:
    grid_n: int
    _spectrum: np.ndarray

    @classmethod
    def create(cls, grid_n: int) -> RunSpectrumAccumulator:
        if grid_n <= 0:
            raise ValueError("grid_n must be positive")
        return cls(grid_n, np.zeros((grid_n, grid_n), dtype=np.complex128))

    def add_run(self, run: HorizontalRun) -> None:
        n = self.grid_n
        if run.y >= n or run.x1 > n:
            raise ValueError("run lies outside spectrum domain")
        ky = np.arange(n)
        kx = np.arange(n)
        wy = np.exp(-2j * np.pi * ky * run.y / n)
        xs = np.arange(run.x0, run.x1)
        sx = np.exp(-2j * np.pi * kx[:, None] * xs[None, :] / n).sum(axis=1)
        self._spectrum += wy[:, None] * sx[None, :]

    def extend(self, runs: Iterable[HorizontalRun]) -> None:
        for run in runs:
            self.add_run(run)

    def normalized_spectrum(self) -> np.ndarray:
        n = self.grid_n
        return self._spectrum / (n * n)


def coherent_spectrum_from_layout(
    *,
    normalized_kernel_spectrum: np.ndarray,
    normalized_layout_spectrum: np.ndarray,
) -> np.ndarray:
    kernels = np.asarray(normalized_kernel_spectrum)
    layout = np.asarray(normalized_layout_spectrum)
    if kernels.ndim != 3 or layout.ndim != 2:
        raise ValueError("expected kernel bank (K,N,N) and layout spectrum (N,N)")
    if kernels.shape[1:] != layout.shape:
        raise ValueError("kernel/layout spectrum shapes do not match")
    n = layout.shape[0]
    out: np.ndarray = (n * n) * kernels * layout[None, :, :]
    return out
