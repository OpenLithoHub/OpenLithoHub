"""Global finite-layout spectral sufficient statistics for B04.

The proof-facing mask is a finite exact-vector pixel-center indicator.  For a
finite set of arbitrary physical spatial-frequency nodes, the complete layout
can be streamed once into Fourier moments

    M_q = sum_{(x,y) occupied}
          exp(-2*pi*i*(fx_q*(x+1/2)*p + fy_q*(y+1/2)*p)).

Horizontal runs admit a closed geometric-sum contribution.  Because every
physical run in the finite layout is included, no spatial exterior is omitted
and the theorem-facing *spatial halo error is exactly zero* for the declared
finite plane-wave quadrature model.

This does not certify the quadrature as an exact representation of continuous
Hopkins diffraction.  Pupil/source quadrature error is a separate, explicit
proof obligation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from openlithohub.verify.interface_runs import HorizontalRun


@dataclass(frozen=True)
class SpectralNode:
    """Physical spatial frequency in cycles / nm."""

    fx_per_nm: float
    fy_per_nm: float


class GlobalExactRunSource(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def layout_hash(self) -> str: ...

    @property
    def pixel_size_nm(self) -> float: ...

    @property
    def backend_kind(self) -> str: ...

    def iter_runs_for_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
    ) -> Iterable[HorizontalRun]: ...


@dataclass(frozen=True)
class GlobalFiniteSpectralContract:
    layout_hash: str
    backend_kind: str
    source_shape_px: tuple[int, int]
    pixel_size_nm: float
    complete_physical_layout: bool = True
    periodic_mask_copies: bool = False
    dense_loader_used: bool = False
    omitted_spatial_exterior: bool = False
    semantics: str = "FINITE_EXACT_VECTOR_PIXEL_CENTER_MEASURE"

    def __post_init__(self) -> None:
        if min(self.source_shape_px) <= 0:
            raise ValueError("source shape must be positive")
        if self.pixel_size_nm <= 0:
            raise ValueError("pixel size must be positive")
        if not self.complete_physical_layout:
            raise ValueError("global zero-halo contract requires the complete finite layout")
        if self.periodic_mask_copies:
            raise ValueError("finite-layout contract forbids periodic mask copies")
        if self.dense_loader_used:
            raise ValueError("finite-layout proof contract forbids dense loader")
        if self.omitted_spatial_exterior:
            raise ValueError("zero-halo contract is invalid when spatial exterior is omitted")

    @property
    def spatial_halo_error_upper(self) -> float:
        return 0.0


@dataclass(frozen=True)
class GlobalFiniteSpectralSummary:
    nodes: tuple[SpectralNode, ...]
    moments: np.ndarray
    run_count: int
    occupied_pixels: int
    contract: GlobalFiniteSpectralContract

    def __post_init__(self) -> None:
        if self.moments.shape != (len(self.nodes),):
            raise ValueError("moment vector length does not match spectral nodes")
        if self.run_count < 0 or self.occupied_pixels < 0:
            raise ValueError("run/pixel counts must be non-negative")


def _run_x_phase_sum(
    *,
    fx_per_nm: float,
    pixel_nm: float,
    x0: int,
    x1: int,
) -> complex:
    """Stable sum over pixel centers x+1/2 for one horizontal run."""

    if x0 < 0 or x1 <= x0:
        raise ValueError("invalid run")
    length = x1 - x0
    theta = 2.0 * math.pi * fx_per_nm * pixel_nm

    # If exp(-i*theta) is numerically unity, all run pixels have the same
    # phase.  The half-pixel center offset remains explicit.
    if abs(math.sin(theta / 2.0)) < 1e-14:
        return length * complex(
            math.cos(-theta * (x0 + 0.5)),
            math.sin(-theta * (x0 + 0.5)),
        )

    # Sum_{n=0}^{L-1} exp(-i theta (x0+n+1/2))
    # = exp(-i theta (x0+L/2)) * sin(L theta/2)/sin(theta/2).
    ratio = math.sin(length * theta / 2.0) / math.sin(theta / 2.0)
    phase = -theta * (x0 + length / 2.0)
    return ratio * complex(math.cos(phase), math.sin(phase))


def run_moment(
    run: HorizontalRun,
    node: SpectralNode,
    *,
    pixel_nm: float,
) -> complex:
    if run.y < 0:
        raise ValueError("run y must be non-negative")
    xsum = _run_x_phase_sum(
        fx_per_nm=node.fx_per_nm,
        pixel_nm=pixel_nm,
        x0=run.x0,
        x1=run.x1,
    )
    yphase = -2.0 * math.pi * node.fy_per_nm * pixel_nm * (run.y + 0.5)
    return xsum * complex(math.cos(yphase), math.sin(yphase))


def global_finite_spectral_summary(
    source: GlobalExactRunSource,
    *,
    nodes: Sequence[SpectralNode],
) -> GlobalFiniteSpectralSummary:
    """Stream every physical run into a finite arbitrary-frequency summary."""

    if not nodes:
        raise ValueError("at least one spectral node is required")
    height, width = source.shape
    bbox = (0, 0, width, height)
    runs = tuple(source.iter_runs_for_bbox(bbox))

    moments = np.zeros(len(nodes), dtype=np.complex128)
    occupied = 0
    for run in runs:
        if not (0 <= run.y < height and 0 <= run.x0 < run.x1 <= width):
            raise ValueError("source returned a run outside the finite layout")
        occupied += int(run.x1 - run.x0)
        for q, node in enumerate(nodes):
            moments[q] += run_moment(
                run,
                node,
                pixel_nm=source.pixel_size_nm,
            )

    contract = GlobalFiniteSpectralContract(
        layout_hash=source.layout_hash,
        backend_kind=source.backend_kind,
        source_shape_px=source.shape,
        pixel_size_nm=source.pixel_size_nm,
    )
    return GlobalFiniteSpectralSummary(
        nodes=tuple(nodes),
        moments=moments,
        run_count=len(runs),
        occupied_pixels=occupied,
        contract=contract,
    )


@dataclass(frozen=True)
class PlaneWaveMode:
    """One coherent finite-quadrature mode."""

    weight: float
    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("mode weight must be non-negative")
        if self.amplitudes.ndim != 1:
            raise ValueError("mode amplitudes must be one-dimensional")


@dataclass(frozen=True)
class IntensityJet:
    value: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dxx: np.ndarray
    dxy: np.ndarray
    dyy: np.ndarray


def _field_derivative(
    summary: GlobalFiniteSpectralSummary,
    amplitudes: np.ndarray,
    points_nm: np.ndarray,
    *,
    alpha_x: int,
    alpha_y: int,
) -> np.ndarray:
    if amplitudes.shape != summary.moments.shape:
        raise ValueError("amplitude vector does not match spectral summary")
    if alpha_x < 0 or alpha_y < 0:
        raise ValueError("derivative orders must be non-negative")

    pts = np.asarray(points_nm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_nm must have shape (M,2)")

    fx = np.array([n.fx_per_nm for n in summary.nodes], dtype=np.float64)
    fy = np.array([n.fy_per_nm for n in summary.nodes], dtype=np.float64)
    phase = np.exp(2j * np.pi * (pts[:, 0:1] * fx[None, :] + pts[:, 1:2] * fy[None, :]))
    derivative_factor = (2j * np.pi * fx) ** alpha_x * (2j * np.pi * fy) ** alpha_y
    coeff = amplitudes * summary.moments * derivative_factor
    out: np.ndarray = phase @ coeff
    return out


def evaluate_intensity_jet(
    summary: GlobalFiniteSpectralSummary,
    modes: Sequence[PlaneWaveMode],
    *,
    points_nm: np.ndarray,
) -> IntensityJet:
    """Evaluate value/gradient/Hessian of a finite plane-wave intensity model."""

    if not modes:
        raise ValueError("at least one coherent mode is required")
    m = np.asarray(points_nm).shape[0]
    value = np.zeros(m, dtype=np.float64)
    dx = np.zeros(m, dtype=np.float64)
    dy = np.zeros(m, dtype=np.float64)
    dxx = np.zeros(m, dtype=np.float64)
    dxy = np.zeros(m, dtype=np.float64)
    dyy = np.zeros(m, dtype=np.float64)

    for mode in modes:
        if mode.amplitudes.shape != summary.moments.shape:
            raise ValueError("mode amplitude length mismatch")
        e = _field_derivative(summary, mode.amplitudes, points_nm, alpha_x=0, alpha_y=0)
        ex = _field_derivative(summary, mode.amplitudes, points_nm, alpha_x=1, alpha_y=0)
        ey = _field_derivative(summary, mode.amplitudes, points_nm, alpha_x=0, alpha_y=1)
        exx = _field_derivative(summary, mode.amplitudes, points_nm, alpha_x=2, alpha_y=0)
        exy = _field_derivative(summary, mode.amplitudes, points_nm, alpha_x=1, alpha_y=1)
        eyy = _field_derivative(summary, mode.amplitudes, points_nm, alpha_x=0, alpha_y=2)
        w = mode.weight
        value += w * np.abs(e) ** 2
        dx += 2.0 * w * np.real(np.conj(e) * ex)
        dy += 2.0 * w * np.real(np.conj(e) * ey)
        dxx += 2.0 * w * (np.abs(ex) ** 2 + np.real(np.conj(e) * exx))
        dxy += 2.0 * w * np.real(np.conj(ex) * ey + np.conj(e) * exy)
        dyy += 2.0 * w * (np.abs(ey) ** 2 + np.real(np.conj(e) * eyy))

    return IntensityJet(
        value=value,
        dx=dx,
        dy=dy,
        dxx=dxx,
        dxy=dxy,
        dyy=dyy,
    )
