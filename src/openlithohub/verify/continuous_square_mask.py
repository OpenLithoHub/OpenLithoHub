"""Continuous square-aperture lift of exact-vector run geometry.

Each occupied exact-vector pixel is interpreted as the physical square
``[xp,(x+1)p) x [yp,(y+1)p)``. Canonical horizontal runs therefore become
disjoint rectangles. With Fourier convention

    Mhat(f) = integral M(r) exp(-2*pi*i*f.r) dr,

one run has an exact separable rectangle transform. Equivalently,

    Mhat(fx,fy) = p^2 sinc_pi(p fx) sinc_pi(p fy) M_point(fx,fy),

where M_point is the pixel-center moment used by Increment 26.

The representation bridge is zero only for the declared square-pixel mask
semantics. Upstream GDS quantization remains explicit provenance.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from openlithohub.verify.interface_runs import HorizontalRun


class ContinuousExactRunSource(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def pixel_size_nm(self) -> float: ...

    @property
    def layout_hash(self) -> str: ...

    @property
    def backend_kind(self) -> str: ...

    def iter_runs_for_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
    ) -> Iterable[HorizontalRun]: ...


@dataclass(frozen=True)
class ContinuousSquareMaskContract:
    layout_hash: str
    backend_kind: str
    source_shape_px: tuple[int, int]
    pixel_size_nm: float
    square_pixel_aperture: bool = True
    periodic_mask_copies: bool = False
    omitted_spatial_exterior: bool = False
    dense_loader_used: bool = False
    semantics: str = "FINITE_EXACT_VECTOR_SQUARE_APERTURE_MASK"

    def __post_init__(self) -> None:
        if min(self.source_shape_px) <= 0:
            raise ValueError("source shape must be positive")
        if self.pixel_size_nm <= 0:
            raise ValueError("pixel size must be positive")
        if not self.square_pixel_aperture:
            raise ValueError("continuous lift requires square-pixel aperture semantics")
        if self.periodic_mask_copies:
            raise ValueError("finite continuous mask forbids periodic copies")
        if self.omitted_spatial_exterior:
            raise ValueError("continuous lift requires complete declared finite mask")
        if self.dense_loader_used:
            raise ValueError("proof-facing continuous lift forbids dense loader")

    @property
    def representation_bridge_upper(self) -> float:
        return 0.0


def sinc_pi_real(x: float) -> float:
    """Normalized sinc ``sin(pi*x)/(pi*x)`` with removable value at zero."""
    if x == 0.0:
        return 1.0
    pix = math.pi * x
    return math.sin(pix) / pix


def rectangle_fourier_transform(
    *,
    x0_nm: float,
    x1_nm: float,
    y0_nm: float,
    y1_nm: float,
    fx_per_nm: float,
    fy_per_nm: float,
) -> complex:
    if not (x1_nm > x0_nm and y1_nm > y0_nm):
        raise ValueError("rectangle must have positive area")
    wx = x1_nm - x0_nm
    wy = y1_nm - y0_nm
    cx = (x0_nm + x1_nm) / 2.0
    cy = (y0_nm + y1_nm) / 2.0
    xterm = wx * sinc_pi_real(fx_per_nm * wx) * cmath.exp(-2j * math.pi * fx_per_nm * cx)
    yterm = wy * sinc_pi_real(fy_per_nm * wy) * cmath.exp(-2j * math.pi * fy_per_nm * cy)
    return xterm * yterm


def run_square_aperture_transform(
    run: HorizontalRun,
    *,
    pixel_nm: float,
    fx_per_nm: float,
    fy_per_nm: float,
) -> complex:
    if pixel_nm <= 0:
        raise ValueError("pixel_nm must be positive")
    return rectangle_fourier_transform(
        x0_nm=run.x0 * pixel_nm,
        x1_nm=run.x1 * pixel_nm,
        y0_nm=run.y * pixel_nm,
        y1_nm=(run.y + 1) * pixel_nm,
        fx_per_nm=fx_per_nm,
        fy_per_nm=fy_per_nm,
    )


def continuous_square_mask_transform(
    source: ContinuousExactRunSource,
    *,
    fx_per_nm: float,
    fy_per_nm: float,
) -> complex:
    """Evaluate the exact continuous square-aperture mask transform."""
    h, w = source.shape
    total = 0j
    for run in source.iter_runs_for_bbox((0, 0, w, h)):
        if not (0 <= run.y < h and 0 <= run.x0 < run.x1 <= w):
            raise ValueError("source returned a run outside the finite layout")
        total += run_square_aperture_transform(
            run,
            pixel_nm=source.pixel_size_nm,
            fx_per_nm=fx_per_nm,
            fy_per_nm=fy_per_nm,
        )
    return total


def continuous_square_mask_contract(
    source: ContinuousExactRunSource,
) -> ContinuousSquareMaskContract:
    return ContinuousSquareMaskContract(
        layout_hash=source.layout_hash,
        backend_kind=source.backend_kind,
        source_shape_px=source.shape,
        pixel_size_nm=source.pixel_size_nm,
    )
