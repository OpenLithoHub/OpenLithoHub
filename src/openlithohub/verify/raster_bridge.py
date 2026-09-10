"""Discrete raster-bridge diagnostics for B04.

This module does not assert that the canonical dense loader and the
pixel-center exact-vector source are equivalent.  It measures the disagreement
and, when possible, certifies a finite grid-neighbourhood enclosure.

A Chebyshev-radius-r enclosure means every occupied pixel center of one mask
lies within r grid steps of an occupied center in the other mask.  Converted
to Euclidean center distance this is at most ``sqrt(2) * r * pixel_nm``.

This is a discrete mask bridge only.  It is not by itself an EPE, contour
Hausdorff, Hopkins/SOCS, or process-window certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryRasterBridge:
    reference_ones: int
    candidate_ones: int
    false_positive_pixels: int
    false_negative_pixels: int
    candidate_in_reference_dilation_px: int | None
    reference_in_candidate_dilation_px: int | None

    @property
    def symmetric_difference_pixels(self) -> int:
        return self.false_positive_pixels + self.false_negative_pixels

    @property
    def exact_equal(self) -> bool:
        return self.symmetric_difference_pixels == 0

    @property
    def mutual_chebyshev_radius_px(self) -> int | None:
        if (
            self.candidate_in_reference_dilation_px is None
            or self.reference_in_candidate_dilation_px is None
        ):
            return None
        return max(
            self.candidate_in_reference_dilation_px,
            self.reference_in_candidate_dilation_px,
        )

    def euclidean_center_hausdorff_upper_nm(
        self,
        *,
        pixel_nm: float,
    ) -> float | None:
        radius = self.mutual_chebyshev_radius_px
        if radius is None:
            return None
        return math.sqrt(2.0) * radius * pixel_nm


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    src = np.asarray(mask, dtype=bool)
    if radius == 0:
        return src.copy()
    out = src.copy()
    for _ in range(radius):
        prev = out
        grown = prev.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                y0 = max(0, -dy)
                y1 = min(prev.shape[0], prev.shape[0] - dy)
                x0 = max(0, -dx)
                x1 = min(prev.shape[1], prev.shape[1] - dx)
                grown[
                    y0 + dy : y1 + dy,
                    x0 + dx : x1 + dx,
                ] |= prev[y0:y1, x0:x1]
        out = grown
    return out


def _minimal_enclosure_radius(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_radius: int,
) -> int | None:
    """Smallest r with ``target subset dilation(source,r)``."""
    src = np.asarray(source, dtype=bool)
    tgt = np.asarray(target, dtype=bool)
    if src.shape != tgt.shape:
        raise ValueError("raster shapes differ")
    if not tgt.any():
        return 0
    for radius in range(max_radius + 1):
        if np.all(~tgt | _dilate_chebyshev(src, radius)):
            return radius
    return None


def compare_binary_rasters(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_radius: int = 4,
) -> BinaryRasterBridge:
    ref = np.asarray(reference, dtype=bool)
    cand = np.asarray(candidate, dtype=bool)
    if ref.shape != cand.shape:
        raise ValueError("raster shapes differ")
    return BinaryRasterBridge(
        reference_ones=int(ref.sum()),
        candidate_ones=int(cand.sum()),
        false_positive_pixels=int(np.logical_and(cand, ~ref).sum()),
        false_negative_pixels=int(np.logical_and(ref, ~cand).sum()),
        candidate_in_reference_dilation_px=_minimal_enclosure_radius(
            ref, cand, max_radius=max_radius
        ),
        reference_in_candidate_dilation_px=_minimal_enclosure_radius(
            cand, ref, max_radius=max_radius
        ),
    )
