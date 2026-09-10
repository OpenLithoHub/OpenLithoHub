"""Theorem-facing optical transfer for mask/raster perturbations.

For a SOCS model

    F(m)(x) = sum_j w_j |E_j(m;x)|^2,

let ``delta_m = m_tilde - m`` and
``D_j = E_j(delta_m)``.  The exact identity is

    F(m + delta_m) - F(m)
      = sum_j w_j [2 Re(E_j(m) conjugate(D_j)) + |D_j|^2].

Hence any modewise bounds

    |E_j(m)| <= A_j,   |D_j| <= B_j

on a core imply

    |Delta F| <= sum_j w_j (2 A_j B_j + B_j^2).

This module deliberately separates that intensity bridge from contour transfer.
A continuous level-set statement additionally requires a certified
transversality lower bound kappa on the full intensity band.

No raster-Hausdorff distance is silently reinterpreted as EPE.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SOCSModeBridge:
    weight: float
    reference_field_upper: float
    perturbation_field_upper: float

    def __post_init__(self) -> None:
        if self.weight < 0 or self.reference_field_upper < 0 or self.perturbation_field_upper < 0:
            raise ValueError("SOCS bridge inputs must be non-negative")

    @property
    def intensity_contribution_upper(self) -> float:
        a = self.reference_field_upper
        b = self.perturbation_field_upper
        return self.weight * (2.0 * a * b + b * b)


@dataclass(frozen=True)
class OpticalIntensityBridge:
    intensity_upper: float
    provenance: str

    def __post_init__(self) -> None:
        if self.intensity_upper < 0:
            raise ValueError("intensity upper bound must be non-negative")


def socs_intensity_bridge_upper(
    modes: Sequence[SOCSModeBridge],
    *,
    provenance: str,
) -> OpticalIntensityBridge:
    upper = math.fsum(m.intensity_contribution_upper for m in modes)
    return OpticalIntensityBridge(
        intensity_upper=upper,
        provenance=provenance,
    )


@dataclass(frozen=True)
class ContourTransfer:
    intensity_upper: float
    kappa_lower_per_nm: float
    contour_hausdorff_upper_nm: float

    @property
    def epe_upper_nm(self) -> float:
        """Continuous symmetric contour-EPE upper under the same contour model."""
        return self.contour_hausdorff_upper_nm

    @property
    def paired_cd_error_upper_nm(self) -> float:
        """CD bound when opposite edges remain uniquely paired in oriented slabs."""
        return 2.0 * self.contour_hausdorff_upper_nm


def contour_transfer_from_intensity_bridge(
    *,
    intensity_upper: float,
    kappa_lower_per_nm: float,
) -> ContourTransfer:
    """Apply the sharp band-transversality transfer ``delta/kappa``.

    The caller is responsible for proving that ``kappa_lower_per_nm`` holds on
    the entire threshold band of width at least ``intensity_upper`` around the
    reference level set.  A nominal-contour gradient sample is insufficient.
    """
    if intensity_upper < 0:
        raise ValueError("intensity upper bound must be non-negative")
    if kappa_lower_per_nm <= 0:
        raise ValueError("kappa lower bound must be positive")
    upper = intensity_upper / kappa_lower_per_nm
    return ContourTransfer(
        intensity_upper=intensity_upper,
        kappa_lower_per_nm=kappa_lower_per_nm,
        contour_hausdorff_upper_nm=upper,
    )


@dataclass(frozen=True)
class ProcessWindowBridgeBudget:
    process_contour_upper_nm: float
    reconstruction_upper_nm: float
    optical_raster_bridge_upper_nm: float
    other_bridge_upper_nm: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.process_contour_upper_nm,
                self.reconstruction_upper_nm,
                self.optical_raster_bridge_upper_nm,
                self.other_bridge_upper_nm,
            )
            < 0
        ):
            raise ValueError("error-budget terms must be non-negative")

    @property
    def total_upper_nm(self) -> float:
        return math.fsum(
            (
                self.process_contour_upper_nm,
                self.reconstruction_upper_nm,
                self.optical_raster_bridge_upper_nm,
                self.other_bridge_upper_nm,
            )
        )

    def passes(self, tolerance_nm: float) -> bool:
        if tolerance_nm < 0:
            raise ValueError("tolerance must be non-negative")
        return self.total_upper_nm <= tolerance_nm
