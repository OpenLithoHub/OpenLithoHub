"""Certified halo contracts for proof-carrying lithography verification.

This module is deliberately separate from ``workflow.halo``.  The workflow
helper chooses a practical overlap from node OIR and model receptive field;
this module only emits CERTIFIED_SUFFICIENT when an explicit error tail has
been proved for a theorem-facing core/loaded-region geometry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class HaloStatus(str, Enum):
    CERTIFIED_SUFFICIENT = "CERTIFIED_SUFFICIENT"
    PARTIAL = "PARTIAL"
    OPEN = "OPEN"


@dataclass(frozen=True)
class HaloTailPoint:
    h_px: int
    lower_intensity: float
    upper_intensity: float

    def __post_init__(self) -> None:
        if self.h_px < 0:
            raise ValueError("h_px must be non-negative")
        if self.lower_intensity < 0 or self.upper_intensity < 0:
            raise ValueError("tail bounds must be non-negative")
        if self.lower_intensity > self.upper_intensity:
            raise ValueError("lower tail bound exceeds upper tail bound")


@dataclass(frozen=True)
class MinimalHaloBracket:
    epsilon_intensity: float
    necessary_h_min_px: int
    certified_sufficient_h_px: int | None
    status: HaloStatus


@dataclass(frozen=True)
class CoreHaloGeometry:
    """Explicit theorem geometry; do not infer this from workflow overlap."""

    core_width_px: int
    core_height_px: int
    loaded_width_px: int
    loaded_height_px: int
    margin_left_px: int
    margin_right_px: int
    margin_top_px: int
    margin_bottom_px: int

    @property
    def verifier_halo_px(self) -> int:
        return min(
            self.margin_left_px,
            self.margin_right_px,
            self.margin_top_px,
            self.margin_bottom_px,
        )

    def supports(self, h_px: int) -> bool:
        return self.verifier_halo_px >= h_px


@dataclass(frozen=True)
class FinalHaloRequirement:
    physics_h_px: int
    model_h_px: int
    verifier_h_px: int
    final_h_px: int
    status: HaloStatus
    note: str = ""


def socs_absolute_tail_upper(
    weights: Sequence[float],
    mode_l1: Sequence[float],
    mode_tail_l1: Sequence[float],
) -> float:
    """Sufficient intensity error bound for arbitrary |delta mask| <= 1."""
    if not (len(weights) == len(mode_l1) == len(mode_tail_l1)):
        raise ValueError("mode arrays must have equal length")
    total = 0.0
    for w, a, t in zip(weights, mode_l1, mode_tail_l1, strict=True):
        if w < 0 or a < 0 or t < 0:
            raise ValueError("weights/L1 bounds must be non-negative")
        total += w * (2.0 * a * t + t * t)
    return total


def unrestricted_binary_tail_lower(
    weights: Sequence[float],
    mode_tail_l1_lower: Sequence[float],
) -> float:
    """Necessary worst-case lower bound for arbitrary binary exterior masks.

    For one coherent mode with tail coefficients a_r, averaging the positive
    part of Re(exp(-i theta) a_r) over theta shows that some binary selector
    has coherent amplitude at least sum|a_r|/pi.  SOCS positivity then gives
    w_j*(T_j/pi)^2 for that mode.
    """
    if len(weights) != len(mode_tail_l1_lower):
        raise ValueError("mode arrays must have equal length")
    if not weights:
        raise ValueError("at least one mode is required")
    vals = []
    for w, t in zip(weights, mode_tail_l1_lower, strict=True):
        if w < 0 or t < 0:
            raise ValueError("weights/tails must be non-negative")
        vals.append(w * (t / math.pi) ** 2)
    return max(vals)


def minimal_halo_bracket(
    profile: Sequence[HaloTailPoint],
    epsilon_intensity: float,
) -> MinimalHaloBracket:
    if epsilon_intensity < 0:
        raise ValueError("epsilon_intensity must be non-negative")
    if not profile:
        raise ValueError("tail profile must not be empty")
    ordered = sorted(profile, key=lambda p: p.h_px)
    bad = [p.h_px for p in ordered if p.lower_intensity > epsilon_intensity]
    necessary = max(bad) + 1 if bad else ordered[0].h_px
    sufficient = next(
        (p.h_px for p in ordered if p.upper_intensity <= epsilon_intensity),
        None,
    )
    if sufficient is not None and sufficient < necessary:
        raise ValueError("inconsistent halo profile")
    if sufficient is None:
        status = HaloStatus.OPEN
    elif sufficient == necessary:
        status = HaloStatus.CERTIFIED_SUFFICIENT
    else:
        status = HaloStatus.PARTIAL
    return MinimalHaloBracket(
        epsilon_intensity=epsilon_intensity,
        necessary_h_min_px=necessary,
        certified_sufficient_h_px=sufficient,
        status=status,
    )


def resolve_final_halo(
    *,
    physics_h_px: int,
    model_h_px: int,
    verifier_bracket: MinimalHaloBracket,
) -> FinalHaloRequirement:
    if min(physics_h_px, model_h_px) < 0:
        raise ValueError("halo requirements must be non-negative")
    if verifier_bracket.certified_sufficient_h_px is None:
        return FinalHaloRequirement(
            physics_h_px=physics_h_px,
            model_h_px=model_h_px,
            verifier_h_px=verifier_bracket.necessary_h_min_px,
            final_h_px=max(physics_h_px, model_h_px, verifier_bracket.necessary_h_min_px),
            status=HaloStatus.OPEN,
            note="no theorem-facing sufficient verifier halo is currently proved",
        )
    vh = verifier_bracket.certified_sufficient_h_px
    return FinalHaloRequirement(
        physics_h_px=physics_h_px,
        model_h_px=model_h_px,
        verifier_h_px=vh,
        final_h_px=max(physics_h_px, model_h_px, vh),
        status=verifier_bracket.status,
    )


def halo_duplicate_compute_ratio(core_edge_px: int, halo_px: int) -> float:
    if core_edge_px <= 0 or halo_px < 0:
        raise ValueError("invalid core/halo geometry")
    c = float(core_edge_px)
    h = float(halo_px)
    return ((c + 2.0 * h) ** 2 - c**2) / c**2
