"""Halo policies (RFC 0008, prompt §4-6).

A ``HaloPolicy`` answers "how much context halo does this tile need?" and
must always explain itself: every answer carries the halo, its provenance,
an optional rigorous error bound, and a certification status.  Policies
composing means taking the max halo; the strictest provenance/certification
status wins.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch

from openlithohub.models.base import LithographyModel
from openlithohub.workflow.process_node import ProcessNodeConfig

DEFAULT_HALO_PX = 128
_HALO_ROUND_PX = 8


class HaloStatus(str, Enum):
    FIXED = "FIXED"
    PHYSICS_ESTIMATED = "PHYSICS_ESTIMATED"
    EMPIRICALLY_STABLE = "EMPIRICALLY_STABLE"
    CERTIFIED_SUFFICIENT = "CERTIFIED_SUFFICIENT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class HaloRequirement:
    halo_px: int
    provenance: str
    error_bound: float | None
    certified: bool
    reason: str
    status: HaloStatus = HaloStatus.FIXED

    def __post_init__(self) -> None:
        if self.halo_px < 0:
            raise ValueError(f"halo_px must be nonnegative, got {self.halo_px}")
        if self.status is HaloStatus.CERTIFIED_SUFFICIENT and self.error_bound is None:
            raise ValueError("CERTIFIED_SUFFICIENT requires an error_bound")


def combine_requirements(*requirements: HaloRequirement | None) -> HaloRequirement:
    """Max-combine halo requirements; strictest provenance wins.

    Any non-certified participant demotes the combination, and any
    INCONCLUSIVE participant makes the whole answer inconclusive — a halo
    is only as trustworthy as its weakest contributor.
    """
    live = [r for r in requirements if r is not None]
    if not live:
        raise ValueError("combine_requirements needs at least one requirement")
    halo = max(r.halo_px for r in live)
    provenance = "+".join(r.provenance for r in live)
    bounds = [r.error_bound for r in live if r.error_bound is not None]
    error_bound = max(bounds) if bounds else None
    if any(r.status is HaloStatus.INCONCLUSIVE for r in live):
        status = HaloStatus.INCONCLUSIVE
        certified = False
    elif all(r.status is HaloStatus.CERTIFIED_SUFFICIENT for r in live):
        status = HaloStatus.CERTIFIED_SUFFICIENT
        certified = all(r.certified for r in live)
    else:
        status = HaloStatus.PHYSICS_ESTIMATED
        certified = False
    reason = "; ".join(r.reason for r in live)
    return HaloRequirement(
        halo_px=halo,
        provenance=provenance,
        error_bound=error_bound,
        certified=certified,
        reason=reason,
        status=status,
    )


@dataclass(frozen=True)
class HaloContext:
    """Everything a policy may need to size a halo."""

    pixel_nm: float
    tile_core_px: int
    node: ProcessNodeConfig | None = None
    model: LithographyModel | None = None
    kernel: torch.Tensor | None = None
    kernel_tolerance: float | None = None


@runtime_checkable
class HaloPolicy(Protocol):
    name: str

    def required_halo(self, context: HaloContext) -> HaloRequirement: ...


def _round_up(px: int) -> int:
    return ((px + _HALO_ROUND_PX - 1) // _HALO_ROUND_PX) * _HALO_ROUND_PX


class LegacyFixedHaloPolicy:
    """Reproduce the pre-RFC-0005 fixed halo (default 128 px)."""

    name = "legacy_fixed"

    def __init__(self, halo_px: int = DEFAULT_HALO_PX) -> None:
        if halo_px < 0:
            raise ValueError(f"halo_px must be nonnegative, got {halo_px}")
        self._halo_px = halo_px

    def required_halo(self, context: HaloContext) -> HaloRequirement:
        return HaloRequirement(
            halo_px=min(self._halo_px, max(context.tile_core_px - 1, 0)),
            provenance="legacy_fixed",
            error_bound=None,
            certified=False,
            reason=f"fixed legacy halo of {self._halo_px} px (backward compatibility)",
            status=HaloStatus.FIXED,
        )


class PhysicalInteractionHaloPolicy:
    """``h = max(OIR_px, receptive-field px)``, RFC-0005 logic moved here."""

    name = "physical_interaction"

    def __init__(self, round_up_to: int = _HALO_ROUND_PX) -> None:
        self._round = max(1, round_up_to)

    def required_halo(self, context: HaloContext) -> HaloRequirement:
        parts: list[str] = []
        px = 0
        if context.node is not None and context.pixel_nm > 0:
            oir_px = math.ceil(context.node.optical_radius_nm / context.pixel_nm)
            parts.append(f"optical interaction radius {oir_px} px")
            px = max(px, oir_px)
        if context.model is not None:
            rf = getattr(context.model, "RECEPTIVE_FIELD_PX", 0) or 0
            if rf:
                parts.append(f"model receptive field {rf} px")
            px = max(px, rf)
        halo = _round_up(px) if self._round > 1 else px
        return HaloRequirement(
            halo_px=min(halo, max(context.tile_core_px - 1, 0)),
            provenance="physical_interaction",
            error_bound=None,
            certified=False,
            reason="max(" + (", ".join(parts) if parts else "no physics source") + ")",
            status=HaloStatus.PHYSICS_ESTIMATED,
        )


class KernelTailHaloPolicy:
    """Pick the smallest halo whose kernel tail mass <= tolerance.

    ``T(h) = sum_{|x|>h} |K(x)| <= eps`` over the sampled spatial kernel.
    When the sample truly bounds the full kernel, the emitted halo is
    CERTIFIED_SUFFICIENT with ``tail mass`` as the error bound; if the
    caller cannot vouch for the sample, pass ``trusted_sample=False`` and
    the answer stays PHYSICS_ESTIMATED (never pretend certification).
    """

    name = "kernel_tail"

    def __init__(self, tolerance: float = 1e-3, trusted_sample: bool = True) -> None:
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        self._tolerance = float(tolerance)
        self._trusted = bool(trusted_sample)

    def required_halo(self, context: HaloContext) -> HaloRequirement:
        kernel = context.kernel
        if kernel is None:
            return HaloRequirement(
                halo_px=DEFAULT_HALO_PX,
                provenance="kernel_tail",
                error_bound=None,
                certified=False,
                reason="no kernel sample available; fell back to legacy default",
                status=HaloStatus.INCONCLUSIVE,
            )
        k = kernel.detach().abs()
        total = float(k.sum())
        if total <= 0.0:
            return HaloRequirement(
                halo_px=DEFAULT_HALO_PX,
                provenance="kernel_tail",
                error_bound=None,
                certified=False,
                reason="degenerate (all-zero) kernel sample",
                status=HaloStatus.INCONCLUSIVE,
            )
        # Box-aligned tail sweep: the halo keeps a (2h+1)^2 central box, so
        # the tail at halo h is exactly kernel_tail_mass(k, h). Find the
        # smallest h whose tail meets the tolerance.
        max_r = min(k.shape[0], k.shape[1]) // 2
        best_h: int | None = None
        for h in range(0, max_r + 1):
            if kernel_tail_mass(k, h) <= self._tolerance:
                best_h = h
                break
        if best_h is None:
            # even keeping the whole sampled kernel leaves the tolerance
            # unmet only when the sample is unnormalizable noise; report the
            # tail at the largest box.
            tail = kernel_tail_mass(k, max_r)
            return HaloRequirement(
                halo_px=_round_up(max_r),
                provenance="kernel_tail",
                error_bound=tail,
                certified=False,
                reason=(
                    f"kernel tail {tail:.3e} still exceeds tolerance "
                    f"{self._tolerance:.3e} at the largest sampled radius"
                ),
                status=HaloStatus.INCONCLUSIVE,
            )
        halo = min(_round_up(best_h), max(context.tile_core_px - 1, 0))
        return HaloRequirement(
            halo_px=halo,
            provenance="kernel_tail",
            error_bound=self._tolerance if self._trusted else None,
            certified=self._trusted,
            reason=(f"smallest sampled radius with kernel tail mass <= {self._tolerance:.3e}"),
            status=(
                HaloStatus.CERTIFIED_SUFFICIENT if self._trusted else HaloStatus.PHYSICS_ESTIMATED
            ),
        )


def kernel_tail_mass(kernel: torch.Tensor, halo_px: int) -> float:
    """Fraction of |kernel| mass outside the central ``(2h+1)^2`` box."""
    k = kernel.detach().abs()
    total = float(k.sum())
    if total <= 0.0:
        return 0.0
    cy, cx = k.shape[0] // 2, k.shape[1] // 2
    # Clamp so a halo larger than the kernel radius cannot wrap via
    # negative slicing — beyond the kernel radius the tail is exactly 0.
    y0, y1 = max(0, cy - halo_px), min(k.shape[0], cy + halo_px + 1)
    x0, x1 = max(0, cx - halo_px), min(k.shape[1], cx + halo_px + 1)
    keep = torch.zeros_like(k, dtype=torch.bool)
    keep[y0:y1, x0:x1] = True
    return float(k[~keep].sum()) / total


def estimate_minimum_halo(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    core_bbox_px: tuple[int, int, int, int],
    full_context: torch.Tensor,
    candidate_halos: Iterable[int] = (0, 8, 16, 32, 64, 128),
    tolerance: float = 1e-3,
) -> HaloRequirement:
    """Adaptive empirical halo sizing (prompt §6).

    Compares the forward result restricted to a fixed core as the halo
    grows.  The first candidate whose discrepancy with the final
    (largest-halo) reference drops under ``tolerance`` — and stays there
    for the remaining candidates — is returned as EMPIRICALLY_STABLE.
    This is explicitly *not* a mathematical certificate; a rigorous bound
    needs an analytic kernel-tail or QDM argument.
    """
    x0, y0, x1, y1 = core_bbox_px
    core_h, core_w = y1 - y0, x1 - x0
    candidates = sorted({int(h) for h in candidate_halos if h >= 0})
    if not candidates:
        raise ValueError("need at least one candidate halo")

    core_errs: list[tuple[int, torch.Tensor]] = []
    for halo in candidates:
        tile = full_context[max(0, y0 - halo) : y1 + halo, max(0, x0 - halo) : x1 + halo]
        out = forward_fn(tile)
        # Extract the core response from the (possibly clipped) read region.
        oy = y0 - max(0, y0 - halo)
        ox = x0 - max(0, x0 - halo)
        core_out = out[oy : oy + core_h, ox : ox + core_w]
        core_errs.append((halo, core_out))

    reference_h, reference = core_errs[-1]
    stable_from: int | None = None
    for i in range(len(core_errs) - 1):
        h_i, out_i = core_errs[i]
        disc = float(torch.max(torch.abs(out_i - reference)).item())
        if disc <= tolerance and stable_from is None:
            stable_from = h_i
        elif disc > tolerance:
            stable_from = None
    if stable_from is None:
        return HaloRequirement(
            halo_px=reference_h,
            provenance="empirical_sweep",
            error_bound=None,
            certified=False,
            reason=f"core discrepancy never stabilised under tolerance {tolerance:g}",
            status=HaloStatus.INCONCLUSIVE,
        )
    return HaloRequirement(
        halo_px=stable_from,
        provenance="empirical_sweep",
        error_bound=tolerance,
        certified=False,
        reason=(
            f"core discrepancy <= {tolerance:g} for all halos >= {stable_from} px "
            f"(reference halo {reference_h} px)"
        ),
        status=HaloStatus.EMPIRICALLY_STABLE,
    )
