"""Public ``verify_layout()`` entry point (RFC 0008, philosophy §16).

Provides a single-call proof-carrying verification API that returns a typed
result with status, continuous EPE/Hausdorff upper bounds, error budget,
coverage, and provenance — without exposing internal B04 research
terminology.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from .halo_policy import HaloPolicy
from .pipeline import run_streaming
from .sinks import MetricOnlyTileSink
from .sources import TileSource
from .verification import (
    VerificationPlugin,
)


@dataclass(frozen=True)
class VerificationResult:
    """Public result from ``verify_layout()``.

    Exposes theorem-facing bounds and status without exposing internal
    research terminology.
    """

    status: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    continuous_epe_upper_nm: float | None
    hausdorff_upper_nm: float | None
    error_budget: dict[str, float]
    coverage: str
    inconclusive_tiles: int
    total_tiles: int
    model_provenance: str
    layout_hash: str | None = None
    certificate_hash: str | None = None
    n_refinements: int = 0


def verify_layout(
    source: TileSource,
    *,
    simulator: Callable[[torch.Tensor], torch.Tensor],
    core_size: int = 256,
    halo_policy: HaloPolicy | None = None,
    verifiers: Sequence[VerificationPlugin] = (),
    tolerance_nm: float = 3.0,
    max_halo_px: int = 1024,
    pixel_nm: float = 1.0,
    max_requeues: int = 4,
) -> VerificationResult:
    """Run proof-carrying verification on a full-chip layout.

    Uses the streaming core/halo pipeline.  No full-chip dense raster is
    materialized; peak memory is O(tile area + active batch).

    Returns a :class:`VerificationResult` with theorem-facing status and
    error bounds.
    """
    sink = MetricOnlyTileSink(source.shape)
    report = run_streaming(
        source,
        sink,
        simulator,
        core_size=core_size,
        halo_policy=halo_policy,
        verifiers=verifiers,
        max_halo_px=max_halo_px,
        pixel_nm=pixel_nm,
        max_requeues=max_requeues,
    )
    vres = report.verification
    worst_upper = vres.worst_upper_bound if vres else None
    return VerificationResult(
        status=vres.status if vres else "INCONCLUSIVE",
        continuous_epe_upper_nm=worst_upper,
        hausdorff_upper_nm=worst_upper,
        error_budget=vres.error_budget if vres else {},
        coverage=vres.coverage if vres else "INCONCLUSIVE",
        inconclusive_tiles=vres.n_inconclusive if vres else 0,
        total_tiles=vres.n_tiles if vres else 0,
        model_provenance=(
            f"streaming/{core_size}px/"
            f"{report.halo_requirement.halo_px if report.halo_requirement else '?'}px"
        ),
        n_refinements=report.n_requeued,
    )
