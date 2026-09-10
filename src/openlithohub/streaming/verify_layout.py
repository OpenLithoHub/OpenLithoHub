"""Public ``verify_layout()`` entry point (RFC 0008, philosophy §16).

Provides a single-call proof-carrying verification API that returns a typed
result with status, continuous EPE/Hausdorff upper bounds, error budget,
coverage, work-avoidance accounting, and provenance — without exposing
internal B04 research terminology.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch

from .halo_policy import HaloPolicy
from .pipeline import run_streaming
from .screening import TileScreeningPolicy
from .sinks import MetricOnlyTileSink
from .sources import TileSource
from .verification import (
    MultiVerifierSummary,
    VerificationPlugin,
    project_metric_bound,
)
from .work_accounting import WorkAccounting


@dataclass(frozen=True)
class VerificationResult:
    """Public result from ``verify_layout()``."""

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
    work_accounting: dict[str, float | int] = field(default_factory=dict)


def verify_layout(
    source: TileSource,
    *,
    simulator: Callable[[torch.Tensor], torch.Tensor],
    core_size: int = 256,
    halo_policy: HaloPolicy | None = None,
    verifiers: Sequence[VerificationPlugin] = (),
    screening_policy: TileScreeningPolicy | None = None,
    tolerance_nm: float = 3.0,
    max_halo_px: int = 1024,
    pixel_nm: float = 1.0,
    max_requeues: int = 4,
) -> VerificationResult:
    """Run proof-carrying verification on a full-chip layout.

    An optional fail-closed screening policy may prove cores irrelevant before
    raster-window materialization and forward simulation.  Screening is
    reflected in ``work_accounting``; the underlying pipeline refuses to skip
    plugin verification unless the screen explicitly certifies those
    verifiers.

    R17 C5: ``continuous_epe_upper_nm`` / ``hausdorff_upper_nm`` are typed
    projections of the unique verifier owning each (quantity, unit) metric —
    ``None`` when no verifier owns it, an explicit error when two do.  A
    generic worst-across-metrics bound is never exposed.
    """
    sink = MetricOnlyTileSink(source.shape)
    accounting = WorkAccounting(full_chip_pixels=int(source.shape[0] * source.shape[1]))
    report = run_streaming(
        source,
        sink,
        simulator,
        core_size=core_size,
        halo_policy=halo_policy,
        verifiers=verifiers,
        screening_policy=screening_policy,
        work_accounting=accounting,
        max_halo_px=max_halo_px,
        pixel_nm=pixel_nm,
        max_requeues=max_requeues,
        tolerance_nm=tolerance_nm if len(tuple(verifiers)) == 1 else None,
    )
    vres = report.verification
    multi = isinstance(vres, MultiVerifierSummary)
    # R17 C5: typed projections — the unique verifier owning each
    # (quantity, unit) metric; never a generic worst-across-metrics bound.
    descriptors = {
        identity: descriptor
        for identity, descriptor in report.metric_descriptors.items()
        if descriptor is not None
    }
    if not report.verification_results:
        epe_upper = None
        hausdorff_upper = None
    elif not descriptors:
        # no verifier declares a metric: legacy single-result projection
        # (a multi-verifier summary deliberately carries no numeric bound)
        own_bound = None if multi else (vres.worst_upper_bound if vres else None)
        epe_upper = own_bound
        hausdorff_upper = own_bound
    else:
        epe_upper = project_metric_bound(
            report.verification_results, descriptors, quantity="epe", unit="nm"
        )
        hausdorff_upper = project_metric_bound(
            report.verification_results, descriptors, quantity="hausdorff", unit="nm"
        )
    screen_name = (
        getattr(screening_policy, "name", "none") if screening_policy is not None else "none"
    )
    return VerificationResult(
        status=vres.status if vres else "INCONCLUSIVE",
        continuous_epe_upper_nm=epe_upper,
        hausdorff_upper_nm=hausdorff_upper,
        error_budget=vres.error_budget if (vres and not multi) else {},
        coverage=vres.coverage if (vres and not multi) else "INCONCLUSIVE",
        inconclusive_tiles=vres.n_inconclusive if vres else 0,
        total_tiles=report.n_tiles,
        model_provenance=(
            f"streaming/{core_size}px/"
            f"{report.halo_requirement.halo_px if report.halo_requirement else '?'}px/"
            f"screen={screen_name}"
        ),
        n_refinements=report.n_requeued,
        work_accounting=dict(report.work_accounting),
    )
