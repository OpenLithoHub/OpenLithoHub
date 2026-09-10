"""Streaming full-chip pipeline (RFC 0008, prompt §11).

Runs the per-tile loop::

    optional certified pre-forward screen
        -> TileSource.read_window(core+halo) only for survivors
        -> forward model
        -> optional VerificationPlugin(s)
        -> TileSink.write_core(trusted core only)
        -> discard tile tensors

Peak memory is O(tile area + active batch), never O(full-chip raster).
Every core is written exactly once; an inconclusive verdict re-runs the
same core with a larger halo (bounded by ``max_requeues``) before the
best-effort result is committed.

Increment 28 integrates work accounting into the actual pipeline and adds a
fail-closed screening hook.  Screening may skip expensive work only when its
decision is certified; with verification plugins present, the screen must also
explicitly certify those verifiers or the pipeline falls back to the ordinary
active path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from .core_halo import TileRequest, plan_tile_requests, tiling_overhead
from .geometry import BoundingBox, HaloSpec, halo_actual, read_bbox_for
from .halo_policy import HaloContext, HaloPolicy, HaloRequirement, LegacyFixedHaloPolicy
from .screening import TileScreeningPolicy
from .sinks import TileSink
from .sources import TileSource
from .verification import (
    RefinementRequest,
    StreamingVerificationReducer,
    TileContext,
    TileVerificationResult,
    VerificationContext,
    VerificationPlugin,
)
from .work_accounting import WorkAccounting


@dataclass
class StreamingRunReport:
    n_tiles: int = 0
    n_requeued: int = 0
    halo_requirement: HaloRequirement | None = None
    overhead: dict[str, float] = field(default_factory=dict)
    verification: Any = None
    work_accounting: dict[str, float | int] = field(default_factory=dict)


def _default_refinement(
    request: TileRequest, plugin: VerificationPlugin | None
) -> RefinementRequest:
    """Ask the plugin for refinement advice; fall back to a bigger halo."""
    refine = getattr(plugin, "refine", None)
    if callable(refine):
        advice: RefinementRequest | None = refine(request.tile_id)
        if advice is not None:
            return advice
    extra = max(request.halo.left, request.halo.top, request.halo.right, request.halo.bottom)
    return RefinementRequest(
        tile_id=request.tile_id,
        action="increase_halo",
        extra_halo_px=max(8, extra),
    )


def _grown_request(
    request: TileRequest, refinement: RefinementRequest, domain: tuple[int, int]
) -> TileRequest:
    """Re-plan a tile with more halo (clipped to the real global domain)."""
    max_px = max(1, refinement.extra_halo_px)
    current = max(request.halo.left, request.halo.top, request.halo.right, request.halo.bottom)
    grown = HaloSpec.uniform(current + max_px)
    read = read_bbox_for(
        request.core_bbox,
        grown,
        width=domain[1],
        height=domain[0],
    )
    return TileRequest(
        core_bbox=request.core_bbox,
        read_bbox=read,
        halo=halo_actual(request.core_bbox, read),
        tile_id=request.tile_id,
    )


def _core_slices(request: TileRequest) -> tuple[slice, slice]:
    """Location of the core inside the read-region tensor."""
    y0 = request.core_bbox.y0 - request.read_bbox.y0
    x0 = request.core_bbox.x0 - request.read_bbox.x0
    return (
        slice(y0, y0 + request.core_bbox.height),
        slice(x0, x0 + request.core_bbox.width),
    )


def run_streaming(
    source: TileSource,
    sink: TileSink,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    core_size: int,
    halo_policy: HaloPolicy | None = None,
    verifiers: Iterable[VerificationPlugin] = (),
    screening_policy: TileScreeningPolicy | None = None,
    work_accounting: WorkAccounting | None = None,
    max_halo_px: int = 1024,
    pixel_nm: float = 1.0,
    max_requeues: int = 4,
) -> StreamingRunReport:
    """Process a full chip tile-by-tile under core/halo ownership."""
    policy = halo_policy or LegacyFixedHaloPolicy()
    verifier_list = list(verifiers)

    vctx = VerificationContext(model=forward_fn, pixel_nm=pixel_nm)
    plugin_requirements = []
    for verifier in verifier_list:
        requirement = verifier.required_halo(vctx)
        if requirement is not None:
            plugin_requirements.append(requirement)

    physical = policy.required_halo(HaloContext(pixel_nm=pixel_nm, tile_core_px=core_size))
    if plugin_requirements:
        from .halo_policy import combine_requirements

        requirement = combine_requirements(physical, *plugin_requirements)
    else:
        requirement = physical
    halo_px = min(requirement.halo_px, max_halo_px)

    reducer = StreamingVerificationReducer()
    report = StreamingRunReport(halo_requirement=requirement)
    accounting = work_accounting or WorkAccounting()
    full_pixels = int(source.shape[0] * source.shape[1])
    if accounting.full_chip_pixels == 0:
        accounting.full_chip_pixels = full_pixels
    elif accounting.full_chip_pixels != full_pixels:
        raise ValueError("WorkAccounting.full_chip_pixels does not match the source domain")

    for verifier in verifier_list:
        verifier.prepare(vctx)

    for base in plan_tile_requests(source.shape, core_size, halo_px):
        current = base
        refinements_left = max_requeues
        while True:
            screen_decision = None
            if screening_policy is not None:
                accounting.record_screen_query()
                screen_decision = screening_policy.screen(source, current)

            can_skip = (
                screen_decision is not None
                and screen_decision.status == "SCREENED_OUT"
                and (not verifier_list or screen_decision.certifies_verifiers)
            )
            if can_skip and screen_decision is not None:
                if not screen_decision.certified or screen_decision.fill_value is None:
                    raise ValueError("uncertified screening decision attempted to skip work")

                core_result = torch.full(
                    (
                        current.core_bbox.height,
                        current.core_bbox.width,
                    ),
                    float(screen_decision.fill_value),
                    dtype=torch.float32,
                )
                tile_meta: dict[str, Any] = {
                    "screening_status": "SCREENED_OUT",
                    "screening_policy": getattr(
                        screening_policy, "name", type(screening_policy).__name__
                    ),
                    "screening_reason": screen_decision.reason,
                    "screening_certified": True,
                    "screening_certificate_ref": screen_decision.certificate_ref,
                    **screen_decision.metrics,
                }

                if verifier_list:
                    reducer.add(
                        TileVerificationResult(
                            tile_id=current.tile_id,
                            core_bbox=current.core_bbox,
                            status="PASS",
                            upper_bound=screen_decision.verification_upper_bound,
                            certificate_ref=screen_decision.certificate_ref,
                            metrics=dict(screen_decision.metrics),
                        )
                    )

                sink.write_core(
                    current.tile_id,
                    current.core_bbox,
                    core_result,
                    tile_meta,
                )
                accounting.record_screened_out(
                    current.tile_id,
                    current.core_bbox.area,
                )
                report.n_tiles += 1
                break

            accounting.record_active_core(current.tile_id, current.core_bbox)
            tile = source.read_window(current.read_bbox)
            accounting.record_read_window(current.read_bbox.area)

            source_meta: dict[str, Any] = {}
            metadata_fn = getattr(source, "verification_metadata", None)
            if callable(metadata_fn):
                source_meta = dict(
                    metadata_fn(
                        current.read_bbox,
                        tile_id=current.tile_id,
                        core_bbox=current.core_bbox,
                    )
                )
            if screen_decision is not None:
                source_meta["screening_status"] = screen_decision.status
                source_meta["screening_reason"] = screen_decision.reason

            result = forward_fn(tile)
            accounting.record_forward(current.read_bbox.area)
            ys, xs = _core_slices(current)
            core_result = result[ys, xs]

            # Type already declared on the screened-out path above.
            tile_meta = {}
            refinement: RefinementRequest | None = None
            for verifier in verifier_list:
                tctx = TileContext(
                    tile_id=current.tile_id,
                    core_bbox=current.core_bbox,
                    read_bbox=current.read_bbox,
                    halo=current.halo,
                    tensor=tile,
                    metadata=source_meta,
                )
                verdict = verifier.verify_tile(tctx)
                reducer.add(verdict)
                tile_meta[f"{verifier.name}_status"] = verdict.status
                if verdict.status == "INCONCLUSIVE" and refinement is None:
                    if refinements_left > 0:
                        refinement = _default_refinement(current, verifier)
                    else:
                        tile_meta[f"{verifier.name}_note"] = (
                            "inconclusive; refinement budget exhausted"
                        )

            if refinement is None:
                sink.write_core(
                    current.tile_id,
                    current.core_bbox,
                    core_result,
                    tile_meta,
                )
                report.n_tiles += 1
                break

            refinements_left -= 1
            report.n_requeued += 1
            accounting.record_refinement(
                current.tile_id,
                current.tile_id,
                current.core_bbox.area,
            )
            current = _grown_request(current, refinement, source.shape)
            del tile, result, core_result

    report.overhead = tiling_overhead(plan_tile_requests(source.shape, core_size, halo_px))
    if verifier_list:
        report.verification = reducer.finalize()
    report.work_accounting = accounting.summary()
    return report


__all__ = [
    "BoundingBox",
    "HaloSpec",
    "StreamingRunReport",
    "run_streaming",
]
