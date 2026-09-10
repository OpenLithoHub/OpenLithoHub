"""Streaming full-chip pipeline (RFC 0008, prompt §11).

Runs the per-tile loop::

    TileSource.read_window(core+halo)
        → forward model
        → optional VerificationPlugin(s)
        → TileSink.write_core(trusted core only)
        → discard tile tensors

Peak memory is O(tile area + active batch), never O(full-chip raster).
Every core is written exactly once; an inconclusive verdict re-runs the
same core with a larger halo (bounded by ``max_requeues``) before the
best-effort result is committed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from .core_halo import TileRequest, plan_tile_requests, tiling_overhead
from .geometry import BoundingBox, HaloSpec, halo_actual, read_bbox_for
from .halo_policy import HaloContext, HaloPolicy, HaloRequirement, LegacyFixedHaloPolicy
from .sinks import TileSink
from .sources import TileSource
from .verification import (
    RefinementRequest,
    StreamingVerificationReducer,
    TileContext,
    VerificationContext,
    VerificationPlugin,
)


@dataclass
class StreamingRunReport:
    n_tiles: int = 0
    n_requeued: int = 0
    halo_requirement: HaloRequirement | None = None
    overhead: dict[str, float] = field(default_factory=dict)
    verification: Any = None


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
    for verifier in verifier_list:
        verifier.prepare(vctx)

    for base in plan_tile_requests(source.shape, core_size, halo_px):
        current = base
        refinements_left = max_requeues
        while True:
            tile = source.read_window(current.read_bbox)
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
            result = forward_fn(tile)
            ys, xs = _core_slices(current)
            core_result = result[ys, xs]

            tile_meta: dict[str, Any] = {}
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
                sink.write_core(current.tile_id, current.core_bbox, core_result, tile_meta)
                report.n_tiles += 1
                break

            refinements_left -= 1
            report.n_requeued += 1
            current = _grown_request(current, refinement, source.shape)
            del tile, result, core_result

    report.overhead = tiling_overhead(plan_tile_requests(source.shape, core_size, halo_px))
    if verifier_list:
        report.verification = reducer.finalize()
    return report


__all__ = [
    "BoundingBox",
    "HaloSpec",
    "StreamingRunReport",
    "run_streaming",
]
