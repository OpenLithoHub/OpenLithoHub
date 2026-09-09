"""Streaming core/halo full-chip architecture (RFC 0008).

Layout growth turns into tile count, not single-allocation size: the
pipeline reads only core+halo windows from a :class:`TileSource`, runs the
forward model, lets optional :class:`VerificationPlugin`\\s inspect the
tile, and commits only the trusted core to a :class:`TileSink`.
"""

from .core_halo import (
    RefinementRequest,
    TileRequest,
    TileScheduler,
    plan_tile_requests,
    plan_tiling,
    subdivide_request,
    tiling_overhead,
)
from .geometry import (
    BoundingBox,
    HaloSpec,
    halo_actual,
    halo_overhead_stats,
    read_bbox_for,
)
from .halo_policy import (
    DEFAULT_HALO_PX,
    HaloContext,
    HaloPolicy,
    HaloRequirement,
    HaloStatus,
    KernelTailHaloPolicy,
    LegacyFixedHaloPolicy,
    PhysicalInteractionHaloPolicy,
    combine_requirements,
    estimate_minimum_halo,
    kernel_tail_mass,
)
from .pipeline import StreamingRunReport, run_streaming
from .sinks import MemmapTileSink, MetricOnlyTileSink, TensorTileSink, TileSink
from .sources import (
    MemmapTensorTileSource,
    SpatialLayoutIndex,
    TensorTileSource,
    TileSource,
    VectorLayoutTileSource,
)
from .verification import (
    GlobalVerificationResult,
    StreamingVerificationReducer,
    TileContext,
    TileVerificationResult,
    VerificationContext,
    VerificationPlugin,
    VerificationRegistry,
    verification_registry,
)

__all__ = [
    "DEFAULT_HALO_PX",
    "BoundingBox",
    "GlobalVerificationResult",
    "HaloContext",
    "HaloPolicy",
    "HaloRequirement",
    "HaloSpec",
    "HaloStatus",
    "KernelTailHaloPolicy",
    "LegacyFixedHaloPolicy",
    "MemmapTensorTileSource",
    "MemmapTileSink",
    "MetricOnlyTileSink",
    "PhysicalInteractionHaloPolicy",
    "RefinementRequest",
    "SpatialLayoutIndex",
    "StreamingRunReport",
    "StreamingVerificationReducer",
    "TensorTileSink",
    "TensorTileSource",
    "TileContext",
    "TileRequest",
    "TileScheduler",
    "TileSink",
    "TileSource",
    "TileVerificationResult",
    "VerificationContext",
    "VerificationPlugin",
    "VerificationRegistry",
    "VectorLayoutTileSource",
    "combine_requirements",
    "estimate_minimum_halo",
    "halo_actual",
    "halo_overhead_stats",
    "kernel_tail_mass",
    "plan_tile_requests",
    "plan_tiling",
    "read_bbox_for",
    "run_streaming",
    "subdivide_request",
    "tiling_overhead",
    "verification_registry",
]
