"""Streaming core/halo full-chip architecture (RFC 0008).

Layout growth turns into tile count, not single-allocation size: the
pipeline reads only core+halo windows from a :class:`TileSource`, can apply
an optional certified pre-forward screen, runs the forward model on survivors,
lets optional :class:`VerificationPlugin`\\s inspect the tile, and commits only
the trusted core to a :class:`TileSink`.
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
from .screening import (
    ExactEmptyContextScreeningPolicy,
    TileScreenDecision,
    TileScreeningPolicy,
)
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
    MultiVerifierSummary,
    StreamingVerificationReducer,
    TileContext,
    TileVerificationResult,
    VerificationContext,
    VerificationPlugin,
    VerificationRegistry,
    VerifierIdentity,
    VerifierSession,
    verification_registry,
)
from .verify_layout import VerificationResult, verify_layout
from .work_accounting import WorkAccounting

__all__ = [
    "DEFAULT_HALO_PX",
    "BoundingBox",
    "ExactEmptyContextScreeningPolicy",
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
    "MultiVerifierSummary",
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
    "TileScreenDecision",
    "TileScreeningPolicy",
    "TileSink",
    "TileSource",
    "TileVerificationResult",
    "VerifierIdentity",
    "VerifierSession",
    "VerificationContext",
    "VerificationPlugin",
    "VerificationRegistry",
    "VerificationResult",
    "VectorLayoutTileSource",
    "WorkAccounting",
    "combine_requirements",
    "estimate_minimum_halo",
    "fold_global",
    "halo_actual",
    "halo_overhead_stats",
    "kernel_tail_mass",
    "make_verifier_session",
    "plan_tile_requests",
    "plan_tiling",
    "read_bbox_for",
    "run_streaming",
    "subdivide_request",
    "tiling_overhead",
    "verification_registry",
    "verify_layout",
]
