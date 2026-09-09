"""Opt-in theorem-facing verification API.

Nothing in this package changes OpenLithoHub's raster EPE/PV-band semantics.
"""

from .boundary import BoundaryIsolationResult, Interval, isolate_roots_on_segment
from .certifier import assemble_continuous_focus_certificate
from .coverage import replay_boundary_coverage, verify_boundary_artifact
from .full_chip import FullChipAggregate, TileProofStatus, aggregate_full_chip_status
from .halo import (
    CoreHaloGeometry,
    FinalHaloRequirement,
    HaloStatus,
    HaloTailPoint,
    MinimalHaloBracket,
    halo_duplicate_compute_ratio,
    minimal_halo_bracket,
    resolve_final_halo,
    socs_absolute_tail_upper,
    unrestricted_binary_tail_lower,
)
from .mvp1 import certify_mvp1_manifest
from .replay import ExpandedBandReplay, replay_expanded_band, sha256_file
from .source_native import (
    BACKEND_ID as SOURCE_NATIVE_BACKEND_ID,
)
from .source_native import (
    FieldEnclosure,
    OutwardRoundedCPUBackend,
    ProcessBox,
    SourceNativeVerificationBackend,
    SpatialDerivativeEnclosure,
)
from .source_snapshot import (
    FORWARD_MODEL_ID as SOURCE_NATIVE_FORWARD_MODEL_ID,
)
from .source_snapshot import (
    SourceSnapshot,
    dyadic_from_float,
    freeze_source_snapshot,
    outward_round_interval,
)
from .spatial import curvature_scale_nm, hidden_loop_excluded, reduce_coverage
from .types import (
    BoundaryRootStatus,
    CellCertificate,
    CellDisposition,
    CertificateStatus,
    CertificateTarget,
    ContinuousFocusCertificate,
    CoverageStatus,
    DependencyRecord,
    ParameterInterval,
    ProofLevel,
    RootBracket,
)

__all__ = [
    "BoundaryIsolationResult",
    "BoundaryRootStatus",
    "CellCertificate",
    "CellDisposition",
    "CertificateStatus",
    "CertificateTarget",
    "ContinuousFocusCertificate",
    "CoreHaloGeometry",
    "CoverageStatus",
    "DependencyRecord",
    "ExpandedBandReplay",
    "FieldEnclosure",
    "FinalHaloRequirement",
    "FullChipAggregate",
    "HaloStatus",
    "HaloTailPoint",
    "Interval",
    "MinimalHaloBracket",
    "OutwardRoundedCPUBackend",
    "ParameterInterval",
    "ProcessBox",
    "ProofLevel",
    "RootBracket",
    "SOURCE_NATIVE_BACKEND_ID",
    "SOURCE_NATIVE_FORWARD_MODEL_ID",
    "SourceNativeVerificationBackend",
    "SourceSnapshot",
    "SpatialDerivativeEnclosure",
    "TileProofStatus",
    "aggregate_full_chip_status",
    "assemble_continuous_focus_certificate",
    "certify_mvp1_manifest",
    "curvature_scale_nm",
    "dyadic_from_float",
    "freeze_source_snapshot",
    "halo_duplicate_compute_ratio",
    "hidden_loop_excluded",
    "isolate_roots_on_segment",
    "minimal_halo_bracket",
    "outward_round_interval",
    "replay_boundary_coverage",
    "replay_expanded_band",
    "replay_reconstruction_artifact",
    "reduce_coverage",
    "resolve_final_halo",
    "sha256_file",
    "socs_absolute_tail_upper",
    "unrestricted_binary_tail_lower",
    "verify_boundary_artifact",
    "verify_reconstruction_artifact",
]
