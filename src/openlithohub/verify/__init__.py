"""Opt-in theorem-facing verification API.

Nothing in this package changes OpenLithoHub's raster EPE/PV-band semantics.
"""

from .boundary import BoundaryIsolationResult, Interval, isolate_roots_on_segment
from .cellwise_fourier import (
    CellTransversalityTransfer,
    LayoutFourierEnvelope,
    LocalCellGeometry,
    local_cell_geometry,
    transfer_center_gradient_to_cell,
)
from .certifier import assemble_continuous_focus_certificate
from .coverage import replay_boundary_coverage, verify_boundary_artifact
from .derivative_runs import (
    CenterJetCertificate,
    ComplexIntervalVector,
    DerivativeRunPrefixArtifact,
    JetComponent,
    center_jet_certificate_from_json,
)
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
from .interface_runs import (
    CyclicRowPrefix,
    HorizontalRun,
    RunCompressionStats,
    RunSource,
    encode_binary_row_runs,
    known_layout_far_coherent,
    run_compression_stats,
    split_runs_outside_chebyshev,
    zero_padded_horizontal_variation,
)
from .mvp1 import certify_mvp1_manifest
from .reconstruction import replay_reconstruction_artifact, verify_reconstruction_artifact
from .replay import ExpandedBandReplay, replay_expanded_band, sha256_file
from .run_spectrum import (
    ComplexDiskGrid,
    RunSpectrumAccumulator,
    coherent_spectrum_from_layout,
)
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
from .streaming import CertificateSink, CoreWindow, LayoutRunSource, TileSource
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
    "CertificateSink",
    "CertificateStatus",
    "CertificateTarget",
    "CellTransversalityTransfer",
    "CenterJetCertificate",
    "ComplexDiskGrid",
    "ComplexIntervalVector",
    "CoreHaloGeometry",
    "CoreWindow",
    "CyclicRowPrefix",
    "ContinuousFocusCertificate",
    "CoverageStatus",
    "DependencyRecord",
    "DerivativeRunPrefixArtifact",
    "ExpandedBandReplay",
    "FieldEnclosure",
    "FinalHaloRequirement",
    "FullChipAggregate",
    "HaloStatus",
    "HaloTailPoint",
    "HorizontalRun",
    "Interval",
    "JetComponent",
    "LayoutFourierEnvelope",
    "LayoutRunSource",
    "LocalCellGeometry",
    "MinimalHaloBracket",
    "OutwardRoundedCPUBackend",
    "ParameterInterval",
    "ProcessBox",
    "ProofLevel",
    "RootBracket",
    "RunCompressionStats",
    "RunSource",
    "RunSpectrumAccumulator",
    "SOURCE_NATIVE_BACKEND_ID",
    "SOURCE_NATIVE_FORWARD_MODEL_ID",
    "SourceNativeVerificationBackend",
    "SourceSnapshot",
    "SpatialDerivativeEnclosure",
    "TileProofStatus",
    "TileSource",
    "aggregate_full_chip_status",
    "assemble_continuous_focus_certificate",
    "center_jet_certificate_from_json",
    "certify_mvp1_manifest",
    "coherent_spectrum_from_layout",
    "curvature_scale_nm",
    "dyadic_from_float",
    "encode_binary_row_runs",
    "freeze_source_snapshot",
    "halo_duplicate_compute_ratio",
    "hidden_loop_excluded",
    "isolate_roots_on_segment",
    "known_layout_far_coherent",
    "local_cell_geometry",
    "minimal_halo_bracket",
    "outward_round_interval",
    "replay_boundary_coverage",
    "replay_expanded_band",
    "replay_reconstruction_artifact",
    "reduce_coverage",
    "resolve_final_halo",
    "run_compression_stats",
    "sha256_file",
    "socs_absolute_tail_upper",
    "split_runs_outside_chebyshev",
    "transfer_center_gradient_to_cell",
    "unrestricted_binary_tail_lower",
    "verify_boundary_artifact",
    "verify_reconstruction_artifact",
    "zero_padded_horizontal_variation",
]
