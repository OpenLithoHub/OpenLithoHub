"""Opt-in theorem-facing verification API.

Nothing in this package changes OpenLithoHub's raster EPE/PV-band semantics.
"""

from .boundary import BoundaryIsolationResult, Interval, isolate_roots_on_segment
from .certifier import assemble_continuous_focus_certificate
from .coverage import replay_boundary_coverage, verify_boundary_artifact
from .mvp1 import certify_mvp1_manifest
from .replay import ExpandedBandReplay, replay_expanded_band, sha256_file
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
    "CoverageStatus",
    "DependencyRecord",
    "ExpandedBandReplay",
    "Interval",
    "ParameterInterval",
    "ProofLevel",
    "RootBracket",
    "assemble_continuous_focus_certificate",
    "certify_mvp1_manifest",
    "curvature_scale_nm",
    "replay_boundary_coverage",
    "verify_boundary_artifact",
    "replay_reconstruction_artifact",
    "verify_reconstruction_artifact",
    "hidden_loop_excluded",
    "isolate_roots_on_segment",
    "reduce_coverage",
    "replay_expanded_band",
    "sha256_file",
]
