"""Typed proof-carrying verification records for B04 / RFC 0007.

This module deliberately keeps theorem-facing status separate from the
existing raster benchmark metrics.  A PASS is a statement about an explicit
certificate and its dependencies, not about foundry sign-off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .model_identity import ModelIdentity


class ProofLevel(str, Enum):
    HEURISTIC = "HEURISTIC"
    NUMERICAL_DIAGNOSTIC = "NUMERICAL-DIAGNOSTIC"
    INTERVAL_CERTIFIED = "INTERVAL-CERTIFIED"
    IMPORTED_QDM_CERTIFIED = "IMPORTED-QDM-CERTIFIED"


class CertificationCapability(str, Enum):
    """What a backend is *engineered* to certify (P-054 repo integration).

    Backend identity and capability identity are separate: a backend may be
    the canonical implementation of a model while still being unable to back
    a rigorous certificate.  Enforced, not documented.
    """

    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    RIGOROUS_INTERVAL = "RIGOROUS_INTERVAL"
    IMPORTED_FROZEN_CERTIFICATE = "IMPORTED_FROZEN_CERTIFICATE"


class DependencyProvenance(str, Enum):
    """Where a dependency's evidence comes from (P-054 re-audit PR-1B).

    A certifying-level dependency may never be anonymous: it either comes
    from a named backend (with declared capability), an imported frozen
    artifact (with content hash), or the explicitly-gated legacy replay
    path.
    """

    BACKEND = "BACKEND"
    IMPORTED_FROZEN = "IMPORTED_FROZEN"
    LEGACY_REPLAY = "LEGACY_REPLAY"


class CertificateTarget(str, Enum):
    LEVEL_SET_STABILITY = "LEVEL_SET_STABILITY"
    EXTRACTED_CONTOUR_EPE = "EXTRACTED_CONTOUR_EPE"
    CRITICAL_SET_STRATIFICATION = "CRITICAL_SET_STRATIFICATION"
    OWNERSHIP_STRATIFICATION = "OWNERSHIP_STRATIFICATION"
    FIXED_TARGET_TOPOLOGY = "FIXED_TARGET_TOPOLOGY"
    FOCUS_DOSE_CHAMBER = "FOCUS_DOSE_CHAMBER"


class CertificateStatus(str, Enum):
    PASS = "PASS"  # noqa: S105 — verification verdict, not a credential
    FAIL = "FAIL"  # noqa: S105
    INCONCLUSIVE = "INCONCLUSIVE"


class CoverageStatus(str, Enum):
    ALL_COMPONENTS_COVERED = "ALL_COMPONENTS_COVERED"
    PARTIAL_COVER = "PARTIAL_COVER"
    INCONCLUSIVE = "INCONCLUSIVE"


class BoundaryRootStatus(str, Enum):
    ROOT_BRACKETS = "ROOT_BRACKETS"
    ROOT_FREE_CERTIFIED = "ROOT_FREE_CERTIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"


class CellDisposition(str, Enum):
    SIGN_DEFINITE_EMPTY = "SIGN_DEFINITE_EMPTY"
    CURVATURE_CERTIFIED_EMPTY = "CURVATURE_CERTIFIED_EMPTY"
    SEEDED = "SEEDED"
    SUBDIVIDE = "SUBDIVIDE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class ParameterInterval:
    name: str
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"invalid interval {self.name}: {self.lo} > {self.hi}")


@dataclass(frozen=True)
class DependencyRecord:
    name: str
    level: ProofLevel
    satisfied: bool
    method: str
    artifact_sha256: str | None = None
    note: str = ""
    # R17/P-054 capability firewall: which backend produced this dependency
    # and what it is engineered to certify.  A certifying-level dependency
    # whose backend declares DIAGNOSTIC_ONLY capability corrupts the proof
    # package and fails closed in the assembler.
    backend_id: str | None = None
    certification_capability: CertificationCapability | None = None
    # P-054 re-audit PR-1B: certifying-level dependencies may never be
    # anonymous.  BACKEND requires backend_id + capability; IMPORTED_FROZEN
    # requires artifact_sha256; LEGACY_REPLAY is admitted only through the
    # explicit legacy replay API.
    provenance: DependencyProvenance | None = None


@dataclass(frozen=True)
class RootBracket:
    lo: float
    hi: float
    face: str | None = None

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("root bracket lo must not exceed hi")


@dataclass(frozen=True)
class CellCertificate:
    cell_id: str
    cell_bbox_nm: tuple[float, float, float, float]
    kappa_lower_per_nm: float
    hessian_upper_per_nm2: float
    cell_diameter_nm: float
    curvature_scale_nm: float
    boundary_root_status: BoundaryRootStatus
    hidden_loop_excluded: bool
    root_brackets: tuple[RootBracket, ...]
    eta_sp_nm: float | None
    disposition: CellDisposition


@dataclass(frozen=True)
class ContinuousFocusCertificate:
    target: CertificateTarget
    upstream_commit: str
    model_id: str
    input_sha256: str
    parameters: tuple[ParameterInterval, ...]
    continuous_focus_contour_upper_nm: float
    continuous_focus_pairwise_diameter_upper_nm: float | None
    tolerance_nm: float
    coverage_status: CoverageStatus
    dependencies: tuple[DependencyRecord, ...]
    status: CertificateStatus
    certified_violation_lower_nm: float | None = None
    proof_artifact_sha256: str | None = None
    nominal_reconstruction_upper_nm: float | None = None
    # P-054 repo integration: the frozen model this certificate names.  None
    # only on the explicit legacy replay path (which records that fact in
    # `note`).
    model_identity: ModelIdentity | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["target"] = self.target.value
        out["status"] = self.status.value
        out["coverage_status"] = self.coverage_status.value
        for dep, raw in zip(self.dependencies, out["dependencies"], strict=True):
            raw["level"] = dep.level.value
        return out
