"""Proof gates for B04 continuous-focus verification.

The central firewall is intentionally asymmetric:
- certified upper bounds may prove PASS;
- FAIL requires an independently certified violation lower bound;
- everything else is INCONCLUSIVE.

P-054 repo integration adds two fail-closed gates:
- **capability ceiling**: a certifying-level dependency produced by a
  backend that declares ``DIAGNOSTIC_ONLY`` capability corrupts the proof
  package (hard error, not a warning);
- **model identity**: a new certificate must name its frozen model via
  :class:`~openlithohub.verify.model_identity.ModelIdentity`; the legacy
  replay path without one keeps the historical pinned commit and says so.
"""

from __future__ import annotations

from .model_identity import ModelIdentity
from .types import (
    CertificateStatus,
    CertificateTarget,
    CertificationCapability,
    ContinuousFocusCertificate,
    CoverageStatus,
    DependencyProvenance,
    DependencyRecord,
    ParameterInterval,
    ProofLevel,
)

# Legacy replay basis ( Increments 11–15 artifacts were produced at this
# commit).  New certificates must pass model_identity explicitly instead of
# silently inheriting this value.
LEGACY_PINNED_UPSTREAM = "348fa5d86d5355465af98e2c4ce3deac60081a4c"

_CERTIFYING_LEVELS = {
    ProofLevel.INTERVAL_CERTIFIED,
    ProofLevel.IMPORTED_QDM_CERTIFIED,
}

_CAPABILITY_CEILING = {
    CertificationCapability.DIAGNOSTIC_ONLY: ProofLevel.NUMERICAL_DIAGNOSTIC,
    CertificationCapability.RIGOROUS_INTERVAL: ProofLevel.INTERVAL_CERTIFIED,
    CertificationCapability.IMPORTED_FROZEN_CERTIFICATE: (ProofLevel.IMPORTED_QDM_CERTIFIED),
}


def dependency_chain_is_certified(dependencies: tuple[DependencyRecord, ...]) -> bool:
    return bool(dependencies) and all(
        d.satisfied and d.level in _CERTIFYING_LEVELS for d in dependencies
    )


def _enforce_capability_ceiling(dependencies: tuple[DependencyRecord, ...]) -> None:
    """Fail closed when a dependency overclaims its backend's capability."""
    for dep in dependencies:
        if dep.level not in _CERTIFYING_LEVELS:
            continue
        if dep.certification_capability is None:
            if dep.backend_id is not None:
                raise ValueError(
                    f"dependency {dep.name!r} declares backend {dep.backend_id!r} "
                    "without a certification capability; certifying-level "
                    "dependencies must state the producing backend's capability"
                )
            continue
        ceiling = _CAPABILITY_CEILING[dep.certification_capability]
        if _level_rank(dep.level) > _level_rank(ceiling):
            raise ValueError(
                f"dependency {dep.name!r} claims {dep.level.value} but its backend "
                f"capability {dep.certification_capability.value} ceilings at "
                f"{ceiling.value}; this corrupts the proof package"
            )


def _level_rank(level: ProofLevel) -> int:
    order = [
        ProofLevel.HEURISTIC,
        ProofLevel.NUMERICAL_DIAGNOSTIC,
        ProofLevel.INTERVAL_CERTIFIED,
        ProofLevel.IMPORTED_QDM_CERTIFIED,
    ]
    return order.index(level)


def _enforce_provenance(dependencies: tuple[DependencyRecord, ...], *, allow_legacy: bool) -> None:
    """Certifying-level dependencies may never be anonymous (PR-1B)."""
    for dep in dependencies:
        if dep.level not in _CERTIFYING_LEVELS:
            continue
        if allow_legacy:
            if dep.provenance not in (None, DependencyProvenance.LEGACY_REPLAY):
                raise ValueError(
                    f"dependency {dep.name!r}: only anonymous/LEGACY_REPLAY "
                    "provenance is admitted on the legacy replay path"
                )
            continue
        if dep.provenance is None:
            raise ValueError(
                f"dependency {dep.name!r} claims {dep.level.value} without "
                "provenance; a new certificate may not inherit authority by "
                "omission — declare BACKEND, IMPORTED_FROZEN provenance or "
                "use the explicit legacy replay API"
            )
        if dep.provenance is DependencyProvenance.BACKEND:
            if not dep.backend_id:
                raise ValueError(f"dependency {dep.name!r}: BACKEND provenance requires backend_id")
            if dep.certification_capability is None:
                raise ValueError(
                    f"dependency {dep.name!r}: BACKEND provenance requires certification_capability"
                )
        elif dep.provenance is DependencyProvenance.IMPORTED_FROZEN:
            if not dep.artifact_sha256:
                raise ValueError(
                    f"dependency {dep.name!r}: IMPORTED_FROZEN provenance requires artifact_sha256"
                )
        elif dep.provenance is DependencyProvenance.LEGACY_REPLAY:
            raise ValueError(
                f"dependency {dep.name!r}: LEGACY_REPLAY provenance is only "
                "admitted through replay_legacy_continuous_focus_certificate"
            )


def _finalize_status(
    *,
    target: CertificateTarget,
    coverage_status: CoverageStatus,
    tolerance_nm: float,
    continuous_focus_contour_upper_nm: float,
    certified_violation_lower_nm: float | None,
    dependencies: tuple[DependencyRecord, ...],
) -> CertificateStatus:
    deps_ok = dependency_chain_is_certified(dependencies)

    # Contradictory certified claims indicate a corrupt proof package, not a
    # choice between PASS and FAIL.
    if (
        deps_ok
        and certified_violation_lower_nm is not None
        and continuous_focus_contour_upper_nm <= tolerance_nm
        and certified_violation_lower_nm > tolerance_nm
    ):
        raise ValueError("certificate contains contradictory certified bounds")

    status = CertificateStatus.INCONCLUSIVE
    if (
        deps_ok
        and certified_violation_lower_nm is not None
        and certified_violation_lower_nm > tolerance_nm
    ):
        status = CertificateStatus.FAIL
    coverage_ok = (
        target is CertificateTarget.LEVEL_SET_STABILITY
        or coverage_status is CoverageStatus.ALL_COMPONENTS_COVERED
    )
    if (
        status is CertificateStatus.INCONCLUSIVE
        and deps_ok
        and coverage_ok
        and continuous_focus_contour_upper_nm <= tolerance_nm
    ):
        status = CertificateStatus.PASS
    return status


def assemble_continuous_focus_certificate(
    *,
    model_id: str,
    target: CertificateTarget = CertificateTarget.EXTRACTED_CONTOUR_EPE,
    input_sha256: str,
    parameters: tuple[ParameterInterval, ...],
    continuous_focus_contour_upper_nm: float,
    continuous_focus_pairwise_diameter_upper_nm: float | None,
    tolerance_nm: float,
    coverage_status: CoverageStatus,
    dependencies: tuple[DependencyRecord, ...],
    certified_violation_lower_nm: float | None = None,
    proof_artifact_sha256: str | None = None,
    nominal_reconstruction_upper_nm: float | None = None,
    model_identity: ModelIdentity,
    note: str = "",
) -> ContinuousFocusCertificate:
    """Assemble a NEW theorem-facing certificate.

    ``model_identity`` is mandatory: there is no API surface where omitting
    the frozen model silently changes what the certificate claims (PR-1B).
    """
    if continuous_focus_contour_upper_nm < 0.0:
        raise ValueError("continuous_focus_contour_upper_nm must be nonnegative")
    if tolerance_nm < 0.0:
        raise ValueError("tolerance_nm must be nonnegative")
    if certified_violation_lower_nm is not None and certified_violation_lower_nm < 0.0:
        raise ValueError("certified_violation_lower_nm must be nonnegative")

    _enforce_capability_ceiling(dependencies)
    _enforce_provenance(dependencies, allow_legacy=False)

    status = _finalize_status(
        target=target,
        coverage_status=coverage_status,
        tolerance_nm=tolerance_nm,
        continuous_focus_contour_upper_nm=continuous_focus_contour_upper_nm,
        certified_violation_lower_nm=certified_violation_lower_nm,
        dependencies=dependencies,
    )

    return ContinuousFocusCertificate(
        target=target,
        # A new certificate names its own frozen model — never the current
        # HEAD, never the legacy pin.
        upstream_commit=model_identity.implementation_commit,
        model_id=model_id,
        input_sha256=input_sha256,
        parameters=parameters,
        continuous_focus_contour_upper_nm=continuous_focus_contour_upper_nm,
        continuous_focus_pairwise_diameter_upper_nm=(continuous_focus_pairwise_diameter_upper_nm),
        tolerance_nm=tolerance_nm,
        coverage_status=coverage_status,
        dependencies=dependencies,
        status=status,
        certified_violation_lower_nm=certified_violation_lower_nm,
        proof_artifact_sha256=proof_artifact_sha256,
        nominal_reconstruction_upper_nm=nominal_reconstruction_upper_nm,
        model_identity=model_identity,
        note=note,
    )


def replay_legacy_continuous_focus_certificate(
    *,
    model_id: str,
    target: CertificateTarget = CertificateTarget.EXTRACTED_CONTOUR_EPE,
    input_sha256: str,
    parameters: tuple[ParameterInterval, ...],
    continuous_focus_contour_upper_nm: float,
    continuous_focus_pairwise_diameter_upper_nm: float | None,
    tolerance_nm: float,
    coverage_status: CoverageStatus,
    dependencies: tuple[DependencyRecord, ...],
    certified_violation_lower_nm: float | None = None,
    proof_artifact_sha256: str | None = None,
    nominal_reconstruction_upper_nm: float | None = None,
    note: str = "",
) -> ContinuousFocusCertificate:
    """Explicit historical replay path (Increments 11–15 artifacts).

    The ONLY place where a certificate may carry the legacy pinned commit
    without a ``ModelIdentity``.  The result is marked as a legacy replay
    in its note.
    """
    if continuous_focus_contour_upper_nm < 0.0:
        raise ValueError("continuous_focus_contour_upper_nm must be nonnegative")
    if tolerance_nm < 0.0:
        raise ValueError("tolerance_nm must be nonnegative")
    if certified_violation_lower_nm is not None and certified_violation_lower_nm < 0.0:
        raise ValueError("certified_violation_lower_nm must be nonnegative")

    _enforce_capability_ceiling(dependencies)
    _enforce_provenance(dependencies, allow_legacy=True)

    status = _finalize_status(
        target=target,
        coverage_status=coverage_status,
        tolerance_nm=tolerance_nm,
        continuous_focus_contour_upper_nm=continuous_focus_contour_upper_nm,
        certified_violation_lower_nm=certified_violation_lower_nm,
        dependencies=dependencies,
    )

    return ContinuousFocusCertificate(
        target=target,
        upstream_commit=LEGACY_PINNED_UPSTREAM,
        model_id=model_id,
        input_sha256=input_sha256,
        parameters=parameters,
        continuous_focus_contour_upper_nm=continuous_focus_contour_upper_nm,
        continuous_focus_pairwise_diameter_upper_nm=(continuous_focus_pairwise_diameter_upper_nm),
        tolerance_nm=tolerance_nm,
        coverage_status=coverage_status,
        dependencies=dependencies,
        status=status,
        certified_violation_lower_nm=certified_violation_lower_nm,
        proof_artifact_sha256=proof_artifact_sha256,
        nominal_reconstruction_upper_nm=nominal_reconstruction_upper_nm,
        model_identity=None,
        note=f"{note} legacy replay without model identity".strip(),
    )
